# JAX at scale on TPUs, outside of Google 

The "outside of Google" part here is very important. It seems like a bunch of the pointy bits I'm about to outline here have been abstracted away inside of Google by tools like XManager and are probably irrelevant. 

### Setup

Firstly, creating TPUs: 

`gcloud alpha compute tpus tpu-vm create helen-pod --zone=us-central2-b --accelerator-type=v4-512 --version=tpu-vm-v4-base --subnetwork=helen`

Check that you can SSH into the host: 

`gcloud alpha compute tpus tpu-vm ssh helen-pod --zone=us-central2-b`

This will land you on the TPU VM, where you have root access and can install whatever dependencies you want. If you're running a single TPU (e.g. a v4-8), you can launch training jobs from here and they will just run on the device. Easy! 

However, it gets a bit trickier if you want to run JAX code on pods.
The SIMD (single instruction, multiple data) setup of JAX means that you must run _the same code everywhere_ and a JAX process on every host. This is why there's a lot of code in t5x which goes like `if jax.process_index() == 0`; there's some code which shouldn't be run multiple times, like checkpointing and making directories. It's also why there is a ridiculous amount of duplicated logs which come out when you run code on all the hosts.

The point, though: running the same code on host requires you to actually copy your code onto every host. If you SSH into the TPU-VM for a pod and just try to launch the training job from there without deploying your code to all hosts then it won't work (even though it works in the case of a v4-8). This is pretty annoying, but luckily there's a gcloud command for running a command on all the hosts in a pod: 

`gcloud compute tpus tpu-vm ssh helen-pod   --zone us-central2-b   --worker=all   --command="whatever normal bash command goes here"`

We'll use this to install JAX on the whole pod: 

```
gcloud compute tpus tpu-vm ssh helen-pod \
  --zone us-central2-b \
  --worker=all \
  --command="pip install 'jax[tpu]>=0.2.16' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
```

You can check that everything went well by running:

`gcloud compute tpus tpu-vm ssh helen-pod   --zone us-central2-b   --worker=all   --command="python3 -c 'import jax; print(jax.device_count())'"`

`device_count` returns the total number of devices across the pod, which means the number of chips (not cores!) — so on a v4-512, you'd expect this to be 256. 

You might want to install other dependencies for whatever reason onto all hosts (e.g. Python libraries for downloading and processing The Pile to TFRecords). The most sensical thing to do is to write a bash setup script which does this in one shot and run that setup script on all of your hosts like above, but if you just want to get up and running quickly, you can toss it into a setup script and then do something like 

`gcloud compute tpus tpu-vm ssh helen-pod   --zone us-central2-b   --worker=all   --command="git clone --branch=hn-tpu https://github.com/mathemakitten/t5x && chmod +x t5x/setup.sh && ./t5x/setup.sh"
`

Then run your code with a run script like:

`gcloud compute tpus tpu-vm ssh helen-pod   --zone us-central2-b   --worker=all   --command="cd t5x && git pull && cd .. && chmod +x t5x/run.sh && ./t5x/run.sh"
`

This is obviously suboptimal, but works in a pinch if it's two o'clock in the morning and you have temporary access to a large pod slice. In all other settings you should write a setup script.

### Performance tuning

Pod slices come in different shapes. If you didn't define the shape by passing the `--topology=4x4x16` flag during TPU creation, you can find out what shape a pod slice is by looking at the logs for `train.py`. A good rule of thumb is that the default shape is blob-like as opposed to deformed; e.g. for 512 chips / 256 cores, the default topology is 4x8x8 (and you would never have something wacky like 64x2x2).

The correct setting for `model_parallel_submesh` is critically important for performance. You should firstly figure out the minimal number of chips which are required to shard your model over and then tune from there (e.g. T5-11B fits on 8x40GB A100s, or a model-parallel submesh of (4, 2, 2, 1), where we need slightly more chips in the TPU case because TPU chips have slightly less HBM than A100s). Critically, the order of dimensions on the mesh is _not_ symmetric. On a v4-512 (4x8x8 topology), swapping the model parallel submesh from `(2, 2, 4, 1)` to `(4, 2, 2, 1)` resulted in a speedup of **35k tokens per second** while all else was held constant, because the first model submesh dimension matched up better with the pod topology, which meant that sharded tensors had to be communicated 
"closer", speeding up comms by _a lot_! 

Empirically, I found that you could push throughput during training by setting a larger global batch size with more microbatches, and this often increased throughput (in terms of tokens/second) over just increasing the batch size. Hooray for super fast interconnects. The best setup here is able to do **321k tokens per second**. 

| HARDWARE    | SIZE | SEQ_LEN | TOPOLOGY | TP           | GBS  | N_MICROBATCHES | TOKENS/SEC                                             | SAMPLES/SEC |
| ----------- | ---- | ------- | -------- | ------------ | ---- | -------------- | ------------------------------------------------------ | ----------- |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (2,4,4,1)    | 384  | 0              | 198053.9916                                            | 386.824202  |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (2,4,4,1)    | 512  | 0              | 193779.1386                                            | 378.47488   |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (2,4,4,1)    | 576  | 0              | 194367.2036                                            | 379.623444  |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (2,4,4,1)    | 640  | 0              | OOM                                                    |             |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (2,4,4,1)    | 576  | 2              | 203656.4476                                            | 397.766499  |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (2,4,4,1)    | 768  | 2              | 202282.7193                                            | 395.083436  |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (2, 4, 2, 1) | 896  | 16             | 233918.7726                                            |             |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (2, 4, 2, 1) | 896  | 8              | 288800.8486                                            | 564.064157  |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (2, 4, 2, 1) | 896  | 4              | 299727.8941                                            | 585.406043  |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (4, 2, 2, 1) | 896  | 4              | 304393.737                                             | 620.9197998 |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (4, 2, 2, 1) | 1344 | 6              | 311404.3091                                            | 608.211541  |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (4, 2, 2, 1) | 1792 | 8              | 314970.7319                                            | 615.177211  |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (4, 2, 2, 1) | 3584 | 16             | 320329.8307                                            | 625.644201  |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (4, 2, 2, 1) | 5376 | 24             | 321811.682                                             | 628.538441  |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (4, 2, 2, 1) | 928  | 8              | 261914.9138                                            | 511.552566  |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (4, 2, 2, 1) | 928  | 4              | 291203.4057                                            | 568.756652  |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (2, 2, 4, 1) | 896  | 4              | 286131.2504                                            | 558.850098  |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (2, 4, 2, 1) | 928  | 4              | 286839.3706                                            | 560.233146  |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (2, 4, 2, 1) | 896  | 2              | OOM                                                    |             |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (2, 2, 2, 1) | 896  | 4              | OOM                                                    |             |
| TPU v4-512  | 11b  | 512     | 4x8x8    | (2, 2, 2, 1) | 864  | 6              | slower than expected and also it OOMs in a weird place |             |                                               |             |
| TPU v4-1024 | 68b  | 512     | 8x8x8    | (4,4,2,1)    | 16   | 0              | 17476.3                                                |             |
| TPU v4-1024 | 68b  | 1024    | 8x8x8    | (8,4,2,1)    | 16   | 0              | 26495.7                                                |             |
| TPU v4-1024 | 68b  | 1024    | 8x8x8    | (4, 4, 4, 1) | 24   | 0              |                                                        |             |
| TPU v4-1024 | 68b  | 1024    | 8x8x8    | (8,4,2,1)    | 24   | 0              | 26279.4                                                |             |
| TPU v4-1024 | 68b  | 1024    | 8x8x8    | (8,4,2,1)    | 48   | 8              | 19787.7                                                |             |
| TPU v4-1024 | 68b  | 1024    | 8x8x8    | (8,4,2,1)    | 64   | 8              | 27040.3                                                |             |
| TPU v4-1024 | 68b  | 1024    | 8x8x8    | (8,4,2,1)    | 192  | 8              | OOM                                                    |             |
| TPU v4-1024 | 68b  | 1024    | 8x8x8    | (8,4,2,1)    | 32   | 0              | OOM   

There are more good notes on performance tuning here, but really, the only way to find out what works best is to try them out.

**A quick, opinionated note on benchmarking**: if you're comparing performance of TPUs vs. GPUs across frameworks and clusters, **use tokens per second instead of FLOPS**! Over-indexing on FLOPS to gauge overall performance is a lossy proxy as best. In the large model training regime you're often not FLOPS-bound but comms-bound or memory-bound, and the different axes you can optimize for on different systems on both the hardware and software level (including data pipelining!) means that the only meaningful comparison is _how many tokens can I push through the model per second_, i.e. _how much faster will my model train_. Big Chip tells you that you can do more matmuls faster, but what if you're not bottlenecked on matmuls, but on moving data between device and host? What if you have a really large model and need to do rematerialization on the backward pass and then you artificially have more matmuls to do? At the end of the day, if you're reading this, your job is likely to _train large models faster_, not to get the highest device utilization, and you should employ all the software and hardware tricks you have to get there, and directly measure the model training speedup—tokens per second tells you directly _how much faster will my model train_, but FLOPS doesn't. Whole system performance matters!

### Outstanding notes

I didn't figure out how to gracefully kill a job running t5x in the time when I was working on this! If you're benchmarking or performance tuning then you can, of course, do something silly like turn off checkpointing everywhere and only run for a few hundred steps to make sure everything is as-expected before kicking off an actual run. Otherwise you will have zombie processes running on the TPU and won't be able to kick off another job and have to shut down the TPU and re-create it.  Don't kill the main TPU process by its pid; I did that by accident and had to nerf the entire pod and start over. 

Logging is also strange: there's a *lot* of extra logs, one copy streaming from all the hosts. I figure that this doesn't bug people inside Google because some part of their infrastructure automagically collapses the log stream into a single readable one, but if you're on the outside, you should update your terminal scrollback to be infinite, and the logger should be changed to only print when `process_index = 0` if it's a duplicate. 

Lastly, there's a bit of commentary in `t5x/partitioning.py` which I didn't fully understand:

```python3
model_parallel_submesh: a HardwareMesh spec, namely (x,y,z,core) on TPU for
  a single model-parallel replica's "tile" in the physical device mesh. The
  first three elements (`x`, `y`, and `z`) should be factors of the pod
  slice; e.g., if you are using df_4x8, then `x` should be a factor of 4
  (one of 1, 2, 4), `y` should be a factor of 8 (one of 1, 2, 4, 8), and `z`
  must be 1, because TPU v3 slices are only 2D. `z` can be >1 for TPU v4
  (and maybe later TPUs) that allow 3D slices. `core` is the number of cores
  to use from each TPU node. As communication is usually fastest inside the
  same node, if you need a tile of more than 1 core, then
  you should first increase `core`: e.g., for TPU v3, (1,1,1,2) is better
    than (2,1,1,1). To pick a good spec, try a few possible values until you
    get high TPU utilization.
```

What does this mean? What is _core_ ("cores to use from each TPU node") referring to here and why would you ever want to set it to be > 1? I couldn't ever get a run working with the core dimension > 1 (it would error out) and I still can't figure out when you would wan this. Maybe someone with a bigger brain could tell me?
