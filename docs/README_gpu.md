# JAX and t5x on GPUs

As of February 2, 2023.

**TLDR**: You maybe don't actually want to run JAX at scale on GPUs unless you have a very fast cluster (and if you have money to have a fast enough cluster to do this and you want to write JAX, you're better off using TPUs). It runs almost right out of the box, but it is limited by the fact that sharding across GPU nodes is limited by the interconnect of your cluster. t5x does not implement pipeline parallelism because it was designed to run on TPUs, and the TPU interconnect is so fast that it doesn't _need_ pipeline parallelism. Unfortunately, on GPUs, being limited to only data and model parallelism means it is out-performed by tools like NVIDIA's Megatron unless your internode communications speed is very fast, since Megatron implements 3D parallelism (pipeline, model, data).  NVIDIA is apparently looking to implement 3D parallelism in t5x and so am I, but who knows when that will happen, or whether it will close the gap. If you're looking to do work on GPUs at scale you are likely better off with something like NVIDIA's Megatron-Nemo, but JAX within a node is fine. 

The t5x partitioning logic requires that the number of model partitions on GPU must be a factor or multiple of the number of local devices on a node (see `get_gpu_mesh` in `t5x/partitioning.py` for the details). TODO: write about some of that non-trivially-opaque hybrid mesh logic.

I ran t5x on GPUs with two setups: on Google Cloud with a single node, and scaled up to 24 nodes on AWS with slurm. 

### Installation

* If you're on Google Cloud, the Deep Learning VM (Tensorflow version) is OK to use. The major upside is that it installs cuDNN and CUDA for you, which is still somehow one of the most painful things to do in 2023. You do **not** need nvidia-docker despite the fact that NVIDIA contributed examples to the repository which used nvidia-docker. It works perfectly fine without Docker. (As much as I love Docker in theory, in practice it adds unnecessary complexity to most machine learning workflows.)
* The easiest way to solve most early permissions problems is to `sudo chown -R $USER /opt/conda/` 
* `gcloud auth login`
* `conda create -n t5x python=3.8.10`
* `pip install -e .`
* If you want to run model training on The Pile, you can also do `python3 -m pip install -e '.[pile]'` to install dependencies which  install and turn The Pile into TFRecords very quickly.
* Install the CUDA-friendly version of jaxlib (different than the version in setup.py): `pip install jaxlib==0.4.1+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
* If you get a 403 Forbidden on the storage bucket, do `gcloud auth login --update-adc` to add the service account to the defaults.

### Some small single-node model training runs 

Run these examples to make sure that everything works as expected.

Ensure to set the environment variables `MODEL_DIR="gs://..."` and `TFDS_DATA_DIR=gs://...`

**Decoder only**

```
python t5x/train.py --gin_file="t5x/examples/decoder_only/examples/base_wmt_from_scratch.gin" --gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR}
```

This will train a model on the pretraining mix of `'europarl_v7', 'commoncrawl', 'newscommentary_v13'`, validate on `'newstest2013'` and test on `'newstest2014'`.

For a decoder (134307072 parameters) which you can fit on a single T4 for debugging purposes:
```
python t5x/train.py --gin_file="t5x/examples/decoder_only/examples/tiny_gpu_single.gin" --gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR}
```

For the same decoder (134307072 parameters) with a larger batch size to scale up on 4 GPUs:

```
python t5x/train.py --gin_file="t5x/examples/decoder_only/examples/tiny_gpu_4x.gin" --gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR}
```

### CUDA shenanigans

As with any GPU cluster, you might run into CUDA issues. Using nvidia-docker would likely solve some of them, but if you are like me and do not want to use Docker, this is how I got up and running on the Deep Learning VM:

* For JAX on GPUs, CUDA 11.4 or newer is required so you will likely have to upgrade it manually
  * NVIDIA instructions [here](https://developer.nvidia.com/cuda-11-4-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=10&target_type=deb_local)
  * `nvcc --version`; `Build cuda_11.4` definitely works.
* Download JAX GPU versions from [here](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html), *not* [here](https://storage.googleapis.com/jax-releases/jax_releases.html) as linked in the original T5X documentation.
* CUDA is probably at `/usr/local/cuda/lib64`, and `jaxlib==0.4.1+cuda11.cudnn86` works without issue.

### XLA flags for GPU
* [NVIDIA](https://github.com/google-research/t5x/pull/952) suggests setting XLA debugging flags with `export XLA_FLAGS='--xla_gpu_simplify_all_fp_conversions --xla_gpu_all_reduce_combine_threshold_bytes=136314880 ${XLA_FLAGS}'`
* As per [XLA documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla.proto), `xla_gpu_simplify_all_fp_conversions` needs to be set because it "allows all floating-point conversions to be simplified, including those that affect the numerics. The `BFloat16Normalization` pass inserts many `f32 -> bf16 -> f32` conversion pairs. These are not removed by the `AlgebraicSimplifier`, as that will only simplify conversions that are no-ops, e.g. `bf16 -> f32 -> bf16`. Removing these improves accuracy."
* In practice, I did not ablate to understand if they made a difference. It doesn't hurt to set them anyway.

### Running t5x on Slurm

On Slurm, you can call `jax.distributed.initialize()` with no arguments and everything just works. See [here](https://jax.readthedocs.io/en/latest/multi_process.html). 

From the JAX docs: `When running on GPUs with Slurm, it is assumed that one process is started per GPU, i.e. each process will be assigned only one visible local device.`

If you're running Slurm I assume that you're likely not running on Google Cloud (otherwise you'd be using Kubernetes!), which means you are likely running RHEL, which means you likely have to modify the environment variable which Tensorflow uses to look for certificates.

See an example of a Slurm submission script in `scripts/train.slurm`.

### Gotchas

Within a node the performance is about the same as Megatron-LM if not a little bit faster. However, the partitioning is tricky to scale across nodes. TODO: drop-in details.

Additionally, I had some issues with S3 and Tensorflow's GFile which I couldn't figure out in the span of two hours in a single night (they looked like authentication issues which I wasn't having on Google Cloud), and given that I was on a tight deadline, I ended up writing data to local storage on the cluster instead. I wouldn't recommend this on shared environments with shared data since random read/writes across a lot of people could slow down your cluster.

### Miscellaneous thoughts
* What's commonly referred to as "tensor parallelism" in the NVIDIA/Pytorch ecosystem is "model parallelism" here. If you're searching the codebase, search for "model parallel".
* `model_parallel_submesh` and `num_partitions` arguments are mutually-exclusive methods for partitioning! See partitioning.md for more details.
* Easiest way to get up and running on a single GPU for debugging is setting `num_partitions = 1` in `partitioning.PjitPartitioner`, which allows you to instantiate the model on a single GPU. You can easily scale this to two GPUs by setting `num_partitions = 1`. 
* Gin configs are great, unless you chain them together by overwriting them several times and lose track of where an attribute gets changed. This is probably the one downside of the gin `include someotherconfig.gin` magic.


### Useful notes if you're coming from PyTorch
* There's no equivalent tool to SeqIO in the Pytorch ecosystem, but luckily, SeqIO works out-of-the-box! You can do underrated things like checkpoint the data pipeline iterator to restore midway through a run, or upsample specific parts of your pretraining mix. You should probably replace whatever data iteration tool you're using with SeqIO right away.
* Model checkpointing in full precision across workers is already implemented, and reshaping them to be restored on a different topology is super easy.