## Log 

### things built or extended 

* [ ] Assert that `loss_per_nonpadding_target_token` makes sense given the vocabulary size (e.g. `ln(30000)` ~ 10.3); this checks if the model was correctly instantiated

### things i like about t5x and SeqIO
* Being able to checkpoint the data pipeline iterator to restore midway through a run is already implemented in SeqIO
* Can control epochs, not only number of training steps (big downside to the usual tf.data style)
* gin configurations!
* Support for cloud storage by default and the insanely great interoperability of `tf.io.gfile` (compared to the Megatron format of storing data as single giant binary files) (also it even supports S3??? wtf)
* Sharded TFRecords (compare to Megatron binary data format)
* Distributed data loading way better than PyTorch. SeqIO is the best data library for machine learning right now by far
* Model checkpointing across workers already implemented, and (reshaping them???)
* Model checkpoints save in full precision by default (Megatron saving weights in fp16 instead of bf16 or fp32 is absolutely fucked)
* So much easier to setup than Megatron-Deepspeed (though that might be a byproduct of working on LUMI)

### things i do not like 
* The stylistic choice to include, inherit and overwrite gin configs makes it sometimes confusing to figure out where your hyperparameters come from, and it's easy to fuck up if you don't replace things everywhere you should be replacing them
* Lack of ready-to-run GPU scripts (this is WIP by NVIDIA though it seems) and unclear install instructions for GPU (NVIDIA uses nvidia-docker but it appears that you might not actually need it; I assume there are optimizations in nvidia-docker which I'm not aware of)
* Unclear if GPU multi-node performance will suffer (t5x does not implement pipeline parallelism because pod interconnect is so good it's not necessary)
* The flags could really use more documentation. It's not clear in what situation I would want to use `trainer.stop_training`, for example.

### outstanding questions
* Does the usual profiler work for GPUs? 
* Does multinode performance suffer? 
* SLURM??? lmao help me god