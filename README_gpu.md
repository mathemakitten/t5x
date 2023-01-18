# notes2self for GPU 

As of January 4, 2023.

### Installation 

* Deep Learning VM is OK to use (it installs cuDNN and CUDA; you do **not** need nvidia-docker), but don't forget to `sudo chown -R $USER /opt/conda/` 
* `gcloud auth login`
* `conda create -n t5x python=3.8.10`
* `pip install -e .`
* `pip install jaxlib==0.4.1+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
* If you get a 403 Forbidden on the storage bucket, do `gcloud auth login --update-adc` to add the service account

### Test training 

Ensure to set `MODEL_DIR="gs://..."` and `TFDS_DATA_DIR=gs://...`

**Decoder only**

```
python t5x/train.py --gin_file="t5x/examples/decoder_only/examples/base_wmt_from_scratch.gin" --gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR}
```

This will train a model on the pretraining mix of `'europarl_v7', 'commoncrawl', 'newscommentary_v13'`, validate on `'newstest2013'` and test on `'newstest2014'`.

For a decoder (134307072 parameters) which you can fit on a single T4 for debugging purposes:
```
python t5x/train.py --gin_file="t5x/examples/decoder_only/examples/tiny_gpu_single.gin" --gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR}
```

TODO(helen): figure out why the loss is so high (1477)

For the same decoder (134307072 parameters) with a larger batch size to scale up on 4 GPUs:

```
python t5x/train.py --gin_file="t5x/examples/decoder_only/examples/tiny_gpu_4x.gin" --gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR}
```

**T5** (have not tested on GPU yet)

```
python t5x/train.py --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin" --gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR}` 
```

### CUDA shenanigans
* `nvcc --version`; `Build cuda_11.4` works for sure
* For JAX on GPUs CUDA 11.4 or newer is required so you will likely have to upgrade it manually
  * NVIDIA instructions [here](https://developer.nvidia.com/cuda-11-4-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=10&target_type=deb_local)
* Download JAX GPU versions from [here](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html), *not* [here](https://storage.googleapis.com/jax-releases/jax_releases.html) as linked in the T5X documentation. 
* CUDA is probably at `/usr/local/cuda/lib64`, and `jaxlib==0.4.1+cuda11.cudnn86` works.

### XLA flags 
* [NVIDIA](https://github.com/google-research/t5x/pull/952) suggests setting XLA debugging flags with `export XLA_FLAGS='--xla_gpu_simplify_all_fp_conversions --xla_gpu_all_reduce_combine_threshold_bytes=136314880 ${XLA_FLAGS}'`
* As per [XLA documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla.proto), `xla_gpu_simplify_all_fp_conversions` needs to be set because it "allows all floating-point conversions to be simplified, including those that affect the numerics. The `BFloat16Normalization` pass inserts many `f32 -> bf16 -> f32` conversion pairs. These are not removed by the `AlgebraicSimplifier`, as that will only simplify conversions that are no-ops, e.g. `bf16 -> f32 -> bf16`. Removing these improves accuracy."

### Multi-node setup
T5X uses `jax.distributed.initialize` (see [here](https://jax.readthedocs.io/en/latest/multi_process.html)) and requires an instance of t5x to be instantiated on each host.

In Slurm environments, you can simply call jax.distributed.initialize() with no arguments. `When running on GPUs with Slurm, it is assumed that one process is started per GPU, i.e. each process will be assigned only one visible local device.`. Note that Slurm jobs need to be run with exporting all the environment variables; e.g. `sbatch --export=ALL train.slurm`

### Implementing a new vocabulary 
Converts the GPT2 tokenizer to a format compatible with T5X. The Vocabulary abstract class is in `seqio/vocabularies.py`.

Vocabularies need to run in the graph if tokenization is to be done on the fly, otherwise tokenize and encode data and write to disk like the rest of the plebs.

### Running on AWS 
* On machines running `Amazon Linux release 2`, the certs are fucked in a way that they aren't on GCP because Tensorflow hardcodes the expected path (see [here](https://github.com/tensorflow/tensorflow/issues/40065) for related issue). Fix with `sudo ln -s /etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt` if you get libcurl errors. 
* For using GFile not working on AWS, not only do you need to do `gcloud auth login --update-adc` to update the service account, you must manually set `os.environ = ["GOOGLE_APPLICATION_CREDENTIALS"]`. This is likely at `~/.config/gcloud`.

### Misc
* `model_parallel_submesh` can be an int (for a single GPU) or a 
* `model_parallel_submesh` and `num_partitions` arguments are mutually-exclusive methods for partitioning! See partitioning.md for more details.
* Easiest way to get up and running on a single GPU for debugging is setting `num_partitions = 1` in `partitioning.PjitPartitioner`.