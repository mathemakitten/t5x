# notes2self for GPU 

As of January 4, 2023.

### Installation 

* Deep Learning VM is OK to use (it installs cuDNN and CUDA), but don't forget to `sudo chown -R $USER /opt/conda/` 
* `gcloud auth login`
* `conda create -n t5x python=3.8.10`
* `pip install -e .`
* If you get a 403 Forbidden on the storage bucket, do `gcloud auth login --update-adc` to add the service account

### Test training 

Ensure to set `MODEL_DIR="gs://..."` and `TFDS_DATA_DIR=gs://...`

**Decoder only**

```
python t5x/train.py --gin_file="t5x/examples/decoder_only/examples/base_wmt_from_scratch.gin" --gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR}
```

This will train a model on the pretraining mix of `'europarl_v7', 'commoncrawl', 'newscommentary_v13'`, validate on `'newstest2013'` and test on `'newstest2014'`.

For a decoder which you can fit on a single T4 for debugging purposes:
```
python t5x/train.py --gin_file="t5x/examples/decoder_only/examples/tiny_single_gpu.gin" --gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR}
```

TODO(helen): figure out why the loss is so high (1477)

**T5** (have not tested on GPU yet)

```
python t5x/train.py --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin" --gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR}` 
```

### CUDA shenanigans
* `nvcc --version`; `Build cuda_11.4.r11.4` works for sure
* For JAX on GPUs CUDA 11.4 or newer is required so you will likely have to upgrade it manually
  * NVIDIA instructions [here](https://developer.nvidia.com/cuda-11-4-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=10&target_type=deb_local)
* Download JAX GPU versions from [here](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html), *not* [here](https://storage.googleapis.com/jax-releases/jax_releases.html) as linked in the T5X documentation. 
* CUDA is probably at `/usr/local/cuda/lib64`, and `jaxlib==0.4.1+cuda11.cudnn86` works.