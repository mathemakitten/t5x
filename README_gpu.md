# notes2self

### Installation 

* Deep Learning VM ok, but `sudo chown -R $USER /opt/conda/` (don't @ me)
* `gcloud auth login`
* `conda create -n t5x python=3.8.10`
* `pip install -e .`
* If you get a 403 Forbidden on the storage bucket, do `gcloud auth login --update-adc` to add the service account

### Test training 
```
MODEL_DIR="gs://..."

python t5x/train.py --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin" --gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir="/home/helen/tensorflow_datasets/wmt_t2t_translate/de-en/1.0.0"
```

### Test CUDA version
* `nvcc --version`
* Apparently for JAX on GPUs CUDA 11.4 or newer is required!
  * See NVIDIA instructions [here](https://developer.nvidia.com/cuda-11-4-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=10&target_type=deb_local)
