## TPU how-to guide, for outside of Google

**CREATE A TPU**

`gcloud alpha compute tpus tpu-vm create helen-1 --zone=us-central2-b --accelerator-type=v4-8 --version=tpu-vm-v4-base --subnetwork=helen`

**SSH INTO THE HOST** 

`gcloud alpha compute tpus tpu-vm ssh helen-pod --zone=us-central2-b`

**ON PODS ONLY, INSTALL JAX EVERYWHERE**
```
gcloud compute tpus tpu-vm ssh helen-pod \
  --zone us-central2-b \
  --worker=all \
  --command="pip install 'jax[tpu]>=0.2.16' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
```

See https://cloud.google.com/tpu/docs/jax-pods#copy_examplepy_to_all_vms_in_the_pod_slice for more details on pod setup.

**SETUP SCRIPT**

`gcloud compute tpus tpu-vm ssh helen-pod   --zone us-central2-b   --worker=all   --command="git clone --branch=hn-tpu2 https://github.com/mathemakitten/t5x && chmod +x t5x/setup.sh && ./t5x/setup.sh"`

**GIT PULL + RUN**

`gcloud compute tpus tpu-vm ssh helen-pod   --zone us-central2-b   --worker=all   --command="cd t5x && git pull && cd .. && chmod +x t5x/run.sh && ./t5x/run.sh"`

## Misc Troubleshooting

**CHECK IF POD SEES ALL THE DEVICES** 

`gcloud compute tpus tpu-vm ssh helen-pod   --zone us-central2-b   --worker=all   --command="python3 -c 'import jax; print(jax.device_count())'"`

This should return the number of devices across all processes (or 8, in the case of a single TPU).