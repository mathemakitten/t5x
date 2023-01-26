cd t5x

python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

python3 -m pip install -e '.[pile]'