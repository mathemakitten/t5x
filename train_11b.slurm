#!/bin/bash -l

#SBATCH --job-name=t5x
#SBATCH --nodes=24
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --hint=nomultithread
#SBATCH --time 100:00:00
#SBATCH --output=/home/ec2-user/t5x/log.txt

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/ec2-user/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ec2-user/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/ec2-user/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ec2-user/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# This needs to run at least once because AWS certs live elsewhere apparently
sudo ln -s /etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt

conda activate t5x

MODEL_DIR=/home/ec2-user/t5x/experiments/models

MODEL_DIR=/gpfswork/helen/experiments/models
TFDS_DATA_DIR=/gpfswork/tfds
PROJECT_DIR=/home/ec2-user/t5x/t5x/experiments
T5X_DIR=/home/ec2-user/t5x

export XLA_FLAGS='--xla_gpu_simplify_all_fp_conversions --xla_gpu_all_reduce_combine_threshold_bytes=136314880 ${XLA_FLAGS}'

export PYTHONPATH="${PYTHONPATH}:/home/ec2-user/t5x/t5x/experiments"
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

srun python t5x/train.py --gin_search_paths=${PROJECT_DIR} --gin_file="t5x/experiments/t5_1_0_11b.gin" \
--gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR} --multiprocess_gpu=True