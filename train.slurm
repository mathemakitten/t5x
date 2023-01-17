#!/bin/bash -l

#SBATCH --job-name=benchmark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --hint=nomultithread
#SBATCH --time 100:00:00
#SBATCH --output=/home/ec2-user/t5x/log.txt

set -x -e

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

# source /home/ec2-user/anaconda3/conda/bin/activate t5x
sudo ln -s /etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt

conda activate t5x

MODEL_DIR=/home/ec2-user/t5x/experiments/models
TFDS_DATA_DIR=/gpfswork/tfds
PROJECT_DIR=/home/ec2-user/t5x/t5x/experiments
T5X_DIR=/home/ec2-user/t5x

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "PYTHONPATH=${PROJECT_DIR} python t5x/train.py --gin_search_paths=${PROJECT_DIR} --gin_file="t5x/experiments/tiny_gpu_8x.gin" --gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR}"