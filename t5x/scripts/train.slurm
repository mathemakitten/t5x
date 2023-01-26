#!/bin/bash -l

#SBATCH --job-name=t5x
#SBATCH --nodes= however many nodes you want to run on
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --hint=nomultithread
#SBATCH --time 100:00:00
#SBATCH --output=/home/ec2-user/t5x/log.txt

conda activate t5x

MODEL_DIR= # wherever you want your model checkpoints to go; e.g. .../t5x/experiments/models
TFDS_DATA_DIR= # wherever you want data written, e.g. /mydir/tfds or a gs:// or s3:// bucket
PROJECT_DIR= # some dir for holding experiment configs, e.g. .../t5x/t5x/experiments
T5X_DIR=  # wherever t5x lives on your instance, e.g. /home/ec2-user/t5x

export XLA_FLAGS='--xla_gpu_simplify_all_fp_conversions --xla_gpu_all_reduce_combine_threshold_bytes=136314880 ${XLA_FLAGS}'

export PYTHONPATH="${PYTHONPATH}:/home/ec2-user/t5x/t5x/experiments"
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

srun python t5x/train.py --gin_search_paths=${PROJECT_DIR} --gin_file="t5x/experiments/t5_1_0_11b.gin" \
--gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR} --multiprocess_gpu=True