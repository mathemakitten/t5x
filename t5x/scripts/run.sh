cd t5x

MODEL_DIR=gs://.../t5x/experiments/11b/model
TFDS_DATA_DIR=gs://.../t5x-test/pile
PROJECT_DIR=/home/.../t5x/t5x/experiments
T5X_DIR=/home/.../t5x

PYTHONPATH=${PROJECT_DIR} python3 t5x/train.py --gin_search_paths=${PROJECT_DIR} --gin_file="t5x/experiments/t5_1_0_11b.gin" --gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR}