cd t5x

export PATH=$PATH:/home/helen_huggingface_co/.local/bin

MODEL_DIR=gs://hugginghelen/t5x/experiments/11b/model
TFDS_DATA_DIR=gs://hugginghelen/t5x-test/pile
PROJECT_DIR=/home/helen_huggingface_co/t5x/t5x/experiments
T5X_DIR=/home/helen_huggingface_co/t5x

PYTHONPATH=${PROJECT_DIR} python3 t5x/train.py --gin_search_paths=${PROJECT_DIR} --gin_file="t5x/experiments/t5_1_0_11b.gin" --gin.MODEL_DIR=\"${MODEL_DIR}\" --tfds_data_dir=${TFDS_DATA_DIR}