#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7

PORT='29501'
MODEL_PATH='/home/Dataset/Models/google/gemma-2-2b-it'
DATA_DIR='axbench/concept500/aug_gpt_prod_2b_l10_v2/generate/'
DUMP_DIR="${DATA_DIR}/outputs/"
CONFIG_PATH='axbench/sweep/byt/axbench/reps.yaml'

mkdir -p "${DUMP_DIR}/inference/"

torchrun --master_port=${PORT} --nproc_per_node=1 axbench/scripts/inference.py \
    --mode steering \
    --config ${CONFIG_PATH} \
    --dump_dir ${DUMP_DIR} \
    --overwrite_data_dir ${DATA_DIR} \
    --overwrite_metadata_dir ${DATA_DIR} \
    --model_name ${MODEL_PATH} \
    --steering_model_name ${MODEL_PATH}
