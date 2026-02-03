#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

PORT='29508'
DATA_DIR='axbench/refusal/refusal_concept'
SEED='42'

# HF_KEY='phi'
# HF_MODEL_PATH='/home/Dataset/Models/microsoft/Phi-3.5-mini-instruct'

# HF_KEY='gemma'
# HF_MODEL_PATH='/home/Dataset/Models/google/gemma-2-2b-it'

HF_KEY='llama'
# HF_MODEL_PATH='/home/Dataset/Models/meta-llama/Llama-3.1-8B-Instruct'
# LAYER='12'
HF_MODEL_PATH='/home/Dataset/Models/meta-llama/Llama-3.1-70B-Instruct'
LAYER='32'

BASE_NAME=$(basename ${HF_MODEL_PATH})

CONFIG_PATH="axbench/sweep/byt/refusal/reps_${HF_KEY}.yaml"

DUMP_DIR="${HOME}/share/pyvene_data/reps/refusal/${BASE_NAME}/${LAYER}"
mkdir -p ${DUMP_DIR}

torchrun --master_port=${PORT} --nproc_per_node=1 axbench/scripts/train.py \
    --seed ${SEED} \
    --config ${CONFIG_PATH} \
    --dump_dir ${DUMP_DIR} \
    --overwrite_data_dir ${DATA_DIR} \
    --model_name ${HF_MODEL_PATH} \
    --layer ${LAYER}
