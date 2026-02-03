#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

PORT='29508'
AXBENCH_CFG='2b_l10_v1'
DATA_DIR="axbench/concept500/prod_${AXBENCH_CFG}/generate"
SEED='42'
LAYER=10

HF_MODEL_PATH='/home/Dataset/Models/google/gemma-2-2b-it'

CONFIG_PATH="axbench/sweep/wuzhengx/reps/experiments/p_vector_dps_g2-2b_axbench.yaml"

DUMP_DIR="${HOME}/share/pyvene_data/reps/axbench/${AXBENCH_CFG}"
mkdir -p ${DUMP_DIR}

torchrun --master_port=${PORT} --nproc_per_node=1 axbench/scripts/train.py \
    --seed ${SEED} \
    --config ${CONFIG_PATH} \
    --dump_dir ${DUMP_DIR} \
    --overwrite_data_dir ${DATA_DIR} \
    --model_name ${HF_MODEL_PATH} \
    --layer ${LAYER}
