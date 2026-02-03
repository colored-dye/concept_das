#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

MODE='steering'

PORT='29508'
DATA_DIR='axbench/ablation'
SEED='42'

# HF_KEY='olmo_1b'
# HF_MODEL_PATH='/home/Dataset/Models/allenai/OLMo-2-0425-1B-Instruct'
# LAYER='8'

# HF_KEY='olmo_7b'
# HF_MODEL_PATH='/home/Dataset/Models/allenai/OLMo-2-1124-7B-Instruct'
# LAYER='16'

HF_KEY='olmo_13b'
HF_MODEL_PATH='/home/Dataset/Models/allenai/OLMo-2-1124-13B-Instruct'
LAYER='20'

# HF_KEY='qwen_3b'
# HF_MODEL_PATH='/home/Dataset/Models/Qwen/Qwen2.5-3B-Instruct'
# LAYER='18'

# HF_KEY='qwen_7b'
# HF_MODEL_PATH='/home/Dataset/Models/Qwen/Qwen2.5-7B-Instruct'
# LAYER='14'

BASE_NAME=$(basename ${HF_MODEL_PATH})

CONFIG_PATH="axbench/ablation/configs/reps_${HF_KEY}.yaml"

DUMP_DIR="${HOME}/share/pyvene_data/reps/ablation/${HF_KEY}/${LAYER}"
INF_DUMP_DIR="${DUMP_DIR}/inference"
mkdir -p ${INF_DUMP_DIR}

torchrun --master_port=${PORT} --nproc_per_node=1 axbench/scripts/inference.py \
    --mode ${MODE} \
    --config ${CONFIG_PATH} \
    --dump_dir ${DUMP_DIR} \
    --overwrite_data_dir ${DATA_DIR} \
    --overwrite_inference_dump_dir ${INF_DUMP_DIR} \
    --overwrite_metadata_dir ${DATA_DIR} \
    --model_name ${HF_MODEL_PATH} \
    --steering_model_name ${HF_MODEL_PATH} \
    --layer ${LAYER} \
    --steering_layer ${LAYER} \
    --seed ${SEED}

echo -e '\n######################################################################'
echo -e "All finished at $(date)"
echo -e '######################################################################\n'
