#!/usr/bin/env bash
#
# Perform inference with trained steering vectors.


export CUDA_VISIBLE_DEVICES=3

PORT='29504'
SCRIPT='../../cdas/scripts/benchmark_eval.py'
DATA_DIR='./data/'

SEED='0'

# BENCHMARK='truthfulqa'
# HF_DATASET_PATH="${HOME}/Dataset/truthful_qa/"
# NUM_SHOTS='3'

BENCHMARK='mmlu'
HF_DATASET_PATH="${HOME}/Dataset/cais/mmlu/"
NUM_SHOTS='5'

#######################################
# LLM configurations
#
#######################################

# HF_CONFIG_KEY='gemma'
# HF_MODEL_PATH='google/gemma-2-2b-it'
# LAYERS=(6 10 14 18 22)

# HF_CONFIG_KEY='phi'
# HF_MODEL_PATH='microsoft/Phi-3.5-mini-instruct'
# LAYER='16'

# HF_CONFIG_KEY='llama_8b'
# HF_CONFIG_KEY='llama'
# HF_MODEL_PATH='meta-llama/Llama-3.1-8B-Instruct'
# LAYER='12'

# HF_CONFIG_KEY='llama_70b'
HF_CONFIG_KEY='llama'
HF_MODEL_PATH='meta-llama/Llama-3.1-70B-Instruct'
LAYER='32'

HF_BASENAME=$(basename ${HF_MODEL_PATH})

MODEL_NAME='pdas_vector'
CONFIG_PATH="./configs/${MODEL_NAME}_${HF_CONFIG_KEY}.yaml"
DUMP_DIR="outputs/${HF_BASENAME}/outputs_${MODEL_NAME}/seed_42/${LAYER}"

mkdir -p ${DUMP_DIR}

echo -e '\n################################################################################################'
echo -e "Benchmark: ${BENCHMARK} || Model: ${HF_BASENAME} || Method: ${MODEL_NAME} || Layer: ${LAYER}"
echo -e '################################################################################################\n'

torchrun --master_port ${PORT} --nproc_per_node=1 ${SCRIPT} \
    --benchmark ${BENCHMARK} \
    --config ${CONFIG_PATH} \
    --overwrite_data_dir ${DATA_DIR} \
    --hf_dataset_path ${HF_DATASET_PATH} \
    --dump_dir ${DUMP_DIR} \
    --model_name ${HF_MODEL_PATH} \
    --steering_model_name ${HF_MODEL_PATH} \
    --num_shots ${NUM_SHOTS} \
    --seed 0

echo -e '\n######################################################################'
echo -e "All finished at $(date)"
echo -e '######################################################################\n'
