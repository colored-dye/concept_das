#!/usr/bin/env bash
#

export CUDA_VISIBLE_DEVICES=3

MODE="$@"

if [[ -z ${MODE} ]]
then
    echo "Evaluation mode is null."
    exit 1
fi

echo "Evaluation mode(s): ${MODE}"


SEED='42'

PORT='29504'
SCRIPT='../../cdas/scripts/evaluate.py'
DATA_DIR='./data/'

TEST_DATA_PATH='./data/contrast_test_jailbreakbench_seed=0.parquet'

# HF_CONFIG_KEY='phi'
# HF_MODEL_PATH='microsoft/Phi-3.5-mini-instruct'
# LAYER=16

# HF_CONFIG_KEY='llama'
# HF_MODEL_PATH='meta-llama/Llama-3.1-8B-Instruct'
# LAYER=12

# HF_CONFIG_KEY='llama_70b'
HF_CONFIG_KEY='llama'
HF_MODEL_PATH='meta-llama/Llama-3.1-70B-Instruct'
LAYER=32

HF_BASENAME=$(basename ${HF_MODEL_PATH})

METHOD='pdas_vector'
DUMP_DIR="outputs/${HF_BASENAME}/outputs_${METHOD}/seed_${SEED}/"
CONFIG_PATH="./configs/${METHOD}_${HF_CONFIG_KEY}.yaml"

echo -e '\n#################################################################################################'
echo -e "Seed: ${SEED} || Model: ${HF_BASENAME} || Method: ${METHOD} || Layer: ${LAYER}"
echo -e '#################################################################################################\n'

LAYER_DUMP_DIR="${DUMP_DIR}/${LAYER}"
mkdir -p ${LAYER_DUMP_DIR}

torchrun --master_port ${PORT} --nproc_per_node=1 ${SCRIPT} \
    --mode ${MODE} \
    --config ${CONFIG_PATH} \
    --overwrite_data_dir ${DATA_DIR} \
    --contrast_test_data_path ${TEST_DATA_PATH} \
    --dump_dir ${LAYER_DUMP_DIR} \
    --model_name ${HF_MODEL_PATH} \
    --steering_model_name ${HF_MODEL_PATH} \
    --layer ${LAYER} \
    --steering_layer ${LAYER} \
    --seed ${SEED}

echo -e '\n######################################################################'
echo -e "All finished at $(date)"
echo -e '######################################################################\n'
