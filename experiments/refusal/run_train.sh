#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

PORT='29504'
DATA_DIR='./data/'
SEED='42'


HF_CONFIG_KEY='phi'
HF_MODEL_PATH='microsoft/Phi-3.5-mini-instruct'
LAYERS=(16)

# HF_CONFIG_KEY='llama'
# HF_MODEL_PATH='meta-llama/Llama-3.1-8B-Instruct'
# LAYERS=(12)

# HF_CONFIG_KEY='llama'
# HF_MODEL_PATH='meta-llama/Llama-3.1-70B-Instruct'
# LAYERS=(32)

HF_BASENAME=$(basename ${HF_MODEL_PATH})

# MODEL_NAME='cdas_vector'
# MODEL_NAME='pdas_vector'
MODEL_NAME='cdas_subspace'
CONFIG_PATH="./configs/${MODEL_NAME}_${HF_CONFIG_KEY}.yaml"
DUMP_DIR="outputs/${HF_BASENAME}/outputs_${MODEL_NAME}/seed_${SEED}"

mkdir -p ${DUMP_DIR}

for layer in ${LAYERS[@]}
do
    echo -e '\n#################################################################################################'
    echo -e "Seed: ${SEED} || Model: ${HF_BASENAME} || Method: ${MODEL_NAME} || Layer: ${layer}"
    echo -e '#################################################################################################\n'

    LAYER_DUMP_DIR="${DUMP_DIR}/${layer}"
    mkdir -p ${LAYER_DUMP_DIR}

    torchrun --master_port=${PORT} --nproc_per_node=1 ../../cdas/scripts/contrast_train.py \
        --config ${CONFIG_PATH} \
        --overwrite_data_dir ${DATA_DIR} \
        --dump_dir ${LAYER_DUMP_DIR} \
        --model_name ${HF_MODEL_PATH} \
        --layer ${layer} \
        --seed ${SEED}

    if [[ ! $? -eq 0 ]]
    then
        echo -e "\n#########################################################################"
        echo -e 'Early stop'
        echo -e "#########################################################################\n"
        exit 1
    fi
done

echo -e '\n######################################################################'
echo -e "All finished at $(date)"
echo -e '######################################################################\n'
