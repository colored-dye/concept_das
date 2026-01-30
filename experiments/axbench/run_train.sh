#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

PORT='29504'

# AXBENCH_CFG='2b_l10_v2'
# AXBENCH_CFG='9b_l20_v1'
AXBENCH_CFG='9b_l31_v1'
MODEL_NAME='das_vector'

if [[ ${AXBENCH_CFG} == 9b_l31* ]]
then
    MODEL_PATH='google/gemma-2-9b-it'
    LAYER='31'
    CONFIG_PATH="./configs/${MODEL_NAME}_gemma_9b.yaml"
elif [[ ${AXBENCH_CFG} == 9b_l20* ]]
then
    MODEL_PATH='google/gemma-2-9b-it'
    LAYER='20'
    CONFIG_PATH="./configs/${MODEL_NAME}_gemma_9b.yaml"
elif [[ ${AXBENCH_CFG} == 2b_l10* ]]
then
    MODEL_PATH='google/gemma-2-2b-it'
    LAYER='10'
    CONFIG_PATH="./configs/${MODEL_NAME}_gemma_2b.yaml"
else
    echo -e "Unknown configutation: ${AXBENCH_CFG}"
    exit 1
fi

echo -e '\n######################################################################'
echo -e "Configuration: ${AXBENCH_CFG}"
echo -e '######################################################################\n'

DATA_DIR="./data/concept500/aug_gpt_prod_${AXBENCH_CFG}/generate/"

DUMP_DIR="outputs/${AXBENCH_CFG}/outputs_${MODEL_NAME}/"

mkdir -p ${DUMP_DIR}

torchrun --master_port=${PORT} --nproc_per_node=1 ../../cdas/scripts/contrast_train.py \
    --config ${CONFIG_PATH} \
    --overwrite_data_dir ${DATA_DIR} \
    --dump_dir ${DUMP_DIR} \
    --model_name ${MODEL_PATH} \
    --layer ${LAYER} \
    --max_concepts 10

echo -e '\n######################################################################'
echo -e "All finished at $(date)"
echo -e '######################################################################\n'
