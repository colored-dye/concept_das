#!/usr/bin/env bash
#
# Perform inference with trained steering vectors.


export CUDA_VISIBLE_DEVICES=2

PORT='29502'

# AXBENCH_CFG='2b_l10_v2'
AXBENCH_CFG='2b_l20_v1'
# AXBENCH_CFG='9b_l20_v1'
# AXBENCH_CFG='9b_l31_v1'

MODEL_NAME='das_vector'

if [[ ${AXBENCH_CFG} == 9b_l20* ]]
then
    MODEL_PATH='google/gemma-2-9b-it'
    CONFIG_PATH="./configs/${MODEL_NAME}_gemma_9b.yaml"
    LAYER='20'
elif [[ ${AXBENCH_CFG} == 9b_l31* ]]
then
    MODEL_PATH='google/gemma-2-9b-it'
    CONFIG_PATH="./configs/${MODEL_NAME}_gemma_9b.yaml"
    LAYER='31'
elif [[ ${AXBENCH_CFG} == 2b_l20* ]]
then
    MODEL_PATH='google/gemma-2-2b-it'
    CONFIG_PATH="./configs/${MODEL_NAME}_gemma_2b.yaml"
    LAYER='20'
elif [[ ${AXBENCH_CFG} == 2b_l10* ]]
then
    MODEL_PATH='google/gemma-2-2b-it'
    CONFIG_PATH="./configs/${MODEL_NAME}_gemma_2b.yaml"
    LAYER='10'
else
    echo -e "Unknown configutation: ${AXBENCH_CFG}"
    exit 1
fi

DATA_DIR="./data/concept500/aug_gpt_prod_${AXBENCH_CFG}/generate/"
TEST_DATA_PATH="./data/concept500/aug_gpt_prod_${AXBENCH_CFG}/generate/contrast_train_data.parquet"
SCRIPT='../../cdas/scripts/contrast_inference.py'

DUMP_DIR="outputs/${AXBENCH_CFG}/outputs_${MODEL_NAME}/"

mkdir -p ${DUMP_DIR}

#######################################
# Inference mode
#
# Options:
#   latent: Concept detection
#   factor: Gather steering factors
#   two_way_steering, positive_steering, negative steering: Steering
#######################################
MODE="$@"

if [[ -z ${MODE} ]]
then
    echo "Inference mode is null."
    exit 1
fi

echo "Inference mode(s): ${MODE}"

torchrun --master_port ${PORT} --nproc_per_node=1 ${SCRIPT} \
    --mode ${MODE} \
    --config ${CONFIG_PATH} \
    --overwrite_data_dir ${DATA_DIR} \
    --contrast_test_data_path ${TEST_DATA_PATH} \
    --dump_dir ${DUMP_DIR} \
    --model_name ${MODEL_PATH}

echo -e '\n######################################################################'
echo -e "All finished at $(date)"
echo -e '######################################################################\n'
