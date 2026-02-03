#!/usr/bin/env bash
#
# Perform inference with trained steering vectors,
# but test data is not contrastive.


export CUDA_VISIBLE_DEVICES=2

PORT='29503'

# AXBENCH_CFG='2b_l10_v2'
AXBENCH_CFG='2b_l20_v1'
# AXBENCH_CFG='9b_l20_v1'
# AXBENCH_CFG='9b_l31_v1'

SCRIPT='../../cdas/scripts/inference.py'
DATA_DIR="./data/concept500/aug_gpt_prod_${AXBENCH_CFG}/generate/"
TEST_DATA_PATH='../../data/alpaca_eval.json'

MODEL_NAME='das_vector'

if [[ ${AXBENCH_CFG} == 9b* ]]
then
    MODEL_PATH='google/gemma-2-9b-it'
    CONFIG_PATH="./configs/${MODEL_NAME}_gemma_9b.yaml"
elif [[ ${AXBENCH_CFG} == 2b* ]]
then
    MODEL_PATH='google/gemma-2-2b-it'
    CONFIG_PATH="./configs/${MODEL_NAME}_gemma_2b.yaml"
else
    echo -e "Unknown configutation: ${AXBENCH_CFG}"
    exit 1
fi

DUMP_DIR="outputs/${AXBENCH_CFG}/outputs_${MODEL_NAME}/"

mkdir -p ${DUMP_DIR}

#######################################
# Inference mode
#
# Options:
#   two_way_steering, positive_steering, negative steering: Steering
#######################################
MODE="positive_steering"

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
    --test_data_path ${TEST_DATA_PATH} \
    --dump_dir ${DUMP_DIR} \
    --model_name ${MODEL_PATH}

echo -e '\n######################################################################'
echo -e "All finished at $(date)"
echo -e '######################################################################\n'
