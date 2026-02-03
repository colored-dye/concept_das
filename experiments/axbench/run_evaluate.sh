#!/usr/bin/env bash
#
# Evaluate inference results.

PORT='29505'

export CUDA_VISIBLE_DEVICES=1

SCRIPT='../../cdas/scripts/evaluate.py'

MODEL_NAME='das_vector'

# AXBENCH_CFG='2b_l10_v2'
# AXBENCH_CFG='2b_l20_v1'
AXBENCH_CFG='9b_l31_v1'

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

echo -e '\n######################################################################'
echo -e "Configuration: ${AXBENCH_CFG}"
echo -e '######################################################################\n'

DUMP_DIR="outputs/${AXBENCH_CFG}/outputs_${MODEL_NAME}/"
mkdir -p ${DUMP_DIR}

#######################################
# Inference mode
#
# Options:
#   steering: Evaluate steering generations.
#######################################
MODE="steering"

if [[ -z ${MODE} ]]
then
    echo "Evaluation mode is null."
    exit 1
fi

echo "Evaluation mode: ${MODE}"

torchrun --master_port ${PORT} --nproc_per_node=1 ${SCRIPT} \
    --mode ${MODE} \
    --config ${CONFIG_PATH} \
    --dump_dir ${DUMP_DIR} \
    --end_concept_id 200

echo -e '\n######################################################################'
echo -e "All finished at $(date)"
echo -e '######################################################################\n'

