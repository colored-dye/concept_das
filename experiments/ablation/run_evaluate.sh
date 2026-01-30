#!/usr/bin/env bash
#
# Evaluate inference results.


PORT=29508
SCRIPT='../../cdas/scripts/evaluate.py'

# MODEL_NAME='kldas_reverse'
MODEL_NAME='cdas_vector'
# MODEL_NAME='reps'

# AXBENCH_CFG='qwen_3b_l18'
# AXBENCH_CFG='qwen_7b_l14'
# AXBENCH_CFG='olmo_1b_l8'
# AXBENCH_CFG='olmo_7b_l16'
AXBENCH_CFG='olmo_13b_l20'

if [[ ${AXBENCH_CFG} == 2b_l20* ]]
then
    MODEL_PATH='google/gemma-2-2b-it'
    LAYER='20'
elif [[ ${AXBENCH_CFG} == qwen_3b_l18* ]]
then
    MODEL_PATH='Qwen/Qwen2.5-3B-Instruct'
    LAYER='18'
    MODEL_KEY='qwen_3b'
elif [[ ${AXBENCH_CFG} == qwen_7b_l14* ]]
then
    MODEL_PATH='Qwen/Qwen2.5-7B-Instruct'
    LAYER='14'
    MODEL_KEY='qwen_7b'
elif [[ ${AXBENCH_CFG} == olmo_1b_l8* ]]
then
    MODEL_PATH='allenai/OLMo-2-0425-1B-Instruct'
    LAYER='8'
    MODEL_KEY='olmo_1b'
elif [[ ${AXBENCH_CFG} == olmo_7b_l16* ]]
then
    MODEL_PATH='allenai/OLMo-2-1124-7B-Instruct/'
    LAYER='16'
    MODEL_KEY='olmo_7b'
elif [[ ${AXBENCH_CFG} == olmo_13b_l20* ]]
then
    MODEL_PATH='allenai/OLMo-2-1124-13B-Instruct/'
    LAYER='20'
    MODEL_KEY='olmo_13b'
else
    echo -e "Unknown configutation: ${AXBENCH_CFG}"
    exit 1
fi

CONFIG_PATH="./configs/${MODEL_NAME}_${MODEL_KEY}.yaml"

DUMP_DIR="outputs/${AXBENCH_CFG}/outputs_${MODEL_NAME}/"
mkdir -p ${DUMP_DIR}

#######################################
# Inference mode
#
# Options:
#   steering: Evaluate steering generations.
#######################################
MODE="$@"

if [[ -z ${MODE} ]]
then
    echo "Evaluation mode is null."
    exit 1
fi

echo -e '\n######################################################################'
echo -e "Configuration: ${AXBENCH_CFG}"
echo -e '######################################################################\n'
echo "Evaluation mode: ${MODE}"

torchrun --master_port ${PORT} --nproc_per_node=1 ${SCRIPT} \
    --mode ${MODE} \
    --config ${CONFIG_PATH} \
    --dump_dir ${DUMP_DIR}
