#!/usr/bin/env bash
#
# Perform inference with trained steering vectors,
# but test data is not contrastive.


export CUDA_VISIBLE_DEVICES=3

PORT='29506'
SCRIPT='../../cdas/scripts/inference.py'

# AXBENCH_CFG='2b_l20_v1'
# AXBENCH_CFG='qwen_3b_l18'
# AXBENCH_CFG='olmo_1b_l8'
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

echo -e '\n######################################################################'
echo -e "Configuration: ${AXBENCH_CFG}"
echo -e '######################################################################\n'

DATA_DIR="./data/concept500/aug_gpt_prod_9b_l20_v1/generate/"
TEST_DATA_PATH='../../data/alpaca_eval.json'

# MODEL_NAME='kldas_reverse'
MODEL_NAME='cdas_vector'

CONFIG_PATH="./configs/${MODEL_NAME}_${MODEL_KEY}.yaml"
DUMP_DIR="outputs/${AXBENCH_CFG}/outputs_${MODEL_NAME}/"

mkdir -p ${DUMP_DIR}

#######################################
# Inference mode
#
# Options:
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
    --test_data_path ${TEST_DATA_PATH} \
    --dump_dir ${DUMP_DIR} \
    --model_name ${MODEL_PATH} \
    --layer ${LAYER} \
    --steering_layer ${LAYER}
