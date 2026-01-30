#!/usr/bin/env bash
#
# Perform inference with trained interventions.


export CUDA_VISIBLE_DEVICES=2

SEED='42'
PORT='29504'
SCRIPT='../../cdas/scripts/contrast_inference.py'
DATA_DIR='./data/'
TEST_DATA_PATH='./data/contrast_test_jailbreakbench_seed=0.parquet'


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

MODEL_NAME='cdas_vector'
# MODEL_NAME='pdas_vector'
CONFIG_PATH="./configs/${MODEL_NAME}_${HF_CONFIG_KEY}.yaml"
DUMP_DIR="outputs/${HF_BASENAME}/outputs_${MODEL_NAME}/seed_${SEED}"

mkdir -p ${DUMP_DIR}


#######################################
# Inference mode
#
# Options:
#   latent: Concept detection
#   factor: Gather steering factors
#   two_way_steering, positive_steering, negative_steering: Steering
#######################################
MODE="$@"

if [[ -z ${MODE} ]]
then
    echo "Inference mode is null."
    exit 1
fi

echo "Inference mode(s): ${MODE}"


for layer in ${LAYERS[@]}
do
    echo -e '\n#################################################################################################'
    echo -e "Seed: ${SEED} || Model: ${HF_BASENAME} || Method: ${MODEL_NAME} || Layer: ${layer}"
    echo -e '#################################################################################################\n'

    LAYER_DUMP_DIR="${DUMP_DIR}/${layer}"
    mkdir -p ${LAYER_DUMP_DIR}

    torchrun --master_port ${PORT} --nproc_per_node=1 ${SCRIPT} \
        --mode ${MODE} \
        --config ${CONFIG_PATH} \
        --overwrite_data_dir ${DATA_DIR} \
        --contrast_test_data_path ${TEST_DATA_PATH} \
        --dump_dir ${LAYER_DUMP_DIR} \
        --model_name ${HF_MODEL_PATH} \
        --steering_model_name ${HF_MODEL_PATH} \
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
