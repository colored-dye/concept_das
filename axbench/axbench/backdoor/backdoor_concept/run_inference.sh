#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

PORT='29506'

SEED='42'
# SEED='43'

MODEL_PATH="${HOME}/share/Model/cot_backdoor_model"
DATA_DIR='axbench/backdoor/backdoor_concept'
CONFIG_PATH='axbench/sweep/byt/backdoor/reps.yaml'
LAYERS=(8 12 16 20 24 28)

DUMP_DIR="${HOME}/share/pyvene_data/reps/backdoor/seed_${SEED}/outputs/"

for layer in ${LAYERS[@]}
do
    echo -e '\n######################################################################'
    echo -e "Layer ${layer}"
    echo -e '######################################################################\n'

    LAYER_DUMP_DIR="${DUMP_DIR}/${layer}"
    mkdir -p ${LAYER_DUMP_DIR}

    torchrun --master_port=${PORT} --nproc_per_node=1 axbench/scripts/inference.py \
        --seed ${SEED} \
        --mode steering \
        --config ${CONFIG_PATH} \
        --dump_dir ${LAYER_DUMP_DIR} \
        --overwrite_data_dir ${DATA_DIR} \
        --overwrite_metadata_dir ${DATA_DIR} \
        --model_name ${MODEL_PATH} \
        --steering_model_name ${MODEL_PATH} \
        --layer ${layer} \
        --steering_layer ${layer}

    if [[ ! $? -eq 0 ]]
    then
        echo -e "\n#########################################################################"
        echo -e 'Early stop'
        echo -e "#########################################################################\n"
        exit 1
    fi
done
