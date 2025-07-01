#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
DS_CONFIG=${3:-"configs/deepspeed/ds_config_zero2.json"}
PORT=${PORT:-28509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
deepspeed --num_gpus=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --deepspeed $DS_CONFIG \
    ${@:4}