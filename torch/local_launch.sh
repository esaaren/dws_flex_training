#!/bin/bash

export RANK=0
export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29400

export HF_HUB_ENABLE_HF_TRANSFER=1
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d fsdp.py