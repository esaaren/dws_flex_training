#!/bin/bash

export RANK=$JOB_COMPLETION_INDEX

torchrun --nnodes=$NODES --nproc_per_node=$PROC_PER_NODE --node_rank=$JOB_COMPLETION_INDEX --rdzv_id=$JOB_ID --rdzv_endpoint=$MASTER_ENDPOINT --rdzv_backend=c10d $PYTORCH_SCRIPT_NAME
