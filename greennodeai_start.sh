#!/bin/bash

echo "==================ENV=================="
pip install -r requirements.txt
echo "==================ENV END=================="

echo "==================TEST FLASH ATTN 2=================="
python check_torch.py
echo "==================[DONE]=================="


nccl_version=$(cat /usr/include/nccl.h | grep 'define NCCL_MAJOR' | awk '{print $3}')
echo "NCCL Version: $nccl_version"

echo "Pytorch torch.distributed.run"
nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "nproc_per_node $nproc_per_node"

echo "==================FIRST TEST DISTRIBUTED TRAINING=================="

python -m torch.distributed.run --nproc_per_node $nproc_per_node \
    --nnodes $WORLD_SIZE \
    --master_addr ${MASTER_ADDR} \
    --master_port 7777 \
    t.py 150 10
echo "==================[DONE]=================="


echo "==================SECOND TEST DISTRIBUTED TRAINING LLM=================="

NCCL_DEBUG_FILE=/tmp/nccl.%h.%p  NCCL_DEBUG=INFO UCX_LOG_LEVEL=DEBUG UCX_LOG_FILE=/tmp/latest.log torchrun --nnodes $WORLD_SIZE \
  --nproc_per_node $nproc_per_node \
  --master_addr ${MASTER_ADDR} --master_port 7777 \
  run_fsdp_qlora.py --config llama_3_8b_fsdp_qlora.yaml

echo "==================[DONE]=================="
