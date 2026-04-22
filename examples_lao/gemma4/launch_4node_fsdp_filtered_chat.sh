#!/bin/bash
# Usage: bash examples_lao/gemma4/launch_4node_fsdp_filtered_chat.sh <node_rank>
set -euo pipefail
NODE_RANK=${1:?'Usage: bash launch_4node_fsdp_filtered_chat.sh <node_rank 0-3>'}
MASTER_ADDR=10.178.157.101
MASTER_PORT=29500
REPO_ROOT=/llm-align/liuchonghan/Automodel_lao
CONFIG=examples_lao/gemma4/gemma4_31b_sft_4node_fsdp_filtered_chat.yaml
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IFACE=$(ip -4 route get ${MASTER_ADDR} 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' | head -1 || true)
IFACE=${IFACE:-eth0}
export GLOO_SOCKET_IFNAME=${IFACE}
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_READ=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
echo "Using gloo interface: ${IFACE}"
cd ${REPO_ROOT}
torchrun --nproc-per-node=8 --nnodes=4 --node_rank=${NODE_RANK} --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv-id=gemma4-sft-fsdp-filtered-chat -m nemo_automodel.cli.app ${CONFIG}
