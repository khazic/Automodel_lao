#!/bin/bash
# Usage: bash examples_lao/gemma4/launch_4node_filtered_chat.sh <node_rank>
# Example on master node: bash examples_lao/gemma4/launch_4node_filtered_chat.sh 0
set -euo pipefail
NODE_RANK=${1:?'Usage: bash launch_4node_filtered_chat.sh <node_rank 0-3>'}
MASTER_ADDR=10.178.157.101
MASTER_PORT=29500
REPO_ROOT=/llm-align/liuchonghan/Automodel_lao
CONFIG=examples_lao/gemma4/gemma4_31b_sft_4node_tp4_filtered_chat.yaml
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Force IPv4 — k8s clusters often try IPv6 first (errno 97) when resolving hostnames
IFACE=$(ip -4 route get ${MASTER_ADDR} 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' | head -1 || true)
IFACE=${IFACE:-eth0}
export NCCL_SOCKET_IFNAME=${IFACE}
export GLOO_SOCKET_IFNAME=${IFACE}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
echo "Using network interface: ${IFACE}"

cd ${REPO_ROOT}
torchrun --nproc-per-node=8 --nnodes=4 --node_rank=${NODE_RANK} --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv-id=gemma4-sft-filtered-chat -m nemo_automodel.cli.app ${CONFIG}
