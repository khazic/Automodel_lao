#!/usr/bin/env bash
# Two-node launch for the P-EAGLE / EAGLE-3.1 recipe, forcing NCCL onto
# Ethernet (no InfiniBand). Run this ON EACH of the two pods.
#
#   node 0 (rendezvous host):  ./launch_eagle3_2node.sh 0 <MASTER_ADDR>
#   node 1:                    ./launch_eagle3_2node.sh 1 <MASTER_ADDR>
#
# MASTER_ADDR = the reachable IP/hostname of node 0 (e.g. `hostname -i` on node 0).
set -euo pipefail

NODE_RANK=${1:?usage: launch_eagle3_2node.sh <node_rank: 0|1> <master_addr>}
MASTER_ADDR=${2:?usage: launch_eagle3_2node.sh <node_rank: 0|1> <master_addr>}
MASTER_PORT=${MASTER_PORT:-13742}
CONFIG=${CONFIG:-examples/speculative/p-eagle/qwen_peagle_perfectblend_eagle31.yaml}

# GPUs per node + default-route NIC (the Ethernet one in a k8s pod), auto-detected.
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
NET_IF=${NET_IF:-$(ip -o -4 route show to default | awk '{print $5}' | head -1)}

# --- force NCCL + gloo rendezvous onto Ethernet, disable IB ---
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="$NET_IF"
export GLOO_SOCKET_IFNAME="$NET_IF"
export NCCL_NET=Socket
# fail loudly instead of hanging a dead rank; bump dist_env.timeout_minutes too.
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# remove this once you've confirmed it prints "NET/Socket" (not "NET/IB"):
export NCCL_DEBUG=INFO

# wandb: config uses mode: offline so no key is needed at runtime. For online,
# export WANDB_API_KEY in your shell or ~/.bashrc before launch -- never commit it.
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# keep intra-cluster rendezvous traffic off the corporate proxy
export no_proxy="${no_proxy:-},127.0.0.1,localhost,${MASTER_ADDR}"

echo "node_rank=${NODE_RANK} nproc=${NPROC_PER_NODE} nic=${NET_IF} master=${MASTER_ADDR}:${MASTER_PORT}"

torchrun \
  --nnodes=2 \
  --node_rank="${NODE_RANK}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m nemo_automodel.cli.app "${CONFIG}"
