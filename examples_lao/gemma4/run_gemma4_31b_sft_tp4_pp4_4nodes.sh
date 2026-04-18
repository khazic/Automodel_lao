#!/bin/bash
# Launch Gemma4 31B VLM SFT — 4-node 32×GPU TP4×PP4×DP2 test.
#
# Topology: tp_size=4, pp_size=4, dp_size=2 (32 GPUs total)
#
# Usage — run on EACH node:
#   NODE_RANK=0 MASTER_ADDR=<node0_ip> bash examples_lao/gemma4/run_gemma4_31b_sft_tp4_pp4_4nodes.sh
#   NODE_RANK=1 MASTER_ADDR=<node0_ip> bash examples_lao/gemma4/run_gemma4_31b_sft_tp4_pp4_4nodes.sh
#   NODE_RANK=2 MASTER_ADDR=<node0_ip> bash examples_lao/gemma4/run_gemma4_31b_sft_tp4_pp4_4nodes.sh
#   NODE_RANK=3 MASTER_ADDR=<node0_ip> bash examples_lao/gemma4/run_gemma4_31b_sft_tp4_pp4_4nodes.sh
#
# Override paths via env vars, e.g.:
#   REPO_ROOT=/your/path NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash ...

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/llm-align/liuchonghan/Automodel_lao}
CONFIG=${CONFIG:-examples_lao/gemma4/gemma4_31b_sft_tp4_pp4.yaml}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${NNODES:-4}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:?Must set MASTER_ADDR to node-0 IP}
MASTER_PORT=${MASTER_PORT:-29500}
TORCHRUN=${TORCHRUN:-/usr/local/bin/torchrun}

export TMPDIR=${TMPDIR:-/llm-align/liuchonghan/tmp}
export TEMP=${TEMP:-/llm-align/liuchonghan/tmp}
export TMP=${TMP:-/llm-align/liuchonghan/tmp}
export HF_HOME=${HF_HOME:-/llm-align/liuchonghan/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/llm-align/liuchonghan/hf_cache/hub}
export TORCH_HOME=${TORCH_HOME:-/llm-align/liuchonghan/torch_cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/llm-align/liuchonghan/torch_extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/llm-align/liuchonghan/triton_cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/llm-align/liuchonghan/xdg_cache}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-/llm-align/liuchonghan/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/llm-align/liuchonghan/wandb/cache}
export HF_TOKEN=${HF_TOKEN:-}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

mkdir -p \
    "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" \
    "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" \
    "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" \
    "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

echo "========================================"
echo "HOSTNAME=$(hostname)"
echo "NODE_RANK=${NODE_RANK}  /  NNODES=${NNODES}"
echo "MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "CONFIG=${CONFIG}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}  (tp4 x pp4 x dp2 = 32 GPUs)"
echo "========================================"

cd "${REPO_ROOT}"

"${TORCHRUN}" \
    --nnodes="${NNODES}" \
    --node-rank="${NODE_RANK}" \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    -m nemo_automodel.cli.app \
    "${CONFIG}"
