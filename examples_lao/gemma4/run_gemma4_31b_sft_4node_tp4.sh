#!/bin/bash
# Launch Gemma4 31B SFT — 4-node 32×GPU TP4 recipe.
#
# Run this script on EVERY node (node 0~3).
# Set NODE_RANK to the node index before running:
#
#   Node 0 (master): NODE_RANK=0 bash run_gemma4_31b_sft_4node_tp4.sh
#   Node 1:          NODE_RANK=1 bash run_gemma4_31b_sft_4node_tp4.sh
#   Node 2:          NODE_RANK=2 bash run_gemma4_31b_sft_4node_tp4.sh
#   Node 3:          NODE_RANK=3 bash run_gemma4_31b_sft_4node_tp4.sh

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/llm-align/liuchonghan/Automodel_lao}
CONFIG=${CONFIG:-examples_lao/gemma4/gemma4_31b_sft_4node_tp4.yaml}
VENV_DIR=${VENV_DIR:-${REPO_ROOT}/.venv}

# Force-override any platform-injected MASTER_ADDR/PORT (e.g. K8s env vars).
MASTER_ADDR=10.178.157.101
MASTER_PORT=8881
NNODES=${NNODES:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:?'NODE_RANK must be set (0=master, 1, 2, 3)'}

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
# Synchronise CUDA ops so errors point to the actual failing kernel (debug only)
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}

mkdir -p \
    "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" \
    "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" \
    "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" \
    "${WANDB_DIR}" "${WANDB_CACHE_DIR}" \
    /llm-align/liuchonghan/ckpt_automodel/gemma4_31b_sft_4node_tp4

echo "========================================"
echo "HOSTNAME=$(hostname)"
echo "NODE_RANK=${NODE_RANK} / ${NNODES}"
echo "MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "CONFIG=${CONFIG}"
echo "========================================"

cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"

torchrun \
    --nnodes="${NNODES}" \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m nemo_automodel.cli.app \
    "${CONFIG}"
