#!/bin/bash
# Launch Gemma4 31B VLM SFT — single-node 8×GPU TP4×PP2 test.
#
# Topology: tp_size=4, pp_size=2, dp_size=1 (8 GPUs total)
#
# Usage:
#   bash examples_lao/gemma4/run_gemma4_31b_sft_tp4_pp2.sh
#
# Override paths via env vars, e.g.:
#   REPO_ROOT=/your/path bash examples_lao/gemma4/run_gemma4_31b_sft_tp4_pp2.sh

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/llm-align/liuchonghan/Automodel_lao}
CONFIG=${CONFIG:-examples_lao/gemma4/gemma4_31b_sft_tp4_pp2.yaml}
VENV_DIR=${VENV_DIR:-${REPO_ROOT}/.venv}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

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
# Uncomment for synchronous CUDA errors (slower but shows exact crash location):
# export CUDA_LAUNCH_BLOCKING=1

mkdir -p \
    "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" \
    "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" \
    "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" \
    "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

echo "========================================"
echo "HOSTNAME=$(hostname)"
echo "CONFIG=${CONFIG}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}  (tp4 x pp2 = 8 GPUs)"
echo "========================================"

cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"

torchrun \
    --standalone \
    --nproc-per-node="${NPROC_PER_NODE}" \
    -m nemo_automodel.cli.app \
    "${CONFIG}"
