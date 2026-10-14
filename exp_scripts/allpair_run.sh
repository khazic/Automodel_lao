#!/bin/bash
# One model through the all-pair suite on one 8-GPU node: free the occupied GPUs, decode with
# allpair_gen_eval.py (one worker per GPU), merge, then score with the suite's eval_allpair.py.
# park_guard re-occupies the node afterwards (its JOB_RE matches "eval.py", so it stays away
# while the workers run).
#
#   bash allpair_run.sh <tag> <weights dir> <out dir> <ngram: 0|1>
#
# Site paths come from the environment:
#   AM_REPO      Automodel checkout on PYTHONPATH (the n-gram code the checkpoint was trained with)
#   CONDA_SH     conda.sh to source, CONDA_ENV the env to activate
#   LD_EXTRA     optional LD_LIBRARY_PATH prefix
#   PROMPTS      all-pair prompt JSONL
#   EVAL_HOME    dir holding the scorer's env.sh, EVAL_SCRIPT the eval_allpair.py path
set -uo pipefail
TAG=${1:?tag}
WEIGHTS=${2:?weights dir}
OUT=${3:?out dir}
NGRAM=${4:?0 or 1}
: "${AM_REPO:?}" "${CONDA_SH:?}" "${CONDA_ENV:?}" "${PROMPTS:?}" "${EVAL_HOME:?}" "${EVAL_SCRIPT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
LOG=$OUT/run.log
say() { echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

say "START $TAG host=$(hostname) weights=$WEIGHTS ngram=$NGRAM code=$(git -C "$AM_REPO" log --oneline -1)"
# Occupiers are keep_gpu_busy.sh -> gpu_keepalive_aggressive.py -> multiprocessing children
# (reparented to PID 1); every layer holds GPU memory, so kill all three.
pkill -9 -f keep_gpu_busy.sh
pkill -9 -f gpu_keepalive_aggressive
sleep 5
pkill -9 -f 'multiprocessing.(spawn|resource_tracker)'
busy=8
for _ in $(seq 1 60); do
    busy=0
    for m in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); do
        [ "$m" -gt 1000 ] && busy=$((busy + 1))
    done
    [ "$busy" = 0 ] && break
    sleep 10
done
if [ "$busy" != 0 ]; then
    say "RUN_DONE rc=gpus_busy ($busy GPUs still hold memory)"
    exit 1
fi
# Evict any occupier started from elsewhere while the workers run.
(
    while kill -0 $$ 2>/dev/null; do
        for p in $(pgrep -f gpu_keepalive_aggressive); do
            echo "[$(date +%T)] evicting occupier $p" >> "$OUT/occupier_evictions.log"
            pkill -9 -P "$p"
            kill -9 "$p"
        done
        pkill -9 -f keep_gpu_busy.sh
        sleep 20
    done
) &
WATCH=$!

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH=${LD_EXTRA:+$LD_EXTRA:}${LD_LIBRARY_PATH:-}
export PYTHONPATH=$AM_REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
FLAG=""
[ "$NGRAM" = 1 ] && FLAG=--ngram
pids=()
for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g python "$HERE/allpair_gen_eval.py" gen --weights "$WEIGHTS" --prompts "$PROMPTS" \
        --out "$OUT/part_$g.jsonl" --rank "$g" --world 8 $FLAG > "$OUT/gen_$g.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done
kill "$WATCH" 2>/dev/null
say "generation finished rc=$rc"
if [ "$rc" != 0 ]; then
    say "RUN_DONE rc=gen_failed"
    exit 1
fi

python "$HERE/allpair_gen_eval.py" merge --prompts "$PROMPTS" --out "$OUT/allpair_$TAG" "$OUT"/part_*.jsonl 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = 0 ] || { say "RUN_DONE rc=merge_failed"; exit 1; }
conda deactivate
(cd "$EVAL_HOME" && . ./env.sh && python "$EVAL_SCRIPT" "$OUT/allpair_$TAG") > "$OUT/bleu_$TAG.txt" 2>&1
rc=$?
grep -E "BLEU mean|directions scored" "$OUT/bleu_$TAG.txt" | tee -a "$LOG"
say "RUN_DONE rc=$rc"
