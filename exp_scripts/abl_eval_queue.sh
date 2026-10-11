#!/bin/bash
# Job queue for abl_eval.py on spare (occupied) GPUs.
#
#   abl_eval_queue.sh init          write one job per (model, task) into $E/queue
#   abl_eval_queue.sh worker <gpu>  free this GPU's occupier, run jobs until the
#                                   queue is empty, then occupy the GPU again
#
# Jobs are claimed with mkdir (atomic on the shared disk), so one worker per GPU
# on any number of nodes can share the queue. A job that fails is retried by
# the next free worker, at most 3 attempts.
set -u
L=/llm-align/liuchonghan
R=$L/runs/gemma4_ngram
E=$R/eval_abl
W=$E/weights
CODE=$L/Automodel_ngram_eval

# name|weights|run yaml (- = no n-gram branch)
MODELS=(
  "base_e2b|/llm-align/open_models/gemma4/gemma-4-E2B|-"
  "control_14999|$W/control_14999|-"
  "main_24999|$W/main_24999|$R/e2b_ngram_mala.yaml"
  "tablelr10_14999|$W/tablelr10_14999|$R/abl/e2b_abl_tablelr10.yaml"
  "frozenbase_14999|$W/frozenbase_14999|$R/abl/e2b_abl_frozenbase.yaml"
  "ngram2_14999|$W/ngram2_14999|$R/abl/e2b_abl_ngram2.yaml"
  "ngram4_14999|$W/ngram4_14999|$R/abl/e2b_abl_ngram4.yaml"
  "layer0_14999|$W/layer0_14999|$R/abl/e2b_abl_layer0.yaml"
  "layer8_14999|$W/layer8_14999|$R/abl/e2b_abl_layer8.yaml"
  "table0.25b_14999|$W/table0.25b_14999|$R/abl/e2b_abl_table0.25b.yaml"
  "table4b_4999|$W/table4b_4999|$R/abl/e2b_abl_table4b.yaml"
)
# Longest tasks first so the tail of the queue is short jobs.
TASKS=("1:mt" "2:lmeval" "3:indist" "4:heldout")

say() { echo "[$(date +%F_%T)] $*"; }

init() {
    mkdir -p "$E/queue" "$E/claims" "$E/done" "$E/attempts" "$E/results" "$E/logs"
    for t in "${TASKS[@]}"; do
        for m in "${MODELS[@]}"; do
            IFS='|' read -r name weights yaml <<< "$m"
            job="${t%%:*}_${name}__${t#*:}"
            [ -e "$E/done/$job" ] || echo "$name|$weights|$yaml|${t#*:}" > "$E/queue/$job.job"
        done
    done
    ls "$E/queue" | wc -l
}

free_gpu() {
    local g=$1 pid
    for pid in $(ps -eo pid,args | grep -E "^ *[0-9]+ bash (\S*/)?keep_gpu_busy\.sh \"?$g\"? " | awk '{print $1}'); do
        kill -9 -- "-$(ps -o pgid= -p "$pid" | tr -d ' ')" 2>/dev/null
    done
    for _ in $(seq 1 60); do
        [ "$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && return 0
        sleep 5
    done
    say "GPU $g still busy after 5 min"
    return 1
}

occupy_gpu() {
    local g=$1
    ( cd $L/openrlhf && exec setsid nohup bash $L/openrlhf/keep_gpu_busy.sh "$g" 0.8 0.8 >> "$L/runs/zhanka_$(hostname -I | cut -d' ' -f1)_gpu$g.log" 2>&1 < /dev/null & )
}

worker() {
    local g=$1 f job name weights yaml task n
    source $L/env/etc/profile.d/conda.sh && conda activate automodel
    export PATH=$L/envs/ngram_eval/bin:$PATH PYTHONPATH=$CODE:$CODE/exp_scripts PYTHONUNBUFFERED=1
    export LD_LIBRARY_PATH=$L/env/envs/automodel/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
    export HF_HOME=$L/hf_cache HF_DATASETS_CACHE=$L/hf_cache/datasets TOKENIZERS_PARALLELISM=false
    export http_proxy=http://public-proxy.qihoo.net:3128 https_proxy=http://public-proxy.qihoo.net:3128 no_proxy=api.360.cn
    export CUDA_VISIBLE_DEVICES=$g
    free_gpu "$g" || { occupy_gpu "$g"; return 1; }
    while true; do
        job=""
        for f in "$E"/queue/*.job; do
            [ -e "$f" ] || break
            job=$(basename "$f" .job)
            mkdir "$E/claims/$job" 2>/dev/null && break
            job=""
        done
        [ -z "$job" ] && break
        IFS='|' read -r name weights yaml task < "$E/queue/$job.job"
        n=$(( $(cat "$E/attempts/$job" 2>/dev/null || echo 0) + 1 ))
        echo "$n" > "$E/attempts/$job"
        say "GPU $g: $job (attempt $n)"
        args=(--name "$name" --weights "$weights" --task "$task" --out "$E/results/${name}__$task.json")
        [ "$yaml" != "-" ] && args+=(--yaml "$yaml")
        if python -u $CODE/exp_scripts/abl_eval.py "${args[@]}" > "$E/logs/$job.log" 2>&1; then
            say "GPU $g: $job ok"
            mv "$E/queue/$job.job" "$E/done/$job"
        elif [ "$n" -ge 3 ]; then
            say "GPU $g: $job failed $n times, giving up"
            mv "$E/queue/$job.job" "$E/done/$job.failed"
        else
            say "GPU $g: $job failed, requeued"
            rmdir "$E/claims/$job"
        fi
    done
    say "GPU $g: queue empty, re-occupying"
    occupy_gpu "$g"
}

case "${1:-}" in
    init) init ;;
    worker) worker "$2" ;;
    *) echo "usage: $0 init | worker <gpu>" >&2; exit 2 ;;
esac
