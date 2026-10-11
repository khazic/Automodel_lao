#!/bin/bash
# Finish the per-model translation jobs of abl_eval_queue.sh faster by splitting
# the directions that are still missing into small jobs over every spare GPU.
#
#   abl_mt_split.sh plan            chunk the missing directions of every running mt
#                                   job (finished ones are read from its log) into
#                                   $E/queue_mt; the monolithic job is retired
#   abl_mt_split.sh worker <gpu>    run chunk jobs on this GPU (two workers per GPU is
#                                   fine: one decode stream leaves it mostly idle);
#                                   occupy the GPU again once nothing runs on it
#   abl_mt_split.sh merge           write results/<model>__mt.json from the log and
#                                   the chunk results once all 24 directions exist
#
# Uses its own queue dir so workers of abl_eval_queue.sh, which hold their code in
# memory, never pick up these jobs.
set -u
L=/llm-align/liuchonghan
E=$L/runs/gemma4_ngram/eval_abl
CODE=$L/Automodel_ngram_eval
Q=$E/queue_mt
CODES="zsm_Latn ind_Latn tha_Thai vie_Latn khm_Khmr lao_Laoo mya_Mymr tam_Taml tgl_Latn bod_Tibt uig_Arab khk_Cyrl"
CHUNK=2

say() { echo "[$(date +%F_%T)] $*"; }

plan() {
    mkdir -p "$Q" "$E/claims_mt" "$E/done_mt" "$E/attempts_mt" "$E/results_mt"
    local spec job name weights yaml task log todo d k i
    for spec in "$E"/queue/1_*__mt.job; do
        [ -e "$spec" ] || continue
        job=$(basename "$spec" .job)
        IFS='|' read -r name weights yaml task < "$spec"
        log=$E/logs/$job.log
        todo=()
        for d in $CODES; do
            for k in "eng->$d" "$d->eng"; do
                grep -qF "mt $k: chrF++" "$log" 2>/dev/null || todo+=("$k")
            done
        done
        i=0
        while [ $i -lt ${#todo[@]} ]; do
            dirs=$(IFS=,; echo "${todo[*]:$i:$CHUNK}")
            echo "$name|$weights|$yaml|$dirs" > "$Q/${name}__p$(printf %02d $((i / CHUNK))).job"
            i=$((i + CHUNK))
        done
        # Keep the spec for merge; the old worker must not re-run this job.
        mv "$spec" "$E/done/$job.split"
        say "$name: ${#todo[@]} directions left -> $(( (${#todo[@]} + CHUNK - 1) / CHUNK )) jobs"
    done
}

occupy_when_idle() {
    local g=$1
    while nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
    ps -eo args | grep -qE "^bash (\S*/)?keep_gpu_busy\.sh \"?$g\"? " && return
    ( cd $L/openrlhf && exec setsid nohup bash $L/openrlhf/keep_gpu_busy.sh "$g" 0.8 0.8 >> "$L/runs/zhanka_$(hostname -I | cut -d' ' -f1)_gpu$g.log" 2>&1 < /dev/null & )
}

worker() {
    local g=$1 f job name weights yaml dirs n pid
    source $L/env/etc/profile.d/conda.sh && conda activate automodel
    export PATH=$L/envs/ngram_eval/bin:$PATH PYTHONPATH=$CODE:$CODE/exp_scripts PYTHONUNBUFFERED=1
    export LD_LIBRARY_PATH=$L/env/envs/automodel/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
    export HF_HOME=$L/hf_cache TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$g
    for pid in $(ps -eo pid,args | grep -E "^ *[0-9]+ bash (\S*/)?keep_gpu_busy\.sh \"?$g\"? " | awk '{print $1}'); do
        kill -9 -- "-$(ps -o pgid= -p "$pid" | tr -d ' ')" 2>/dev/null
    done
    while true; do
        job=""
        for f in "$Q"/*.job; do
            [ -e "$f" ] || break
            job=$(basename "$f" .job)
            mkdir "$E/claims_mt/$job" 2>/dev/null && break
            job=""
        done
        [ -z "$job" ] && break
        IFS='|' read -r name weights yaml dirs < "$Q/$job.job"
        n=$(( $(cat "$E/attempts_mt/$job" 2>/dev/null || echo 0) + 1 ))
        echo "$n" > "$E/attempts_mt/$job"
        say "GPU $g: $job ($dirs, attempt $n)"
        args=(--name "$name" --weights "$weights" --task mt --mt-dirs "$dirs" --out "$E/results_mt/$job.json")
        [ "$yaml" != "-" ] && args+=(--yaml "$yaml")
        if python -u $CODE/exp_scripts/abl_eval.py "${args[@]}" > "$E/logs/mt_$job.log" 2>&1; then
            mv "$Q/$job.job" "$E/done_mt/$job"
        elif [ "$n" -ge 3 ]; then
            say "GPU $g: $job failed $n times, giving up"
            mv "$Q/$job.job" "$E/done_mt/$job.failed"
        else
            rmdir "$E/claims_mt/$job"
        fi
    done
    say "GPU $g: queue empty"
    occupy_when_idle "$g"
}

merge() {
    python3 - "$E" <<'EOF'
import glob, json, os, re, sys

E = sys.argv[1]
want = 24
for spec in sorted(glob.glob(f"{E}/done/1_*__mt.split")):
    job = os.path.basename(spec)[: -len(".split")]
    name, weights, yaml_path, _ = open(spec).read().strip().split("|")
    chrf, samples = {}, {}
    for line in open(f"{E}/logs/{job}.log", errors="replace"):
        m = re.search(r"mt (\S+->\S+): chrF\+\+ ([0-9.]+)", line)
        if m:
            chrf[m.group(1)] = float(m.group(2))
    for part in glob.glob(f"{E}/results_mt/*{name}__p*.json"):
        r = json.load(open(part))
        chrf.update(r["chrf"])
        samples.update(r["samples"])
    if len(chrf) < want:
        print(f"{name}: {len(chrf)}/{want} directions, not merged")
        continue
    out = {"chrf": chrf, "samples": samples, "name": name, "task": "mt", "weights": weights}
    json.dump(out, open(f"{E}/results/{name}__mt.json", "w"), ensure_ascii=False, indent=1)
    print(f"{name}: merged {len(chrf)} directions")
EOF
}

case "${1:-}" in
    plan) plan ;;
    worker) worker "$2" ;;
    merge) merge ;;
    *) echo "usage: $0 plan | worker <gpu> | merge" >&2; exit 2 ;;
esac
