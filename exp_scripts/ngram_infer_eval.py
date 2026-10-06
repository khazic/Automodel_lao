"""Load the E2B n-gram (Engram) mala checkpoint on one GPU, compare per-language
loss against the base E2B and against itself with the n-gram delta switched off,
then print greedy continuations.

usage: python ngram_infer_eval.py --ckpt <consolidated dir> --base <E2B dir> --data <jsonl> --out <json>
"""

import argparse
import glob
import json
import os
import random
import time

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration

NGRAM = dict(ngram_size=3, heads_per_ngram=8, head_dim=96, rows_per_head=666000, layer_index=1, eos_token_ids=[1, 2])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build(base, weights_dir, with_ngram):
    cfg = AutoConfig.from_pretrained(base)
    cfg.torch_dtype = torch.bfloat16
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        model = Gemma4ForConditionalGeneration(cfg, ngram_config=NGRAM if with_ngram else None)
    torch.set_default_dtype(torch.float32)
    sd = {}
    for f in sorted(glob.glob(os.path.join(weights_dir, "*.safetensors"))):
        sd.update(load_file(f, device="cuda"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    # The base release still ships k/v projections for the KV-shared layers,
    # which the HF model does not build.
    unexpected = [k for k in unexpected if not k.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.k_norm.weight"))]
    log(f"loaded {len(sd)} tensors from {weights_dir}; missing={missing[:8]} ({len(missing)}) unexpected={unexpected[:8]} ({len(unexpected)})")
    if missing or unexpected:
        raise SystemExit("state dict mismatch")
    model.tie_weights()
    return model.eval()


def sample_docs(path, per_source, seed=0):
    rng = random.Random(seed)
    size = os.path.getsize(path)
    docs = {}
    with open(path, "rb") as fh:
        for _ in range(4000):
            fh.seek(rng.randrange(size))
            fh.readline()
            line = fh.readline()
            try:
                row = json.loads(line)
            except ValueError:
                continue
            src, text = row.get("source"), row.get("text")
            if not src or not text or len(text) < 400:
                continue
            if src.startswith("code_"):
                src = "code"
            bucket = docs.setdefault(src, [])
            if len(bucket) < per_source:
                bucket.append(text)
    return docs


@torch.no_grad()
def doc_nll(model, ids):
    x = torch.tensor([ids], device="cuda")
    logits = model(input_ids=x, use_cache=False).logits.float()
    loss = torch.nn.functional.cross_entropy(logits[0, :-1], x[0, 1:], reduction="sum")
    return loss.item(), len(ids) - 1


@torch.no_grad()
def greedy(model, ids, new_tokens, eos):
    x = torch.tensor([ids], device="cuda")
    for _ in range(new_tokens):
        nxt = model(input_ids=x, use_cache=False).logits[0, -1].argmax()
        x = torch.cat([x, nxt.view(1, 1)], dim=1)
        if nxt.item() in eos:
            break
    return x[0, len(ids):].tolist()


def set_ngram(model, enabled):
    model._ngram_hook_handle.remove()
    ngram = model.ngram
    if enabled:
        layer = model.model.language_model.layers[ngram.config.layer_index]
        model._ngram_hook_handle = layer.register_forward_pre_hook(ngram.decoder_layer_pre_hook, with_kwargs=True)
    else:
        model._ngram_hook_handle = model.model.language_model.layers[0].register_forward_pre_hook(lambda *a, **k: None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", type=int, default=24)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--new-tokens", type=int, default=160)
    ap.add_argument("--resume", action="store_true", help="skip models already in --out")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    docs = sample_docs(args.data, args.per_source)
    log("sampled: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(docs.items())))
    enc = {k: [[tok.bos_token_id] + tok(t, add_special_tokens=False).input_ids[: args.max_len - 1] for t in v] for k, v in docs.items()}

    prompts = []
    for src in ["malay", "indonesian", "thai", "vietnamese", "khmer", "lao", "myanmar", "tamil", "filipino", "tibetan", "uighur", "mongolian"]:
        if src in docs:
            prompts.append((src, [tok.bos_token_id] + tok(docs[src][0], add_special_tokens=False).input_ids[:48]))
    for src, text in [
        ("en_qa", "Question: What is the capital of Malaysia, and what is it known for?\nAnswer:"),
        ("zh_qa", "问题：请简要介绍一下泰国的首都。\n回答："),
        ("translate", "Translate the following English sentence into Malay.\nEnglish: The weather is very hot today, so we stayed at home.\nMalay:"),
        ("translate_th", "Translate the following English sentence into Thai.\nEnglish: Where is the nearest train station?\nThai:"),
    ]:
        prompts.append((src, [tok.bos_token_id] + tok(text, add_special_tokens=False).input_ids))

    eos = {1, 2, 106}
    result = {"loss": {}, "gen": {}}
    runs = [("ngram_64999", args.ckpt, True), ("base_e2b", args.base, False)]
    if args.resume and os.path.exists(args.out):
        with open(args.out) as fh:
            result = json.load(fh)
        runs = [r for r in runs if r[0] not in result["loss"]]
    for name, weights, with_ngram in runs:
        model = build(args.base, weights, with_ngram)
        variants = [("", True), ("_ngram_off", False)] if with_ngram else [("", None)]
        for suffix, flag in variants:
            if flag is not None:
                set_ngram(model, flag)
            tag = name + suffix
            per = {}
            t0 = time.time()
            for src, seqs in enc.items():
                tot, n = 0.0, 0
                for ids in seqs:
                    a, b = doc_nll(model, ids)
                    tot, n = tot + a, n + b
                per[src] = tot / n
            result["loss"][tag] = per
            log(f"{tag}: loss done in {time.time() - t0:.0f}s, mean {sum(per.values()) / len(per):.4f}")
            if flag is not False:
                gens = {}
                for src, ids in prompts:
                    out = greedy(model, ids, args.new_tokens, eos)
                    gens[src] = {"prompt": tok.decode(ids[1:]), "output": tok.decode(out)}
                result["gen"][tag] = gens
                log(f"{tag}: generation done")
            with open(args.out, "w") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=1)
        del model
        torch.cuda.empty_cache()

    srcs = sorted(enc)
    tags = list(result["loss"])
    print("\n%-26s" % "source" + "".join("%16s" % t for t in tags))
    for s in srcs:
        print("%-26s" % s + "".join("%16.4f" % result["loss"][t][s] for t in tags))
    print("%-26s" % "MEAN" + "".join("%16.4f" % (sum(result["loss"][t].values()) / len(srcs)) for t in tags))
    for tag, gens in result["gen"].items():
        print(f"\n===== {tag}")
        for src, g in gens.items():
            print(f"--- [{src}] PROMPT: {g['prompt']!r}\n    OUTPUT: {g['output']!r}")
    log("EVAL_DONE")


if __name__ == "__main__":
    main()
