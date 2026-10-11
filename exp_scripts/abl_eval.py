"""Evaluate one model of the E2B n-gram (Engram) mala ablation on one task.

Models are the untrained gemma-4-E2B base, the no-n-gram control arm, the main
n-gram run and the ablation arms. Each (model, task) pair writes one JSON file;
abl_report.py aggregates them.

Tasks:
  heldout  FLORES-200 devtest per-language loss (sentences of one source URL are
           joined into a document); n-gram arms also report the loss with the
           n-gram delta switched off.
  indist   per-source loss on documents sampled from the training corpus, the
           same sample as ngram_infer_eval.py (seed 0, 24 per source).
  mt       FLORES-200 3-shot translation, English <-> each target language,
           greedy decoding, chrF++.
  lmeval   English zero-shot benchmarks through lm-evaluation-harness.

usage: python abl_eval.py --name <tag> --weights <dir> [--yaml <run yaml>] --task <task> --out <json>
"""

import argparse
import json
import os
import time

import pandas as pd
import torch
import yaml
from ngram_infer_eval import build, doc_nll, greedy, log, sample_docs, set_ngram
from transformers import AutoTokenizer

BASE = "/llm-align/open_models/gemma4/gemma-4-E2B"
FLORES = "/llm-align/liuchonghan/datasets/flores200_ms"
MALA = "/llm-align/duyimin/multi_lang/mala/Qwen2.5-7B-8Langs-24Btokens.jsonl"

# FLORES code -> language name used in the translation prompt.
TARGETS = {
    "zsm_Latn": "Malay",
    "ind_Latn": "Indonesian",
    "tha_Thai": "Thai",
    "vie_Latn": "Vietnamese",
    "khm_Khmr": "Khmer",
    "lao_Laoo": "Lao",
    "mya_Mymr": "Burmese",
    "tam_Taml": "Tamil",
    "tgl_Latn": "Filipino",
    "bod_Tibt": "Tibetan",
    "uig_Arab": "Uyghur",
    "khk_Cyrl": "Mongolian",
}
HELDOUT_LANGS = ["eng_Latn", "zho_Hans", *TARGETS]
LMEVAL_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "boolq", "mmlu"]


def flores(lang, split):
    return pd.read_parquet(os.path.join(FLORES, lang, split + ".parquet"))


def ngram_variants(model, with_ngram):
    """Yield (suffix, flag) pairs: on/off for n-gram models, a single pass otherwise."""
    if not with_ngram:
        yield "", None
        return
    for suffix, flag in (("", True), ("_ngram_off", False)):
        set_ngram(model, flag)
        yield suffix, flag
    set_ngram(model, True)


def corpus_loss(model, docs):
    tot, n = 0.0, 0
    for ids in docs:
        a, b = doc_nll(model, ids)
        tot, n = tot + a, n + b
    return tot / n


def task_heldout(model, tok, with_ngram, args):
    docs = {}
    for lang in HELDOUT_LANGS:
        df = flores(lang, "devtest")
        sep = "" if lang == "zho_Hans" else " "
        texts = df.groupby("URL", sort=False)["sentence"].apply(sep.join).tolist()
        docs[lang] = [
            [tok.bos_token_id] + tok(t, add_special_tokens=False).input_ids[: args.max_len - 1] for t in texts
        ]
    out = {}
    for suffix, _ in ngram_variants(model, with_ngram):
        out["loss" + suffix] = {lang: corpus_loss(model, d) for lang, d in docs.items()}
        log(f"heldout{suffix}: mean {sum(out['loss' + suffix].values()) / len(docs):.4f}")
    return out


def task_indist(model, tok, with_ngram, args):
    raw = sample_docs(MALA, 24)
    docs = {
        k: [[tok.bos_token_id] + tok(t, add_special_tokens=False).input_ids[: args.max_len - 1] for t in v]
        for k, v in raw.items()
    }
    out = {}
    for suffix, _ in ngram_variants(model, with_ngram):
        out["loss" + suffix] = {src: corpus_loss(model, d) for src, d in docs.items()}
        log(f"indist{suffix}: mean {sum(out['loss' + suffix].values()) / len(docs):.4f}")
    return out


def task_mt(model, tok, with_ngram, args):
    import sacrebleu

    eng_dev, eng_test = flores("eng_Latn", "dev")["sentence"], flores("eng_Latn", "devtest")["sentence"]
    eos = {1, 2, 106}
    newline = set(tok("\n", add_special_tokens=False).input_ids) | set(tok("\n\n", add_special_tokens=False).input_ids)
    out = {"chrf": {}, "samples": {}}
    for code, name in TARGETS.items():
        dev, test = flores(code, "dev")["sentence"], flores(code, "devtest")["sentence"]
        for src_name, tgt_name, src_dev, tgt_dev, src_test, tgt_test in (
            ("English", name, eng_dev, dev, eng_test, test),
            (name, "English", dev, eng_dev, test, eng_test),
        ):
            shots = "".join(f"{src_name}: {src_dev[i]}\n{tgt_name}: {tgt_dev[i]}\n\n" for i in range(args.shots))
            head = f"Translate the following sentences from {src_name} to {tgt_name}.\n\n" + shots
            hyps = []
            for i in range(args.mt_n):
                ids = [tok.bos_token_id] + tok(
                    head + f"{src_name}: {src_test[i]}\n{tgt_name}:", add_special_tokens=False
                ).input_ids
                gen = greedy(model, ids, args.new_tokens, eos | newline)
                hyps.append(tok.decode(gen, skip_special_tokens=True).split("\n")[0].strip())
            refs = list(tgt_test[: args.mt_n])
            key = f"{'eng' if src_name == 'English' else code}->{'eng' if tgt_name == 'English' else code}"
            out["chrf"][key] = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
            out["samples"][key] = [{"hyp": h, "ref": r} for h, r in zip(hyps[:5], refs[:5])]
            log(f"mt {key}: chrF++ {out['chrf'][key]:.2f}")
    return out


def task_lmeval(model, tok, with_ngram, args):
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size, max_length=2048, add_bos_token=True)
    res = lm_eval.simple_evaluate(model=lm, tasks=LMEVAL_TASKS, num_fewshot=0, log_samples=False)
    out = {}
    for task, metrics in res["results"].items():
        out[task] = {
            k.split(",")[0]: v
            for k, v in metrics.items()
            if k.split(",")[0] in ("acc", "acc_norm") and isinstance(v, float)
        }
    return {"results": out}


TASKS = {"heldout": task_heldout, "indist": task_indist, "mt": task_mt, "lmeval": task_lmeval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--yaml", help="run yaml; its model.ngram_config builds the n-gram branch (absent = plain model)")
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--mt-n", type=int, default=100, help="devtest sentences per translation direction")
    ap.add_argument("--new-tokens", type=int, default=192)
    # 262k-vocab logits: a batch of 8 x 2048 tokens is already ~8 GB in bf16.
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    ngram = None
    if args.yaml:
        with open(args.yaml) as fh:
            ngram = yaml.safe_load(fh)["model"].get("ngram_config")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = build(BASE, args.weights, ngram)
    t0 = time.time()
    result = TASKS[args.task](model, tok, ngram is not None, args)
    result.update(
        name=args.name, task=args.task, weights=args.weights, ngram_config=ngram, seconds=round(time.time() - t0)
    )
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    log(f"{args.name} {args.task} done in {result['seconds']}s")


if __name__ == "__main__":
    main()
