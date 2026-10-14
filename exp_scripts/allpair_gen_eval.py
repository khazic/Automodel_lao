"""Batched greedy generation for the all-pair translation suite, one GPU per process.

Every prompt is rendered as a single user turn with the checkpoint's chat template and
decoded greedily (the suite's usual temperature 0 / max_tokens 1012) without a KV cache:
the n-gram (Engram) branch hashes raw ids and runs a causal conv over the whole sequence,
which incremental decoding would break. Without a cache each row is an independent causal
forward, so rows are right padded and the padding never reaches the real tokens. Every
step sorts the active rows by length and splits them into chunks under a token budget,
so one long row does not pad the others. Finished rows are refilled from the queue.

  gen    decode rows idx % world == rank into an {"idx", "res"} JSONL (resumable)
  merge  join the parts back into the {"prompt", "res"} JSONL that eval_allpair.py reads
"""

import argparse
import glob
import json
import os
import time
from collections import deque

# The n-gram defaults of the E2B mala runs (e2b_ngram_mala.yaml and the engram SFT).
NGRAM = dict(ngram_size=3, heads_per_ngram=8, head_dim=96, rows_per_head=666000, layer_index=1, eos_token_ids=[1, 2])
# generation_config.json of gemma-4-E2B-it: <eos>, <turn|> and token 50.
EOS = (1, 106, 50)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build(weights, with_ngram):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration

    cfg = AutoConfig.from_pretrained(weights)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        model = Gemma4ForConditionalGeneration(cfg, ngram_config=NGRAM if with_ngram else None)
    torch.set_default_dtype(torch.float32)
    sd = {}
    for f in sorted(glob.glob(os.path.join(weights, "*.safetensors"))):
        sd.update(load_file(f, device="cuda"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    # The base release still ships k/v projections for the KV-shared layers, which the HF model does not build.
    unexpected = [
        k
        for k in unexpected
        if not k.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.k_norm.weight"))
    ]
    log(f"loaded {len(sd)} tensors from {weights}")
    log(f"missing={missing[:8]} ({len(missing)}) unexpected={unexpected[:8]} ({len(unexpected)})")
    if missing or unexpected:
        raise SystemExit("state dict mismatch")
    if with_ngram and not any(".ngram." in k for k in sd):
        raise SystemExit("--ngram given but the checkpoint has no n-gram weights")
    model.tie_weights()
    log(f"attn_implementation={model.config.text_config._attn_implementation} ngram={model.ngram is not None}")
    return model.eval()


class Stepper:
    """Next greedy token for each row of a right-padded batch.

    Logits are only needed at each row's own last position, which differs per row, so the
    final-norm output is captured with a hook and only those positions go through lm_head
    (softcapping is monotonic and leaves the argmax unchanged).
    """

    def __init__(self, model, pad_id):
        self.model = model
        self.pad_id = pad_id
        self.hidden = None
        model.model.language_model.norm.register_forward_hook(self._capture)

    def _capture(self, _module, _args, out):
        self.hidden = out

    def __call__(self, seqs):
        import torch

        lens = torch.tensor([len(s) for s in seqs])
        x = torch.full((len(seqs), int(lens.max())), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
        self.model(input_ids=x.cuda(non_blocking=True), use_cache=False, logits_to_keep=1)
        last = self.hidden[torch.arange(len(seqs), device="cuda"), lens.cuda() - 1]
        return self.model.lm_head(last).argmax(-1).tolist()


def self_check(model, step, rows):
    """The padded batch path must pick the same next token as a plain unpadded forward."""
    import torch

    batched = step([r[1] for r in rows])
    for (idx, seq), b in zip(rows, batched):
        x = torch.tensor([seq], device="cuda")
        ref = int(model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].argmax())
        log(f"self-check idx {idx} len {len(seq)}: batched {b} unpadded {ref} {'OK' if b == ref else 'MISMATCH'}")


def gen(args):
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.weights)
    with open(args.prompts) as fh:
        prompts = [json.loads(line)["prompt"] for line in fh if line.strip()]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["idx"])
                except ValueError:
                    pass  # a line cut off by a crash; that row is decoded again
    mine = [i for i in range(len(prompts)) if i % args.world == args.rank and i not in done]
    log(f"rank {args.rank}/{args.world}: {len(mine)} rows to decode, {len(done)} already in {args.out}")
    if not mine:
        return

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

    log(f"rendered prompt {mine[0]}: {chat(prompts[mine[0]])!r}")
    queue = [(i, tok(chat(prompts[i]), add_special_tokens=False).input_ids) for i in mine]
    queue.sort(key=lambda r: -len(r[1]))

    model = build(args.weights, args.ngram)
    step = Stepper(model, tok.pad_token_id if tok.pad_token_id is not None else 0)
    eos = set(EOS)
    with torch.inference_mode():
        self_check(model, step, [queue[0], queue[len(queue) // 2], queue[-1]])

        pending = deque(queue)
        active = []  # [idx, prompt_len, ids]
        out = open(args.out, "a")
        n_done, n_tok, t0, last_log, unflushed = 0, 0, time.time(), time.time(), 0
        while pending or active:
            while pending and len(active) < args.max_rows:
                idx, ids = pending.popleft()
                active.append([idx, len(ids), list(ids)])
            active.sort(key=lambda r: len(r[2]))
            start = 0
            while start < len(active):
                end = start + 1
                while end < len(active) and (end + 1 - start) * len(active[end][2]) <= args.token_budget:
                    end += 1
                chunk = active[start:end]
                for r, t in zip(chunk, step([r[2] for r in chunk])):
                    r[2].append(t)
                start = end
            n_tok += len(active)
            still = []
            for r in active:
                new = r[2][r[1] :]
                if new[-1] in eos or len(new) >= args.max_new_tokens:
                    text = tok.decode(new, skip_special_tokens=True)
                    out.write(json.dumps({"idx": r[0], "res": text}, ensure_ascii=False) + "\n")
                    n_done += 1
                    unflushed += 1
                else:
                    still.append(r)
            active = still
            if unflushed >= 200:
                out.flush()
                unflushed = 0
            if time.time() - last_log > 60:
                el = time.time() - t0
                eta = (len(queue) - n_done) / max(n_done / el, 1e-9) / 60
                log(f"{n_done}/{len(queue)} rows, {n_tok / el:.0f} tok/s, {len(active)} active, ETA {eta:.0f} min")
                last_log = time.time()
        out.close()
    log(f"DONE {n_done} rows in {(time.time() - t0) / 60:.1f} min")


def merge(args):
    with open(args.prompts) as fh:
        prompts = [json.loads(line)["prompt"] for line in fh if line.strip()]
    res = {}
    for path in args.parts:
        with open(path) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                res[row["idx"]] = row["res"]
    missing = [i for i in range(len(prompts)) if i not in res]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(prompts)} rows missing, first {missing[:5]}")
    with open(args.out, "w") as fh:
        for i, p in enumerate(prompts):
            fh.write(json.dumps({"prompt": p, "res": res[i]}, ensure_ascii=False) + "\n")
    log(f"wrote {len(prompts)} rows to {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--weights", required=True, help="consolidated HF checkpoint dir")
    g.add_argument("--tokenizer", help="tokenizer/chat template dir (default: --weights)")
    g.add_argument("--prompts", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--rank", type=int, default=0)
    g.add_argument("--world", type=int, default=1)
    g.add_argument("--ngram", action="store_true", help="build the n-gram branch (engram checkpoints)")
    g.add_argument("--max-new-tokens", type=int, default=1012)
    g.add_argument("--max-rows", type=int, default=384)
    g.add_argument("--token-budget", type=int, default=49152, help="padded tokens per forward")
    m = sub.add_parser("merge")
    m.add_argument("--prompts", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("parts", nargs="+")
    args = ap.parse_args()
    gen(args) if args.cmd == "gen" else merge(args)


if __name__ == "__main__":
    main()
