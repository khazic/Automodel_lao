"""Score unfinished all-pair runs on the rows every model has already decoded.

eval_allpair.py needs all 91080 rows. While allpair_gen_eval.py is still running, this scores each
direction on the intersection of finished rows, with eval_allpair.py's own shard parsing, references
and bleu_one (imported, not copied), so models are compared on identical sentences. The generator
decodes longest prompts first, so the subset leans long and only the comparison is meaningful, not
the absolute values.

usage: python allpair_partial_bleu.py --scorer-dir <dir of eval_allpair.py> --prompts <prompt jsonl>
           <name>=<part glob> [<name>=<part glob> ...]
"""

import argparse
import glob
import json
import sys
from collections import OrderedDict
from multiprocessing import Pool


def load_parts(pattern):
    res = {}
    for path in glob.glob(pattern):
        with open(path) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # the line being written right now
                res[row["idx"]] = row["res"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer-dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("models", nargs="+", help="name=glob of that model's part_*.jsonl")
    args = ap.parse_args()
    sys.path.insert(0, args.scorer_dir)
    import eval_allpair as ea

    size = ea.DEFAULT_SHARD_SIZE
    langs = {idx: lang for idx, lang, _ in ea.read_shards(args.prompts, size)}
    refs = {lang: ea.read_reference(ea.DEFAULT_REF_DIR, lang, size)[1] for lang in set(langs.values())}
    outs = OrderedDict((spec.split("=", 1)[0], load_parts(spec.split("=", 1)[1])) for spec in args.models)
    common = set.intersection(*(set(o) for o in outs.values()))
    print(f"rows finished by every model: {len(common)} / {len(langs) * size}")

    tasks = []
    for name, out in outs.items():
        for shard, lang in sorted(langs.items()):
            rows = sorted(i for i in common if i // size == shard)
            if rows:
                cands = [out[i] for i in rows]
                ref = [refs[lang][i % size] for i in rows]
                tasks.append(((name, len(rows)), (shard, lang, "", cands, ref, ea.DEFAULT_TOKENIZER_LANG)))
    with Pool(args.workers) as pool:
        scored = pool.map(ea.bleu_one, [t[1] for t in tasks])

    table = OrderedDict((name, {}) for name in outs)
    for (name, n), (shard, lang, score, err) in zip((t[0] for t in tasks), scored):
        if err:
            raise SystemExit(f"{name} shard {shard}: {err}")
        table[name][shard] = (lang, score, n)

    names = list(outs)
    print(f"\ndirections scored: {len(table[names[0]])}   (BLEU x100, same rows for every model)")
    print(f"{'':12s}" + "".join(f"{n:>10s}" for n in names))
    print(
        f"{'mean':12s}"
        + "".join(f"{100 * sum(s for _, s, _ in table[n].values()) / len(table[n]):10.2f}" for n in names)
    )
    by_lang = OrderedDict()
    for shard, (lang, _, _) in sorted(table[names[0]].items()):
        by_lang.setdefault(lang, []).append(shard)
    print("\nper target language:")
    for lang, shards in sorted(by_lang.items()):
        avg = [100 * sum(table[n][s][1] for s in shards) / len(shards) for n in names]
        rows = sum(table[names[0]][s][2] for s in shards)
        print(f"  {lang:10s}" + "".join(f"{a:10.2f}" for a in avg) + f"   ({len(shards)} dirs, {rows} rows)")


if __name__ == "__main__":
    main()
