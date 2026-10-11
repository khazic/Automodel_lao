"""Aggregate abl_eval.py results into one Markdown report.

Every metric table lists each model with its delta against the untrained base
E2B and against the no-n-gram control arm. Loss is mean nats per token (lower is
better); chrF++ and accuracy are higher-is-better.

usage: python abl_report.py <results dir> <report.md>
"""

import glob
import json
import os
import sys

ORDER = [
    "base_e2b",
    "control_14999",
    "main_24999",
    "tablelr10_14999",
    "frozenbase_14999",
    "ngram2_14999",
    "ngram4_14999",
    "layer0_14999",
    "layer8_14999",
    "table0.25b_14999",
    "table4b_4999",
]


def load(results_dir):
    res = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        res.setdefault(r["name"], {})[r["task"]] = r
    return res


def mean(d):
    return sum(d.values()) / len(d) if d else float("nan")


def fmt(x, nd):
    return "-" if x is None else f"{x:.{nd}f}"


def table(title, note, cols, rows, nd, lower_better):
    """rows: name -> {col: value}; adds delta-vs-base/control columns on the first col."""
    lines = [f"## {title}", "", note, "", "| model | " + " | ".join(cols) + " | Δ base | Δ control |"]
    lines.append("|" + "---|" * (len(cols) + 3))
    key = cols[0]
    ref_base = rows.get("base_e2b", {}).get(key)
    ref_ctrl = rows.get("control_14999", {}).get(key)
    for name in [n for n in ORDER if n in rows] + sorted(set(rows) - set(ORDER)):
        v = rows[name]
        x = v.get(key)
        db = None if x is None or ref_base is None else x - ref_base
        dc = None if x is None or ref_ctrl is None else x - ref_ctrl
        lines.append(
            f"| {name} | " + " | ".join(fmt(v.get(c), nd) for c in cols) + f" | {fmt(db, nd)} | {fmt(dc, nd)} |"
        )
    arrow = "lower is better" if lower_better else "higher is better"
    lines += ["", f"Deltas are on `{key}` ({arrow}).", ""]
    return lines


def main():
    res = load(sys.argv[1])
    out = ["# E2B Engram mala ablation: evaluation report", ""]
    out.append(
        "Checkpoints: arms at step 14999, main run at 24999, table4b at 4999 (steps differ, see model names); "
        "base_e2b is the untrained gemma-4-E2B; control is the same training without the n-gram branch."
    )
    out.append("")

    for task, title, note in (
        (
            "heldout",
            "Held-out loss (FLORES-200 devtest)",
            "Mean NLL per token over 14 languages; never seen in training.",
        ),
        (
            "indist",
            "In-distribution loss (training corpus sample)",
            "Same 24-docs-per-source sample as the 10-06 eval; part of it has been trained on.",
        ),
    ):
        rows, langs = {}, []
        for name, r in res.items():
            if task not in r:
                continue
            on = r[task]["loss"]
            langs = sorted(on)
            rows[name] = {"mean": mean(on), "mean (ngram off)": mean(r[task].get("loss_ngram_off", {})) or None}
            rows[name].update(on)
        if rows:
            cols = ["mean", "mean (ngram off)"] + (langs if task == "heldout" else [])
            out += table(title, note, cols, rows, 4, True)

    rows = {}
    for name, r in res.items():
        if "mt" not in r:
            continue
        chrf = r["mt"]["chrf"]
        to_x = {k: v for k, v in chrf.items() if k.startswith("eng->")}
        to_en = {k: v for k, v in chrf.items() if k.endswith("->eng")}
        rows[name] = {"avg": mean(chrf), "en->xx": mean(to_x), "xx->en": mean(to_en)}
        rows[name].update({k.replace("eng->", "en→"): v for k, v in to_x.items()})
    if rows:
        cols = ["avg", "en->xx", "xx->en"] + sorted(k for k in next(iter(rows.values())) if k.startswith("en→"))
        out += table(
            "Translation (FLORES-200, 3-shot, chrF++)",
            "100 devtest sentences per direction, greedy.",
            cols,
            rows,
            2,
            False,
        )

    rows = {}
    for name, r in res.items():
        if "lmeval" not in r:
            continue
        accs = {t: m.get("acc_norm", m.get("acc")) for t, m in r["lmeval"]["results"].items() if m}
        core = {t: v for t, v in accs.items() if not t.startswith("mmlu_")}
        rows[name] = {"avg": mean(core), **core}
    if rows:
        cols = ["avg"] + sorted(k for k in next(iter(rows.values())) if k != "avg")
        out += table("English zero-shot (lm-eval)", "acc_norm where defined, else acc.", cols, rows, 4, False)

    with open(sys.argv[2], "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
