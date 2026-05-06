# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate xiaoyuzhong parquet dataset for Gemma4 SFT training.

Usage (raw format check only, no tokenizer needed):
    python validate_dataset.py --files /path/to/*.parquet

Usage (with label-mask check, requires Gemma4 processor):
    python validate_dataset.py --files /path/to/*.parquet \
        --processor /llm-align/open_models/gemma4/gemma-4-31B-it \
        --max_length 4096 \
        --label_check_samples 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Step 1: raw parquet / messages format checks (no tokenizer)
# ---------------------------------------------------------------------------


def check_messages_column(path: str) -> dict:
    table = pq.read_table(path)
    n_rows = table.num_rows
    if "messages" not in table.column_names:
        return {"path": path, "n_rows": n_rows, "has_messages_col": False}
    return {"path": path, "n_rows": n_rows, "has_messages_col": True, "columns": table.column_names}


def validate_sample(messages, idx: int) -> list[str]:
    """Return list of issue strings for a single sample (empty = OK)."""
    issues = []
    if not isinstance(messages, list):
        issues.append(f"sample[{idx}] messages is not a list: {type(messages)}")
        return issues

    roles_seen = Counter()
    for j, msg in enumerate(messages):
        if not isinstance(msg, dict):
            issues.append(f"sample[{idx}] msg[{j}] is not a dict: {type(msg)}")
            continue
        role = msg.get("role", "")
        if role not in {"system", "user", "assistant"}:
            issues.append(f"sample[{idx}] msg[{j}] unknown role: {repr(role)}")
        content = msg.get("content", "")
        if isinstance(content, list):
            issues.append(
                f"sample[{idx}] msg[{j}] content is a list "
                f"(will be stringified by _to_conversation): {repr(content)[:80]}"
            )
        elif content is None or str(content).strip() == "":
            issues.append(f"sample[{idx}] msg[{j}] role={role!r} has empty content")
        roles_seen[role] += 1

    if roles_seen.get("user", 0) == 0:
        issues.append(f"sample[{idx}] has no user turn")
    if roles_seen.get("assistant", 0) == 0:
        issues.append(f"sample[{idx}] has no assistant turn")

    return issues


def check_parquet_file(path: str, max_samples: int | None = None) -> dict:
    table = pq.read_table(path, columns=["messages"])
    messages_col = table["messages"].to_pylist()

    n_total = len(messages_col)
    n_check = n_total if max_samples is None else min(n_total, max_samples)

    all_issues: list[str] = []
    role_dist: Counter = Counter()
    content_list_count = 0

    for i in range(n_check):
        raw = messages_col[i]
        issues = validate_sample(raw, i)
        for iss in issues:
            if "content is a list" in iss:
                content_list_count += 1
            else:
                all_issues.append(iss)
        if isinstance(raw, list):
            for msg in raw:
                if isinstance(msg, dict):
                    role_dist[msg.get("role", "unknown")] += 1

    return {
        "path": path,
        "n_total": n_total,
        "n_checked": n_check,
        "n_issues": len(all_issues),
        "content_is_list_count": content_list_count,
        "role_distribution": dict(role_dist),
        "sample_issues": all_issues[:20],  # show first 20
    }


# ---------------------------------------------------------------------------
# Step 2: token length distribution (no processor needed via rough estimate)
# ---------------------------------------------------------------------------


def estimate_token_lengths(path: str, max_samples: int = 2000) -> dict:
    """Rough estimate: count characters and divide by 3 (Gemma4 ~3 chars/token for Latin)."""
    table = pq.read_table(path, columns=["messages"])
    messages_col = table["messages"].to_pylist()
    n = min(len(messages_col), max_samples)

    char_lengths = []
    for i in range(n):
        raw = messages_col[i]
        if not isinstance(raw, list):
            continue
        total_chars = sum(len(str(msg.get("content", ""))) for msg in raw if isinstance(msg, dict))
        char_lengths.append(total_chars)

    if not char_lengths:
        return {}

    char_lengths.sort()
    p50 = char_lengths[len(char_lengths) // 2]
    p95 = char_lengths[int(len(char_lengths) * 0.95)]
    p99 = char_lengths[int(len(char_lengths) * 0.99)]
    p_max = char_lengths[-1]

    return {
        "n_samples": n,
        "char_p50": p50,
        "char_p95": p95,
        "char_p99": p99,
        "char_max": p_max,
        "rough_token_p50": p50 // 3,
        "rough_token_p95": p95 // 3,
        "rough_token_p99": p99 // 3,
        "rough_token_max": p_max // 3,
    }


# ---------------------------------------------------------------------------
# Step 3: label mask check — requires processor
# ---------------------------------------------------------------------------


def check_label_masks(paths: list[str], processor_path: str, max_length: int, n_samples: int) -> dict:
    """Load processor and check that build_labels_from_template produces non-empty supervision."""
    try:
        from transformers import AutoProcessor
    except ImportError:
        return {"error": "transformers not available"}

    try:
        from nemo_automodel.components.datasets.vlm.collate_fns import build_labels_from_template
    except ImportError:
        return {"error": "nemo_automodel not in PYTHONPATH"}

    print(f"\n[label_check] Loading processor from {processor_path} ...")
    processor = AutoProcessor.from_pretrained(processor_path)
    tokenizer = getattr(processor, "tokenizer", processor)

    # Collect samples from all files
    from parquet_messages_dataset import _to_conversation  # local import

    all_conversations = []
    for path in paths:
        table = pq.read_table(path, columns=["messages"])
        for row in table["messages"].to_pylist():
            try:
                conv = _to_conversation({"messages": row})["conversation"]
                all_conversations.append(conv)
            except Exception:
                pass
        if len(all_conversations) >= n_samples:
            break

    all_conversations = all_conversations[:n_samples]
    print(f"[label_check] Checking {len(all_conversations)} samples ...")

    n_all_masked = 0
    n_partial = 0
    token_lengths = []

    batch_size = 8
    for start in range(0, len(all_conversations), batch_size):
        batch_convs = all_conversations[start : start + batch_size]
        original_side = getattr(tokenizer, "truncation_side", "right")
        tokenizer.truncation_side = "left"
        try:
            out = processor.apply_chat_template(
                batch_convs,
                tokenize=True,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_dict=True,
            )
        finally:
            tokenizer.truncation_side = original_side

        labels = build_labels_from_template(out["input_ids"], batch_convs, processor)
        for i in range(labels.shape[0]):
            row_labels = labels[i]
            valid = (row_labels != -100).sum().item()
            pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
            token_lengths.append(out["input_ids"][i].ne(pad_id).sum().item())
            if valid == 0:
                n_all_masked += 1
            elif valid < 3:
                n_partial += 1

    token_lengths.sort()
    n = len(token_lengths)
    return {
        "n_checked": n,
        "n_all_labels_masked": n_all_masked,
        "n_almost_masked_lt3_tokens": n_partial,
        "token_len_p50": token_lengths[n // 2],
        "token_len_p95": token_lengths[int(n * 0.95)],
        "token_len_p99": token_lengths[int(n * 0.99)],
        "token_len_max": token_lengths[-1],
        "pct_hitting_max_length": sum(1 for tl in token_lengths if tl >= max_length) / n * 100,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Validate xiaoyuzhong parquet dataset")
    parser.add_argument("--files", nargs="+", required=True, help="Parquet file paths")
    parser.add_argument(
        "--max_samples_per_file", type=int, default=5000, help="Max samples to check per file for format validation"
    )
    parser.add_argument(
        "--processor", type=str, default=None, help="Path to Gemma4 processor/tokenizer for label-mask check"
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--label_check_samples", type=int, default=500, help="Number of samples to use for label mask check"
    )
    args = parser.parse_args()

    all_paths = []
    for pattern in args.files:
        expanded = sorted(Path(".").glob(pattern)) if "*" in pattern else [Path(pattern)]
        all_paths.extend(str(p) for p in expanded)

    if not all_paths:
        print("No files found.")
        sys.exit(1)

    print("=" * 70)
    print(f"Validating {len(all_paths)} file(s)")
    print("=" * 70)

    total_rows = 0
    total_issues = 0
    total_content_list = 0
    summary_rows = []

    for path in all_paths:
        col_info = check_messages_column(path)
        if not col_info["has_messages_col"]:
            print(f"\n[FAIL] {path}: NO 'messages' column! columns={col_info.get('columns')}")
            total_issues += 1
            continue

        result = check_parquet_file(path, max_samples=args.max_samples_per_file)
        total_rows += result["n_total"]
        total_issues += result["n_issues"]
        total_content_list += result["content_is_list_count"]

        status = "OK" if result["n_issues"] == 0 else "WARN"
        print(
            f"\n[{status}] {Path(path).name}"
            f"  rows={result['n_total']}"
            f"  checked={result['n_checked']}"
            f"  issues={result['n_issues']}"
            f"  content_as_list={result['content_is_list_count']}"
            f"  roles={result['role_distribution']}"
        )
        if result["sample_issues"]:
            for iss in result["sample_issues"]:
                print(f"       {iss}")

        lengths = estimate_token_lengths(path, max_samples=2000)
        if lengths:
            print(
                f"       rough_tokens  p50={lengths['rough_token_p50']}"
                f"  p95={lengths['rough_token_p95']}"
                f"  p99={lengths['rough_token_p99']}"
                f"  max={lengths['rough_token_max']}"
            )
        summary_rows.append(result)

    print("\n" + "=" * 70)
    print(f"SUMMARY  total_rows={total_rows}  total_issues={total_issues}  content_as_list={total_content_list}")
    print("=" * 70)

    if args.processor:
        print("\n[label_check] Starting label-mask verification ...")
        label_result = check_label_masks(all_paths, args.processor, args.max_length, args.label_check_samples)
        print(json.dumps(label_result, indent=2))
        if label_result.get("n_all_labels_masked", 0) > 0:
            print(
                f"\n[WARN] {label_result['n_all_labels_masked']} samples have ALL labels masked — "
                "build_labels_from_template found no assistant turn to supervise!"
            )
    else:
        print("\n(Skip label-mask check — pass --processor to enable)")


if __name__ == "__main__":
    main()
