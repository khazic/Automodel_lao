"""Convert a prompt/res JSONL file to a parquet file with an openai-style
``messages`` column that the TP4 Gemma4 SFT recipe can consume.

Each input line is a JSON object. Recognized keys:

- ``system`` (optional): if present and non-empty, becomes a ``system`` turn.
- ``prompt`` (required): becomes a ``user`` turn.
- ``res`` (required; alias ``response``): becomes an ``assistant`` turn.
- Everything else is ignored (e.g. ``skill``).

Usage:

    python3 examples_lao/gemma4/jsonl_to_messages_parquet.py \
        --input  /path/to/input.jsonl \
        --output /path/to/output.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _build_messages(row: dict) -> list[dict] | None:
    prompt = (row.get("prompt") or "").strip()
    res = (row.get("res") or row.get("response") or "").strip()
    if not prompt or not res:
        return None

    messages: list[dict] = []

    system = row.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system.strip()})

    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": res})
    return messages


def convert(input_path: str, output_path: str) -> tuple[int, int, int]:
    """Convert ``input_path`` (JSONL) to ``output_path`` (parquet).

    Returns a tuple ``(kept, skipped_empty, skipped_invalid_json)``.
    """
    rows: list[dict] = []
    kept = 0
    skipped_empty = 0
    skipped_invalid = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped_invalid += 1
                continue

            messages = _build_messages(obj)
            if messages is None:
                skipped_empty += 1
                continue

            rows.append({"messages": messages})
            kept += 1

    table = pa.Table.from_pylist(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")
    return kept, skipped_empty, skipped_invalid


def main() -> None:
    ap = argparse.ArgumentParser(description="JSONL (prompt/res) -> messages-schema parquet.")
    ap.add_argument("--input", "-i", required=True, help="Input JSONL file.")
    ap.add_argument("--output", "-o", required=True, help="Output parquet file.")
    args = ap.parse_args()

    kept, empty, invalid = convert(args.input, args.output)
    print(f"wrote {kept} samples -> {args.output} "
          f"(skipped {empty} empty, {invalid} invalid JSON)")


if __name__ == "__main__":
    main()
