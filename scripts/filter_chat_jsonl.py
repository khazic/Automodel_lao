#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Filter a chat JSONL, keeping only rows with >= 1 non-empty assistant turn.

Usage:
    python filter_chat_jsonl.py in.jsonl out.jsonl
"""
import json
import sys


def has_assistant(messages):
    if not isinstance(messages, list):
        return False
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                return True
    return False


def main():
    src, dst = sys.argv[1], sys.argv[2]
    kept = dropped = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if has_assistant(row.get("messages")):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1
    print(f"kept={kept} dropped={dropped} -> {dst}")


if __name__ == "__main__":
    main()
