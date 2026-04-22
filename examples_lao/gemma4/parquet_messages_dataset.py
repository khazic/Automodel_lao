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

import json
import logging
import random

import pandas as pd

from nemo_automodel.components.datasets.vlm.collate_fns import gemma4_prefix_collate_fn

logger = logging.getLogger(__name__)


def _convert_messages_to_conversation(messages):
    """Convert a list of role/content message dicts to NeMo conversation format."""
    conversation = []
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("user", "assistant", "system"):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            content_list = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            content_list = []
            for item in content:
                if isinstance(item, str):
                    content_list.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    content_list.append(item)
        else:
            content_list = [{"type": "text", "text": str(content)}]
        conversation.append({"role": role, "content": content_list})
    return conversation


def make_parquet_messages_dataset(path_or_dataset, split="train", shuffle_seed=None, **kwargs):
    """Load parquet file(s) with chat messages and return conversation-format examples.

    Each row must contain a "messages" (or "conversation"/"conversations") column
    with a list of {"role": ..., "content": ...} dicts.

    Args:
        path_or_dataset: Path or list of paths to parquet files.
        split: Unused; kept for API consistency with other dataset loaders.
        shuffle_seed: If set, shuffle the resulting list with this seed.

    Returns:
        list[dict]: Each dict has a "conversation" key in NeMo VLM format.
    """
    if isinstance(path_or_dataset, str):
        path_or_dataset = [path_or_dataset]

    dfs = []
    for path in path_or_dataset:
        logger.info("Loading parquet: %s", path)
        dfs.append(pd.read_parquet(path))

    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    if "messages" in df.columns:
        msg_col = "messages"
    elif "conversation" in df.columns:
        msg_col = "conversation"
    elif "conversations" in df.columns:
        msg_col = "conversations"
    else:
        raise ValueError(f"No messages column found. Available columns: {list(df.columns)}")

    examples = []
    for row in df[msg_col]:
        if isinstance(row, str):
            row = json.loads(row)
        conversation = _convert_messages_to_conversation(row)
        if conversation:
            examples.append({"conversation": conversation})

    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(examples)

    logger.info("Loaded %d examples from %s", len(examples), path_or_dataset)
    return examples


def gemma4_prefix_truncating_collate_fn(examples, processor, max_length=4096):
    """Gemma4 collate function with thinking-channel prefix injection and sequence truncation.

    Wraps gemma4_prefix_collate_fn with a fixed max_length so that sequences
    longer than max_length are truncated rather than causing OOM.

    Args:
        examples: List of dataset examples (each with a "conversation" key).
        processor: The Gemma4 processor instance.
        max_length: Maximum token sequence length; sequences are truncated to this.

    Returns:
        dict: Batched tensors ready for model forward.
    """
    return gemma4_prefix_collate_fn(examples, processor, max_length=max_length)
