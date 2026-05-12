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

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from datasets import load_dataset

from nemo_automodel.components.datasets.vlm.collate_fns import (
    HAVE_QWEN_VL_UTILS,
    MISSING_QWEN_VL_UTILS_MSG,
    _count_media_per_sample,
    _ensure_rgb,
    build_labels_from_template,
    mask_fake_vision_tokens_batch,
)


def _as_list(path_or_dataset: str | Sequence[str]) -> list[str]:
    if isinstance(path_or_dataset, str):
        return [path_or_dataset]
    return [str(path) for path in path_or_dataset]


def _to_conversation(example: dict) -> dict:
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Expected each sample to contain a `messages` list.")

    conversation = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue

        content = message.get("content", "")
        if content is None:
            content = ""

        conversation.append(
            {
                "role": role,
                "content": [{"type": "text", "text": str(content)}],
            }
        )

    if not conversation:
        raise ValueError("Conversation is empty after message normalization.")

    return {"conversation": conversation}


def make_parquet_messages_dataset(
    path_or_dataset: str | Sequence[str],
    split: str = "train",
    shuffle_seed: int | None = 42,
    limit_dataset_samples: int | None = None,
    **kwargs,
):
    """Load parquet files with a `messages` column as text-only VLM conversations."""

    data_files = _as_list(path_or_dataset)
    missing = [path for path in data_files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing parquet file(s): {missing}")

    dataset = load_dataset("parquet", data_files=data_files, split=split)

    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    if limit_dataset_samples is not None:
        dataset = dataset.select(range(min(limit_dataset_samples, len(dataset))))

    return dataset.map(_to_conversation, remove_columns=dataset.column_names)


def qwen3_6_prefix_truncating_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Qwen3.6 VLM collate for text-only SFT data with left-side truncation."""

    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    conversations = _ensure_rgb([example["conversation"] for example in examples])
    tokenizer = getattr(processor, "tokenizer", processor)

    original_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        processor_kwargs = {
            "tokenize": True,
            "padding": "max_length" if max_length is not None else True,
            "truncation": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        if max_length is not None:
            processor_kwargs["max_length"] = max_length
        batch = processor.apply_chat_template(conversations, **processor_kwargs)
    finally:
        tokenizer.truncation_side = original_truncation_side

    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    if "pixel_values_videos" in batch:
        batch["pixel_values_videos"] = batch["pixel_values_videos"].to(torch.bfloat16)

    labels = build_labels_from_template(batch["input_ids"], conversations, processor)
    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key in list(batch.keys()):
        value = batch[key]
        if isinstance(value, torch.Tensor) and value.shape == input_shape and key != "labels":
            batch[key] = value[:, :-1]

    fake_indices = [i for i, example in enumerate(examples) if example.get("_injected_fake")]
    if fake_indices:
        mask_fake_vision_tokens_batch(batch, processor, fake_indices)

    image_counts, video_counts = _count_media_per_sample(conversations)
    if any(count > 0 for count in image_counts):
        batch["n_images_per_sample"] = torch.tensor(image_counts, dtype=torch.long)
    if any(count > 0 for count in video_counts):
        batch["n_videos_per_sample"] = torch.tensor(video_counts, dtype=torch.long)

    return batch
