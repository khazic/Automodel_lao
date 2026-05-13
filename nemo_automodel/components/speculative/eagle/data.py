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

"""Data helpers for minimal EAGLE-3 training."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset


def _stack_batch(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack a batch of pre-padded unshifted chat samples."""
    batch = {}
    for key in ("input_ids", "loss_mask", "attention_mask"):
        batch[key] = torch.tensor([feature[key] for feature in features], dtype=torch.long)
    return batch


def build_eagle3_dataloader(
    *,
    data_path: str,
    tokenizer,
    seq_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    split: str | None = None,
    distributed: bool = False,
    shuffle_seed: int | None = 42,
) -> DataLoader:
    """Build a dataloader backed by the repo's chat formatting utilities."""
    dataset = ChatDataset(
        data_path,
        tokenizer=tokenizer,
        split=split,
        seq_length=seq_length,
        padding="max_length",
        truncation=True,
        shuffle_seed=shuffle_seed,
        unshifted=True,
    )
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_stack_batch,
        drop_last=False,
    )


def build_eagle3_token_mapping(
    dataloader: DataLoader,
    *,
    target_vocab_size: int,
    draft_vocab_size: int | None,
    special_token_ids: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build draft-vocab mapping tensors from supervised token frequency.

    Returns:
        Tuple ``(selected_token_ids, selected_token_mask)`` where:
        - ``selected_token_ids`` has shape ``[draft_vocab_size]``
        - ``selected_token_mask`` has shape ``[target_vocab_size]``
    """
    if draft_vocab_size is None or draft_vocab_size >= target_vocab_size:
        selected_token_ids = torch.arange(target_vocab_size, dtype=torch.long)
        selected_token_mask = torch.ones(target_vocab_size, dtype=torch.bool)
        return selected_token_ids, selected_token_mask

    counter: Counter[int] = Counter()
    for batch in dataloader:
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"].bool()
        supervised_ids = input_ids[loss_mask]
        counter.update(supervised_ids.tolist())

    selected: list[int] = []
    seen: set[int] = set()
    for token_id in special_token_ids or []:
        if token_id is None or token_id < 0 or token_id >= target_vocab_size or token_id in seen:
            continue
        selected.append(int(token_id))
        seen.add(int(token_id))

    for token_id, _count in counter.most_common():
        if token_id in seen or token_id < 0 or token_id >= target_vocab_size:
            continue
        selected.append(token_id)
        seen.add(token_id)
        if len(selected) >= draft_vocab_size:
            break

    for token_id in range(target_vocab_size):
        if len(selected) >= draft_vocab_size:
            break
        if token_id not in seen:
            selected.append(token_id)
            seen.add(token_id)

    selected_token_ids = torch.tensor(selected[:draft_vocab_size], dtype=torch.long)
    selected_token_mask = torch.zeros(target_vocab_size, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True
    return selected_token_ids, selected_token_mask
