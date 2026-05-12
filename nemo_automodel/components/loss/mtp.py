# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


@dataclass(frozen=True)
class MTPLogits:
    """Logits for one multi-token-prediction head."""

    logits: torch.Tensor
    target_offset: int


def _full_tensor_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


def _normalize_mtp_logits(mtp_logits: torch.Tensor | MTPLogits | Iterable[torch.Tensor | MTPLogits]) -> list[MTPLogits]:
    if isinstance(mtp_logits, torch.Tensor):
        return [MTPLogits(logits=mtp_logits, target_offset=1)]
    if isinstance(mtp_logits, MTPLogits):
        return [mtp_logits]

    normalized = []
    for index, item in enumerate(mtp_logits):
        if isinstance(item, MTPLogits):
            normalized.append(item)
        else:
            normalized.append(MTPLogits(logits=item, target_offset=index + 1))
    return normalized


class MultiTokenPredictionCrossEntropy(nn.Module):
    """Cross entropy for next-token and multi-token-prediction logits.

    The regular language-model logits are trained against ``labels`` as usual.
    Each MTP head is trained against the same labels shifted left by its
    ``target_offset``.  The returned loss keeps the recipe's summed-loss
    contract: when ``num_label_tokens`` is provided, the combined loss is
    normalized once by that value.
    """

    def __init__(
        self,
        mtp_loss_weight: float = 1.0,
        include_main_loss: bool = True,
        fp32_upcast: bool = True,
        ignore_index: int = -100,
        reduction: str = "sum",
    ):
        super().__init__()
        self.mtp_loss_weight = mtp_loss_weight
        self.include_main_loss = include_main_loss
        self.fp32_upcast = fp32_upcast
        self.ignore_index = ignore_index
        self.reduction = reduction

    def _ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = _full_tensor_if_needed(logits)
        labels = _full_tensor_if_needed(labels)
        if labels.device != logits.device:
            labels = labels.to(logits.device)
        if self.fp32_upcast:
            logits = logits.float()
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mtp_logits: torch.Tensor | MTPLogits | Iterable[torch.Tensor | MTPLogits] | None = None,
        num_label_tokens: int | None = None,
    ) -> torch.Tensor:
        if self.reduction != "sum" and num_label_tokens is not None:
            raise AssertionError("num_label_tokens is only supported when reduction is 'sum'")

        total = self._ce(logits, labels) if self.include_main_loss else logits.new_tensor(0.0)

        if mtp_logits is None:
            raise ValueError("MultiTokenPredictionCrossEntropy requires `mtp_logits`.")

        for item in _normalize_mtp_logits(mtp_logits):
            if item.target_offset <= 0:
                raise ValueError(f"MTP target_offset must be positive, got {item.target_offset}.")
            if labels.shape[1] <= item.target_offset:
                continue
            mtp_labels = labels[:, item.target_offset :]
            aligned_logits = item.logits[:, : mtp_labels.shape[1], :]
            total = total + self.mtp_loss_weight * self._ce(aligned_logits, mtp_labels)

        if num_label_tokens is not None:
            if num_label_tokens == 0:
                return total * 0.0
            total = total / num_label_tokens
        return total
