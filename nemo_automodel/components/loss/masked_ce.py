# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


class MaskedCrossEntropy(nn.Module):
    def __init__(self, fp32_upcast: bool = True, ignore_index: int = -100, reduction: str = "sum"):
        """
        Masked cross-entropy loss.

        Args:
            fp32_upcast (bool): if True it will cast logits to float32 before computing
                cross entropy. Default: True.
            ignore_index (int): label to ignore in CE calculation. Defaults to -100.
            reduction (str): type of reduction. Defaults to "sum".
        """
        super().__init__()
        self.fp32_upcast = fp32_upcast
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_label_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the masked cross-entropy loss between logits and targets.

        If a mask is provided, the loss is computed per element, multiplied by the mask,
        and then averaged. If no mask is provided, the standard cross-entropy loss is used.

        Args:
            logits (torch.Tensor): The predicted logits with shape [batch_size, seq_len, vocab_size] where C is the number of classes.
            labels (torch.Tensor): The ground truth class indices with shape [batch_size, seq_len].
            mask (torch.Tensor, optional): A tensor that masks the loss computation. Items marked with
                1 will be used to calculate loss, otherwise ignored. Must be broadcastable to the shape
                of the loss. Defaults to None.

        Returns:
            torch.Tensor: The computed loss as a scalar tensor.
        """
        # this may happen with CPUOffloadPolicy
        if labels.device != logits.device:
            labels = labels.to(logits.device)  # pragma: no cover
        # reshape to (N, C) and (N,) respectively
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        if mask is not None:
            with torch.no_grad():
                if mask.device != labels.device:
                    mask = mask.to(labels.device)  # pragma: no cover
                labels.masked_fill_(mask.view(-1) == 0, self.ignore_index)
                del mask
        if self.fp32_upcast:
            logits = logits.float()

        if isinstance(logits, DTensor):
            logits = logits.full_tensor()

        if isinstance(labels, DTensor):
            labels = labels.full_tensor()

        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        # --- TEMP DIAGNOSTIC (once, from whichever rank the loss lands on)
        import os
        import sys

        if not getattr(MaskedCrossEntropy, "_dbg_done", False):
            MaskedCrossEntropy._dbg_done = True
            _rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
            print(f"[ce-diag] firing on rank={_rank}", file=sys.stderr, flush=True)
            valid = labels != self.ignore_index
            n_valid = int(valid.sum())
            nan = bool(torch.isnan(logits).any())
            inf = bool(torch.isinf(logits).any())
            print(
                f"[ce-diag] logits shape={tuple(logits.shape)} dtype={logits.dtype} "
                f"min={float(logits.min()):.2f} max={float(logits.max()):.2f} "
                f"mean={float(logits.mean()):.3f} std={float(logits.std()):.3f} "
                f"nan={nan} inf={inf}",
                file=sys.stderr, flush=True,
            )
            if n_valid > 0:
                vlog = logits[valid]
                vlab = labels[valid]
                pred = vlog.argmax(dim=-1)
                match = (pred == vlab).float().mean().item()
                # log-prob at the correct label vs uniform baseline
                logp = torch.log_softmax(vlog, dim=-1)
                correct_logp = logp.gather(-1, vlab.unsqueeze(-1)).squeeze(-1)
                print(
                    f"[ce-diag] n_valid_labels={n_valid} argmax_match={match*100:.1f}% "
                    f"mean_correct_logp={float(correct_logp.mean()):.3f} "
                    f"median_correct_logp={float(correct_logp.median()):.3f} "
                    f"(random baseline logp = -{torch.log(torch.tensor(logits.size(-1))).item():.3f})",
                    file=sys.stderr, flush=True,
                )
                # Sample of first 5 valid positions: predicted top-3 vs target
                sample = min(5, int(n_valid))
                for i in range(sample):
                    top_v, top_i = vlog[i].topk(3)
                    print(
                        f"[ce-diag] pos{i}: target_tok={int(vlab[i])} "
                        f"top3=[{' '.join(f'{int(t)}({float(v):.1f})' for t, v in zip(top_i, top_v))}]",
                        file=sys.stderr, flush=True,
                    )
        # ---------------------------------------------------------------
        if num_label_tokens is not None:
            assert self.reduction == "sum", "num_label_tokens is only supported when reduction is 'sum'"
            if num_label_tokens == 0:
                return loss * 0.0
            loss = loss / num_label_tokens
        return loss
