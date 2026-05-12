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

import torch
import torch.nn.functional as F

from nemo_automodel.components.loss.mtp import MTPLogits, MultiTokenPredictionCrossEntropy


def test_mtp_loss_adds_shifted_auxiliary_loss():
    labels = torch.tensor([[0, 1, 2, -100]])
    logits = torch.randn(1, 4, 5)
    mtp_logits = torch.randn(1, 3, 5)
    loss_fn = MultiTokenPredictionCrossEntropy(mtp_loss_weight=0.5)

    loss = loss_fn(logits=logits, labels=labels, mtp_logits=MTPLogits(mtp_logits, target_offset=1))

    expected_main = F.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1), ignore_index=-100, reduction="sum")
    expected_mtp = F.cross_entropy(
        mtp_logits.reshape(-1, 5),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    assert torch.allclose(loss, expected_main + 0.5 * expected_mtp)


def test_mtp_loss_normalizes_by_num_label_tokens_once():
    labels = torch.tensor([[0, 1, 2]])
    logits = torch.randn(1, 3, 5)
    mtp_logits = torch.randn(1, 2, 5)
    loss_fn = MultiTokenPredictionCrossEntropy()

    loss = loss_fn(logits=logits, labels=labels, mtp_logits=[mtp_logits], num_label_tokens=3)
    unnormalized = loss_fn(logits=logits, labels=labels, mtp_logits=[mtp_logits])

    assert torch.allclose(loss, unnormalized / 3)
