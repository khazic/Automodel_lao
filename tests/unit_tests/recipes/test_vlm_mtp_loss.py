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

from nemo_automodel.components.loss.mtp import MTPLogits, MultiTokenPredictionCrossEntropy
from nemo_automodel.recipes.vlm.finetune import calculate_loss


def test_calculate_loss_forwards_mtp_logits():
    labels = torch.tensor([[0, 1, 2]])
    logits = torch.randn(1, 3, 5)
    mtp_logits = [MTPLogits(torch.randn(1, 2, 5), target_offset=1)]
    loss_fn = MultiTokenPredictionCrossEntropy()

    loss = calculate_loss(loss_fn, logits=logits, labels=labels, mtp_logits=mtp_logits, num_label_tokens=3)

    assert loss.ndim == 0
