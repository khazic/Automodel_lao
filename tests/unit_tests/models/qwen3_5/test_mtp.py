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
import torch.nn as nn

from nemo_automodel.components.models.qwen3_5.mtp import compute_qwen3_5_mtp_logits


class _IdentityMtpLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states + 1


class _Rotary(nn.Module):
    def forward(self, hidden_states, position_ids):
        return hidden_states, hidden_states


class _FakeQwen36(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(16, 4)
        self.model.language_model.rotary_emb = _Rotary()
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.mtp = nn.ModuleDict(
            {
                "pre_fc_norm_embedding": nn.RMSNorm(4),
                "pre_fc_norm_hidden": nn.RMSNorm(4),
                "fc": nn.Linear(8, 4, bias=False),
                "layers": nn.ModuleList([_IdentityMtpLayer(), _IdentityMtpLayer()]),
                "norm": nn.RMSNorm(4),
            }
        )

    def get_output_embeddings(self):
        return self.lm_head


def test_compute_qwen3_5_mtp_logits_from_hf_style_module():
    model = _FakeQwen36()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    hidden_states = torch.randn(1, 5, 4)

    mtp_logits = compute_qwen3_5_mtp_logits(model, input_ids, hidden_states)

    assert mtp_logits is not None
    assert [item.target_offset for item in mtp_logits] == [1, 2]
    assert mtp_logits[0].logits.shape == (1, 4, 16)
    assert mtp_logits[1].logits.shape == (1, 3, 16)
