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

import torch
from transformers import LlamaConfig

from nemo_automodel.components.speculative.eagle.core import _compute_target_distribution
from nemo_automodel.components.speculative.eagle.data import build_eagle3_token_mapping
from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel
from nemo_automodel.components.speculative.eagle.loss import masked_soft_cross_entropy
from nemo_automodel.components.speculative.eagle.target import _shift_left_with_zero


def test_masked_soft_cross_entropy_normalizes_by_valid_positions():
    logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0]]],
        dtype=torch.float32,
    )
    target_probs = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    position_mask = torch.tensor([[[1], [0]]], dtype=torch.bool)

    loss = masked_soft_cross_entropy(logits=logits, target_probs=target_probs, position_mask=position_mask)
    expected = -torch.log_softmax(logits[0, 0], dim=-1)[0]
    torch.testing.assert_close(loss, expected)


def test_compute_target_distribution_uses_selected_vocab_mask():
    target_logits = torch.tensor(
        [[[0.1, 2.0, 0.0], [0.1, 0.2, 3.0]]],
        dtype=torch.float32,
    )
    selected_token_ids = torch.tensor([0, 1], dtype=torch.long)
    selected_token_mask = torch.tensor([True, True, False], dtype=torch.bool)
    loss_mask = torch.tensor([[1, 1]], dtype=torch.long)

    target_probs, position_mask = _compute_target_distribution(
        target_logits=target_logits,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        loss_mask=loss_mask,
    )

    assert target_probs.shape == (1, 2, 2)
    assert position_mask.tolist() == [[[True], [False]]]


def test_llama_eagle3_draft_forward_shape():
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    config.torch_dtype = torch.float32
    config.draft_vocab_size = 16
    config.target_hidden_size = 32
    model = LlamaEagle3DraftModel(config)

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    hidden_states = model(
        input_ids=input_ids,
        projected_hidden_states=model.project_hidden_states(aux_hidden_states),
        attention_mask=attention_mask,
    )
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (batch_size, seq_len, config.hidden_size)
    assert logits.shape == (batch_size, seq_len, config.draft_vocab_size)


def test_build_eagle3_token_mapping_keeps_requested_vocab_size():
    class DummyLoader:
        def __iter__(self):
            yield {
                "input_ids": torch.tensor([[5, 9, 9, 3]], dtype=torch.long),
                "loss_mask": torch.tensor([[0, 1, 1, 1]], dtype=torch.long),
            }

    selected_ids, selected_mask = build_eagle3_token_mapping(
        DummyLoader(),
        target_vocab_size=16,
        draft_vocab_size=4,
        special_token_ids=[0, 1],
    )

    assert selected_ids.shape == (4,)
    assert selected_mask.shape == (16,)
    assert selected_ids[0].item() == 0
    assert selected_ids[1].item() == 1


def test_target_shift_matches_reference_padding_behavior():
    tensor = torch.tensor([[[1.0], [2.0], [3.0]]])
    shifted = _shift_left_with_zero(tensor)
    expected = torch.tensor([[[2.0], [3.0], [0.0]]])
    torch.testing.assert_close(shifted, expected)

    mask = torch.tensor([[1, 0, 1]], dtype=torch.long)
    shifted_mask = _shift_left_with_zero(mask)
    expected_mask = torch.tensor([[0, 1, 0]], dtype=torch.long)
    torch.testing.assert_close(shifted_mask, expected_mask)
