# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for the batch-one Transformers speculative benchmark."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from tools.transformers_vlm_spec_bench import _acceptance_lengths, _greedy_cached_forward


class _TinyPositionModel(nn.Module):
    """Return shape-compatible multimodal position IDs for cached decoding."""

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values,
        **kwargs,
    ) -> torch.Tensor:
        """Build position IDs for one incremental token.

        Args:
            input_ids: Tensor of shape [1, query_sequence].
            inputs_embeds: Tensor of shape [1, query_sequence, hidden].
            attention_mask: Tensor of shape [1, prefix_sequence + query_sequence].
            past_key_values: Target cache covering ``prefix_sequence`` tokens.
            **kwargs: Unused multimodal position metadata.

        Returns:
            Tensor of shape [3, 1, query_sequence].
        """
        del inputs_embeds, attention_mask, kwargs
        start = past_key_values.get_seq_length()
        positions = torch.arange(start, start + input_ids.shape[1], device=input_ids.device)
        return positions.view(1, 1, -1).expand(3, 1, -1)


class _TinyCachedTarget(nn.Module):
    """Deterministic target whose next token is the current token plus one."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyPositionModel()
        self.embeddings = nn.Embedding(8, 4)
        self.query_lengths: list[int] = []
        self.cache_ids: list[int] = []
        self.cache_positions: list[list[int] | None] = []

    def get_input_embeddings(self) -> nn.Module:
        """Return the target token embeddings."""
        return self.embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        *,
        next_sequence_length: int,
        past_key_values,
        attention_mask: torch.Tensor,
        is_first_iteration: bool,
        use_cache: bool,
        **kwargs,
    ) -> dict[str, object]:
        """Slice the uncached query tokens from a full sequence.

        Args:
            input_ids: Tensor of shape [1, full_sequence].
            next_sequence_length: Number of trailing query tokens.
            past_key_values: Target cache covering the processed prefix.
            attention_mask: Tensor of shape [1, full_sequence].
            is_first_iteration: Whether this is the prompt prefill.
            use_cache: Whether the target should update its cache.
            **kwargs: Unused processor tensors.

        Returns:
            Mapping containing ``input_ids`` of shape [1, query_sequence], the
            full attention mask, and the persistent target cache.
        """
        del is_first_iteration, kwargs
        return {
            "input_ids": input_ids[:, -next_sequence_length:],
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ):
        """Append query tokens to the cache and return successor logits.

        Args:
            input_ids: Tensor of shape [1, query_sequence].
            past_key_values: Target cache mutated to append ``query_sequence`` tokens.
            attention_mask: Tensor of shape [1, processed_sequence + query_sequence].
            cache_position: Tensor of shape [query_sequence], or ``None`` during prefill.
            **kwargs: Unused target forward arguments.

        Returns:
            Object containing logits of shape [1, query_sequence, vocab] and the
            mutated target cache.
        """
        del attention_mask, kwargs
        query_length = input_ids.shape[1]
        cache_tensor = torch.zeros((1, 1, query_length, 1), device=input_ids.device)
        past_key_values.update(cache_tensor, cache_tensor, 0)
        self.query_lengths.append(query_length)
        self.cache_ids.append(id(past_key_values))
        self.cache_positions.append(None if cache_position is None else cache_position.tolist())
        logits = torch.full((1, query_length, 8), -1000.0, device=input_ids.device)
        logits.scatter_(-1, input_ids.add(1).remainder(8).unsqueeze(-1), 0.0)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def test_greedy_cached_forward_reuses_cache_and_decodes_one_token_at_a_time() -> None:
    """The target prefills once and then uses one-token cached forwards."""
    target = _TinyCachedTarget()

    generated = _greedy_cached_forward(
        target,
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        },
        max_new_tokens=6,
        eos_token_id=5,
    )

    assert generated == [3, 4, 5]
    assert target.query_lengths == [2, 1, 1]
    assert len(set(target.cache_ids)) == 1
    assert target.cache_positions == [None, [2], [3]]


def test_acceptance_lengths_separates_official_tau_from_emitted_tokens() -> None:
    """Official tau excludes the one guaranteed target token emitted per round."""
    assert _acceptance_lengths(9, 4) == (2.25, 3.25)
    assert _acceptance_lengths(0, 0) == (None, None)
