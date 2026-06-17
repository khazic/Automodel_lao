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

"""Cross-attention and gated merge modules for P-EAGLE KV cache reuse.

Implements the KVShot-style gated hybrid architecture (arXiv:2604.26412):
each draft layer runs self-attention (existing COD flex_attention) and
cross-attention (to target KV cache) in parallel, merging them via a
learned gate. The cross-attention lets the draft attend directly into
the target model's KV cache at each layer, providing richer long-range
context than hidden-state reuse alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers import PretrainedConfig

from nemo_automodel.components.models.llama.rope_utils import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from nemo_automodel.components.speculative.eagle.peagle_draft import _peagle_flex_attention


class PeagleCrossAttention(nn.Module):
    """Cross-attention from draft hidden states to target KV cache.

    Query is projected from the draft layer's hidden states with its own
    ``cross_q_proj`` and receives RoPE at the draft reference positions
    (anchor_pos + depth). Key and Value come directly from the target
    model's captured attention KV (already post-RoPE at original positions).
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.cross_q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.cross_o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_k: torch.Tensor,
        target_v: torch.Tensor,
        position_ids: torch.Tensor,
        cross_block_mask,
    ) -> torch.Tensor:
        """Cross-attention forward.

        Args:
            hidden_states: Draft layer hidden states [B, S_draft, H].
            target_k: Target attention K [B, num_kv_heads, S_target, head_dim] (post-RoPE).
            target_v: Target attention V [B, num_kv_heads, S_target, head_dim].
            position_ids: Draft reference positions [B, S_draft] for RoPE on Q.
            cross_block_mask: flex_attention block mask (Q_LEN=S_draft, KV_LEN=S_target).

        Returns:
            Cross-attention output [B, S_draft, H].
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.cross_q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)

        if self.num_key_value_groups > 1:
            target_k = target_k.repeat_interleave(self.num_key_value_groups, dim=1)
            target_v = target_v.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = _peagle_flex_attention(q, target_k, target_v, block_mask=cross_block_mask, scale=self.scaling)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.cross_o_proj(attn_output)


class PeagleGatedMerge(nn.Module):
    """Gated delta merge for self-attention and cross-attention outputs.

    Implements: h = n + sigmoid(W_g * [n; o; delta] + b_g) * W_delta * delta
    where delta = o - n, n = self-attn output, o = cross-attn output.

    Initialized so the gate starts near zero (sigmoid(-2) ~ 0.12),
    making the initial behavior approximate pure self-attention.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.gate_proj = nn.Linear(3 * hidden_size, hidden_size, bias=True)
        self.delta_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)
        nn.init.normal_(self.delta_proj.weight, std=0.01)

    def forward(self, self_attn_out: torch.Tensor, cross_attn_out: torch.Tensor) -> torch.Tensor:
        """Gated merge of self-attention and cross-attention outputs.

        Args:
            self_attn_out: [B, S, H] from the existing self-attention path.
            cross_attn_out: [B, S, H] from the cross-attention path.

        Returns:
            Merged output [B, S, H].
        """
        delta = cross_attn_out - self_attn_out
        gate_input = torch.cat([self_attn_out, cross_attn_out, delta], dim=-1)
        g = torch.sigmoid(self.gate_proj(gate_input))
        return self_attn_out + g * self.delta_proj(delta)


def create_peagle_cross_attn_mask_mod(
    draft_anchor_pos: torch.Tensor,
    draft_depth: torch.Tensor,
    target_seq_len: int,
):
    """Build a flex_attention mask_mod for cross-attention to target KV.

    All draft depths can only attend to target KV at positions <= anchor_pos.
    This matches inference where the target KV cache only contains tokens up
    to the last verified position (the anchor), regardless of draft depth.

    Args:
        draft_anchor_pos: [total_sampled] anchor positions per COD element.
        draft_depth: [total_sampled] depth per COD element.
        target_seq_len: Length of the target KV sequence.

    Returns:
        A mask_mod callable for create_block_mask.
    """

    def cross_mask_mod(_b, _h, q_idx, kv_idx):
        q_ref_pos = draft_anchor_pos[q_idx]
        return kv_idx <= q_ref_pos

    return cross_mask_mod


def build_peagle_cross_block_mask(
    anchor_pos: torch.Tensor,
    depth: torch.Tensor,
    target_seq_len: int,
) -> object:
    """Construct the cross-attention block mask for one P-EAGLE sequence.

    Args:
        anchor_pos: [total_sampled] anchor positions.
        depth: [total_sampled] COD depths.
        target_seq_len: Full target sequence length.

    Returns:
        A flex_attention block_mask object.
    """
    mask_mod = create_peagle_cross_attn_mask_mod(
        draft_anchor_pos=anchor_pos,
        draft_depth=depth,
        target_seq_len=target_seq_len,
    )
    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=anchor_pos.shape[0],
        KV_LEN=target_seq_len,
        device=anchor_pos.device,
    )
