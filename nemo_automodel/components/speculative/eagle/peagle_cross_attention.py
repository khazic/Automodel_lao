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

"""KV-reuse cross-attention for P-EAGLE (experimental).

This adds the GLIDE / LongSpec-style "re-attend to the target KV cache" signal to
the P-EAGLE parallel drafter. Each draft element forms its own query and attends
to the target model's pre-computed key/value pairs for the *verified* prefix,
instead of relying solely on the single compressed target hidden state that the
EAGLE-3 ``fc`` path provides.

Two deliberate choices keep this minimal and sidestep the training-pipeline
bottlenecks identified by KVShot (arXiv:2604.26412):

* **No draft-side K/V projections.** The cross-attention reuses the target K/V
  *directly* (only ``q_proj`` / ``o_proj`` are learned), so there is no sparse
  ``W_K`` / ``W_V`` gradient to optimize -- the draft's whole capacity goes to
  query estimation.
* **Zero-initialized ``o_proj``, no sigmoid gate.** At init the branch outputs
  exactly zero (the drafter == baseline P-EAGLE). ``o_proj`` still receives a
  dense gradient from step one, so it leaves the zero state immediately and the
  branch activates -- unlike a multiplicative fusion gate, which is driven back
  toward zero and stays starved (the gate-collapse failure mode KVShot reports).

Because ``draft_config`` is cloned from the target config, the draft shares the
target's ``head_dim`` / ``num_key_value_heads`` / RoPE settings, so the target
K/V tensors slot directly into the draft's attention without any projection or
re-heading. The cross-query is rotated at the element's reference position
(``position_ids``) while the target keys carry their own absolute-position RoPE
from the target forward, so the dot product encodes the relative offset exactly
as the target's own attention would.

Caveat: this assumes the target's key RoPE matches the draft's
:class:`LlamaRotaryEmbedding` (true when the draft config is cloned from a
standard Llama/Qwen target). Targets with a divergent RoPE variant (partial
rotary, custom scaling, MLA) need their key rotation matched before reuse. The
target K/V must come from a full-attention layer -- a sliding-window layer's
cache is truncated and would break the ``j <= anchor`` window.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nemo_automodel.components.models.common import initialize_rms_norm_module
from nemo_automodel.components.models.llama.rope_utils import LlamaRotaryEmbedding, rotate_half


def build_cross_additive_mask(
    anchor_pos: torch.Tensor,
    row_length: torch.Tensor,
    total_seq_len: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the additive cross-attention mask over target prefix positions.

    A draft element of a rollout anchored at ``a`` attends to the target KV at
    positions ``0..a`` -- the verified prefix that exists at inference time. All
    depths of the rollout share the same window (deeper positions ``a+1, a+2, ...``
    are being predicted, so their target KV does not exist yet). Padding target
    positions (``>= row_length``) are excluded.

    Args:
        anchor_pos: Chain-start position per draft element, shape ``[n]``.
        row_length: Valid (unpadded) length of the document, shape ``[1]``.
        total_seq_len: Number of target key positions (original sequence length).
        dtype: Floating dtype of the additive mask (matches the attention logits).

    Returns:
        Additive mask of shape ``[1, 1, n, total_seq_len]``: ``0`` where attention
        is allowed and the dtype's most-negative value where it is masked.
    """
    device = anchor_pos.device
    key_pos = torch.arange(total_seq_len, device=device)
    allowed = (key_pos.unsqueeze(0) <= anchor_pos.unsqueeze(1)) & (key_pos.unsqueeze(0) < row_length)
    neg_inf = torch.finfo(dtype).min
    return torch.where(allowed, torch.zeros((), device=device, dtype=dtype), torch.full((), neg_inf, dtype=dtype))[
        None, None
    ]


class PeagleCrossAttention(nn.Module):
    """Cross-attention from draft elements to the target model's reused KV cache.

    Pre-norms the running hidden state, projects a query (rotated at the element's
    reference ``position_ids``), and attends to the target's pre-computed K/V
    (GQA-expanded to the query head count). ``o_proj`` is zero-initialized so the
    branch contributes nothing at init and is added as a residual correction by
    the host decoder layer.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        attention_bias = getattr(config, "attention_bias", False)
        self.input_layernorm = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=attention_bias)
        # Zero-init the output projection so the cross-attention branch starts as a
        # no-op (the drafter reduces to baseline P-EAGLE) and grows only if the KV
        # signal helps. o_proj still gets a dense gradient from step one, so it
        # leaves zero immediately -- avoiding the gated-residual collapse KVShot
        # observed, where a multiplicative gate is driven back toward zero.
        nn.init.zeros_(self.o_proj.weight)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        target_kv: tuple[torch.Tensor, torch.Tensor],
        cross_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attend draft elements to the target prefix KV.

        Args:
            hidden_states: Running draft hidden states, shape ``[1, n, H]``.
            position_ids: Reference position of each element, shape ``[1, n]``;
                used to rotate the cross-query.
            target_kv: ``(key, value)`` from the target cache, each shaped
                ``[1, num_key_value_heads, total_seq_len, head_dim]`` and already
                carrying the target's absolute-position RoPE on the keys.
            cross_mask: Additive mask ``[1, 1, n, total_seq_len]`` from
                :func:`build_cross_additive_mask`.

        Returns:
            The cross-attention correction, shape ``[1, n, H]`` (zero at init).
        """
        target_k, target_v = target_kv
        batch_size, seq_len, _ = hidden_states.shape
        x = self.input_layernorm(hidden_states)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(x, position_ids)
        q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))

        if self.num_key_value_groups > 1:
            target_k = target_k.repeat_interleave(self.num_key_value_groups, dim=1)
            target_v = target_v.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(q, target_k.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + cross_mask
        attn_probs = torch.softmax(attn_weights.float(), dim=-1).to(q.dtype)
        attn_output = torch.matmul(attn_probs, target_v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)
