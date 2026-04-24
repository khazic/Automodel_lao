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

"""DeepSeek V4 Attention Layer.

V4 replaces MLA (Multi-head Latent Attention) with GQA + Q/O LoRA:
  - Q  : hidden -> wq_a -> q_norm -> wq_b -> [n_heads, head_dim]
  - KV : hidden -> wkv -> split -> kv_norm(K), V unchanged
  - RoPE applied to the last qk_rope_head_dim dimensions of Q and K.
  - Attention: standard GQA with num_key_value_heads=1 (MQA).
  - O  : flat_output -> wo_a (block-diagonal grouped) -> wo_b -> hidden

The first num_hash_layers use hash-clustering (HC) attention for dynamic
token grouping. This is currently a stub; HC is replaced by full causal
attention and the HC parameters (hc_attn_base/fn/scale) are stored for
checkpoint compatibility only.

Layers with compress_ratios[i] > 0 are intended to use a compressed /
sliding-window KV, with an attention sink token prepended. The sliding-
window masking is not yet implemented; all layers use full causal attention.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.deepseek_v3.rope_utils import (
    apply_rotary_emb_qk,
    yarn_get_mscale,
)
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config


class GroupedLinear(nn.Module):
    """Block-diagonal grouped linear projection.

    Splits in_features into `groups` equal groups, projects each group
    independently with weight [out_features//groups, in_features//groups],
    then concatenates the results. The full weight is stored as a single
    [out_features, in_features] parameter with a block-diagonal structure
    so that it is compatible with HuggingFace checkpoint shapes.
    """

    def __init__(self, in_features: int, out_features: int, groups: int):
        super().__init__()
        assert in_features % groups == 0, f"in_features {in_features} not divisible by groups {groups}"
        assert out_features % groups == 0, f"out_features {out_features} not divisible by groups {groups}"
        self.groups = groups
        self.in_per_group = in_features // groups
        self.out_per_group = out_features // groups
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        # [..., groups * in_per_group] -> [..., groups, in_per_group]
        x_g = x.reshape(*shape[:-1], self.groups, self.in_per_group)
        # weight -> [groups, out_per_group, in_per_group]
        w = self.weight.reshape(self.groups, self.out_per_group, self.in_per_group)
        # Grouped matmul: [..., g, in] @ [g, out, in]^T -> [..., g, out]
        out = torch.einsum("...gi,goi->...go", x_g, w)
        return out.reshape(*shape[:-1], self.groups * self.out_per_group)

    def init_weights(self, init_std: float = 0.02) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=init_std)


class DeepseekV4Attention(nn.Module):
    """GQA attention with Q/O LoRA for DeepSeek V4.

    Weight layout (HF names in parentheses):
      wq_a  (attn.wq_a) : hidden_size -> q_lora_rank
      q_norm (attn.q_norm): RMSNorm on q_lora_rank
      wq_b  (attn.wq_b) : q_lora_rank -> n_heads * head_dim
      wkv   (attn.wkv)  : hidden_size -> n_kv_heads * 2 * head_dim  (K||V)
      kv_norm(attn.kv_norm): RMSNorm on head_dim, applied to K
      wo_a  (attn.wo_a) : n_heads * head_dim -> o_lora_rank  (grouped)
      wo_b  (attn.wo_b) : o_lora_rank -> hidden_size
      attn_sink(attn.attn_sink): learnable sink [1, 1, n_kv_heads, 2*head_dim]
    """

    def __init__(self, config: DeepseekV4Config, backend: BackendConfig):
        super().__init__()

        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.o_groups = config.o_groups

        self.backend = backend
        self.rope_fusion = backend.rope_fusion
        linear_impl = backend.linear
        rms_norm_impl = backend.rms_norm
        hidden_size = config.hidden_size

        # Q LoRA
        self.wq_a = initialize_linear_module(linear_impl, hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = initialize_rms_norm_module(rms_norm_impl, self.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = initialize_linear_module(linear_impl, self.q_lora_rank, self.n_heads * self.head_dim, bias=False)

        # Combined KV projection: hidden -> [K || V] where each is n_kv_heads * head_dim
        self.wkv = initialize_linear_module(linear_impl, hidden_size, self.n_kv_heads * self.head_dim * 2, bias=False)
        # kv_norm normalizes the K part before RoPE
        self.kv_norm = initialize_rms_norm_module(rms_norm_impl, self.head_dim, eps=config.rms_norm_eps)

        # O LoRA (grouped block-diagonal first step, then linear)
        self.wo_a = GroupedLinear(
            in_features=self.n_heads * self.head_dim,
            out_features=self.o_lora_rank,
            groups=self.o_groups,
        )
        self.wo_b = initialize_linear_module(linear_impl, self.o_lora_rank, hidden_size, bias=False)

        # Attention sink: a learnable [K||V] token prepended to every KV sequence.
        # Shape: [1, 1, n_kv_heads, 2 * head_dim] so it broadcasts over batch/seq.
        self.register_parameter(
            "attn_sink",
            nn.Parameter(torch.zeros(1, 1, self.n_kv_heads, self.head_dim * 2)),
        )

        # Softmax scale with optional YaRN correction
        self.softmax_scale = self.head_dim**-0.5
        rope_params = getattr(config, "rope_scaling", None)
        if rope_params and all(k in rope_params for k in ("factor", "mscale", "original_max_position_embeddings")):
            factor = rope_params["factor"]
            mscale = rope_params["mscale"]
            original_seq_len = rope_params["original_max_position_embeddings"]
            if config.max_position_embeddings > original_seq_len:
                mscale = yarn_get_mscale(factor, mscale)
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # num_gqa_groups = n_heads / n_kv_heads (64 for V4 Flash since n_kv_heads=1)
        num_gqa_groups = self.n_heads // self.n_kv_heads if self.n_kv_heads > 1 else None
        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=backend.attn,
            num_attention_heads=self.n_heads,
            num_qk_channels=self.head_dim,
            num_v_channels=self.head_dim,
            softmax_scale=self.softmax_scale,
            num_gqa_groups=num_gqa_groups,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if x.dim() == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
            bsz, seq_len = 1, num_tokens
        else:
            qkv_format = "bshd"
            bsz, seq_len, _ = x.shape

        # --- Q path ---
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        if qkv_format == "thd":
            q = q.view(num_tokens, self.n_heads, self.head_dim)
        else:
            q = q.view(bsz, seq_len, self.n_heads, self.head_dim)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # --- KV path ---
        kv = self.wkv(x)  # [..., n_kv_heads * 2 * head_dim]
        if qkv_format == "thd":
            kv = kv.view(num_tokens, self.n_kv_heads, self.head_dim * 2)
        else:
            kv = kv.view(bsz, seq_len, self.n_kv_heads, self.head_dim * 2)
        k_full, v = torch.split(kv, self.head_dim, dim=-1)

        # Apply kv_norm to the entire K vector (head_dim), then split nope/rope
        k_full = self.kv_norm(k_full)
        k_nope, k_pe = torch.split(k_full, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # --- RoPE ---
        cu_seqlens = attn_kwargs.get("cu_seqlens", None)
        q_pe, k_pe = apply_rotary_emb_qk(
            q_pe,
            k_pe,
            freqs_cis,
            format=qkv_format,
            rope_fusion=self.rope_fusion,
            cu_seqlens=cu_seqlens,
            cp_size=attn_kwargs.get("cp_size", 1),
            cp_rank=attn_kwargs.get("cp_rank", 0),
        )

        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        # --- Attention sink ---
        # Prepend the learnable sink [K||V] as position 0 of the KV sequence.
        # sink shape: [1, 1, n_kv_heads, 2*head_dim] -> broadcast over batch
        sink_kv = self.attn_sink.expand(bsz if qkv_format == "bshd" else 1, 1, -1, -1)
        sink_k, sink_v = torch.split(sink_kv, self.head_dim, dim=-1)
        if qkv_format == "thd":
            sink_k = sink_k.view(1, self.n_kv_heads, self.head_dim)
            sink_v = sink_v.view(1, self.n_kv_heads, self.head_dim)
            k = torch.cat([sink_k, k], dim=0)
            v = torch.cat([sink_v, v], dim=0)
        else:
            k = torch.cat([sink_k, k], dim=1)
            v = torch.cat([sink_v, v], dim=1)

        # attention_mask must account for the extra sink position if provided
        # TODO: extend mask by one column on the left when attention_mask is not None

        q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, self.backend.attn, **attn_kwargs
        )
        x = self.attn_func(q, k, v, **_attn_kwargs)
        x = postprocess_output_for_attn(x, self.backend.attn)

        # Remove sink position from output if returned (only affects k/v, not q/output)
        # Output shape is [..., n_heads, head_dim] - no sink dimension here.

        # --- O LoRA ---
        flatten_dim = 2 if qkv_format == "bshd" else 1
        x_flat = x.flatten(flatten_dim)  # [..., n_heads * head_dim]
        x_lora = self.wo_a(x_flat)  # [..., o_lora_rank]
        return self.wo_b(x_lora)  # [..., hidden_size]

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        for linear in (self.wq_a, self.wq_b, self.wkv, self.wo_b):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        self.wo_a.init_weights(init_std)
        for norm in (self.q_norm, self.kv_norm):
            norm.reset_parameters()
        nn.init.zeros_(self.attn_sink)
