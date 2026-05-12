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

from typing import Any

import torch

from nemo_automodel.components.loss.mtp import MTPLogits

_MTP_OUTPUT_NAMES = (
    "mtp_logits",
    "nextn_logits",
    "next_n_logits",
    "multi_token_logits",
    "speculative_logits",
)


def extract_mtp_logits(output: Any) -> list[MTPLogits] | None:
    """Extract MTP logits already returned by a model forward pass."""

    for name in _MTP_OUTPUT_NAMES:
        value = getattr(output, name, None)
        if value is None and isinstance(output, dict):
            value = output.get(name)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            return [MTPLogits(value, 1)]
        return [item if isinstance(item, MTPLogits) else MTPLogits(item, idx + 1) for idx, item in enumerate(value)]
    return None


def _get_language_model(model: torch.nn.Module) -> torch.nn.Module | None:
    inner_model = getattr(model, "model", None)
    if inner_model is None:
        return None
    return getattr(inner_model, "language_model", None)


def _slice_position_ids(position_ids: torch.Tensor | None, end: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if position_ids is None:
        return None, None
    if position_ids.ndim == 2:
        text_position_ids = position_ids[:, :end]
        return text_position_ids, text_position_ids[None, ...].expand(4, text_position_ids.shape[0], -1)
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0, :, :end]
        return text_position_ids, position_ids[:, :, :end]
    if position_ids.ndim == 3 and position_ids.shape[0] == 3:
        return None, position_ids[:, :, :end]
    return None, position_ids[..., :end]


def _build_mtp_attention_mask(
    language_model: torch.nn.Module,
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    text_position_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    layer_type = getattr(layer, "layer_type", None)
    if layer_type == "linear_attention":
        return attention_mask
    if attention_mask is None:
        return None

    try:
        from transformers.masking_utils import create_causal_mask
    except ImportError:
        return None

    return create_causal_mask(
        config=language_model.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )


def compute_qwen3_5_mtp_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor | None,
    hidden_states: torch.Tensor | None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> list[MTPLogits] | None:
    """Compute Qwen3.5/Qwen3.6 MTP logits from HF-style ``model.mtp`` modules.

    Qwen3.6 checkpoints ship a ``mtp`` module, while the current Transformers
    training forward returns only the main logits.  This helper mirrors the
    model-local module structure so AutoModel recipes can train the MTP weights
    without forking the HF model class.
    """

    mtp = getattr(model, "mtp", None)
    language_model = _get_language_model(model)
    if mtp is None or language_model is None:
        return None
    if input_ids is None:
        raise ValueError("Qwen3.6 MTP training requires `input_ids`; `inputs_embeds` alone is not enough.")
    if hidden_states is None:
        raise ValueError("Qwen3.6 MTP training requires main-model hidden states. Pass `output_hidden_states=True`.")

    layers = getattr(mtp, "layers", None)
    if layers is None or len(layers) == 0:
        return None

    embed_tokens = getattr(language_model, "embed_tokens", None)
    rotary_emb = getattr(language_model, "rotary_emb", None)
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else getattr(model, "lm_head")
    if embed_tokens is None or rotary_emb is None or lm_head is None:
        raise ValueError("Qwen3.6 MTP training requires embed_tokens, rotary_emb, and lm_head modules.")

    mtp_hidden = hidden_states
    mtp_outputs: list[MTPLogits] = []
    for index, layer in enumerate(layers):
        target_offset = index + 1
        if input_ids.shape[1] <= target_offset:
            break

        current_hidden = mtp_hidden[:, :-1, :]
        target_embeddings = embed_tokens(input_ids[:, target_offset:])
        current_hidden = mtp.fc(
            torch.cat(
                [
                    mtp.pre_fc_norm_hidden(current_hidden),
                    mtp.pre_fc_norm_embedding(target_embeddings),
                ],
                dim=-1,
            )
        )

        text_position_ids, mtp_position_ids = _slice_position_ids(position_ids, -target_offset)
        if mtp_position_ids is None:
            mtp_position_ids = torch.arange(
                current_hidden.shape[1], device=current_hidden.device, dtype=torch.long
            ).view(1, 1, -1)
            mtp_position_ids = mtp_position_ids.expand(4, current_hidden.shape[0], -1)
            text_position_ids = mtp_position_ids[0]

        position_embeddings = rotary_emb(current_hidden, mtp_position_ids)
        mtp_attention_mask = attention_mask[:, :-target_offset] if attention_mask is not None else None
        mtp_attention_mask = _build_mtp_attention_mask(
            language_model,
            layer,
            current_hidden,
            mtp_attention_mask,
            text_position_ids,
        )
        current_hidden = layer(
            current_hidden,
            position_embeddings=position_embeddings,
            attention_mask=mtp_attention_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
        )
        current_hidden = mtp.norm(current_hidden)
        mtp_outputs.append(MTPLogits(lm_head(current_hidden), target_offset))
        mtp_hidden = current_hidden

    return mtp_outputs
