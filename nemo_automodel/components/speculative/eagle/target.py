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

"""Target-model wrapper for minimal EAGLE-3 training."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from nemo_automodel.components.datasets.llm.packed_sequence import build_block_causal_additive_mask
from nemo_automodel.components.speculative.eagle.backend import Eagle3TargetBackend


def _shift_left_with_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Shift a batched sequence tensor left and zero-fill the tail.

    This matches the reference EAGLE-3 target preparation used by SpecForge:
    sequence-aligned tensors are shifted with ``padding(..., left=False)``.
    See SpecForge ``eagle3_target_model.py`` around the target preparation
    logic referenced by the user.
    """
    tail = torch.zeros_like(tensor[:, :1])
    return torch.cat((tensor[:, 1:], tail), dim=1)


def _layer_key_value(cache, layer_idx: int):
    """Return the ``(key, value)`` tensors for ``layer_idx`` from a KV cache.

    Works across transformers cache APIs: the >=4.54 / 5.x ``DynamicCache``
    exposes per-layer ``cache.layers[i].keys / .values`` and is no longer
    subscriptable, while older versions kept parallel ``key_cache`` /
    ``value_cache`` lists (and the legacy tuple cache is plain-indexable).
    """
    layers = getattr(cache, "layers", None)
    if layers is not None:
        layer = layers[layer_idx]
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is not None and values is not None:
            return keys, values
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        return key_cache[layer_idx], value_cache[layer_idx]
    entry = cache[layer_idx]
    return entry[0], entry[1]


@dataclass
class Eagle3TargetBatch:
    """Target-model supervision for one draft-training batch.

    Carries exactly one supervision encoding (validated in ``__post_init__``),
    both consumed directly by ``Eagle3TrainerModule.forward``:

    - ``logits`` -- the target's full-vocab logits; the draft-vocab projection
      happens trainer-side. Used by the co-located backend, where the tensor
      never leaves the GPU.
    - ``target_probs`` + ``position_mask`` -- the already-projected draft-vocab
      distribution, so a backend that computes it itself (e.g. a remote server)
      only transfers draft-vocab-sized tensors.
    """

    aux_hidden_states: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    logits: torch.Tensor | None = None
    target_probs: torch.Tensor | None = None
    position_mask: torch.Tensor | None = None
    # Packing metadata (None unless packing is enabled), unshifted slot frame:
    # per-document position_ids / seq_lens (block-causal mask) and doc_remaining
    # (gates cross-document TTT supervision).
    position_ids: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    doc_remaining: torch.Tensor | None = None
    # KV cache reuse: list of (K, V) tuples from target attention layers.
    # Each K/V is [B, num_kv_heads, T, head_dim]. None when kv_reuse is disabled.
    target_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    def __post_init__(self) -> None:
        has_logits = self.logits is not None
        has_precomputed = self.target_probs is not None and self.position_mask is not None
        if has_logits == has_precomputed:
            raise ValueError(
                "Eagle3TargetBatch requires exactly one supervision source: either "
                "`logits` (full-vocab, projected trainer-side) or both `target_probs` "
                "and `position_mask` (precomputed over the draft vocab)."
            )

    def to_trainer_inputs(self) -> dict[str, torch.Tensor]:
        """Return kwargs for ``Eagle3TrainerModule.forward``, dispatching on
        whichever supervision encoding this batch carries."""
        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "loss_mask": self.loss_mask,
            "aux_hidden_states": self.aux_hidden_states,
        }
        if self.logits is not None:
            inputs["target_logits"] = self.logits
        else:
            inputs["target_probs"] = self.target_probs
            inputs["position_mask"] = self.position_mask
        if self.seq_lens is not None:
            inputs["position_ids"] = self.position_ids
            inputs["seq_lens"] = self.seq_lens
            inputs["doc_remaining"] = self.doc_remaining
        if self.target_kv is not None:
            inputs["target_kv"] = self.target_kv
        return inputs


class HFEagle3TargetModel(Eagle3TargetBackend):
    """Co-located backend that captures three auxiliary hidden states from a causal LM."""

    def __init__(
        self,
        model: nn.Module,
        aux_layer_ids: Sequence[int] | None = None,
        kv_reuse_layer_ids: Sequence[int] | None = None,
    ):
        self.model = model.eval()
        candidate_ids = list(aux_layer_ids) if aux_layer_ids is not None else self._default_aux_layer_ids()
        self.aux_layer_ids = self._validate_aux_layer_ids(candidate_ids)
        self.kv_reuse_layer_ids = self._validate_kv_reuse_layer_ids(kv_reuse_layer_ids)

    def _default_aux_layer_ids(self) -> list[int]:
        # EAGLE-3 default 3-layer recipe (low / mid / high).
        #
        # The downstream draft model's ``fc`` projection is sized for
        # exactly ``num_aux_hidden_states`` layers (default 3) of
        # concatenated target hidden states. Silently deduplicating
        # collisions on shallow targets would yield fewer than 3
        # captured tensors and crash later inside the draft ``fc`` with
        # a confusing shape-mismatch error -- raise here instead so the
        # caller picks 3 distinct in-bounds ids that match the draft
        # config.
        num_layers = self.model.config.num_hidden_layers
        candidates = [1, num_layers // 2 - 1, num_layers - 4]
        if any(c < 0 or c >= num_layers for c in candidates) or len(set(candidates)) != 3:
            raise ValueError(
                f"Target model has num_hidden_layers={num_layers}, which is too shallow "
                f"for the default EAGLE-3 aux recipe {candidates}. Pass aux_layer_ids "
                f"explicitly (must be 3 distinct in-bounds layer indices, matching the "
                f"draft model's num_aux_hidden_states)."
            )
        return candidates

    def _validate_aux_layer_ids(self, aux_layer_ids: Sequence[int]) -> list[int]:
        """Validate aux-layer selection before any forward hooks are registered."""
        num_layers = self.model.config.num_hidden_layers
        aux_layer_ids = list(aux_layer_ids)
        if len(aux_layer_ids) != 3:
            raise ValueError(
                f"EAGLE-3 expects exactly 3 aux_layer_ids, but got {len(aux_layer_ids)}: "
                f"{aux_layer_ids}. This must match the draft model's num_aux_hidden_states."
            )
        if len(set(aux_layer_ids)) != len(aux_layer_ids):
            raise ValueError(
                f"EAGLE-3 aux_layer_ids must be distinct, but got {aux_layer_ids}. "
                "Duplicate ids would collapse the captured aux hidden states."
            )
        for layer_id in aux_layer_ids:
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(f"aux layer id {layer_id} is out of bounds for model with {num_layers} layers")
        return aux_layer_ids

    def _validate_kv_reuse_layer_ids(
        self, kv_reuse_layer_ids: Sequence[int] | None, num_draft_layers: int | None = None
    ) -> list[int] | None:
        """Validate and default KV reuse layer selection."""
        if kv_reuse_layer_ids is None:
            return None
        num_layers = self.model.config.num_hidden_layers
        kv_reuse_layer_ids = list(kv_reuse_layer_ids)
        for layer_id in kv_reuse_layer_ids:
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(f"kv_reuse layer id {layer_id} is out of bounds for model with {num_layers} layers")
        if num_draft_layers is not None and len(kv_reuse_layer_ids) != num_draft_layers:
            raise ValueError(
                f"kv_reuse_layer_ids has {len(kv_reuse_layer_ids)} entries but draft model has "
                f"{num_draft_layers} layers. They must match 1:1 (draft layer i uses target KV from "
                f"kv_reuse_layer_ids[i])."
            )
        return kv_reuse_layer_ids

    def set_kv_reuse_layer_ids(self, kv_reuse_layer_ids: Sequence[int], num_draft_layers: int | None = None) -> None:
        """Set KV reuse layer IDs after construction (e.g. when derived from draft config)."""
        self.kv_reuse_layer_ids = self._validate_kv_reuse_layer_ids(kv_reuse_layer_ids, num_draft_layers)

    def _get_transformer_layers(self) -> list[nn.Module]:
        """Return decoder layers as an ordered list indexable by integer.

        Supports both the HuggingFace layouts (where ``layers`` is a
        ``ModuleList``) and AutoModel's custom-impl layouts (where
        ``layers`` is a ``ModuleDict`` keyed by ``str(i)``). Returning a
        plain list normalizes the access pattern for downstream
        ``register_forward_hook`` calls.
        """
        # Common HF causal-LM layouts:
        #   model.model.layers              (Llama, Qwen, Mistral, Gemma, Phi, ...)
        #   model.layers                    (some VLM text backbones exposed directly)
        #   model.transformer.h             (GPT2 / Falcon-style)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            container = self.model.model.layers
        elif hasattr(self.model, "layers"):
            container = self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            container = self.model.transformer.h
        else:
            raise ValueError("Unsupported model structure for EAGLE-3 aux-layer capture")
        if isinstance(container, nn.ModuleDict):
            # AutoModel custom impls use ModuleDict keyed by ``str(i)``.
            return [container[str(i)] for i in range(len(container))]
        return list(container)

    @staticmethod
    def _get_attention_module(layer: nn.Module) -> nn.Module:
        """Return the self-attention submodule of a decoder layer."""
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        if hasattr(layer, "attention"):
            return layer.attention
        raise ValueError(
            f"Cannot find attention submodule on {type(layer).__name__}; expected attribute 'self_attn' or 'attention'."
        )

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the target model input embeddings."""
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        doc_remaining: torch.Tensor | None = None,
    ) -> Eagle3TargetBatch:
        """Run the target model and capture aux hidden states plus logits.

        With ``seq_lens`` (packing), the target runs with a ``[B, 1, T, T]``
        block-causal mask and per-document ``position_ids`` so its outputs respect
        document boundaries; the packing metadata is forwarded unshifted to the
        trainer. ``seq_lens=None`` keeps the original 2D-mask path.
        """
        layers = self._get_transformer_layers()
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def _make_hook(layer_id: int):
            def _hook(_module, _inputs, outputs):
                captured[layer_id] = outputs[0] if isinstance(outputs, tuple) else outputs

            return _hook

        for layer_id in self.aux_layer_ids:
            if layer_id < 0 or layer_id >= len(layers):
                raise ValueError(f"aux layer id {layer_id} is out of bounds for model with {len(layers)} layers")
            handles.append(layers[layer_id].register_forward_hook(_make_hook(layer_id)))

        # AutoModel's custom causal LMs only declare ``input_ids``,
        # ``attention_mask``, ``position_ids``, ``padding_mask`` and a
        # ``**attn_kwargs`` catch-all; the HF flags below mean nothing to
        # them and are dropped to keep the call site honest.
        forward_params = inspect.signature(self.model.forward).parameters
        extra_kwargs = {
            name: False for name in ("output_hidden_states", "output_attentions") if name in forward_params
        }
        # When KV reuse is active, enable the cache so we can extract the real
        # post-RoPE/post-norm K and V directly from the model's own cache object.
        # This is correct for all architectures (Llama, Qwen3 with k_norm, etc.)
        # because the cache receives KV after the full attention projection pipeline.
        use_cache_for_kv = bool(self.kv_reuse_layer_ids) and "use_cache" in forward_params
        if "use_cache" in forward_params:
            extra_kwargs["use_cache"] = use_cache_for_kv

        # Packing isolates documents per attention backend; the mask strategy
        # differs because FlashAttention has no 4D-mask code path:
        #   * SDPA / eager consume the [B, 1, T, T] block-causal additive mask.
        #   * FlashAttention infers per-document cu_seqlens from the reset points
        #     in a per-document ``position_ids`` and is passed ``attention_mask=None``.
        #     Feeding FA the 4D additive mask instead drives its unpad gather out
        #     of bounds: the mask flattens to B*T*T entries and the gather indexes
        #     a B*T-row tensor with them. transformers only packs from position_ids
        #     at batch size 1 (see ``_is_packed_sequence``).
        target_attention_mask = attention_mask
        if seq_lens is not None:
            if position_ids is None or "position_ids" not in forward_params:
                raise ValueError(
                    "EAGLE-3 sequence packing requires per-document position_ids, but none were "
                    "provided or the target model's forward does not accept a `position_ids` argument."
                )
            extra_kwargs["position_ids"] = position_ids
            attn_impl = getattr(self.model.config, "_attn_implementation", None) or ""
            if "flash" in attn_impl:
                if input_ids.shape[0] != 1:
                    raise ValueError(
                        "EAGLE-3 sequence packing with a FlashAttention target only supports "
                        f"micro_batch_size=1 (got {input_ids.shape[0]}). FlashAttention infers "
                        "document boundaries from per-document position_ids, which transformers "
                        "packs only at batch size 1. Set micro_batch_size=1 or load the target "
                        "with attn_implementation='sdpa'."
                    )
                # attention_mask=None + per-document position_ids -> FA varlen packing.
                target_attention_mask = None
            else:
                param_dtype = next(self.model.parameters()).dtype
                target_attention_mask = build_block_causal_additive_mask(
                    seq_lens,
                    seq_length=input_ids.shape[1],
                    dtype=param_dtype,
                    device=input_ids.device,
                )

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=target_attention_mask,
                **extra_kwargs,
            )
        finally:
            for handle in handles:
                handle.remove()

        if len(captured) != len(self.aux_layer_ids):
            raise RuntimeError(
                f"Expected {len(self.aux_layer_ids)} captured aux layers but got {len(captured)}: {sorted(captured)}"
            )

        aux_hidden_states = torch.cat([captured[layer_id] for layer_id in self.aux_layer_ids], dim=-1)

        target_kv = None
        if self.kv_reuse_layer_ids and use_cache_for_kv:
            past_kv = outputs.past_key_values if hasattr(outputs, "past_key_values") else None
            if past_kv is None:
                raise RuntimeError(
                    "kv_reuse is enabled but the target model did not return past_key_values. "
                    "Ensure the target model supports use_cache=True."
                )
            target_kv = [
                tuple(t.detach() for t in _layer_key_value(past_kv, lid)) for lid in self.kv_reuse_layer_ids
            ]

        # HF causal LM outputs wrap logits in a dataclass; AutoModel's
        # custom causal LM returns the logits tensor directly.
        target_logits = outputs.logits if hasattr(outputs, "logits") else outputs
        shifted_logits = _shift_left_with_zero(target_logits)
        shifted_input_ids = _shift_left_with_zero(input_ids)
        shifted_loss_mask = _shift_left_with_zero(loss_mask)
        return Eagle3TargetBatch(
            aux_hidden_states=aux_hidden_states,
            logits=shifted_logits,
            input_ids=shifted_input_ids,
            attention_mask=attention_mask,
            loss_mask=shifted_loss_mask,
            position_ids=position_ids,
            seq_lens=seq_lens,
            doc_remaining=doc_remaining,
            target_kv=target_kv,
        )
