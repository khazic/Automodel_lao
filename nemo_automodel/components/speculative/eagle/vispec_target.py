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

"""Target-model wrapper for ViSpec draft training on a vision-language target.

Where the EAGLE-1/2 wrapper hands the draft token ids, ViSpec hands it the
target's **embedding-layer output**: at image positions there is no token
embedding to look up, only the vision tower's projected features, and those
features are exactly what the draft's image adaptor compresses. The wrapper
therefore returns the target's layer-0 hidden states alongside the usual
last-hidden-state / logits supervision, plus the image-token mask that tells the
draft which positions to compress.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

from nemo_automodel.components.speculative.eagle.target_v12 import (
    _shift_left_with_zero,
    _to_full_tensor,
)


class _VispecPreallocatedLayer(DynamicLayer):
    """Dynamic-cache-compatible layer backed by reusable K/V storage."""

    _MIN_FREE_TOKENS = 256

    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Copy one initialized dynamic layer into reusable storage.

        Args:
            keys: Key cache tensor of shape [batch, heads, sequence, head_dim].
            values: Value cache tensor of shape [batch, heads, sequence, head_dim].
        """
        super().__init__()
        if keys.shape != values.shape:
            raise ValueError("ViSpec target key and value cache tensors must have matching shapes.")
        if keys.ndim != 4:
            raise ValueError("ViSpec target cache tensors must have shape [batch, heads, sequence, head_dim].")
        self.dtype = keys.dtype
        self.device = keys.device
        self._length = keys.shape[-2]
        capacity = self._length + self._MIN_FREE_TOKENS
        storage_shape = (*keys.shape[:-2], capacity, keys.shape[-1])
        self._key_storage = torch.empty(storage_shape, dtype=keys.dtype, device=keys.device)
        self._value_storage = torch.empty(storage_shape, dtype=values.dtype, device=values.device)
        self._key_storage[..., : self._length, :].copy_(keys)
        self._value_storage[..., : self._length, :].copy_(values)
        self.is_initialized = True

    @property
    def keys(self) -> torch.Tensor | None:
        """Return a [batch, heads, sequence, head_dim] logical key-cache view."""
        if not hasattr(self, "_key_storage") or self._key_storage is None:
            return None
        return self._key_storage[..., : self._length, :]

    @keys.setter
    def keys(self, value: torch.Tensor | None) -> None:
        self._key_storage = value

    @property
    def values(self) -> torch.Tensor | None:
        """Return a [batch, heads, sequence, head_dim] logical value-cache view."""
        if not hasattr(self, "_value_storage") or self._value_storage is None:
            return None
        return self._value_storage[..., : self._length, :]

    @values.setter
    def values(self, value: torch.Tensor | None) -> None:
        self._value_storage = value

    def _ensure_capacity(self, required_length: int) -> None:
        if required_length <= self._key_storage.shape[-2]:
            return
        capacity = max(required_length, self._key_storage.shape[-2] * 2)
        storage_shape = (*self._key_storage.shape[:-2], capacity, self._key_storage.shape[-1])
        keys = torch.empty(storage_shape, dtype=self.dtype, device=self.device)
        values = torch.empty(storage_shape, dtype=self.dtype, device=self.device)
        keys[..., : self._length, :].copy_(self._key_storage[..., : self._length, :])
        values[..., : self._length, :].copy_(self._value_storage[..., : self._length, :])
        self._key_storage = keys
        self._value_storage = values

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append K/V tensors without reallocating while reserved storage fits.

        Args:
            key_states: Key tensor of shape [batch, heads, query, head_dim].
            value_states: Value tensor of shape [batch, heads, query, head_dim].
            *args: Unused Transformers cache arguments.
            **kwargs: Unused Transformers cache keyword arguments.

        Returns:
            Logical key and value views of shape [batch, heads, sequence + query,
            head_dim] that alias the layer-owned storage.
        """
        del args, kwargs
        if key_states.shape != value_states.shape or key_states.shape[:-2] != self._key_storage.shape[:-2]:
            raise ValueError("Appended ViSpec target key/value tensors must match the allocated cache layout.")
        if key_states.shape[-1] != self._key_storage.shape[-1]:
            raise ValueError("Appended ViSpec target key/value tensors must preserve head_dim.")
        end = self._length + key_states.shape[-2]
        self._ensure_capacity(end)
        self._key_storage[..., self._length : end, :].copy_(key_states)
        self._value_storage[..., self._length : end, :].copy_(value_states)
        self._length = end
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Return the logical K/V length after appending ``query_length`` tokens."""
        return self._length + query_length, 0

    def get_seq_length(self) -> int:
        """Return the logical cached sequence length."""
        return self._length

    def get_max_cache_shape(self) -> int:
        """Return ``-1`` because storage grows geometrically when necessary."""
        return -1

    def crop(self, max_length: int) -> None:
        """Reduce logical length while retaining allocated storage."""
        if max_length < 0:
            max_length = self._length - abs(max_length)
        self._length = min(self._length, max_length)


@dataclass
class VispecTargetBatch:
    """Target-model outputs needed by :class:`VispecTrainerModule`.

    Attributes:
        inputs_embeds: Tensor of shape [batch, sequence, hidden] -- the target's
            embedding-layer output with vision features already spliced in,
            shifted left by one position.
        input_hidden_states: Tensor of shape [batch, sequence, hidden] -- the
            target's last hidden state, not shifted (the draft's input feature).
        target_logits: Tensor of shape [batch, sequence, vocab], shifted left.
        attention_mask: Tensor of shape [batch, sequence]; 1 for real tokens.
        loss_mask: Tensor of shape [batch, sequence], shifted left.
        image_mask: Bool tensor of shape [batch, sequence], shifted left so it
            aligns with ``inputs_embeds``.
    """

    inputs_embeds: torch.Tensor
    input_hidden_states: torch.Tensor
    target_logits: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    image_mask: torch.Tensor


@dataclass
class VispecGenerationState:
    """State owned by cached batch-one ViSpec generation.

    Attributes:
        input_ids: Tensor of shape [1, sequence] containing every token already
            processed by the target cache.
        attention_mask: Tensor of shape [1, sequence], with one for every
            processed token.
        inputs_embeds: Tensor of shape [1, sequence, hidden] containing the
            target embedding-layer output after vision features are merged.
        input_hidden_states: Tensor of shape [1, sequence, hidden] containing
            the target's final hidden states.
        image_mask: Bool tensor of shape [1, sequence], true at image-token
            positions in the processed prefix.
        next_token_logits: Tensor of shape [1, vocab] predicting the next token.
        past_key_values: Target KV cache covering exactly ``sequence`` tokens.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    inputs_embeds: torch.Tensor
    input_hidden_states: torch.Tensor
    image_mask: torch.Tensor
    next_token_logits: torch.Tensor
    past_key_values: Cache


@dataclass
class VispecTreeTargetOutput:
    """Target outputs for one flattened speculative tree.

    Attributes:
        token_ids: Tensor of shape [1, tree] in tree-node index order.
        inputs_embeds: Tensor of shape [1, tree, hidden] containing target token
            embeddings for the flattened tree.
        hidden_states: Tensor of shape [1, tree, hidden] containing target final
            hidden states for the flattened tree.
        logits: Tensor of shape [1, tree, vocab], where each node predicts its
            next token.
        prefix_length: Number of cached tokens before the tree forward. Tree KV
            entries occupy physical cache positions starting at this offset.
    """

    token_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    hidden_states: torch.Tensor
    logits: torch.Tensor
    prefix_length: int


class HFVispecTargetModel:
    """Expose embedding-layer, last-hidden-state, and logit supervision from a VLM target.

    Args:
        model: The frozen vision-language target model.
        image_token_id: Token id the target uses as an image placeholder; every
            position holding it carries a vision feature instead of a token
            embedding.
    """

    def __init__(self, model: nn.Module, *, image_token_id: int):
        self.model = model.eval()
        self.image_token_id = int(image_token_id)
        # The base model is fixed for this wrapper's lifetime, so resolve what its
        # forward accepts once instead of rebuilding a Signature on every
        # micro-batch of the training loop.
        forward_params = inspect.signature(self.model.model.forward).parameters
        self._accepted_params = frozenset(forward_params)
        # A VLM base model declares its vision tensors explicitly but funnels the
        # HF-generic flags through a ``**kwargs`` catch-all, so a plain
        # ``name in parameters`` test never matches them and the flags are
        # silently dropped, leaving ``use_cache`` on its config default and
        # allocating a full-sequence KV cache on every capture forward.
        has_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in forward_params.values())
        self._extra_kwargs = {
            name: False
            for name in ("output_attentions", "use_cache")
            if name in self._accepted_params or has_var_keyword
        }

    def get_input_embeddings(self) -> nn.Module:
        """Return the target model input embeddings."""
        return self.model.get_input_embeddings()

    def get_lm_head(self) -> nn.Module:
        """Return the target model lm_head."""
        return self.model.lm_head

    @staticmethod
    def _preallocate_generation_cache(cache: Cache) -> Cache:
        """Replace full-attention dynamic layers with reusable K/V storage.

        Args:
            cache: Target cache whose initialized K/V tensors have shape
                [batch, heads, sequence, head_dim]. The returned cache is the
                same container with its layer objects replaced in place.

        Returns:
            The input cache with reusable layer-owned K/V storage.
        """
        for layer_index, layer in enumerate(cache.layers):
            if isinstance(layer, _VispecPreallocatedLayer) or not layer.is_initialized:
                continue
            if not isinstance(layer, DynamicLayer) or layer.is_sliding:
                raise TypeError("ViSpec target cache preallocation supports full-attention dynamic layers only.")
            cache.layers[layer_index] = _VispecPreallocatedLayer(layer.keys, layer.values)
        return cache

    @torch.no_grad()
    def prefill_generation(self, model_inputs: dict[str, torch.Tensor]) -> VispecGenerationState:
        """Prefill the target cache and capture ViSpec prefix features.

        Args:
            model_inputs: Processor output containing ``input_ids`` and
                ``attention_mask`` tensors of shape [1, sequence]. Qwen vision
                inputs use ``pixel_values`` of shape [patches, patch_features],
                ``image_grid_thw`` of shape [images, 3], and optional
                ``mm_token_type_ids`` of shape [1, sequence].

        Returns:
            Cached state whose tensors cover the complete prompt but not the
            first generated token.
        """
        input_ids = model_inputs.get("input_ids")
        attention_mask = model_inputs.get("attention_mask")
        if input_ids is None or attention_mask is None:
            raise ValueError("Cached ViSpec generation requires input_ids and attention_mask.")
        if input_ids.shape[0] != 1 or attention_mask.shape != input_ids.shape:
            raise ValueError("Cached ViSpec generation requires matching batch-one input ids and attention mask.")

        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=1,
            do_sample=False,
            repetition_penalty=1.0,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_logits=True,
        )
        if generated.past_key_values is None or generated.hidden_states is None or generated.logits is None:
            raise RuntimeError("The target did not return cache, hidden states, and logits during ViSpec prefill.")
        prompt_hidden_states = generated.hidden_states[0]
        if not prompt_hidden_states:
            raise RuntimeError("The target returned no hidden states during ViSpec prefill.")
        cache = generated.past_key_values
        if cache.get_seq_length() != input_ids.shape[1]:
            raise RuntimeError(
                "ViSpec prefill cache must cover the prompt only: "
                f"expected {input_ids.shape[1]} tokens, got {cache.get_seq_length()}."
            )
        cache = self._preallocate_generation_cache(cache)

        return VispecGenerationState(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=prompt_hidden_states[0],
            input_hidden_states=prompt_hidden_states[-1],
            image_mask=input_ids.eq(self.image_token_id),
            next_token_logits=generated.logits[0],
            past_key_values=cache,
        )

    @torch.no_grad()
    def forward_tree_generation(
        self,
        state: VispecGenerationState,
        *,
        token_ids: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
    ) -> VispecTreeTargetOutput:
        """Verify a flattened speculative tree in one cached target forward.

        Args:
            state: Cached state whose tensor fields use the layouts documented
                by :class:`VispecGenerationState`.
            token_ids: Tensor of shape [1, tree] in tree-node index order.
            tree_attention_mask: Bool tensor of shape [tree, tree]. A true value
                allows a query node to attend to the corresponding tree node.
            tree_position_ids: Tensor of shape [tree] containing each node's
                depth relative to the root, whose depth is zero.

        Returns:
            Target features and logits for every tree node. The target appends
            all tree K/V entries to ``state.past_key_values`` in place; callers
            must immediately call :meth:`commit_tree_generation` to retain only
            one accepted path.
        """
        tree_size = token_ids.shape[1]
        if token_ids.shape[0] != 1 or tree_attention_mask.shape != (tree_size, tree_size):
            raise ValueError("ViSpec tree tokens must be [1, tree] with a matching [tree, tree] attention mask.")
        if tree_position_ids.shape != (tree_size,):
            raise ValueError("ViSpec tree position ids must have shape [tree].")
        if not isinstance(state.past_key_values, DynamicCache):
            raise TypeError("ViSpec tree cache compaction currently requires a Transformers DynamicCache target.")

        prefix_length = state.past_key_values.get_seq_length()
        if prefix_length != state.input_ids.shape[1]:
            raise RuntimeError("ViSpec target features and KV cache must cover the same prefix length.")
        prefix_allowed = state.attention_mask.bool().view(1, 1, 1, prefix_length).expand(1, 1, tree_size, -1)
        tree_allowed = tree_attention_mask.bool().view(1, 1, tree_size, tree_size)
        allowed = torch.cat((prefix_allowed, tree_allowed), dim=-1)
        mask_dtype = state.inputs_embeds.dtype
        attention_bias = torch.zeros(allowed.shape, dtype=mask_dtype, device=token_ids.device)
        attention_bias.masked_fill_(~allowed, torch.finfo(mask_dtype).min)

        position_hook = getattr(self.model.model, "compute_3d_position_ids", None)
        if not callable(position_hook):
            raise RuntimeError("The target must expose compute_3d_position_ids for multimodal tree verification.")
        root_token = token_ids[:, :1]
        root_attention_mask = torch.cat((state.attention_mask, state.attention_mask.new_ones((1, 1))), dim=1)
        position_kwargs: dict[str, object] = {
            "input_ids": root_token,
            "inputs_embeds": self.get_input_embeddings()(root_token),
            "image_grid_thw": None,
            "video_grid_thw": None,
            "attention_mask": root_attention_mask,
            "past_key_values": state.past_key_values,
            "second_per_grid_ts": None,
            "mm_token_type_ids": None,
        }
        accepted_position_kwargs = inspect.signature(position_hook).parameters
        root_position_ids = position_hook(
            **{key: value for key, value in position_kwargs.items() if key in accepted_position_kwargs}
        )
        if root_position_ids is None:
            raise RuntimeError("The target did not return multimodal position ids for the ViSpec tree root.")
        position_ids = root_position_ids[..., -1:] + tree_position_ids.view(1, 1, tree_size)

        outputs = self.model(
            input_ids=token_ids,
            attention_mask=attention_bias,
            position_ids=position_ids,
            past_key_values=state.past_key_values,
            use_cache=True,
            output_hidden_states=True,
            logits_to_keep=tree_size,
            return_dict=True,
        )
        if outputs.past_key_values is None or outputs.hidden_states is None or outputs.logits is None:
            raise RuntimeError(
                "The target did not return cache, hidden states, and logits for ViSpec tree verification."
            )
        if outputs.past_key_values.get_seq_length() != prefix_length + tree_size:
            raise RuntimeError("ViSpec tree verification appended an unexpected number of target cache entries.")
        return VispecTreeTargetOutput(
            token_ids=token_ids,
            inputs_embeds=outputs.hidden_states[0],
            hidden_states=outputs.hidden_states[-1],
            logits=outputs.logits,
            prefix_length=prefix_length,
        )

    @torch.no_grad()
    def commit_tree_generation(
        self,
        state: VispecGenerationState,
        tree_output: VispecTreeTargetOutput,
        *,
        accepted_tree_indices: torch.Tensor,
    ) -> VispecGenerationState:
        """Compact one accepted tree path into the persistent target cache.

        Args:
            state: Cached prefix state mutated by :meth:`forward_tree_generation`.
            tree_output: Flattened target output whose tensor layouts are
                documented by :class:`VispecTreeTargetOutput`.
            accepted_tree_indices: Tensor of shape [accepted] containing tree-node
                indices in root-to-leaf order. At least the root must be accepted.

        Returns:
            New state covering the accepted path. Cache tensors are compacted and
            cropped in place; the returned cache aliases ``state.past_key_values``.
        """
        if accepted_tree_indices.ndim != 1 or accepted_tree_indices.numel() < 1:
            raise ValueError("ViSpec cache commit requires at least one accepted tree index.")
        tree_size = tree_output.token_ids.shape[1]
        minimum_index, maximum_index = (
            torch.stack((accepted_tree_indices.min(), accepted_tree_indices.max())).cpu().tolist()
        )
        if minimum_index < 0 or maximum_index >= tree_size:
            raise ValueError("Accepted ViSpec tree indices are outside the verified tree.")
        cache = state.past_key_values
        if not isinstance(cache, DynamicCache):
            raise TypeError("ViSpec tree cache compaction currently requires a Transformers DynamicCache target.")
        accepted_count = accepted_tree_indices.numel()
        physical_indices = accepted_tree_indices + tree_output.prefix_length
        for layer in cache.layers:
            if not layer.is_initialized:
                continue
            selected_keys = layer.keys.index_select(-2, physical_indices.to(layer.keys.device))
            selected_values = layer.values.index_select(-2, physical_indices.to(layer.values.device))
            layer.keys[..., tree_output.prefix_length : tree_output.prefix_length + accepted_count, :].copy_(
                selected_keys
            )
            layer.values[..., tree_output.prefix_length : tree_output.prefix_length + accepted_count, :].copy_(
                selected_values
            )
        cache.crop(tree_output.prefix_length + accepted_count)

        accepted_tokens = tree_output.token_ids.index_select(1, accepted_tree_indices)
        accepted_inputs_embeds = tree_output.inputs_embeds.index_select(1, accepted_tree_indices)
        accepted_hidden_states = tree_output.hidden_states.index_select(1, accepted_tree_indices)
        input_ids = torch.cat((state.input_ids, accepted_tokens), dim=1)
        attention_mask = torch.cat((state.attention_mask, state.attention_mask.new_ones((1, accepted_count))), dim=1)
        last_tree_index = accepted_tree_indices[-1:]
        return VispecGenerationState(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=torch.cat((state.inputs_embeds, accepted_inputs_embeds), dim=1),
            input_hidden_states=torch.cat((state.input_hidden_states, accepted_hidden_states), dim=1),
            image_mask=torch.cat((state.image_mask, torch.zeros_like(accepted_tokens, dtype=torch.bool)), dim=1),
            next_token_logits=tree_output.logits.index_select(1, last_tree_index)[:, 0],
            past_key_values=cache,
        )

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **multimodal_inputs: torch.Tensor,
    ) -> VispecTargetBatch:
        """Run the frozen target once and assemble the draft's supervision.

        Args:
            input_ids: Tensor of shape [batch, sequence].
            attention_mask: Tensor of shape [batch, sequence]; 1 for real tokens.
            loss_mask: Tensor of shape [batch, sequence]; 1 at supervised positions.
            **multimodal_inputs: The processor's vision tensors for this batch
                (e.g. ``pixel_values`` of shape [patches, patch_dim] and
                ``image_grid_thw`` of shape [images, 3] for Qwen2.5-VL). Keys the
                target's forward does not declare are dropped.

        Returns:
            VispecTargetBatch, with every tensor on the target's device.
        """
        base_model = self.model.model
        accepted = {name: value for name, value in multimodal_inputs.items() if name in self._accepted_params}
        extra_kwargs = self._extra_kwargs

        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **accepted,
            **extra_kwargs,
        )
        hidden_states = outputs.hidden_states
        # HF emits ``num_layers + 1`` states: index 0 is the embedding output
        # (post vision-merge for a VLM), index -1 the post-final-norm state.
        inputs_embeds = hidden_states[0]
        last_hidden_states = hidden_states[-1]
        logits = _to_full_tensor(self.model.lm_head(last_hidden_states))

        return VispecTargetBatch(
            inputs_embeds=_shift_left_with_zero(inputs_embeds),
            input_hidden_states=last_hidden_states,
            target_logits=_shift_left_with_zero(logits),
            attention_mask=attention_mask,
            loss_mask=_shift_left_with_zero(loss_mask),
            image_mask=_shift_left_with_zero((input_ids == self.image_token_id).to(input_ids.dtype)).bool(),
        )
