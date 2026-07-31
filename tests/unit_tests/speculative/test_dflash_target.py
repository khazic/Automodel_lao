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

"""Unit tests for ``HFDFlashTargetModel`` (decoder hidden-state capture).

Covers layer-id validation, the ``_get_transformer_layers`` model-structure
dispatch (HF ``ModuleList`` vs AutoModel custom ``ModuleDict``), and the
forward-hook capture producing the correctly concatenated context features --
including that the HF-only flags are not forwarded to a custom backbone whose
``forward`` does not declare them.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput

from nemo_automodel.components.speculative.dflash.target import (
    DFlashTargetBatch,
    HFDFlashTargetModel,
    HFMultimodalDFlashTargetModel,
)

_FORBIDDEN_HF_FLAGS = {"output_hidden_states", "output_attentions", "use_cache"}
_VOCAB = 32
_HIDDEN = 16
_LAYERS = 4


class _FakeHFBackbone(nn.Module):
    """HuggingFace-style backbone: ``ModuleList`` layers, explicit HF flags."""

    def __init__(self, embed: nn.Embedding) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(_HIDDEN, _HIDDEN) for _ in range(_LAYERS)])
        self._embed = embed

    def forward(
        self, input_ids, attention_mask=None, output_hidden_states=False, output_attentions=False, use_cache=False
    ):
        h = self._embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return (h,)


class _FakeHFCausalLM(nn.Module):
    """HF causal-LM stand-in: ``self.model.layers`` is a ``ModuleList``."""

    def __init__(self) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"num_hidden_layers": _LAYERS})
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.model = _FakeHFBackbone(self.embed_tokens)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self, input_ids, attention_mask=None, output_hidden_states=False, output_attentions=False, use_cache=False
    ):
        h = self.model(input_ids, attention_mask=attention_mask)[0]
        return CausalLMOutput(logits=self.lm_head(h))


class _FakeCustomBackbone(nn.Module):
    """AutoModel custom-impl backbone: ``ModuleDict`` layers + ``**attn_kwargs``."""

    def __init__(self, embed: nn.Embedding) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({str(i): nn.Linear(_HIDDEN, _HIDDEN) for i in range(_LAYERS)})
        self._embed = embed

    def forward(self, input_ids, attention_mask=None, **attn_kwargs):
        leaked = _FORBIDDEN_HF_FLAGS & set(attn_kwargs)
        if leaked:
            raise AssertionError(f"HF flag leaked to custom backbone: {leaked}")
        h = self._embed(input_ids)
        for layer in self.layers.values():
            h = layer(h)
        return h


class _FakeCustomCausalLM(nn.Module):
    """AutoModel custom causal-LM: ``self.model.layers`` is a ``ModuleDict``."""

    def __init__(self) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"num_hidden_layers": _LAYERS})
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.model = _FakeCustomBackbone(self.embed_tokens)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, input_ids, attention_mask=None, **attn_kwargs):
        h = self.model(input_ids, attention_mask=attention_mask, **attn_kwargs)
        return self.lm_head(h)


class _FakeMultimodalBackbone(nn.Module):
    """VLM backbone that replaces image-token embeddings from pixel features."""

    def __init__(self, embed: nn.Embedding) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([nn.Linear(_HIDDEN, _HIDDEN) for _ in range(_LAYERS)])
        self._embed = embed

    def forward(self, input_ids, pixel_values):
        """Run a fake multimodal forward.

        Args:
            input_ids: Long tensor of shape [batch, sequence].
            pixel_values: Tensor of shape [batch, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        hidden = self._embed(input_ids) + pixel_values[:, None, :]
        for layer in self.language_model.layers:
            hidden = layer(hidden)
        return hidden


class _FakeMultimodalLM(nn.Module):
    """Image-text target with language layers below ``model.language_model``."""

    def __init__(self) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"num_hidden_layers": _LAYERS})
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.model = _FakeMultimodalBackbone(self.embed_tokens)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        position_ids=None,
        use_cache=False,
    ):
        """Return logits for a processor-shaped image-text batch.

        Args:
            input_ids: Long tensor of shape [batch, sequence].
            attention_mask: Optional tensor of shape [batch, sequence].
            pixel_values: Tensor of shape [batch, hidden].
            image_grid_thw: Long tensor of shape [images, 3].
            use_cache: Whether to return a KV cache; unused by this test model.

        Returns:
            Causal-LM output with logits of shape [batch, sequence, vocab].
        """
        hidden = self.model(input_ids, pixel_values)
        self.forward_inputs = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
            "position_ids": position_ids,
        }
        return CausalLMOutput(logits=self.lm_head(hidden))


def _batch(batch: int = 2, seq: int = 8):
    input_ids = torch.randint(0, _VOCAB, (batch, seq))
    attn = torch.ones(batch, seq, dtype=torch.long)
    loss = torch.ones(batch, seq, dtype=torch.long)
    return input_ids, attn, loss


# --- layer-id validation ---


def test_rejects_empty_layer_ids():
    with pytest.raises(ValueError, match="at least one"):
        HFDFlashTargetModel(_FakeHFCausalLM(), target_layer_ids=[])


def test_rejects_out_of_bounds_layer_id():
    with pytest.raises(ValueError, match="out of bounds"):
        HFDFlashTargetModel(_FakeHFCausalLM(), target_layer_ids=[0, _LAYERS])


# --- _get_transformer_layers dispatch ---


def test_get_layers_modulelist():
    target = HFDFlashTargetModel(_FakeHFCausalLM(), target_layer_ids=[0, 2])
    layers = target._get_transformer_layers()
    assert len(layers) == _LAYERS
    assert all(isinstance(layer, nn.Linear) for layer in layers)


def test_get_layers_moduledict_is_ordered():
    fake = _FakeCustomCausalLM()
    target = HFDFlashTargetModel(fake, target_layer_ids=[0, 2])
    layers = target._get_transformer_layers()
    assert [layers[i] for i in range(_LAYERS)] == [fake.model.layers[str(i)] for i in range(_LAYERS)]


# --- hook capture / concatenation ---


def test_generate_batch_concatenates_selected_layers():
    layer_ids = [1, 3]
    target = HFDFlashTargetModel(_FakeHFCausalLM(), target_layer_ids=layer_ids)
    input_ids, attn, loss = _batch(batch=2, seq=8)
    out = target.generate_batch(input_ids, attn, loss)
    assert isinstance(out, DFlashTargetBatch)
    # context features = the selected layers' hidden states concatenated on the feature dim
    assert out.hidden_states.shape == (2, 8, len(layer_ids) * _HIDDEN)
    # DFlash does NOT shift the supervision tensors (unlike EAGLE-3)
    assert torch.equal(out.input_ids, input_ids)
    assert torch.equal(out.loss_mask, loss)


def test_generate_batch_drops_hf_flags_for_custom_backbone():
    # The custom backbone raises if any HF-only flag leaks through.
    target = HFDFlashTargetModel(_FakeCustomCausalLM(), target_layer_ids=[0, 2])
    input_ids, attn, loss = _batch()
    out = target.generate_batch(input_ids, attn, loss)
    assert out.hidden_states.shape == (2, 8, 2 * _HIDDEN)


# --- teacher-logit capture (JetSpec forward-KL distillation) ---


def test_capture_logits_off_by_default():
    """DFlash does not need teacher logits; the wrapper leaves them None by default."""
    target = HFDFlashTargetModel(_FakeHFCausalLM(), target_layer_ids=[1, 3])
    out = target.generate_batch(*_batch(batch=2, seq=8))
    assert out.logits is None


def test_capture_logits_returns_hf_output_logits():
    """capture_logits=True keeps the HF output's ``.logits`` (full-vocab teacher dist)."""
    model = _FakeHFCausalLM()
    target = HFDFlashTargetModel(model, target_layer_ids=[1, 3], capture_logits=True)
    input_ids, attn, loss = _batch(batch=2, seq=8)
    out = target.generate_batch(input_ids, attn, loss)
    assert out.logits is not None
    assert out.logits.shape == (2, 8, _VOCAB)
    assert torch.isfinite(out.logits).all()


def test_capture_logits_handles_bare_tensor_return():
    """A custom backbone that returns a bare logits tensor (no ``.logits``) is captured as-is."""
    target = HFDFlashTargetModel(_FakeCustomCausalLM(), target_layer_ids=[0, 2], capture_logits=True)
    out = target.generate_batch(*_batch(batch=2, seq=8))
    assert out.logits is not None
    assert out.logits.shape == (2, 8, _VOCAB)


def test_multimodal_target_captures_language_layers_with_pixels():
    target = HFMultimodalDFlashTargetModel(_FakeMultimodalLM(), target_layer_ids=[1, 3])
    input_ids, attention_mask, loss_mask = _batch(batch=2, seq=8)
    loss_mask[:, :4] = 0
    pixel_values = torch.randn(2, _HIDDEN)

    output = target.generate_batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        pixel_values=pixel_values,
        image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]),
    )

    assert output.hidden_states.shape == (2, 8, 2 * _HIDDEN)
    assert torch.equal(output.input_ids, input_ids)
    assert torch.equal(output.loss_mask, loss_mask)
    assert output.loss_mask[:, :4].count_nonzero() == 0


def test_multimodal_target_forwards_qwen3_vl_processor_tensors():
    model = _FakeMultimodalLM()
    target = HFMultimodalDFlashTargetModel(model, target_layer_ids=[1, 3])
    input_ids, attention_mask, loss_mask = _batch(batch=2, seq=8)
    pixel_values = torch.randn(2, _HIDDEN)
    video_pixels = torch.randn(3, _HIDDEN)
    video_grid = torch.tensor([[1, 1, 3]])
    token_types = torch.zeros_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)

    output = target.generate_batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        pixel_values=pixel_values,
        image_grid_thw=torch.tensor([[1, 1, 2]]),
        pixel_values_videos=video_pixels,
        video_grid_thw=video_grid,
        mm_token_type_ids=token_types,
        position_ids=position_ids,
    )

    assert torch.equal(model.forward_inputs["pixel_values_videos"], video_pixels)
    assert torch.equal(model.forward_inputs["video_grid_thw"], video_grid)
    assert torch.equal(model.forward_inputs["mm_token_type_ids"], token_types)
    assert torch.equal(model.forward_inputs["position_ids"], position_ids)
    assert torch.equal(output.position_ids, position_ids)
