# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU tests for the post-hoc Gemma4 n-gram embedding and its decoder hook."""

import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration
from nemo_automodel.components.models.gemma4_moe.ngram import (
    Gemma4NGramConfig,
    Gemma4NGramEmbedding,
    Gemma4NGramInjection,
    ngram_layer_multipliers,
)

EOS = 1


def _tiny_ngram_config(**overrides) -> Gemma4NGramConfig:
    defaults = dict(
        ngram_size=3,
        heads_per_ngram=2,
        head_dim=8,
        rows_per_head=101,
        layer_index=1,
        eos_token_ids=(EOS,),
        conv_kernel_size=3,
    )
    defaults.update(overrides)
    return Gemma4NGramConfig(**defaults)


def _dense_text_config(**overrides) -> Gemma4TextConfig:
    defaults = dict(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=3,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        enable_moe_block=False,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=32,
        hidden_activation="gelu_pytorch_tanh",
        hidden_size_per_layer_input=4,
        vocab_size_per_layer_input=64,
        num_kv_shared_layers=0,
        eos_token_id=EOS,
        pad_token_id=0,
        bos_token_id=2,
        torch_dtype="float32",
    )
    defaults.update(overrides)
    return Gemma4TextConfig(**defaults)


def _cpu_backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _build_dense_model(ngram_config: Gemma4NGramConfig | dict | None) -> Gemma4ForConditionalGeneration:
    torch.manual_seed(0)
    config = Gemma4Config(text_config=_dense_text_config(), vision_config=None, audio_config=None)
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    model = Gemma4ForConditionalGeneration(config, backend=_cpu_backend(), ngram_config=ngram_config)
    model.eval()
    return model


class TestConfig:
    def test_head_layout_uses_distinct_primes_below_bound(self):
        config = _tiny_ngram_config(rows_per_head=101, heads_per_ngram=2)
        assert config.num_heads == 4
        assert config.head_sizes == (101, 97, 89, 83)
        assert config.head_offsets == (0, 101, 198, 287)
        assert config.num_rows == 370
        assert config.embed_dim == 32

    def test_scalar_eos_is_normalized_to_tuple(self):
        assert _tiny_ngram_config(eos_token_ids=EOS).eos_token_ids == (EOS,)

    @pytest.mark.parametrize(
        "overrides",
        [dict(ngram_size=1), dict(heads_per_ngram=0), dict(head_dim=0), dict(rows_per_head=1), dict(layer_index=-1)],
    )
    def test_invalid_values_are_rejected(self, overrides):
        with pytest.raises(ValueError):
            _tiny_ngram_config(**overrides)

    def test_build_rejects_layer_index_past_decoder(self):
        with pytest.raises(ValueError, match="out of range"):
            _tiny_ngram_config(layer_index=3).build(hidden_size=8, num_hidden_layers=3, rms_norm_eps=1e-6, dtype=None)

    def test_multipliers_are_deterministic_odd_int64(self):
        first = ngram_layer_multipliers(3, seed=7)
        assert first == ngram_layer_multipliers(3, seed=7)
        assert first != ngram_layer_multipliers(3, seed=8)
        assert all(value % 2 == 1 for value in first)
        assert all(-(1 << 63) <= value < (1 << 63) for value in first)


class TestEmbedding:
    def test_rows_stay_inside_each_head(self):
        config = _tiny_ngram_config()
        embedding = Gemma4NGramEmbedding(config)
        ids = torch.randint(0, 64, (2, 11))
        rows = embedding.hash_input_ids(ids)
        assert rows.shape == (2, 11, config.num_heads)
        for head, (size, offset) in enumerate(zip(config.head_sizes, config.head_offsets)):
            assert rows[..., head].min() >= offset
            assert rows[..., head].max() < offset + size

    def test_context_resets_after_eos(self):
        embedding = Gemma4NGramEmbedding(_tiny_ngram_config())
        # Token 5 preceded by the same two tokens (7, 9) once at the sequence start and
        # once right after an EOS: both positions must hash identically, while the
        # same token with the real preceding context (3, 4) hashes differently.
        ids = torch.tensor([[7, 9, 5, 3, 4, 5, EOS, 7, 9, 5]])
        rows = embedding.hash_input_ids(ids)
        assert torch.equal(rows[0, 2], rows[0, 9])
        assert not torch.equal(rows[0, 2], rows[0, 5])
        # The first token after EOS has no context, like the first token overall.
        assert torch.equal(rows[0, 0], rows[0, 7])
        assert torch.equal(rows[0, 1], rows[0, 8])

    def test_bigram_heads_ignore_the_third_token(self):
        config = _tiny_ngram_config()
        embedding = Gemma4NGramEmbedding(config)
        base = torch.tensor([[11, 12, 13]])
        other = torch.tensor([[21, 12, 13]])
        rows_base, rows_other = embedding.hash_input_ids(base), embedding.hash_input_ids(other)
        bigram = slice(0, config.heads_per_ngram)
        trigram = slice(config.heads_per_ngram, config.num_heads)
        assert torch.equal(rows_base[0, 2, bigram], rows_other[0, 2, bigram])
        assert not torch.equal(rows_base[0, 2, trigram], rows_other[0, 2, trigram])

    def test_forward_concatenates_heads(self):
        config = _tiny_ngram_config()
        embedding = Gemma4NGramEmbedding(config)
        ids = torch.randint(0, 64, (3, 5))
        values = embedding(ids)
        assert values.shape == (3, 5, config.embed_dim)
        rows = embedding.hash_input_ids(ids)
        expected = embedding.table.weight[rows].flatten(start_dim=-2)
        torch.testing.assert_close(values, expected)

    @pytest.mark.parametrize("ids", [torch.zeros(4), torch.zeros(1, 4, dtype=torch.float32)])
    def test_bad_input_ids_are_rejected(self, ids):
        with pytest.raises(ValueError):
            Gemma4NGramEmbedding(_tiny_ngram_config()).hash_input_ids(ids)


class TestInjection:
    def test_starts_as_zero_delta_and_learns(self):
        config = _tiny_ngram_config()
        injection = Gemma4NGramInjection(config, hidden_size=16)
        hidden = torch.randn(2, 6, 16)
        ids = torch.randint(0, 64, (2, 6))
        assert torch.equal(injection(hidden, ids), torch.zeros(2, 6, 16))
        with torch.no_grad():
            injection.value_proj.weight.normal_()
        delta = injection(hidden, ids)
        assert delta.shape == (2, 6, 16)
        assert delta.abs().sum() > 0
        delta.pow(2).sum().backward()
        table_grad = injection.embedding.table.weight.grad
        assert table_grad is not None
        touched_rows = injection.embedding.hash_input_ids(ids).unique()
        assert table_grad[touched_rows].abs().sum() > 0
        untouched = torch.ones(config.num_rows, dtype=torch.bool)
        untouched[touched_rows] = False
        assert table_grad[untouched].abs().sum() == 0

    def test_convolution_is_causal(self):
        config = _tiny_ngram_config()
        injection = Gemma4NGramInjection(config, hidden_size=16)
        with torch.no_grad():
            injection.value_proj.weight.normal_()
            injection.conv1d.weight.normal_()
        ids = torch.randint(0, 64, (1, 12))
        hidden = torch.randn(1, 12, 16)
        reference = injection(hidden, ids)
        perturbed_hidden = hidden.clone()
        perturbed_hidden[:, 9:] += 1.0
        perturbed = injection(perturbed_hidden, ids)
        torch.testing.assert_close(perturbed[:, :9], reference[:, :9])

    def test_shape_mismatches_are_rejected(self):
        injection = Gemma4NGramInjection(_tiny_ngram_config(), hidden_size=16)
        with pytest.raises(ValueError, match="hidden_states"):
            injection(torch.randn(2, 6, 8), torch.zeros(2, 6, dtype=torch.long))
        with pytest.raises(ValueError, match="input_ids"):
            injection(torch.randn(2, 6, 16), torch.zeros(2, 5, dtype=torch.long))

    def test_hook_requires_stashed_ids_and_returns_new_tensor(self):
        injection = Gemma4NGramInjection(_tiny_ngram_config(), hidden_size=16)
        hidden = torch.randn(1, 4, 16)
        with pytest.raises(RuntimeError, match="input_ids"):
            injection.decoder_layer_pre_hook(torch.nn.Identity(), (hidden,), {})
        injection.stash_input_ids(torch.randint(0, 64, (1, 4)))
        args, kwargs = injection.decoder_layer_pre_hook(torch.nn.Identity(), (hidden, None), {"k": 1})
        assert args[0] is not hidden and torch.equal(args[0], hidden)
        assert args[1] is None and kwargs == {"k": 1}
        args, kwargs = injection.decoder_layer_pre_hook(torch.nn.Identity(), (), {"hidden_states": hidden})
        assert args == () and kwargs["hidden_states"] is not hidden


class TestDenseGemma4Integration:
    def test_dict_config_attaches_module_and_hook(self):
        model = _build_dense_model(
            dict(ngram_size=3, heads_per_ngram=2, head_dim=8, rows_per_head=101, layer_index=1, eos_token_ids=[EOS])
        )
        ngram = model.ngram
        assert isinstance(ngram, Gemma4NGramInjection)
        assert model.model.language_model.ngram is ngram
        assert model._nemo_optional_base_checkpoint_key_prefixes == ("model.language_model.ngram.",)
        keys = [key for key in model.state_dict() if key.startswith("model.language_model.ngram.")]
        assert "model.language_model.ngram.embedding.table.weight" in keys
        assert model.model.language_model.layers[1]._forward_pre_hooks_with_kwargs

    def test_zero_init_matches_model_without_ngram(self):
        with_ngram = _build_dense_model(_tiny_ngram_config())
        without = _build_dense_model(None)
        assert without.ngram is None
        without.load_state_dict({k: v for k, v in with_ngram.state_dict().items() if ".ngram." not in k})
        ids = torch.randint(3, 64, (2, 9))
        with torch.no_grad():
            torch.testing.assert_close(with_ngram(input_ids=ids).logits, without(input_ids=ids).logits)

    def test_trained_table_changes_logits_only_through_hook(self):
        model = _build_dense_model(_tiny_ngram_config())
        ids = torch.randint(3, 64, (2, 9))
        with torch.no_grad():
            before = model(input_ids=ids).logits
            model.ngram.value_proj.weight.normal_()
            after = model(input_ids=ids).logits
        assert not torch.allclose(before, after)
        model._ngram_hook_handle.remove()
        with torch.no_grad():
            torch.testing.assert_close(model(input_ids=ids).logits, before)

    def test_gradient_reaches_table_through_the_decoder(self):
        model = _build_dense_model(_tiny_ngram_config())
        model.train()
        with torch.no_grad():
            model.ngram.value_proj.weight.normal_()
        ids = torch.randint(3, 64, (1, 7))
        model(input_ids=ids).logits.float().pow(2).mean().backward()
        assert model.ngram.embedding.table.weight.grad.abs().sum() > 0

    def test_initialize_weights_restores_zero_delta(self):
        # The checkpointer materializes meta parameters with uninitialized storage
        # and then calls initialize_weights(); the base checkpoint never covers the
        # n-gram module, so that call must be what gives it its zero-delta start.
        model = _build_dense_model(_tiny_ngram_config())
        with torch.no_grad():
            model.ngram.value_proj.weight.normal_()
            model.ngram.conv1d.weight.normal_()
            model.ngram.embedding.table.weight.fill_(float("nan"))
        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
        assert model.ngram.value_proj.weight.dtype == torch.float32
        assert torch.equal(model.ngram.value_proj.weight, torch.zeros_like(model.ngram.value_proj.weight))
        assert torch.equal(model.ngram.conv1d.weight, torch.zeros_like(model.ngram.conv1d.weight))
        assert torch.isfinite(model.ngram.embedding.table.weight).all()
        assert model.ngram.embedding.table.weight.std() > 0

    def test_dropped_optional_keys_reinitialize_the_ngram_module(self):
        # The loader skips initialize_weights() for some models (DTensor embeddings
        # with padding_idx), so the n-gram parameters can still be the zeros left by
        # to_empty(). With table AND value_proj at zero the branch is a fixed point:
        # zero output and zero gradient forever. The loader therefore hands the
        # dropped optional keys to the model, which must re-draw the module.
        model = _build_dense_model(_tiny_ngram_config())
        with torch.no_grad():
            for p in model.ngram.parameters():
                p.zero_()
        model.reset_optional_base_checkpoint_parameters(["model.language_model.ngram.embedding.table.weight"])
        assert model.ngram.embedding.table.weight.std() > 0
        assert model.ngram.key_proj.weight.std() > 0
        assert torch.equal(model.ngram.value_proj.weight, torch.zeros_like(model.ngram.value_proj.weight))
        assert torch.equal(model.ngram.conv1d.weight, torch.zeros_like(model.ngram.conv1d.weight))

    def test_unrelated_optional_keys_leave_the_ngram_module_alone(self):
        model = _build_dense_model(_tiny_ngram_config())
        with torch.no_grad():
            model.ngram.embedding.table.weight.zero_()
        model.reset_optional_base_checkpoint_parameters(["model.language_model.other.weight"])
        assert torch.equal(model.ngram.embedding.table.weight, torch.zeros_like(model.ngram.embedding.table.weight))
        # A model without the module must accept the call as a no-op.
        _build_dense_model(None).reset_optional_base_checkpoint_parameters(["model.language_model.ngram.x"])

    def test_inputs_embeds_only_is_rejected(self):
        model = _build_dense_model(_tiny_ngram_config())
        embeds = torch.randn(1, 4, model.config.text_config.hidden_size)
        with pytest.raises(ValueError, match="input_ids"):
            model(inputs_embeds=embeds)

    def test_moe_variant_is_rejected(self):
        text_config = _dense_text_config(
            enable_moe_block=True, num_experts=4, top_k_experts=2, moe_intermediate_size=16, torch_dtype="bfloat16"
        )
        config = Gemma4Config(text_config=text_config, vision_config=None, audio_config=None)
        with pytest.raises(NotImplementedError, match="dense"):
            Gemma4ForConditionalGeneration(config, backend=_cpu_backend(), ngram_config=_tiny_ngram_config())
