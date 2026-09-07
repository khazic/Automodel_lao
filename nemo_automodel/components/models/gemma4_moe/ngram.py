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

"""Post-hoc Engram-style N-gram Embedding for dense Gemma4 checkpoints.

The module hashes the current token together with its preceding tokens into a
large embedding table (one packed table with several hash heads per n-gram
order) and injects a gated delta into the residual stream in front of one
decoder layer. The construction follows the Qwen3.8-Flash-Next PLE layer, but
for a single residual stream instead of HyperConnection branches, and it is
meant to be *added to an already pretrained model*: the value projection and
the causal convolution start at zero, so the model is bit-identical to the base
checkpoint until the table has been trained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15
_INT64_MASK = (1 << 64) - 1


def _splitmix64(state: int) -> tuple[int, int]:
    """Advance the SplitMix64 generator once.

    Args:
        state: Current 64-bit generator state.

    Returns:
        ``(next_state, output)`` as unsigned 64-bit integers.
    """
    state = (state + _SPLITMIX64_GAMMA) & _INT64_MASK
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _INT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _INT64_MASK
    value ^= value >> 31
    return state, value


def _to_signed_int64(value: int) -> int:
    value &= _INT64_MASK
    return value - (1 << 64) if value >= (1 << 63) else value


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _descending_primes(upper_bound: int, count: int) -> tuple[int, ...]:
    """Return ``count`` distinct primes at or below ``upper_bound``, largest first."""
    primes: list[int] = []
    candidate = int(upper_bound)
    while len(primes) < count and candidate >= 2:
        if _is_prime(candidate):
            primes.append(candidate)
        candidate -= 1
    if len(primes) < count:
        raise ValueError(f"Cannot find {count} distinct primes at or below {upper_bound}")
    return tuple(primes)


def ngram_layer_multipliers(ngram_size: int, seed: int = 0) -> tuple[int, ...]:
    """Return one deterministic odd signed int64 hash multiplier per n-gram position.

    Args:
        ngram_size: Number of token positions that take part in the hash.
        seed: Seed of the SplitMix64 generator.

    Returns:
        ``ngram_size`` signed 64-bit odd integers.
    """
    if ngram_size < 2:
        raise ValueError(f"ngram_size must be at least 2, got {ngram_size}")
    state = int(seed)
    multipliers = []
    for _ in range(ngram_size):
        state, value = _splitmix64(state)
        multipliers.append(_to_signed_int64(value | 1))
    return tuple(multipliers)


@dataclass(frozen=True)
class Gemma4NGramConfig:
    """Shape of the n-gram table and where its delta enters the decoder.

    Args:
        ngram_size: Largest n-gram order including the current token (``3``
            hashes bigrams and trigrams).
        heads_per_ngram: Number of independent hash heads per n-gram order.
        head_dim: Width of every table row; the concatenated embedding has
            ``(ngram_size - 1) * heads_per_ngram * head_dim`` values.
        rows_per_head: Upper bound on rows per hash head. Each head uses a
            distinct prime modulus at or below this value, so the packed table
            has ``sum(head_sizes)`` rows.
        layer_index: Zero-based decoder layer whose input receives the delta.
        eos_token_ids: Tokens that end a segment; n-gram context never crosses
            them, so packed documents do not leak into each other.
        conv_kernel_size: Kernel width of the causal depthwise convolution.
        hash_seed: Seed of the deterministic multiplier generator.
        initializer_range: Standard deviation of the table and key projection.
    """

    ngram_size: int = 3
    heads_per_ngram: int = 4
    head_dim: int = 128
    rows_per_head: int = 1_000_000
    layer_index: int = 1
    eos_token_ids: tuple[int, ...] = (1,)
    conv_kernel_size: int = 4
    hash_seed: int = 0
    initializer_range: float = 0.02
    head_sizes: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be at least 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be positive, got {self.heads_per_ngram}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
        if self.rows_per_head < 2:
            raise ValueError(f"rows_per_head must be at least 2, got {self.rows_per_head}")
        if self.layer_index < 0:
            raise ValueError(f"layer_index must be non-negative, got {self.layer_index}")
        if self.conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be positive, got {self.conv_kernel_size}")
        if self.initializer_range < 0:
            raise ValueError(f"initializer_range must be non-negative, got {self.initializer_range}")
        eos_token_ids = (self.eos_token_ids,) if isinstance(self.eos_token_ids, int) else tuple(self.eos_token_ids)
        if not eos_token_ids:
            raise ValueError("eos_token_ids must contain at least one token id")
        object.__setattr__(self, "eos_token_ids", tuple(int(token) for token in eos_token_ids))
        object.__setattr__(self, "head_sizes", _descending_primes(self.rows_per_head, self.num_heads))

    @property
    def num_heads(self) -> int:
        """Total number of hash heads over all n-gram orders."""
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def embed_dim(self) -> int:
        """Width of the concatenated per-token n-gram embedding."""
        return self.num_heads * self.head_dim

    @property
    def num_rows(self) -> int:
        """Number of rows in the packed table."""
        return sum(self.head_sizes)

    @property
    def head_offsets(self) -> tuple[int, ...]:
        """Row offset of every hash head inside the packed table."""
        offsets = []
        total = 0
        for size in self.head_sizes:
            offsets.append(total)
            total += size
        return tuple(offsets)

    def build(
        self,
        *,
        hidden_size: int,
        num_hidden_layers: int,
        rms_norm_eps: float,
        dtype: torch.dtype | None,
    ) -> Gemma4NGramInjection:
        """Construct the injection module for one decoder.

        Args:
            hidden_size: Width of the decoder residual stream.
            num_hidden_layers: Number of decoder layers, used to validate ``layer_index``.
            rms_norm_eps: Variance epsilon of the gate and convolution norms.
            dtype: Parameter dtype, or ``None`` for the default dtype.

        Returns:
            A freshly initialized :class:`Gemma4NGramInjection`.
        """
        if self.layer_index >= num_hidden_layers:
            raise ValueError(f"layer_index={self.layer_index} is out of range for {num_hidden_layers} decoder layers")
        return Gemma4NGramInjection(self, hidden_size, rms_norm_eps=rms_norm_eps, dtype=dtype)


class Gemma4NGramEmbedding(nn.Module):
    """Hash raw token n-grams into one packed multi-head embedding table.

    Args:
        config: Table shape and hashing settings.
        dtype: Parameter dtype of the table, or ``None`` for the default dtype.
    """

    def __init__(self, config: Gemma4NGramConfig, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.config = config
        self.table = nn.Embedding(config.num_rows, config.head_dim, dtype=dtype)
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(ngram_layer_multipliers(config.ngram_size, config.hash_seed), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("head_sizes", torch.tensor(config.head_sizes, dtype=torch.long), persistent=False)
        self.register_buffer("head_offsets", torch.tensor(config.head_offsets, dtype=torch.long), persistent=False)
        self.register_buffer("eos_token_ids", torch.tensor(config.eos_token_ids, dtype=torch.long), persistent=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Draw the table from ``N(0, initializer_range^2)``."""
        nn.init.normal_(self.table.weight, mean=0.0, std=self.config.initializer_range)

    def _shift_right_within_segment(self, input_ids: torch.Tensor, shift: int) -> torch.Tensor:
        """Read the token ``shift`` positions back without crossing a segment start.

        A segment starts at the first position after an EOS token. Positions whose
        source lies before their segment start (or before the sequence start) read
        the first configured EOS id instead.

        Args:
            input_ids: Integer tensor of shape ``[batch, sequence]``.
            shift: Number of positions to look back; ``0`` returns ``input_ids``.

        Returns:
            Integer tensor of shape ``[batch, sequence]``.
        """
        if shift == 0:
            return input_ids
        batch_size, sequence_length = input_ids.shape
        positions = torch.arange(sequence_length, device=input_ids.device, dtype=torch.long)
        is_eos = torch.isin(input_ids, self.eos_token_ids)
        eos_positions = torch.where(is_eos, positions, positions.new_full((), -1))
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]],
            dim=1,
        )
        positions_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted_ids = input_ids.gather(dim=1, index=gather_positions)
        valid = (positions_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted_ids, self.eos_token_ids[0])

    def hash_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map every token to one packed table row per hash head.

        Hashing multiplies each n-gram position by a fixed odd int64 multiplier,
        XORs the products (relying on wrapping int64 arithmetic), and reduces the
        result modulo each head's prime size before adding the head offset.

        Args:
            input_ids: Integer tensor of shape ``[batch, sequence]``.

        Returns:
            Row ids of shape ``[batch, sequence, num_heads]``, ordered by n-gram
            order (bigram heads first) and then head index; every id lies in
            ``[0, num_rows)``.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}")
        if input_ids.is_floating_point() or input_ids.dtype == torch.bool:
            raise ValueError(f"input_ids must have an integer dtype, got {input_ids.dtype}")
        input_ids = input_ids.to(dtype=torch.long)
        ngram_size = self.config.ngram_size
        heads_per_ngram = self.config.heads_per_ngram
        shifted = [self._shift_right_within_segment(input_ids, shift) for shift in range(ngram_size)]
        blocks = []
        for order in range(2, ngram_size + 1):
            head_start = (order - 2) * heads_per_ngram
            head_end = head_start + heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, order):
                mixed = torch.bitwise_xor(mixed, shifted[position] * self.layer_multipliers[position])
            sizes = self.head_sizes[head_start:head_end]
            offsets = self.head_offsets[head_start:head_end]
            blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes) + offsets)
        return torch.cat(blocks, dim=-1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up the concatenated head values of every token.

        Args:
            input_ids: Integer tensor of shape ``[batch, sequence]``.

        Returns:
            Tensor of shape ``[batch, sequence, embed_dim]`` in the table dtype,
            where the last axis concatenates the heads in hash order.
        """
        rows = self.hash_input_ids(input_ids)
        return self.table(rows).flatten(start_dim=-2)


class Gemma4NGramInjection(nn.Module):
    """Gate looked-up n-gram values against the residual stream into a layer delta.

    Args:
        config: Table shape, injection layer, and initialization settings.
        hidden_size: Width of the decoder residual stream.
        rms_norm_eps: Variance epsilon of the gate and convolution norms.
        dtype: Parameter dtype, or ``None`` for the default dtype.
    """

    def __init__(
        self,
        config: Gemma4NGramConfig,
        hidden_size: int,
        *,
        rms_norm_eps: float = 1e-6,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.config = config
        self.hidden_size = hidden_size
        self.embedding = Gemma4NGramEmbedding(config, dtype=dtype)
        self.key_proj = nn.Linear(config.embed_dim, hidden_size, bias=False, dtype=dtype)
        self.value_proj = nn.Linear(config.embed_dim, hidden_size, bias=False, dtype=dtype)
        self.norm_key = nn.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.norm_query = nn.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.norm_conv = nn.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.conv1d = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=hidden_size,
            bias=False,
            dtype=dtype,
        )
        self.conv_dilation = config.ngram_size
        # Raw token ids of the forward in flight, stashed by the owning model so
        # the decoder-layer pre-hook can hash them, including during activation
        # checkpoint replay in backward.
        self._current_input_ids: torch.Tensor | None = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize so the delta is exactly zero on top of the pretrained model."""
        self.embedding.reset_parameters()
        nn.init.normal_(self.key_proj.weight, mean=0.0, std=self.config.initializer_range)
        nn.init.zeros_(self.value_proj.weight)
        self.norm_key.reset_parameters()
        self.norm_query.reset_parameters()
        self.norm_conv.reset_parameters()
        nn.init.zeros_(self.conv1d.weight)

    def _causal_short_conv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the dilated depthwise convolution with zero left history.

        Args:
            hidden_states: Tensor of shape ``[batch, sequence, hidden]``.

        Returns:
            Tensor of shape ``[batch, sequence, hidden]`` where every position
            only depends on itself and earlier positions.
        """
        left_padding = (self.config.conv_kernel_size - 1) * self.conv_dilation
        channels_first = F.pad(hidden_states.transpose(1, 2), (left_padding, 0))
        convolved = F.conv1d(
            channels_first,
            self.conv1d.weight.to(dtype=channels_first.dtype),
            groups=self.hidden_size,
            dilation=self.conv_dilation,
        )
        return convolved.transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute the delta to add to the residual stream before the injection layer.

        Args:
            hidden_states: Residual stream of shape ``[batch, sequence, hidden]``.
            input_ids: Integer tensor of shape ``[batch, sequence]`` with the raw
                tokenizer ids of the same positions.

        Returns:
            Delta of shape ``[batch, sequence, hidden]`` in the dtype of
            ``hidden_states``; it does not alias either input.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have shape [batch, sequence, {self.hidden_size}], got {tuple(hidden_states.shape)}"
            )
        if input_ids.shape != hidden_states.shape[:2]:
            raise ValueError(
                "input_ids and hidden_states must share [batch, sequence] axes, got "
                f"{tuple(input_ids.shape)} and {tuple(hidden_states.shape)}"
            )
        embeddings = self.embedding(input_ids).to(dtype=hidden_states.dtype)
        key = self.key_proj(embeddings)
        value = self.value_proj(embeddings)
        gate = (self.norm_key(key) * self.norm_query(hidden_states)).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = torch.sigmoid(gate)
        gated_value = gate * value
        return gated_value + self._causal_short_conv(self.norm_conv(gated_value))

    def stash_input_ids(self, input_ids: torch.Tensor | None) -> None:
        """Record the raw ids of the forward in flight for the decoder-layer hook.

        Args:
            input_ids: Integer tensor of shape ``[batch, sequence]``, or ``None``
                to clear the stash. The tensor is detached and kept until the next
                call, so activation-checkpoint replay reads the same ids.
        """
        self._current_input_ids = None if input_ids is None else input_ids.detach()

    def decoder_layer_pre_hook(self, _layer: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Add the n-gram delta to the hidden states entering the injection layer.

        Registered with ``register_forward_pre_hook(..., with_kwargs=True)`` on the
        decoder layer selected by ``config.layer_index``.

        Args:
            _layer: The hooked decoder layer (unused).
            args: Positional layer inputs; ``args[0]`` is the residual stream of
                shape ``[batch, sequence, hidden]`` when passed positionally.
            kwargs: Keyword layer inputs; ``kwargs["hidden_states"]`` holds the
                residual stream when it is passed by keyword instead.

        Returns:
            ``(args, kwargs)`` with the residual stream replaced by a new tensor
            ``hidden_states + delta`` of the same shape; nothing is mutated in place.
        """
        if self._current_input_ids is None:
            raise RuntimeError(
                "The Gemma4 n-gram injection needs raw input_ids: run the model forward with input_ids "
                "(inputs_embeds alone cannot be hashed)"
            )
        if args:
            hidden_states = args[0]
            return (hidden_states + self(hidden_states, self._current_input_ids), *args[1:]), kwargs
        hidden_states = kwargs["hidden_states"]
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = hidden_states + self(hidden_states, self._current_input_ids)
        return args, kwargs

    def extra_repr(self) -> str:
        cfg = self.config
        return (
            f"rows={cfg.num_rows}, heads={cfg.num_heads} ({cfg.ngram_size}-gram x {cfg.heads_per_ngram}), "
            f"head_dim={cfg.head_dim}, hidden={self.hidden_size}, layer_index={cfg.layer_index}"
        )


__all__ = (
    "Gemma4NGramConfig",
    "Gemma4NGramEmbedding",
    "Gemma4NGramInjection",
    "ngram_layer_multipliers",
)
