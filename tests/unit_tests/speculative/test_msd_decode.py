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

"""Unit tests for MSD recursive drafting and posterior verification."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import DynamicCache

from nemo_automodel.components.speculative.eagle.msd_decode import (
    MSDGreedyDecoder,
    MSDTreeDraftGenerator,
    MSDTreeNode,
    MSDTreeProposal,
    accept_or_resample,
    build_msd_tree_layout,
    verify_greedy_tree,
    verify_hf_greedy_tree,
    verify_stochastic_chain,
)
from nemo_automodel.components.speculative.eagle.vispec_decode import VispecCachedGreedyDecoder
from nemo_automodel.components.speculative.eagle.vispec_target import (
    HFVispecTargetModel,
    VispecGenerationState,
    VispecTreeTargetOutput,
)


def _proposal() -> MSDTreeProposal:
    nodes = (
        MSDTreeNode(index=1, parent_index=0, token_id=2, depth=1, log_probability=-0.1),
        MSDTreeNode(index=2, parent_index=1, token_id=3, depth=2, log_probability=-0.2),
        MSDTreeNode(index=3, parent_index=0, token_id=4, depth=1, log_probability=-0.3),
    )
    return MSDTreeProposal(
        root_token_id=1,
        nodes=nodes,
        leaf_indices=(2, 3),
        layout=build_msd_tree_layout(nodes, (2, 3), device=torch.device("cpu")),
    )


def _logits(*token_ids: int, vocab_size: int = 6) -> torch.Tensor:
    logits = torch.full((len(token_ids), vocab_size), -20.0)
    for row, token_id in enumerate(token_ids):
        logits[row, token_id] = 20.0
    return logits


def test_msd_tree_layout_preserves_only_ancestor_attention() -> None:
    """Tree metadata exposes root-to-node attention and retrieval paths."""
    proposal = _proposal()

    assert proposal.candidate_paths() == ((1, 2, 3), (1, 4))
    assert torch.equal(proposal.layout.position_ids, torch.tensor([0, 1, 2, 1]))
    assert torch.equal(proposal.layout.attention_mask[2], torch.tensor([True, True, True, False]))
    assert torch.equal(proposal.layout.attention_mask[3], torch.tensor([True, False, False, True]))
    assert torch.equal(proposal.layout.retrieve_indices, torch.tensor([[0, 1, 2], [0, 3, -1]]))


def test_msd_greedy_verifier_selects_longest_accepted_leaf() -> None:
    """Target verification chooses the tree path with the longest prefix hit."""
    result = verify_greedy_tree(
        _proposal(),
        (
            _logits(1, 2, 0, 5),
            _logits(1, 0, 4),
        ),
    )

    assert result.accepted_token_ids == (1, 2)
    assert result.bonus_token_id == 0
    assert result.accepted_draft_tokens == 1
    assert result.leaf_index == 2


def test_msd_greedy_verifier_rejects_invalid_target_shapes() -> None:
    """A missing bonus distribution fails before posterior selection."""
    with pytest.raises(ValueError, match="candidate_tokens"):
        verify_greedy_tree(_proposal(), (_logits(1), _logits(1, 4, 0)))


def test_msd_lossless_acceptance_and_correction_distribution() -> None:
    """Rejected draft tokens are replaced from the normalized positive residual."""
    target_logits = torch.tensor([4.0, 0.0])
    draft_logits = torch.tensor([0.0, 4.0])

    accepted = accept_or_resample(
        target_logits=target_logits,
        draft_logits=draft_logits,
        candidate_token_id=1,
        random_value=0.0,
    )
    rejected = accept_or_resample(
        target_logits=target_logits,
        draft_logits=draft_logits,
        candidate_token_id=1,
        random_value=0.999,
    )

    assert accepted.accepted and accepted.token_id == 1
    assert not rejected.accepted and rejected.token_id == 0


def test_msd_acceptance_never_emits_a_zero_probability_token() -> None:
    """A candidate the target rules out is rejected even on a zero draw."""
    target_logits = torch.tensor([0.0, -1e30])
    draft_logits = torch.tensor([0.0, 0.0])

    step = accept_or_resample(
        target_logits=target_logits,
        draft_logits=draft_logits,
        candidate_token_id=1,
        random_value=0.0,
    )

    assert not step.accepted and step.token_id == 0

    # Both models underflowing to zero gives a 0/0 ratio, which must not accept.
    ruled_out = accept_or_resample(
        target_logits=torch.tensor([0.0, -1e30]),
        draft_logits=torch.tensor([0.0, -1e30]),
        candidate_token_id=1,
        random_value=0.0,
    )
    assert not ruled_out.accepted and ruled_out.token_id == 0

    # Rounding of the two softmax normalizers can leave the target dominated at
    # every index without the distributions being equal, so the positive
    # residual carries no mass and cannot be normalized into a distribution.
    target_logits = torch.tensor([41.3499, -19.4971, -34.1979, 2.7798])
    draft_logits = torch.tensor([44.5302, 24.0608, -11.3887, 18.6314])
    assert (torch.softmax(target_logits, -1) - torch.softmax(draft_logits, -1)).clamp_min(0).sum() == 0
    drained = accept_or_resample(
        target_logits=target_logits,
        draft_logits=draft_logits,
        candidate_token_id=1,
        random_value=0.999999,
    )
    assert not drained.accepted and drained.token_id == 0

    # A draft-only zero divides to +inf and must still accept.
    draft_only_zero = accept_or_resample(
        target_logits=torch.tensor([0.0, 0.0]),
        draft_logits=torch.tensor([0.0, -1e30]),
        candidate_token_id=1,
        random_value=0.999,
    )
    assert draft_only_zero.accepted and draft_only_zero.token_id == 1

    with pytest.raises(ValueError, match=r"random_value must be in \[0, 1\)"):
        accept_or_resample(
            target_logits=target_logits,
            draft_logits=draft_logits,
            candidate_token_id=1,
            random_value=1.0,
        )


def test_msd_acceptance_rejects_a_generator_off_the_sampling_device() -> None:
    """A device-mismatched generator fails up front instead of mid-verification.

    ``torch.rand`` defaults to CPU while the residual resample follows the
    logits, so a single generator cannot otherwise serve both calls on GPU.
    """
    generator = SimpleNamespace(device=torch.device("cuda"))

    with pytest.raises(ValueError, match="generator on the sampling device"):
        accept_or_resample(
            target_logits=torch.tensor([1.0, 0.0]),
            draft_logits=torch.tensor([0.0, 1.0]),
            candidate_token_id=1,
            generator=generator,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="generator on the sampling device"):
        verify_stochastic_chain(
            candidate_token_ids=(1,),
            target_logits=_logits(1, 0),
            draft_logits=_logits(1, 1),
            generator=generator,  # type: ignore[arg-type]
        )


def test_msd_stochastic_verifier_stops_at_first_rejection_or_samples_bonus() -> None:
    """Linear stochastic verification preserves accepted prefixes and emits one target token."""
    target_logits = _logits(1, 0, 2)
    draft_logits = _logits(1, 1, 0)
    generator = torch.Generator().manual_seed(0)
    result = verify_stochastic_chain(
        candidate_token_ids=(1, 1),
        target_logits=target_logits,
        draft_logits=draft_logits,
        generator=generator,
    )

    assert result.accepted_draft_tokens == 1
    assert result.emitted_token_ids == (1, 0)


class _IdentityFeatureDraft(nn.Module):
    """Draft whose hidden prediction is its final input embedding."""

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return input embeddings while keeping every argument in the graph."""
        return inputs_embeds + 0.0 * (
            target_hidden_states + attention_mask.unsqueeze(-1) + image_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        )


def test_msd_tree_generator_recursively_expands_and_prunes_feature_paths() -> None:
    """Feature drafting recursively grows a bounded top-k candidate tree."""
    embeddings = nn.Embedding.from_pretrained(torch.eye(5), freeze=True)
    lm_head = nn.Linear(5, 5, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(5))
    generator = MSDTreeDraftGenerator(_IdentityFeatureDraft(), lm_head, embeddings)
    proposal = generator.propose(
        shifted_inputs_embeds=torch.zeros(1, 3, 5),
        input_hidden_states=torch.zeros(1, 3, 5),
        attention_mask=torch.ones(1, 3),
        shifted_image_mask=torch.zeros(1, 3, dtype=torch.bool),
        root_token_id=2,
        draft_steps=2,
        top_k=2,
        beam_width=2,
    )

    assert len(proposal.nodes) == 4
    assert all(path[0] == 2 for path in proposal.candidate_paths())

    # Both depth-2 nodes descend from node 1, so beam pruning abandoned node 2
    # mid-tree. It is still a candidate continuation and must be verifiable.
    parent_indices = {node.parent_index for node in proposal.nodes}
    assert parent_indices == {0, 1}
    assert proposal.leaf_indices == (2, 3, 4)
    assert sorted(len(path) for path in proposal.candidate_paths()) == [2, 3, 3]
    # Shorter paths are right-padded with -1 so retrieval stays rectangular.
    assert proposal.layout.retrieve_indices[0].tolist() == [0, 2, -1]


def test_msd_tree_generator_keeps_indices_dense_after_beam_pruning() -> None:
    """Pruned candidates cannot leave holes in tree-attention node indices."""
    embeddings = nn.Embedding.from_pretrained(torch.eye(5), freeze=True)
    lm_head = nn.Linear(5, 5, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(5))
    proposal = MSDTreeDraftGenerator(_IdentityFeatureDraft(), lm_head, embeddings).propose(
        shifted_inputs_embeds=torch.zeros(1, 3, 5),
        input_hidden_states=torch.zeros(1, 3, 5),
        attention_mask=torch.ones(1, 3),
        shifted_image_mask=torch.zeros(1, 3, dtype=torch.bool),
        root_token_id=2,
        draft_steps=2,
        top_k=2,
        beam_width=1,
    )

    assert [node.index for node in proposal.nodes] == [1, 2]
    assert proposal.layout.attention_mask.shape == (3, 3)


def test_msd_tree_generator_rejects_left_padded_prompts() -> None:
    """Left padding would silently make the prefix slice select pad positions."""
    embeddings = nn.Embedding.from_pretrained(torch.eye(5), freeze=True)
    lm_head = nn.Linear(5, 5, bias=False)
    generator = MSDTreeDraftGenerator(_IdentityFeatureDraft(), lm_head, embeddings)

    with pytest.raises(ValueError, match="right-padded attention mask"):
        generator.propose(
            shifted_inputs_embeds=torch.zeros(1, 3, 5),
            input_hidden_states=torch.zeros(1, 3, 5),
            attention_mask=torch.tensor([[0.0, 1.0, 1.0]]),
            shifted_image_mask=torch.zeros(1, 3, dtype=torch.bool),
            root_token_id=2,
            draft_steps=1,
            top_k=1,
            beam_width=1,
        )


class _TinyVerifierModel(nn.Module):
    """Target that predicts a deterministic successor for every input token."""

    def __init__(self) -> None:
        super().__init__()
        self.successor = torch.tensor([1, 2, 3, 0, 0, 0])

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool,
        use_cache: bool,
        **kwargs,
    ):
        """Return logits whose greedy token is the current token's successor."""
        del attention_mask, return_dict, use_cache, kwargs
        logits = torch.full((*input_ids.shape, 6), -20.0)
        next_ids = self.successor.to(input_ids.device)[input_ids]
        logits.scatter_(-1, next_ids.unsqueeze(-1), 20.0)
        return SimpleNamespace(logits=logits)


def test_hf_msd_verifier_replays_multimodal_prompt_paths() -> None:
    """The reference verifier extends text fields while preserving prompt media tensors."""
    result = verify_hf_greedy_tree(
        model=_TinyVerifierModel(),
        model_inputs={
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones(1, 1),
            "mm_token_type_ids": torch.ones(1, 1, dtype=torch.long),
            "pixel_values": torch.ones(1, 3, 2, 2),
        },
        proposal=_proposal(),
    )

    assert result.accepted_token_ids == (1, 2, 3)
    assert result.bonus_token_id == 0
    assert result.accepted_draft_tokens == 2


class _TinyDecodeModel(_TinyVerifierModel):
    """Verifier model with the token embedding and LM head required by the decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(torch.eye(6), freeze=True)
        self.lm_head = nn.Linear(6, 6, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.eye(6))

    def get_input_embeddings(self) -> nn.Module:
        """Expose target token embeddings through the Hugging Face convention."""
        return self.embeddings


class _TinyDecodeTarget:
    """Target wrapper that returns deterministic feature supervision for a prompt."""

    def __init__(self) -> None:
        self.model = _TinyDecodeModel()

    def get_lm_head(self) -> nn.Module:
        """Return the frozen language head."""
        return self.model.lm_head

    def generate_batch(self, **kwargs):
        """Return a minimal MSD target batch aligned to input ids."""
        input_ids = kwargs["model_inputs"]["input_ids"]
        attention_mask = kwargs["model_inputs"]["attention_mask"]
        embeddings = self.model.embeddings(input_ids)
        logits = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False
        ).logits
        shifted_logits = torch.cat((logits[:, 1:], torch.zeros_like(logits[:, :1])), dim=1)
        shifted_embeds = torch.cat((embeddings[:, 1:], torch.zeros_like(embeddings[:, :1])), dim=1)
        return SimpleNamespace(
            inputs_embeds=shifted_embeds,
            input_hidden_states=embeddings,
            attention_mask=attention_mask,
            image_mask=torch.zeros_like(input_ids, dtype=torch.bool),
            target_logits=shifted_logits,
        )


def test_msd_greedy_decoder_runs_recursive_draft_and_tree_verification() -> None:
    """The public decoder composes target prefill, drafting, and verification."""
    target = _TinyDecodeTarget()
    decoder = MSDGreedyDecoder(target, _IdentityFeatureDraft())

    proposal, result = decoder.decode_round(
        model_inputs={"input_ids": torch.tensor([[0, 1]]), "attention_mask": torch.ones(1, 2)},
        draft_steps=1,
        top_k=1,
        beam_width=1,
    )

    assert proposal.root_token_id == 2
    assert proposal.candidate_paths() == ((2, 2),)
    assert result.emitted_token_ids == (2, 3)


class _TinyCachedTarget:
    """Deterministic target implementing the cached ViSpec target contract."""

    def __init__(self) -> None:
        self.model = _TinyDecodeModel()

    def get_lm_head(self) -> nn.Module:
        """Return the identity language head."""
        return self.model.lm_head

    def get_input_embeddings(self) -> nn.Module:
        """Return one-hot token embeddings."""
        return self.model.embeddings

    def prefill_generation(self, model_inputs: dict[str, torch.Tensor]) -> VispecGenerationState:
        """Create a deterministic cached state for a batch-one token prefix.

        Args:
            model_inputs: Mapping containing ``input_ids`` and ``attention_mask``
                tensors of shape [1, sequence].

        Returns:
            Fake state with tensor layouts documented by
            :class:`VispecGenerationState`.
        """
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        embeddings = self.model.embeddings(input_ids)
        next_token = int(self.model.successor[input_ids[0, -1]].item())
        return VispecGenerationState(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=embeddings,
            input_hidden_states=embeddings,
            image_mask=torch.zeros_like(input_ids, dtype=torch.bool),
            next_token_logits=_logits(next_token),
            past_key_values=DynamicCache(),
        )

    def forward_tree_generation(
        self,
        state: VispecGenerationState,
        *,
        token_ids: torch.Tensor,
        tree_attention_mask: torch.Tensor,
        tree_position_ids: torch.Tensor,
    ) -> VispecTreeTargetOutput:
        """Return deterministic logits for one flattened fake target tree.

        Args:
            state: Cached fake target state with tensor layouts documented by
                :class:`VispecGenerationState`.
            token_ids: Tensor of shape [1, tree] in tree-node order.
            tree_attention_mask: Bool tensor of shape [tree, tree].
            tree_position_ids: Tensor of shape [tree] containing node depths.

        Returns:
            Fake target output with layouts documented by
            :class:`VispecTreeTargetOutput`.
        """
        del tree_attention_mask, tree_position_ids
        embeddings = self.model.embeddings(token_ids)
        logits = torch.cat([_logits(int(self.model.successor[token_id].item())) for token_id in token_ids[0]], dim=0)
        return VispecTreeTargetOutput(
            token_ids=token_ids,
            inputs_embeds=embeddings,
            hidden_states=embeddings,
            logits=logits.unsqueeze(0),
            prefix_length=state.input_ids.shape[1],
        )

    def commit_tree_generation(
        self,
        state: VispecGenerationState,
        tree_output: VispecTreeTargetOutput,
        *,
        accepted_tree_indices: torch.Tensor,
    ) -> VispecGenerationState:
        """Append the accepted fake tree path without appending the bonus token.

        Args:
            state: Cached fake target state with tensor layouts documented by
                :class:`VispecGenerationState`.
            tree_output: Fake target output with layouts documented by
                :class:`VispecTreeTargetOutput`.
            accepted_tree_indices: Tensor of shape [accepted] containing the
                root-to-leaf accepted tree-node indices.

        Returns:
            Fake target state extended by the accepted path.
        """
        tokens = tree_output.token_ids.index_select(1, accepted_tree_indices)
        embeddings = tree_output.inputs_embeds.index_select(1, accepted_tree_indices)
        attention_mask = torch.cat((state.attention_mask, torch.ones_like(tokens)), dim=1)
        return VispecGenerationState(
            input_ids=torch.cat((state.input_ids, tokens), dim=1),
            attention_mask=attention_mask,
            inputs_embeds=torch.cat((state.inputs_embeds, embeddings), dim=1),
            input_hidden_states=torch.cat((state.input_hidden_states, embeddings), dim=1),
            image_mask=torch.cat((state.image_mask, torch.zeros_like(tokens, dtype=torch.bool)), dim=1),
            next_token_logits=tree_output.logits[:, accepted_tree_indices[-1]],
            past_key_values=state.past_key_values,
        )


class _TinyCachedDraft(_IdentityFeatureDraft):
    """Identity draft exposing the minimal cached ViSpec inference contract."""

    def _prefill_generation(
        self,
        inputs_embeds: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...], torch.Tensor]:
        """Return the last embedding and a shape-compatible fake KV cache.

        Args:
            inputs_embeds: Tensor of shape [1, sequence, hidden].
            target_hidden_states: Tensor of shape [1, sequence, hidden].
            attention_mask: Tensor of shape [1, sequence].
            image_mask: Bool tensor of shape [1, sequence].

        Returns:
            Last hidden state of shape [1, hidden], one fake layer whose K/V
            tensors have shape [1, 1, sequence, 1], and a global image feature
            of shape [1, hidden].
        """
        del target_hidden_states, attention_mask, image_mask
        cache_tensor = torch.zeros(1, 1, inputs_embeds.shape[1], 1)
        return inputs_embeds[:, -1], ((cache_tensor, cache_tensor),), torch.zeros_like(inputs_embeds[0, :1])

    def _decode_generation(
        self,
        inputs_embeds: torch.Tensor,
        target_hidden_states: torch.Tensor,
        global_image_feature: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Return token embeddings and append shape-only fake cache entries.

        Args:
            inputs_embeds: Tensor of shape [1, query, hidden].
            target_hidden_states: Tensor of shape [1, query, hidden].
            global_image_feature: Tensor of shape [1, hidden].
            position_ids: Long tensor of shape [1, query].
            attention_mask: Additive tensor of shape [1, 1, query, cached + query].
            past_key_values: One fake layer whose K/V tensors have shape
                [1, 1, cached, 1].

        Returns:
            Hidden states of shape [1, query, hidden] and one fake layer whose
            K/V tensors have shape [1, 1, cached + query, 1].
        """
        del target_hidden_states, global_image_feature, position_ids, attention_mask
        cache_tensor = torch.zeros(1, 1, inputs_embeds.shape[1], 1)
        next_cache = tuple(
            ((torch.cat((key, cache_tensor), dim=-2), torch.cat((value, cache_tensor), dim=-2)))
            for key, value in past_key_values
        )
        return inputs_embeds, next_cache


def test_cached_vispec_decoder_commits_accepted_path_and_defers_bonus() -> None:
    """Cached verification commits accepted tokens while leaving its bonus pending."""
    decoder = VispecCachedGreedyDecoder(_TinyCachedTarget(), _TinyCachedDraft())
    decoder.prefill({"input_ids": torch.tensor([[0, 1]]), "attention_mask": torch.ones(1, 2)})

    proposal, result = decoder.decode_round(draft_steps=1, top_k=1, beam_width=1)

    assert proposal.candidate_paths() == ((2, 2),)
    assert result.emitted_token_ids == (2, 3)
    assert decoder.state is not None
    assert decoder.state.input_ids.tolist() == [[0, 1, 2]]


def test_vispec_tree_commit_compacts_selected_dynamic_cache_positions() -> None:
    """Tree commit reuses storage and retains selected K/V entries in path order."""
    cache = DynamicCache()
    prefix_keys = torch.tensor([[[[1.0], [2.0]]]])
    prefix_values = prefix_keys + 10.0
    cache.update(prefix_keys, prefix_values, 0)
    wrapper = object.__new__(HFVispecTargetModel)
    wrapper._preallocate_generation_cache(cache)
    storage_pointer = cache.layers[0].keys.untyped_storage().data_ptr()
    tree_keys = torch.tensor([[[[3.0], [4.0], [5.0]]]])
    tree_values = tree_keys + 10.0
    cache.update(tree_keys, tree_values, 0)
    assert cache.layers[0].keys.untyped_storage().data_ptr() == storage_pointer
    state = VispecGenerationState(
        input_ids=torch.tensor([[7, 8]]),
        attention_mask=torch.ones(1, 2),
        inputs_embeds=torch.zeros(1, 2, 2),
        input_hidden_states=torch.zeros(1, 2, 2),
        image_mask=torch.zeros(1, 2, dtype=torch.bool),
        next_token_logits=torch.zeros(1, 16),
        past_key_values=cache,
    )
    tree_output = VispecTreeTargetOutput(
        token_ids=torch.tensor([[9, 10, 11]]),
        inputs_embeds=torch.arange(6, dtype=torch.float32).view(1, 3, 2),
        hidden_states=torch.arange(6, 12, dtype=torch.float32).view(1, 3, 2),
        logits=torch.arange(48, dtype=torch.float32).view(1, 3, 16),
        prefix_length=2,
    )
    committed = wrapper.commit_tree_generation(
        state,
        tree_output,
        accepted_tree_indices=torch.tensor([0, 2]),
    )

    assert committed.input_ids.tolist() == [[7, 8, 9, 11]]
    assert committed.past_key_values.get_seq_length() == 4
    assert committed.past_key_values.layers[0].keys.flatten().tolist() == [1.0, 2.0, 3.0, 5.0]
    assert committed.past_key_values.layers[0].values.flatten().tolist() == [11.0, 12.0, 13.0, 15.0]
    assert committed.past_key_values.layers[0].keys.untyped_storage().data_ptr() == storage_pointer
    assert torch.equal(committed.next_token_logits, tree_output.logits[:, 2])
