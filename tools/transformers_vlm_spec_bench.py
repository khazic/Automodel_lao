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

"""Batch-one Transformers evaluator for Qwen2.5-VL speculative drafts.

The evaluator deliberately uses the same prompt adapter and generation limits
for baseline, DFlash and ViSpec. The ViSpec path flattens each candidate tree
into one target forward, then compacts the accepted path in the target KV cache.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, PretrainedConfig
from transformers.cache_utils import DynamicCache

DATASETS = (
    ("gqa", 64),
    ("textvqa", 64),
    ("coco_caption", 128),
    ("charxiv_reasoning", 256),
    ("mmmu_pro", 256),
)
VISPEC_DEPTH = 3
VISPEC_TOP_K = 8
VISPEC_TOTAL_TOKEN = 30


def _load_prompt_module():
    path = (
        Path(__file__).parents[1]
        / "artifacts/dflash/full/runs/qwen2_5_vl_5layer_6epoch_a256/epoch_3_step_10000/benchmark/run_first_rows.py"
    )
    spec = importlib.util.spec_from_file_location("first_rows", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark prompt adapter at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_hf_messages(prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI Vision messages into Transformers chat-template messages."""
    messages: list[dict[str, Any]] = []
    for message in prompt:
        content = message.get("content", [])
        hf_content: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "text":
                hf_content.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                hf_content.append({"type": "image", "image": part["image_url"]["url"]})
        messages.append({"role": message["role"], "content": hf_content})
    return messages


def _prepare_inputs(processor, prompt: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    """Build one Qwen-VL processor batch with batch axis one."""
    messages = _to_hf_messages(prompt)
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}


def _load_target(path: str, device: torch.device, attn_implementation: str = "eager"):
    return (
        AutoModelForImageTextToText.from_pretrained(
            path,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation=attn_implementation,
        )
        .to(device)
        .eval()
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _acceptance_lengths(accepted_tokens: int | float, verify_steps: int | float) -> tuple[float | None, float | None]:
    """Return official draft acceptance and actual emitted tokens per round."""
    if verify_steps == 0:
        return None, None
    accept_length = accepted_tokens / verify_steps
    return accept_length, 1.0 + accept_length


def _greedy_cached_forward(
    target: nn.Module,
    model_inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> list[int]:
    """Decode greedily with one persistent target KV cache.

    Args:
        target: Transformers image-text model whose forward method accepts a
            ``DynamicCache`` and Qwen-style multimodal position IDs.
        model_inputs: Processor tensors containing ``input_ids`` and
            ``attention_mask`` of shape [1, prompt_sequence]. Vision tensors
            retain the processor-defined flattened patch layouts.
        max_new_tokens: Maximum number of tokens to generate after the prompt.
        eos_token_id: Token that terminates decoding after it is emitted, or
            ``None`` to decode exactly ``max_new_tokens`` tokens.

    Returns:
        Generated token IDs. The prompt is excluded.
    """
    if max_new_tokens < 1:
        return []
    input_ids = model_inputs.get("input_ids")
    prompt_attention_mask = model_inputs.get("attention_mask")
    if input_ids is None or prompt_attention_mask is None:
        raise ValueError("Cached target decoding requires input_ids and attention_mask.")
    if input_ids.shape[0] != 1 or prompt_attention_mask.shape != input_ids.shape:
        raise ValueError("Cached target decoding requires matching batch-one input ids and attention mask.")

    prompt_length = input_ids.shape[1]
    cache = DynamicCache()
    target_kwargs = {key: value for key, value in model_inputs.items() if key != "input_ids"}
    prefill_kwargs = target.prepare_inputs_for_generation(
        input_ids,
        next_sequence_length=prompt_length,
        past_key_values=cache,
        is_first_iteration=True,
        use_cache=True,
        **target_kwargs,
    )
    prefill_input_ids = prefill_kwargs.pop("input_ids")
    prefill_kwargs["logits_to_keep"] = 1
    prefill_kwargs["return_dict"] = True
    outputs = target(prefill_input_ids, **prefill_kwargs)
    cache = outputs.past_key_values
    if cache is None or cache.get_seq_length() != prompt_length:
        actual_length = None if cache is None else cache.get_seq_length()
        raise RuntimeError(f"Target prefill cache must cover {prompt_length} prompt tokens, got {actual_length}.")

    generated_ids = [int(outputs.logits[:, -1].argmax(dim=-1).item())]
    target_forward_params = inspect.signature(target.forward).parameters
    position_hook = getattr(getattr(target, "model", None), "compute_3d_position_ids", None)
    if not callable(position_hook):
        raise RuntimeError("Cached target decoding requires compute_3d_position_ids on the target base model.")
    position_hook_params = inspect.signature(position_hook).parameters

    while len(generated_ids) < max_new_tokens and (eos_token_id is None or generated_ids[-1] != eos_token_id):
        token_ids = torch.tensor([generated_ids[-1:]], dtype=input_ids.dtype, device=input_ids.device)
        active_length = prompt_length + len(generated_ids)
        full_attention_mask = torch.cat(
            (prompt_attention_mask, prompt_attention_mask.new_ones((1, len(generated_ids)))), dim=1
        )
        full_input_ids = torch.cat(
            (input_ids, torch.tensor([generated_ids], dtype=input_ids.dtype, device=input_ids.device)), dim=1
        )
        generation_kwargs = {
            key: value
            for key, value in target_kwargs.items()
            if key not in {"attention_mask", "mm_token_type_ids", "pixel_values", "pixel_values_videos"}
        }
        step_kwargs = target.prepare_inputs_for_generation(
            full_input_ids,
            next_sequence_length=1,
            past_key_values=cache,
            attention_mask=full_attention_mask,
            is_first_iteration=False,
            use_cache=True,
            **generation_kwargs,
        )
        step_input_ids = step_kwargs.pop("input_ids")
        step_kwargs["attention_mask"] = full_attention_mask
        position_kwargs: dict[str, object] = {
            "input_ids": token_ids,
            "image_grid_thw": None,
            "video_grid_thw": None,
            "inputs_embeds": target.get_input_embeddings()(token_ids),
            "attention_mask": full_attention_mask,
            "past_key_values": cache,
            "second_per_grid_ts": None,
            "mm_token_type_ids": None,
        }
        step_kwargs["position_ids"] = position_hook(
            **{key: value for key, value in position_kwargs.items() if key in position_hook_params}
        )[..., -1:]
        if "cache_position" in target_forward_params:
            step_kwargs["cache_position"] = torch.tensor([active_length - 1], dtype=torch.long, device=input_ids.device)
        step_kwargs["logits_to_keep"] = 1
        step_kwargs["return_dict"] = True
        outputs = target(step_input_ids, **step_kwargs)
        cache = outputs.past_key_values
        if cache is None or cache.get_seq_length() != active_length:
            actual_length = None if cache is None else cache.get_seq_length()
            raise RuntimeError(f"Target decode cache must cover {active_length} processed tokens, got {actual_length}.")
        generated_ids.append(int(outputs.logits[:, -1].argmax(dim=-1).item()))
    return generated_ids


@torch.inference_mode()
def _baseline(
    target,
    processor,
    prompts: list[list[dict[str, Any]]],
    max_new_tokens: int,
    device: torch.device,
    fixed_output_length: bool = False,
) -> dict[str, Any]:
    """Run one-token cached target forwards and report output-token throughput."""
    output_tokens = 0
    reference_outputs: list[list[int]] = []
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    if prompts:
        warmup_inputs = _prepare_inputs(processor, prompts[0], device)
        _greedy_cached_forward(
            target,
            warmup_inputs,
            max_new_tokens=min(8, max_new_tokens),
            eos_token_id=eos_token_id,
        )
    _sync(device)
    start = time.perf_counter()
    for prompt_index, prompt in enumerate(prompts):
        inputs = _prepare_inputs(processor, prompt, device)
        generated = _greedy_cached_forward(
            target,
            inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        output_tokens += len(generated)
        reference_outputs.append(generated)
    _sync(device)
    wall = time.perf_counter() - start
    return {
        "completed": len(prompts),
        "output_tokens": output_tokens,
        "wall_clock_s": wall,
        "tok_s": output_tokens / wall,
        "_reference_outputs": reference_outputs,
    }


def _load_dflash(path: str, target, device: torch.device):
    from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel

    config = AutoConfig.from_pretrained(path)
    draft = Qwen3DFlashDraftModel(config)
    from safetensors.torch import load_file

    checkpoint_dir = Path(path)
    weight_files = sorted(checkpoint_dir.glob("model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors weights found under {checkpoint_dir}")
    state_dict = {}
    for weight_file in weight_files:
        state_dict.update(load_file(str(weight_file)))
    draft.load_state_dict(state_dict, strict=True)
    return draft.to(device=device, dtype=next(target.parameters()).dtype).eval()


@torch.inference_mode()
def _dflash(
    target,
    processor,
    draft,
    prompts,
    max_new_tokens: int,
    device: torch.device,
    reference_outputs=None,
    fixed_output_length: bool = False,
    verification_mode: str = "block",
) -> dict[str, Any]:
    """Run DFlash's block verifier with prompt-only multimodal target inputs."""
    output_tokens = 0
    draft_tokens = 0.0
    accepted_tokens = 0.0
    verify_steps = 0.0
    exact_matches = 0
    matching_tokens = 0
    compared_tokens = 0
    common_prefix_tokens = 0
    target_prefill_seconds = 0.0
    draft_seconds = 0.0
    target_verify_seconds = 0.0
    if prompts:
        warmup_inputs = _prepare_inputs(processor, prompts[0], device)
        warmup_ids = warmup_inputs.pop("input_ids")
        draft.spec_generate(
            target,
            warmup_ids,
            min(8, max_new_tokens),
            stop_token_ids=None,
            temperature=0.0,
            target_kwargs=warmup_inputs,
            sequential_target_verification=verification_mode == "sequential",
        )
    _sync(device)
    start = time.perf_counter()
    for prompt_index, prompt in enumerate(prompts):
        inputs = _prepare_inputs(processor, prompt, device)
        prompt_ids = inputs.pop("input_ids")
        sample_max_new_tokens = (
            len(reference_outputs[prompt_index])
            if fixed_output_length and reference_outputs is not None
            else max_new_tokens
        )
        output, stats = draft.spec_generate(
            target,
            prompt_ids,
            sample_max_new_tokens,
            stop_token_ids=None
            if fixed_output_length
            else (
                [int(processor.tokenizer.eos_token_id)]
                if getattr(processor.tokenizer, "eos_token_id", None) is not None
                else None
            ),
            temperature=0.0,
            target_kwargs=inputs,
            return_stats=True,
            sequential_target_verification=verification_mode == "sequential",
        )
        output_tokens += int(output.shape[1] - prompt_ids.shape[1])
        if reference_outputs is not None:
            generated_ids = output[0, prompt_ids.shape[1] :].tolist()
            reference_ids = reference_outputs[prompt_index]
            exact_matches += int(generated_ids == reference_ids)
            compared_length = min(len(generated_ids), len(reference_ids))
            matching_tokens += sum(
                generated_ids[token_index] == reference_ids[token_index] for token_index in range(compared_length)
            )
            compared_tokens += max(len(generated_ids), len(reference_ids))
            for generated_id, reference_id in zip(generated_ids, reference_ids):
                if generated_id != reference_id:
                    break
                common_prefix_tokens += 1
        draft_tokens += stats["draft_tokens"]
        accepted_tokens += stats["accepted_tokens"]
        verify_steps += stats["verify_steps"]
        target_prefill_seconds += stats["target_prefill_seconds"]
        draft_seconds += stats["draft_seconds"]
        target_verify_seconds += stats["target_verify_seconds"]
    _sync(device)
    wall = time.perf_counter() - start
    accept_length, emitted_tokens_per_step = _acceptance_lengths(accepted_tokens, verify_steps)
    return {
        "completed": len(prompts),
        "output_tokens": output_tokens,
        "wall_clock_s": wall,
        "tok_s": output_tokens / wall,
        "accept_length": accept_length,
        "emitted_tokens_per_step": emitted_tokens_per_step,
        "acceptance_rate": accepted_tokens / draft_tokens if draft_tokens else None,
        "exact_match_count": exact_matches if reference_outputs is not None else None,
        "exact_match_rate": exact_matches / len(prompts) if reference_outputs is not None and prompts else None,
        "token_match_rate": matching_tokens / compared_tokens
        if reference_outputs is not None and compared_tokens
        else None,
        "mean_common_prefix_length": common_prefix_tokens / len(prompts)
        if reference_outputs is not None and prompts
        else None,
        "target_prefill_s": target_prefill_seconds,
        "draft_s": draft_seconds,
        "target_verify_s": target_verify_seconds,
        "unattributed_s": max(0.0, wall - target_prefill_seconds - draft_seconds - target_verify_seconds),
    }


def _load_vispec(path: str, target, device: torch.device):
    from nemo_automodel.components.speculative.eagle.vispec_draft import VispecDraftModel

    config = PretrainedConfig.from_dict(json.loads((Path(path) / "config.json").read_text()))
    if getattr(config, "architectures", []) != ["VispecDraftModel"]:
        raise ValueError(
            "ViSpec evaluator expects a NeMo consolidated checkpoint with architectures=['VispecDraftModel']; "
            "the official JLKang checkpoint uses the original ViSpec key layout and needs a conversion step."
        )
    draft = VispecDraftModel(config)
    from safetensors.torch import load_file

    checkpoint_dir = Path(path)
    weight_files = sorted(checkpoint_dir.glob("model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors weights found under {checkpoint_dir}")
    state_dict = {}
    for weight_file in weight_files:
        state_dict.update(load_file(str(weight_file)))
    draft.load_state_dict(state_dict, strict=True)
    return draft.to(device=device, dtype=next(target.parameters()).dtype).eval()


@torch.inference_mode()
def _vispec(
    target,
    processor,
    draft,
    prompts,
    max_new_tokens: int,
    device: torch.device,
    reference_outputs=None,
    fixed_output_length: bool = False,
) -> dict[str, Any]:
    """Run batch-one ViSpec/MSD rounds with cached target verification."""
    from nemo_automodel.components.speculative.eagle.vispec_decode import VispecCachedGreedyDecoder
    from nemo_automodel.components.speculative.eagle.vispec_target import HFVispecTargetModel

    image_token_id = int(getattr(target.config, "image_token_id"))
    vispec_target = HFVispecTargetModel(target, image_token_id=image_token_id)

    decoder = VispecCachedGreedyDecoder(vispec_target, draft)
    output_tokens = 0
    draft_tokens = 0
    accepted_tokens = 0
    verify_steps = 0
    exact_matches = 0
    matching_tokens = 0
    compared_tokens = 0
    common_prefix_tokens = 0
    if prompts:
        warmup_inputs = _prepare_inputs(processor, prompts[0], device)
        decoder.prefill(warmup_inputs)
        decoder.decode_round(
            draft_steps=VISPEC_DEPTH,
            top_k=VISPEC_TOP_K,
            beam_width=VISPEC_TOTAL_TOKEN - 1,
        )
    _sync(device)
    start = time.perf_counter()
    for prompt_index, prompt in enumerate(prompts):
        model_inputs = _prepare_inputs(processor, prompt, device)
        decoder.prefill(model_inputs)
        sample_max_new_tokens = (
            len(reference_outputs[prompt_index])
            if fixed_output_length and reference_outputs is not None
            else max_new_tokens
        )
        generated = 0
        generated_ids: list[int] = []
        while generated < sample_max_new_tokens:
            proposal, result = decoder.decode_round(
                draft_steps=VISPEC_DEPTH,
                top_k=VISPEC_TOP_K,
                beam_width=VISPEC_TOTAL_TOKEN - 1,
            )
            emitted = list(result.accepted_token_ids[: sample_max_new_tokens - generated])
            eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
            if not fixed_output_length and eos_token_id in emitted:
                emitted = emitted[: emitted.index(eos_token_id) + 1]
            if not emitted:
                break
            generated_ids.extend(emitted)
            generated += len(emitted)
            output_tokens += len(emitted)
            draft_tokens += len(proposal.nodes)
            accepted_tokens += min(result.accepted_draft_tokens, max(0, len(emitted) - 1))
            verify_steps += 1
            if not fixed_output_length and eos_token_id in emitted:
                break
        if reference_outputs is not None:
            reference_ids = reference_outputs[prompt_index]
            exact_matches += int(generated_ids == reference_ids)
            compared_length = min(len(generated_ids), len(reference_ids))
            matching_tokens += sum(
                generated_ids[token_index] == reference_ids[token_index] for token_index in range(compared_length)
            )
            compared_tokens += max(len(generated_ids), len(reference_ids))
            for generated_id, reference_id in zip(generated_ids, reference_ids):
                if generated_id != reference_id:
                    break
                common_prefix_tokens += 1
    _sync(device)
    wall = time.perf_counter() - start
    accept_length, emitted_tokens_per_step = _acceptance_lengths(accepted_tokens, verify_steps)
    return {
        "completed": len(prompts),
        "output_tokens": output_tokens,
        "wall_clock_s": wall,
        "tok_s": output_tokens / wall,
        "accept_length": accept_length,
        "emitted_tokens_per_step": emitted_tokens_per_step,
        "acceptance_rate": accepted_tokens / draft_tokens if draft_tokens else None,
        "exact_match_count": exact_matches if reference_outputs is not None else None,
        "exact_match_rate": exact_matches / len(prompts) if reference_outputs is not None and prompts else None,
        "token_match_rate": matching_tokens / compared_tokens
        if reference_outputs is not None and compared_tokens
        else None,
        "mean_common_prefix_length": common_prefix_tokens / len(prompts)
        if reference_outputs is not None and prompts
        else None,
    }


def main() -> None:
    """Run the requested batch-one speculative decoding benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--mode", choices=("baseline", "dflash", "vispec"), required=True)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--fixed-output-length", action="store_true")
    parser.add_argument("--block-size", type=int, choices=(4, 8, 16))
    parser.add_argument("--draft-layers", type=int, choices=(1, 3, 5))
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--verification-mode", choices=("block", "sequential"), default="block")
    parser.add_argument("--only", choices=[name for name, _ in DATASETS])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt_module = _load_prompt_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = _load_target(args.target, device, args.attn_implementation)
    processor = AutoProcessor.from_pretrained(args.target)
    results: dict[str, Any] = {}
    for name, max_new_tokens in DATASETS:
        if args.only is not None and name != args.only:
            continue
        prompts = prompt_module._load_prompts(
            name, *prompt_module.DATASETS[[item[0] for item in prompt_module.DATASETS].index(name)][1:4]
        )
        prompts = prompts[: args.num_prompts]
        reference_outputs = None
        baseline_result = None
        if args.mode != "baseline":
            baseline_result = _baseline(
                target,
                processor,
                prompts,
                max_new_tokens,
                device,
                fixed_output_length=args.fixed_output_length,
            )
            reference_outputs = baseline_result.pop("_reference_outputs")
        if args.mode == "baseline":
            result = _baseline(
                target,
                processor,
                prompts,
                max_new_tokens,
                device,
                fixed_output_length=args.fixed_output_length,
            )
        elif args.mode == "dflash":
            draft = _load_dflash(args.draft, target, device)
            if args.block_size is not None:
                draft.block_size = args.block_size
            if args.draft_layers is not None:
                draft.layers = draft.layers[: args.draft_layers]
            result = _dflash(
                target,
                processor,
                draft,
                prompts,
                max_new_tokens,
                device,
                reference_outputs,
                fixed_output_length=args.fixed_output_length,
                verification_mode=args.verification_mode,
            )
        else:
            result = _vispec(
                target,
                processor,
                _load_vispec(args.draft, target, device),
                prompts,
                max_new_tokens,
                device,
                reference_outputs,
                fixed_output_length=args.fixed_output_length,
            )
        if args.mode == "baseline":
            result.pop("_reference_outputs", None)
        elif baseline_result is not None:
            result["speedup_vs_target"] = (
                result["tok_s"] / baseline_result["tok_s"] if baseline_result["tok_s"] else None
            )
        results[name] = {
            "max_new_tokens": max_new_tokens,
            "fixed_output_length": args.fixed_output_length,
            "block_size": args.block_size if args.mode == "dflash" else None,
            "draft_layers": args.draft_layers if args.mode == "dflash" else None,
            "attn_implementation": args.attn_implementation,
            "verification_mode": args.verification_mode
            if args.mode == "dflash"
            else ("tree" if args.mode == "vispec" else None),
            "vispec_depth": VISPEC_DEPTH if args.mode == "vispec" else None,
            "vispec_top_k": VISPEC_TOP_K if args.mode == "vispec" else None,
            "vispec_total_token": VISPEC_TOTAL_TOKEN if args.mode == "vispec" else None,
            **result,
        }
        print(json.dumps({name: results[name]}, indent=2), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
