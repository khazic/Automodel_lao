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

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_automodel.components.speculative.bench_multimodal import (
    MultimodalBenchmark,
    adapt_multimodal_row,
    load_multimodal_prompts,
)


@pytest.mark.parametrize(
    ("benchmark", "question_column"),
    [
        (MultimodalBenchmark.GQA, "question"),
        (MultimodalBenchmark.TEXTVQA, "question"),
        (MultimodalBenchmark.COCO_CAPTION, "question"),
        (MultimodalBenchmark.CHARXIV_REASONING, "reasoning_q"),
    ],
)
def test_single_image_adapters_build_openai_vision_messages(benchmark, question_column):
    row = {question_column: "What is shown?", "image": b"jpeg"}
    prompt = adapt_multimodal_row(row, benchmark)
    assert prompt is not None
    assert prompt[0]["role"] == "user"
    assert prompt[0]["content"][0]["type"] == "image_url"
    assert prompt[0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert prompt[0]["content"][1] == {"type": "text", "text": "What is shown?"}


def test_mmmu_pro_interleaves_numbered_images_and_formats_options():
    row = {
        "question": "Compare <image 1> with <image 2>.",
        "options": "['first', 'second', 'third', 'fourth']",
        "image_1": b"one",
        "image_2": b"two",
        **{f"image_{index}": None for index in range(3, 8)},
    }
    prompt = adapt_multimodal_row(row, MultimodalBenchmark.MMMU_PRO)
    assert prompt is not None
    parts = prompt[0]["content"]
    assert [part["type"] for part in parts[:4]] == ["text", "image_url", "text", "image_url"]
    assert "A. first" in parts[-1]["text"]
    assert "Answer with the option letter only." in parts[-1]["text"]


def test_mmmu_pro_rejects_invalid_options():
    row = {"question": "q", "options": "not a list", "image_1": b"image"}
    assert adapt_multimodal_row(row, MultimodalBenchmark.MMMU_PRO) is None


def test_gqa_loader_joins_instruction_and_image_configs():
    calls = []

    def load_rows(input_data, *, split, name, shuffle_seed):
        calls.append((input_data, split, name, shuffle_seed))
        if name.endswith("_images"):
            return [{"id": "image-a", "image": b"jpeg"}]
        return [{"imageId": "image-a", "question": "Is it overcast?"}]

    args = SimpleNamespace(
        benchmark_adapter="gqa",
        input_data="lmms-lab/GQA",
        split="testdev",
        dataset_name="testdev_balanced_instructions",
        shuffle_seed=7,
        num_prompts=4,
    )
    prompts = load_multimodal_prompts(args, load_rows)
    assert len(prompts) == 1
    assert calls == [
        ("lmms-lab/GQA", "testdev", "testdev_balanced_instructions", 7),
        ("lmms-lab/GQA", "testdev", "testdev_balanced_images", None),
    ]


def test_gqa_loader_requires_instruction_config():
    args = SimpleNamespace(
        benchmark_adapter="gqa",
        input_data="lmms-lab/GQA",
        split="testdev",
        dataset_name="testdev_balanced_images",
        shuffle_seed=None,
        num_prompts=1,
    )
    with pytest.raises(ValueError, match="instructions"):
        load_multimodal_prompts(args, lambda *args, **kwargs: [])
