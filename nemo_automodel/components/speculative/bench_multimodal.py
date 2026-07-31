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

"""Multimodal dataset adapters for speculative-decoding HTTP benchmarks."""

from __future__ import annotations

import ast
import base64
import io
import re
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable


class MultimodalBenchmark(str, Enum):
    """Supported benchmark row schemas."""

    GQA = "gqa"
    TEXTVQA = "textvqa"
    COCO_CAPTION = "coco_caption"
    CHARXIV_REASONING = "charxiv_reasoning"
    MMMU_PRO = "mmmu_pro"


def _image_data_url(image: Any) -> str:
    """Convert an HF image value, local path, URL, or bytes into an image URL."""
    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:image/")):
            return image
        path = Path(image)
        if path.is_file():
            suffix = path.suffix.lower().lstrip(".") or "jpeg"
            mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
            return f"data:image/{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
        raise ValueError(f"Image string is neither a URL nor an existing file: {image!r}")

    if isinstance(image, dict):
        if image.get("bytes") is not None:
            image = image["bytes"]
        elif image.get("path"):
            return _image_data_url(image["path"])
        elif image.get("src"):
            return _image_data_url(image["src"])

    if isinstance(image, (bytes, bytearray)):
        return f"data:image/jpeg;base64,{base64.b64encode(image).decode()}"

    if hasattr(image, "save"):
        output = io.BytesIO()
        image_format = str(getattr(image, "format", None) or "JPEG").upper()
        if image_format not in {"JPEG", "PNG", "WEBP"}:
            image_format = "JPEG"
        if image_format == "JPEG" and getattr(image, "mode", "RGB") not in {"RGB", "L"}:
            image = image.convert("RGB")
        image.save(output, format=image_format)
        mime = "jpeg" if image_format == "JPEG" else image_format.lower()
        return f"data:image/{mime};base64,{base64.b64encode(output.getvalue()).decode()}"

    raise ValueError(f"Unsupported image value of type {type(image).__name__}")


def _vision_message(text: str, images: list[Any]) -> list[dict[str, Any]] | None:
    """Build one OpenAI Vision user message, preserving numbered image positions."""
    if not isinstance(text, str) or not text.strip() or not images:
        return None

    image_urls = [_image_data_url(image) for image in images]
    parts: list[dict[str, Any]] = []
    cursor = 0
    used: set[int] = set()
    for match in re.finditer(r"<image\s+(\d+)>", text, flags=re.IGNORECASE):
        if match.start() > cursor and text[cursor : match.start()].strip():
            parts.append({"type": "text", "text": text[cursor : match.start()]})
        image_index = int(match.group(1)) - 1
        if 0 <= image_index < len(image_urls):
            parts.append({"type": "image_url", "image_url": {"url": image_urls[image_index]}})
            used.add(image_index)
        cursor = match.end()

    if cursor == 0:
        parts.extend({"type": "image_url", "image_url": {"url": url}} for url in image_urls)
    else:
        parts.extend(
            {"type": "image_url", "image_url": {"url": url}}
            for index, url in enumerate(image_urls)
            if index not in used
        )
    if text[cursor:].strip():
        parts.append({"type": "text", "text": text[cursor:]})
    return [{"role": "user", "content": parts}]


def _parse_options(value: Any) -> list[str]:
    """Normalize MMMU-Pro's serialized options list."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(value, list) or not all(isinstance(option, str) for option in value):
        return []
    return value


def adapt_multimodal_row(row: dict[str, Any], benchmark: MultimodalBenchmark) -> list[dict[str, Any]] | None:
    """Convert one supported benchmark row into OpenAI Vision messages."""
    if benchmark in {MultimodalBenchmark.GQA, MultimodalBenchmark.TEXTVQA, MultimodalBenchmark.COCO_CAPTION}:
        return _vision_message(row.get("question"), [row.get("image")] if row.get("image") is not None else [])
    if benchmark is MultimodalBenchmark.CHARXIV_REASONING:
        return _vision_message(row.get("reasoning_q"), [row.get("image")] if row.get("image") is not None else [])
    if benchmark is MultimodalBenchmark.MMMU_PRO:
        options = _parse_options(row.get("options"))
        if not options:
            return None
        labels = [chr(ord("A") + index) for index in range(len(options))]
        options_text = "\n".join(f"{label}. {option}" for label, option in zip(labels, options))
        question = f"{row.get('question', '')}\n\nOptions:\n{options_text}\nAnswer with the option letter only."
        images = [row.get(f"image_{index}") for index in range(1, 8)]
        return _vision_message(question, [image for image in images if image is not None])
    raise ValueError(f"Unsupported multimodal benchmark: {benchmark}")


def load_multimodal_prompts(
    args: Any,
    load_rows: Callable[..., Iterable[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """Load and adapt a multimodal benchmark, including GQA's split image table."""
    benchmark = MultimodalBenchmark(args.benchmark_adapter)
    rows = load_rows(
        args.input_data,
        split=args.split,
        name=args.dataset_name,
        shuffle_seed=args.shuffle_seed,
    )

    image_by_id: dict[str, Any] | None = None
    if benchmark is MultimodalBenchmark.GQA:
        if not args.dataset_name or not args.dataset_name.endswith("_instructions"):
            raise ValueError("GQA requires an *_instructions dataset_name so its matching image config can be joined.")
        image_rows = load_rows(
            args.input_data,
            split=args.split,
            name=args.dataset_name.removesuffix("_instructions") + "_images",
            shuffle_seed=None,
        )
        image_by_id = {row["id"]: row["image"] for row in image_rows}

    prompts: list[list[dict[str, Any]]] = []
    for raw_row in rows:
        row = dict(raw_row)
        if image_by_id is not None:
            row["image"] = image_by_id.get(row.get("imageId"))
        prompt = adapt_multimodal_row(row, benchmark)
        if prompt is not None:
            prompts.append(prompt)
        if len(prompts) >= args.num_prompts:
            break
    return prompts
