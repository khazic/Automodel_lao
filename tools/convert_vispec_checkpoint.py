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

"""Convert the released ViSpec draft weights to NeMo's checkpoint contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file


def convert(source: Path, destination: Path) -> None:
    """Convert one official ViSpec safetensors file into a NeMo draft export.

    The released checkpoint calls the image adaptor ``imadpt``. Its first
    decoder layer omits input RMSNorm and the model omits final RMSNorm, so the
    converted config preserves those architectural choices instead of
    inventing normalization parameters.
    """
    source_config = json.loads((source / "config.json").read_text())
    source_files = sorted(source.glob("model*.safetensors"))
    if len(source_files) != 1:
        raise ValueError(f"Expected one official model*.safetensors file, found {source_files}")
    source_state = load_file(str(source_files[0]))
    state = {}
    for key, value in source_state.items():
        converted_key = "img_adaptor.query" if key == "imadpt.q" else key.replace("imadpt.", "img_adaptor.")
        state[converted_key] = value

    config = dict(source_config)
    config["architectures"] = ["VispecDraftModel"]
    config["draft_num_hidden_layers"] = int(config.get("num_hidden_layers", 1))
    config["vispec_num_query_tokens"] = int(source_state["imadpt.q"].shape[0])
    config["vocab_size"] = int(source_state["embed_tokens.weight"].shape[0])
    config["fc_bias"] = True
    config["qkv_bias"] = True
    config["draft_skip_first_input_norm"] = True
    config["draft_apply_final_norm"] = False
    config["draft_use_sdpa_attention"] = True
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(state, str(destination / "model.safetensors"))


def main() -> None:
    """Parse command-line arguments and convert a ViSpec checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
    print(f"Wrote converted ViSpec checkpoint to {args.destination}")


if __name__ == "__main__":
    main()
