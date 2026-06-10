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

"""Export an LLM DCP checkpoint back to consolidated Hugging Face safetensors.

This tool reuses the regular LLM fine-tuning recipe to rebuild the distributed
model topology, restore a full-parameter DCP checkpoint, and save a new
checkpoint in Hugging Face-compatible safetensors format.

Example:
    torchrun --nproc-per-node=8 tools/export_llm_dcp_to_hf.py \
        --checkpoint-dir /llm-align/liuchonghan/runs/foo/checkpoints/epoch_0_step_17999 \
        --output-dir /llm-align/liuchonghan/runs/foo/hf_export
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import torch

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

_CHECKPOINT_DIR_RE = re.compile(r"epoch_(\d+)_step_(\d+)$")
_TRACKING_KEYS = ("wandb", "mlflow", "comet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for DCP -> HF export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to the saved training checkpoint directory, e.g. epoch_0_step_17999.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive the exported checkpoint root.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config.yaml path. Defaults to <checkpoint-dir>/config.yaml.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help=(
            "Optional override for model.pretrained_model_name_or_path. Use this only when the original base "
            "model path recorded in config.yaml is no longer valid."
        ),
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Explicit epoch number for the exported checkpoint directory name. Defaults to the source checkpoint.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Explicit step number for the exported checkpoint directory name. Defaults to the source checkpoint.",
    )
    parser.add_argument(
        "--save-consolidated",
        choices=("every", "final"),
        default="every",
        help="When to export consolidated Hugging Face weights. Default: every.",
    )
    return parser.parse_args(argv)


def infer_config_path(checkpoint_dir: str) -> Path:
    """Return the default config.yaml path for a checkpoint directory."""
    return Path(checkpoint_dir) / "config.yaml"


def infer_epoch_step(checkpoint_dir: str) -> tuple[int, int]:
    """Parse epoch/step numbers from a checkpoint directory name."""
    match = _CHECKPOINT_DIR_RE.search(Path(checkpoint_dir).name)
    if match is None:
        raise ValueError(
            "Could not infer epoch/step from checkpoint directory name. Pass both --epoch and --step explicitly."
        )
    return int(match.group(1)), int(match.group(2))


def resolve_epoch_step(checkpoint_dir: str, epoch: int | None, step: int | None) -> tuple[int, int]:
    """Resolve the exported epoch/step from explicit flags or the source checkpoint name."""
    inferred_epoch, inferred_step = infer_epoch_step(checkpoint_dir)
    return (
        inferred_epoch if epoch is None else epoch,
        inferred_step if step is None else step,
    )


def disable_tracking_loggers(cfg: ConfigNode) -> None:
    """Remove remote experiment trackers from the loaded config."""
    for key in _TRACKING_KEYS:
        cfg.__dict__.pop(key, None)


def build_export_config(args: argparse.Namespace) -> ConfigNode:
    """Load config.yaml and apply export-specific overrides."""
    config_path = Path(args.config) if args.config is not None else infer_config_path(args.checkpoint_dir)
    cfg = parse_args_and_load_config(
        str(config_path),
        argv=[
            "--checkpoint.restore_from",
            args.checkpoint_dir,
            "--checkpoint.checkpoint_dir",
            args.output_dir,
            "--checkpoint.model_save_format",
            "safetensors",
            "--checkpoint.save_consolidated",
            args.save_consolidated,
        ],
    )
    if args.model_name_or_path is not None:
        cfg.set_by_dotted("model.pretrained_model_name_or_path", args.model_name_or_path)
    disable_tracking_loggers(cfg)
    return cfg


def close_trainer(trainer: TrainFinetuneRecipeForNextTokenPrediction | None) -> None:
    """Best-effort cleanup for recipe-owned resources."""
    if trainer is None:
        return
    if hasattr(trainer, "metric_logger_train"):
        trainer.metric_logger_train.close()
    if hasattr(trainer, "metric_logger_valid"):
        for logger in trainer.metric_logger_valid.values():
            logger.close()
    if hasattr(trainer, "checkpointer"):
        trainer.checkpointer.close()


def main(argv: Sequence[str] | None = None) -> None:
    """Restore a DCP checkpoint and re-save it as Hugging Face safetensors."""
    args = parse_args(argv)
    export_epoch, export_step = resolve_epoch_step(args.checkpoint_dir, args.epoch, args.step)
    cfg = build_export_config(args)

    trainer: TrainFinetuneRecipeForNextTokenPrediction | None = None
    try:
        trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
        trainer.setup()
        trainer.save_checkpoint(epoch=export_epoch, step=export_step, train_loss=0.0, val_loss=None)
    finally:
        close_trainer(trainer)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        output_dir = Path(args.output_dir) / f"epoch_{export_epoch}_step_{export_step}" / "model" / "consolidated"
        print(f"HF export is ready under: {output_dir}")


if __name__ == "__main__":
    main()
