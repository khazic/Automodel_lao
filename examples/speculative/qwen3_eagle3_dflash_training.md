# Qwen3-8B EAGLE3 / DFlash Training

Run from the Automodel repository root:

```bash
conda activate automodel
```

## Data

Download the regenerated PerfectBlend parquet dataset:

```bash
huggingface-cli download \
    jihwan1205/perfectblend-qwen3-8b-regen \
    --repo-type dataset \
    --local-dir cache/dataset/perfectblend-qwen3-8b-regen
```

Both training configs use:

```yaml
train_data_path: ./cache/dataset/perfectblend-qwen3-8b-regen/data
```

## EAGLE3

Config:

```text
examples/speculative/eagle3/qwen3_eagle3_perfectblend.yaml
```

Train:

```bash
python -m nemo_automodel.cli.app \
    examples/speculative/eagle3/qwen3_eagle3_perfectblend.yaml \
    --nproc-per-node 8
```

Output:

```text
./outputs/eagle3_qwen3_8b
./outputs/eagle3_qwen3_8b/checkpoints
```

## DFlash

Config:

```text
examples/speculative/dflash/qwen3_dflash.yaml
```

Train:

```bash
python -m nemo_automodel.cli.app \
    examples/speculative/dflash/qwen3_dflash.yaml \
    --nproc-per-node 8
```

Output:

```text
./outputs/dflash_qwen3_8b
./outputs/dflash_qwen3_8b/checkpoints
```

Adjust `--nproc-per-node` to the number of GPUs used for the run.
