# NodeConceptLLM connector pretrain and SFT

The `speedup` branch uses one training entry point, `main.py`, for connector
pretraining and LoRA SFT. Example launchers are in `reproduced_run_scripts/`.

## Reproducibility and W&B

Every launch writes these files under `--model_save_path` before model/data
loading starts:

- `command.txt`: exact Python argv, shell quoted;
- `args.json`: every resolved argument, including defaults and dataset paths.

Those two root files describe the latest launch. Immutable copies for every
launch (including resumes into the same output directory) are retained below
`invocations/<UTC timestamp>-<pid>/`.

Trainer sends loss, learning rate, evaluation loss, token accuracy, throughput,
and the full resolved config to the W&B project selected by
`--wandb_project`. Set `WANDB_MODE=offline` for a disconnected node.

## Datasets

Pretrain always loads the complete on-disk `all_connector_pretrain`; SFT always
loads `all_downstream_dataset_32`, unless an explicit full dataset path is given
with `--dataset_dir`. `--dataset_names arxiv pubmed` (comma-separated also works)
filters rows by the `dataset_name` column after that full dataset is loaded.

The current `all_connector_pretrain` artifact does **not** contain a
`dataset_name` column. Its default run therefore uses all rows. Passing
`--dataset_names` against that artifact deliberately fails instead of silently
ignoring the requested subset. A future rebuilt combined artifact with
provenance works without code changes.

Rebuild that artifact once from its provenance-preserving component folders:

```bash
python utils/rebuild_connector_pretrain_with_names.py \
  --source_dir ../../LLM4Graph_tracked/datasets_local/connector_pretrain \
  --output_dir ../../LLM4Graph_tracked/datasets_local/all_connector_pretrain_with_names
```

By default `_pretrain` is removed from each child directory name, so
`arxiv_pretrain/` becomes `dataset_name=arxiv`. Use `--strip_suffix ''` to keep
the exact directory names. The tool refuses an existing output path, writes to
a unique sibling temporary directory, records `provenance_manifest.json`, and
renames only after a successful full save. It never edits the old 206 GB
artifact. The temporary and final paths must therefore be on the same
filesystem; interrupted partial output is intentionally retained for manual
inspection/removal.

Use the rebuilt artifact and select a subset like this:

```bash
PRETRAIN_DATASET_DIR=../../LLM4Graph_tracked/datasets_local/all_connector_pretrain_with_names \
  bash reproduced_run_scripts/1B_connector_training.sh --dataset_names arxiv pubmed
```

For direct `main.py` launches, the environment variables
`NODECONCEPT_PRETRAIN_DATASET` and `NODECONCEPT_SFT_DATASET` override the two
default full-dataset paths.

Tokenization/filtering uses deterministic Arrow caches under
`--preprocessing_cache_dir` (default: `<model_save_path>/dataset_cache`). In DDP,
rank zero creates a missing cache first and other ranks reuse it. Reuse a stable
cache directory across runs to avoid rescanning the 206/391 GB datasets.

## Lightweight checkpoints

- Pretrain checkpoints contain `node_embedding_connect.safetensors` and a small
  `connector_config.json`; the frozen base LLM is never copied.
- SFT checkpoints contain PEFT LoRA adapter files only.
- `--combine_training` additionally trains/saves the connector during SFT.
- Optimizer, scheduler, and RNG state are omitted by default. Add
  `--save_resume_state` when exact interruption recovery matters, then resume
  with `--resume_from_checkpoint checkpoint-N`.

The pretrain connector feeds directly into SFT without packaging a full model:

```bash
CONNECTOR_DIR=ckpts/speedup/1b_connector/final \
  bash reproduced_run_scripts/1B_tasks_training.sh
```

## Performance defaults

The launchers use standard BF16 autocast, SDPA attention, fused AdamW, length
grouping, persistent dataloader workers, and low-frequency bounded evaluation.
Evaluation gathers argmax token IDs rather than full vocabulary logits. Frozen
LLM parameters are excluded from optimization/DDP synchronization, and graph
embeddings are projected in one batched connector call.

Use `--precision fp16` on GPUs without BF16. Use
`--attn_implementation flash_attention_2` only when the installed PyTorch/CUDA
and `flash-attn` versions are compatible. The 3B scripts require a local 3B base
model at the path shown in those scripts.
