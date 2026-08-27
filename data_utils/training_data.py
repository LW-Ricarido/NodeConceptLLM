"""Dataset loading and one-time preprocessing for connector pretraining/SFT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

from accelerate import PartialState
from datasets import Dataset, load_from_disk


def parse_dataset_names(values: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Accept both ``--dataset_names a b`` and ``--dataset_names a,b``."""
    if not values:
        return None
    names = [name.strip() for value in values for name in value.split(",") if name.strip()]
    return sorted(set(names)) or None


def _deterministic_sample_indices(size: int, limit: int) -> list[int]:
    """Select evenly spaced rows without depending on a random seed."""
    if limit <= 0 or size <= limit:
        return list(range(size))
    # Select the midpoint of each of ``limit`` equal-width buckets. Integer
    # arithmetic makes this stable across processes, Python versions, and seeds.
    return [((2 * index + 1) * size) // (2 * limit) for index in range(limit)]


def _cache_path(cache_dir: Path, dataset: Dataset, payload: dict, operation: str) -> str:
    digest_payload = {"fingerprint": dataset._fingerprint, "operation": operation, **payload}
    digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"{operation}-{digest}.arrow")


def _filter(
    dataset: Dataset,
    predicate,
    *,
    input_columns: list[str],
    cache_dir: Path,
    payload: dict,
    num_proc: int,
    operation: str,
) -> Dataset:
    return dataset.filter(
        predicate,
        input_columns=input_columns,
        num_proc=max(1, num_proc),
        cache_file_name=_cache_path(cache_dir, dataset, payload, operation),
        desc=operation,
    )


def _normalise_example(example: dict, tokenizer, embedding_mask_id: int) -> dict:
    embeds = example.get("unaligned_input_embeds")
    if embeds is None:
        embeds = example.get("unalgined_input_embeds")
    if embeds is None:
        raise ValueError("Row has neither unaligned_input_embeds nor unalgined_input_embeds")

    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ],
        return_tensors="pt",
    )[0]
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["question"]}],
        return_tensors="pt",
        add_generation_prompt=True,
    )[0]
    labels = full.clone()
    labels[: len(prompt)] = -100
    input_ids = full.tolist()
    return {
        "input_ids": input_ids,
        "labels": labels.tolist(),
        "length": len(input_ids),
        "embedding_positions": [i for i, token in enumerate(input_ids) if token == embedding_mask_id],
        "unaligned_input_embeds": embeds,
        "dataset_name": example.get("dataset_name", "unknown"),
    }


def _tokenize(
    dataset: Dataset,
    tokenizer,
    embedding_mask_id: int,
    *,
    cache_dir: Path,
    payload: dict,
    num_proc: int,
    max_length: int,
    operation: str,
) -> Dataset:
    tokenized = dataset.map(
        _normalise_example,
        fn_kwargs={"tokenizer": tokenizer, "embedding_mask_id": embedding_mask_id},
        remove_columns=dataset.column_names,
        num_proc=max(1, num_proc),
        cache_file_name=_cache_path(cache_dir, dataset, payload, f"{operation}-tokenize"),
        desc=f"{operation}-tokenize",
    )
    return _filter(
        tokenized,
        lambda length: length <= max_length,
        input_columns=["length"],
        cache_dir=cache_dir,
        payload=payload,
        num_proc=num_proc,
        operation=f"{operation}-length",
    )


def prepare_datasets(
    *,
    task: str,
    dataset_path: str,
    dataset_names: Optional[list[str]],
    tokenizer,
    embedding_mask_id: int,
    cache_dir: str,
    preprocessing_num_workers: int,
    max_length: int,
    eval_samples: int,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Load the complete on-disk dataset, select rows, and reuse preprocessing caches.

    ``main_process_first`` makes rank zero populate each deterministic Arrow cache;
    the remaining DDP ranks then memory-map the same cache rather than repeating the
    expensive 200-400 GB scan/tokenization work.
    """
    state = PartialState()
    names = parse_dataset_names(dataset_names)
    cache_path = Path(cache_dir)
    payload = {
        "task": task,
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_names": names,
        "tokenizer": getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__),
        "embedding_mask_id": embedding_mask_id,
        "max_length": max_length,
        "eval_samples": eval_samples,
        "seed": seed,
    }

    with state.main_process_first():
        all_dataset = load_from_disk(dataset_path)
        if not isinstance(all_dataset, Dataset):
            raise TypeError(f"Expected a Dataset at {dataset_path}, got {type(all_dataset).__name__}")

        if names:
            if "dataset_name" not in all_dataset.column_names:
                raise ValueError(
                    f"--dataset_names={names} was requested, but {dataset_path} has no "
                    "dataset_name column. Rebuild the combined pretrain dataset with provenance "
                    "or omit --dataset_names; filtering is never silently ignored."
                )
            selected = set(names)
            all_dataset = _filter(
                all_dataset,
                lambda name: name in selected,
                input_columns=["dataset_name"],
                cache_dir=cache_path,
                payload=payload,
                num_proc=preprocessing_num_workers,
                operation="dataset-names",
            )

        if task == "sft":
            if "split_set" not in all_dataset.column_names:
                raise ValueError(f"SFT dataset {dataset_path} must contain split_set")

            has_arxiv = "dataset_name" in all_dataset.column_names and any(
                str(name).lower() == "arxiv" for name in set(all_dataset["dataset_name"])
            )
            train_dataset = _filter(
                all_dataset,
                lambda split: split == "train",
                input_columns=["split_set"],
                cache_dir=cache_path,
                payload=payload,
                num_proc=preprocessing_num_workers,
                operation="train-split",
            )
            if has_arxiv:
                required_columns = {"dataset_name", "task_type"}
                missing = required_columns.difference(all_dataset.column_names)
                if missing:
                    raise ValueError(
                        f"SFT dataset contains arxiv rows but is missing required columns: {sorted(missing)}"
                    )
                eval_dataset = _filter(
                    all_dataset,
                    lambda split, dataset_name, task_type: (
                        split == "valid"
                        and str(dataset_name).lower() == "arxiv"
                        and task_type == "classification"
                    ),
                    input_columns=["split_set", "dataset_name", "task_type"],
                    cache_dir=cache_path,
                    payload={**payload, "eval_selection": "arxiv-classification-valid"},
                    num_proc=preprocessing_num_workers,
                    operation="eval-arxiv-classification",
                )
            else:
                eval_dataset = _filter(
                    all_dataset,
                    lambda split: split == "valid",
                    input_columns=["split_set"],
                    cache_dir=cache_path,
                    payload={**payload, "eval_selection": "valid"},
                    num_proc=preprocessing_num_workers,
                    operation="eval-valid-split",
                )
            if eval_samples > 0:
                eval_dataset = eval_dataset.select(
                    _deterministic_sample_indices(len(eval_dataset), eval_samples)
                )
        else:
            if len(all_dataset) <= eval_samples:
                raise ValueError(f"Pretrain dataset has {len(all_dataset)} rows, not enough for {eval_samples} eval rows")
            # Contiguous slices remain zero-copy Arrow views. A full shuffle would
            # build a multi-million-row indices table before a 200-row evaluation.
            eval_dataset = all_dataset.select(range(eval_samples))
            train_dataset = all_dataset.select(range(eval_samples, len(all_dataset)))

        train_dataset = _tokenize(
            train_dataset,
            tokenizer,
            embedding_mask_id,
            cache_dir=cache_path,
            payload=payload,
            num_proc=preprocessing_num_workers,
            max_length=max_length,
            operation="train",
        )
        eval_dataset = _tokenize(
            eval_dataset,
            tokenizer,
            embedding_mask_id,
            cache_dir=cache_path,
            payload=payload,
            num_proc=preprocessing_num_workers,
            max_length=max_length,
            operation="eval",
        )

    if not len(train_dataset):
        raise ValueError("No training rows remain after dataset/split/length filtering")
    if not len(eval_dataset):
        raise ValueError("No evaluation rows remain after dataset/split/length filtering")
    return train_dataset, eval_dataset
