"""Deterministic generate-time evaluation for GraphAdapter SFT checkpoints."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from safetensors.torch import load_file
from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from torch.utils.data import DataLoader

from data_utils.prompt_str import embedding_mask_str
from models.graphAdapter import GraphAdapter4CausalLM, HEAD_BIN, HEAD_SAFE


def deterministic_indices(size: int, limit: int) -> list[int]:
    if limit <= 0 or size <= limit:
        return list(range(size))
    return [((2 * i + 1) * size) // (2 * limit) for i in range(limit)]


def load_connector(model, checkpoint: Path) -> None:
    safe_path = checkpoint / HEAD_SAFE
    bin_path = checkpoint / HEAD_BIN
    if safe_path.exists():
        state = load_file(str(safe_path), device="cpu")
    elif bin_path.exists():
        state = torch.load(bin_path, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"No connector weights found in {checkpoint}")
    model.node_embedding_connect.load_state_dict(state, strict=True)


def prepare_rows(dataset, tokenizer, embedding_mask_id: int, indices: list[int]) -> list[dict]:
    rows = []
    for index in indices:
        source = dataset[index]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": source["question"]}],
            return_tensors="pt",
            add_generation_prompt=True,
        )[0].tolist()
        rows.append(
            {
                "source_index": index,
                "input_ids": prompt,
                "answer": source["answer"],
                "dataset_name": source.get("dataset_name"),
                "task_type": source.get("task_type"),
                "unaligned_input_embeds": source.get(
                    "unaligned_input_embeds", source.get("unalgined_input_embeds")
                ),
            }
        )
        if rows[-1]["unaligned_input_embeds"] is None:
            raise ValueError(f"Row {index} has no graph embeddings")
        if embedding_mask_id not in prompt:
            raise ValueError(f"Row {index} contains no {embedding_mask_str} token")
    return rows


class GenerateCollator:
    def __init__(self, tokenizer, embedding_mask_id: int):
        self.tokenizer = tokenizer
        self.embedding_mask_id = embedding_mask_id

    def __call__(self, examples):
        input_ids = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            {"input_ids": [example["input_ids"] for example in examples]},
            return_tensors="pt",
            padding_side="left",
            pad_to_multiple_of=8,
        )
        input_ids["attention_mask"] = (input_ids["input_ids"] != self.tokenizer.pad_token_id).long()
        input_ids["unaligned_inputs_embeds"] = [
            torch.as_tensor(example["unaligned_input_embeds"], dtype=torch.float32)
            for example in examples
        ]
        input_ids["embedding_positions"] = [
            torch.nonzero(row == self.embedding_mask_id).flatten() for row in input_ids["input_ids"]
        ]
        input_ids["source_index"] = [example["source_index"] for example in examples]
        input_ids["answer"] = [example["answer"] for example in examples]
        return input_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--checkpoint", required=True, help="SFT final/checkpoint directory")
    parser.add_argument("--connector_checkpoint", default=None, help="Pretrain connector directory if SFT checkpoint does not contain it")
    parser.add_argument("--connector_dim", type=int, default=768)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset", required=True, help="Dataset saved with datasets.save_to_disk")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--split_set", default="valid")
    parser.add_argument("--task_type", default="classification")
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Generation seed; selection is seed-independent")
    return parser


def main(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    with open(output_dir / "command.txt", "w", encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable, *sys.argv[1:]]) + "\n")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    mask_tokens = tokenizer.encode(embedding_mask_str, add_special_tokens=False)
    if len(mask_tokens) != 1:
        raise ValueError(f"{embedding_mask_str!r} must map to one token")
    embedding_mask_id = mask_tokens[0]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]

    base = GraphAdapter4CausalLM.from_pretrained(
        base_model_dir=args.base_model,
        embedding_mask_id=embedding_mask_id,
        connector_dim=args.connector_dim,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    checkpoint = Path(args.checkpoint)
    if (checkpoint / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, str(checkpoint), is_trainable=False)
    elif (checkpoint / "config.json").exists() and (checkpoint / "base_model").exists():
        model = GraphAdapter4CausalLM.from_pretrained(
            model_dir=str(checkpoint), torch_dtype=dtype, attn_implementation=args.attn_implementation
        )
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint}")

    if (checkpoint / HEAD_SAFE).exists() or (checkpoint / HEAD_BIN).exists():
        load_connector(model.get_base_model() if isinstance(model, PeftModel) else model, checkpoint)
    elif args.connector_checkpoint:
        load_connector(model.get_base_model() if isinstance(model, PeftModel) else model, Path(args.connector_checkpoint))
    else:
        raise FileNotFoundError("SFT checkpoint has no connector; provide --connector_checkpoint")

    dataset = load_from_disk(args.dataset)
    if args.dataset_name is not None:
        dataset = dataset.filter(lambda name: str(name).lower() == args.dataset_name.lower(), input_columns=["dataset_name"])
    if args.split_set is not None:
        dataset = dataset.filter(lambda split: split == args.split_set, input_columns=["split_set"])
    if args.task_type is not None:
        dataset = dataset.filter(lambda task: task == args.task_type, input_columns=["task_type"])
    indices = deterministic_indices(len(dataset), args.eval_samples)
    rows = prepare_rows(dataset, tokenizer, embedding_mask_id, indices)
    dataloader = DataLoader(
        rows,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=GenerateCollator(tokenizer, embedding_mask_id),
    )

    device = torch.device(args.device)
    # Connector checkpoints are commonly stored as float32 even when the base
    # model is loaded in bf16/fp16.  Cast the complete generation model so the
    # connector's Linear receives embeddings with the same dtype as its weight.
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model.eval()
    generation_config = model.generation_config
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    generation_config.pad_token_id = tokenizer.pad_token_id
    results = []
    with torch.inference_mode():
        with tqdm(total=len(rows), desc="Generate eval", unit="dp") as progress:
            for batch in dataloader:
                source_indices = batch.pop("source_index")
                answers = batch.pop("answer")
                input_length = batch["input_ids"].shape[1]
                batch = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                generated = model.generate(
                    max_new_tokens=args.max_new_tokens,
                    generation_config=generation_config,
                    **batch,
                )
                completions = generated[:, input_length:]
                texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
                for source_index, answer, text_value in zip(source_indices, answers, texts):
                    results.append(
                        {
                            "source_index": source_index,
                            "answer": answer,
                            "generated": text_value,
                            "label_in_output": answer.strip().lower() in text_value.strip().lower(),
                        }
                    )
                progress.update(len(answers))

    correct = sum(item["label_in_output"] for item in results)
    summary = {
        "checkpoint": str(checkpoint),
        "dataset": str(args.dataset),
        "eval_rows": len(results),
        "label_in_output_correct": correct,
        "label_in_output_accuracy": correct / max(len(results), 1),
        "selection_seed_independent": True,
    }
    with open(output_dir / "results.jsonl", "w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    main(parsed)
