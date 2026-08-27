"""Fast, reproducible connector pretraining and LoRA SFT entry point."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import wandb
from accelerate import PartialState
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file
from transformers import AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig

from data_utils.data_collators import InstructEmbedsPretrainCollator
from data_utils.prompt_str import embedding_mask_str
from data_utils.training_data import parse_dataset_names, prepare_datasets
from models.graphAdapter import GraphAdapter4CausalLM, HEAD_SAFE
from trainers.lightweight_sft_trainer import LightweightSFTTrainer


DEFAULT_PRETRAIN_DATASET = os.environ.get(
    "NODECONCEPT_PRETRAIN_DATASET",
    "../../LLM4Graph_tracked/datasets_local/all_connector_pretrain",
)
DEFAULT_SFT_DATASET = os.environ.get(
    "NODECONCEPT_SFT_DATASET",
    "../../LLM4Graph_tracked/datasets_local/all_downstream_dataset_32",
)


class WandbResolvedArgsCallback(TrainerCallback):
    """Put non-TrainingArguments CLI values in the W&B run config as well."""

    def __init__(self, resolved_args: dict):
        self.resolved_args = resolved_args

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero and wandb.run is not None:
            wandb.config.update(self.resolved_args, allow_val_change=True)


def record_invocation(args: argparse.Namespace) -> None:
    """Persist both the exact argv and every resolved/defaulted parameter."""
    if not PartialState().is_main_process:
        return
    output_dir = Path(args.model_save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    invocation_dir = output_dir / "invocations" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{os.getpid()}"
    invocation_dir.mkdir(parents=True, exist_ok=False)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]) + "\n"
    for directory in (output_dir, invocation_dir):
        with open(directory / "command.txt", "w", encoding="utf-8") as handle:
            handle.write(command)
        with open(directory / "args.json", "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True)


def load_connector_weights(model: GraphAdapter4CausalLM, path: str) -> None:
    connector_path = Path(path)
    if connector_path.is_dir():
        safe_path = connector_path / HEAD_SAFE
        pt_path = connector_path / "node_embedding_connect.pt"
    else:
        safe_path = connector_path if connector_path.suffix == ".safetensors" else Path("__missing__")
        pt_path = connector_path
    if safe_path.exists():
        state = load_file(str(safe_path), device="cpu")
    elif pt_path.exists():
        loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
        state = loaded.state_dict() if hasattr(loaded, "state_dict") else loaded
    else:
        raise FileNotFoundError(f"No connector checkpoint found at {path}")
    model.node_embedding_connect.load_state_dict(state, strict=True)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    # Prevent Trainer from gathering [batch, sequence, vocabulary] tensors.
    return logits.argmax(dim=-1)


def token_accuracy(eval_prediction):
    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids
    predictions = predictions[:, :-1]
    labels = labels[:, 1:]
    mask = labels != -100
    correct = ((predictions == labels) & mask).sum()
    total = mask.sum()
    return {"token_accuracy": float(correct / max(int(total), 1))}


def label_in_output_accuracy(eval_prediction, tokenizer):
    """Count an example correct when its reference answer occurs in the output.

    Trainer evaluation uses teacher-forced argmax predictions.  Aligning the
    argmax tokens with the non-masked label positions gives a deterministic,
    inexpensive approximation of the standalone generation-time check.
    """
    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids
    correct = 0
    for predicted, target in zip(predictions, labels):
        target_mask = target != -100
        target_text = tokenizer.decode(target[target_mask], skip_special_tokens=True).strip().lower()
        if not target_text:
            continue
        predicted_text = tokenizer.decode(
            predicted[:-1][target_mask[1:]], skip_special_tokens=True
        ).strip().lower()
        correct += int(target_text in predicted_text)
    return {"label_in_output_accuracy": correct / max(len(labels), 1)}


def build_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir or args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    embedding_mask_tokens = tokenizer.encode(embedding_mask_str, add_special_tokens=False)
    if len(embedding_mask_tokens) != 1:
        raise ValueError(f"{embedding_mask_str!r} must map to one token, got {embedding_mask_tokens}")
    embedding_mask_id = embedding_mask_tokens[0]

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]
    load_kwargs = {"torch_dtype": dtype, "attn_implementation": args.attn_implementation}
    if args.task == "pretrain":
        model = GraphAdapter4CausalLM.from_pretrained(
            base_model_dir=args.model_dir,
            embedding_mask_id=embedding_mask_id,
            connector_dim=args.connector_dim,
            **load_kwargs,
        )
        if args.connector_path:
            load_connector_weights(model, args.connector_path)
    else:
        if args.connector_path:
            # Direct pretrain -> SFT hand-off without packaging/copying the frozen LLM.
            model = GraphAdapter4CausalLM.from_pretrained(
                base_model_dir=args.model_dir,
                embedding_mask_id=embedding_mask_id,
                connector_dim=args.connector_dim,
                **load_kwargs,
            )
            load_connector_weights(model, args.connector_path)
        else:
            model = GraphAdapter4CausalLM.from_pretrained(
                model_dir=args.model_dir,
                connector_dim=args.connector_dim,
                **load_kwargs,
            )
            # A packaged connector checkpoint is authoritative for this wrapper.
            embedding_mask_id = model.embedding_mask_id

    for parameter in model.base_model.parameters():
        parameter.requires_grad = False

    model.base_model.generation_config.do_sample = False
    model.base_model.generation_config.top_p = 1.0
    model.base_model.generation_config.temperature = 1.0

    if args.task == "sft":
        if not args.use_LoRA:
            raise ValueError("SFT must use --use_LoRA so checkpoints contain only a LoRA adapter")
        if args.load_local_adapter:
            model = PeftModel.from_pretrained(model, args.load_local_adapter, is_trainable=True)
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=args.r_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=args.lora_target_modules,
                ),
            )
        if args.combine_training:
            connector = model.get_base_model().node_embedding_connect
            for parameter in connector.parameters():
                parameter.requires_grad = True
        model.print_trainable_parameters()
    return model, tokenizer, embedding_mask_id


def train(args: argparse.Namespace) -> None:
    state = PartialState()
    set_seed(args.seed)
    record_invocation(args)
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    model, tokenizer, embedding_mask_id = build_model_and_tokenizer(args)
    dataset_names = parse_dataset_names(args.dataset_names)
    train_dataset, eval_dataset = prepare_datasets(
        task=args.task,
        dataset_path=args.dataset_dir,
        dataset_names=dataset_names,
        tokenizer=tokenizer,
        embedding_mask_id=embedding_mask_id,
        cache_dir=args.preprocessing_cache_dir,
        preprocessing_num_workers=args.preprocessing_num_workers,
        max_length=args.max_length,
        eval_samples=args.eval_samples,
        seed=args.seed,
    )
    if state.is_main_process:
        print(
            json.dumps(
                {
                    "loaded_dataset": str(Path(args.dataset_dir).resolve()),
                    "dataset_names": dataset_names or "ALL",
                    "train_rows": len(train_dataset),
                    "eval_rows": len(eval_dataset),
                },
                indent=2,
            )
        )
        tokenizer.save_pretrained(args.model_save_path)

    use_eval = (
        args.eval_strategy != "no"
        and len(eval_dataset) > 0
        and (args.eval_strategy != "steps" or args.eval_steps > 0)
    )
    use_save = args.save_strategy != "no" and (args.save_strategy != "steps" or args.save_steps > 0)
    training_args = SFTConfig(
        output_dir=args.model_save_path,
        logging_dir=args.log_dir,
        run_name=args.run_name,
        report_to=["wandb"],
        per_device_train_batch_size=args.train_size,
        per_device_eval_batch_size=args.eval_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epoch,
        max_steps=args.max_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy=args.eval_strategy if use_eval else "no",
        eval_steps=args.eval_steps if use_eval and args.eval_strategy == "steps" else None,
        eval_accumulation_steps=args.eval_accumulation_steps,
        save_strategy=args.save_strategy if use_save else "no",
        save_steps=args.save_steps if use_save and args.save_strategy == "steps" else 500,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        save_only_model=not args.save_resume_state,
        remove_unused_columns=False,
        label_names=["labels"],
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        tf32=args.tf32,
        optim=args.optim,
        max_length=args.max_length,
        group_by_length=args.group_by_length,
        length_column_name="length",
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.gradient_checkpointing,
        dataset_kwargs={"skip_prepare_dataset": True},
        seed=args.seed,
        data_seed=args.seed,
    )
    collator = InstructEmbedsPretrainCollator(
        tokenizer=tokenizer,
        mlm=False,
        mask_eos=args.mask_eos,
        use_fp16=args.precision == "fp16",
        pad_to_multiple_of=8,
    )
    trainer = LightweightSFTTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if use_eval else None,
        processing_class=tokenizer,
        compute_metrics=(
            (lambda evaluation: {
                **token_accuracy(evaluation),
                **label_in_output_accuracy(evaluation, tokenizer),
            })
            if use_eval and args.label_in_output
            else token_accuracy if use_eval else None
        ),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if use_eval else None,
        callbacks=[WandbResolvedArgsCallback(vars(args))],
        training_task=args.task,
        save_connector_with_lora=args.combine_training,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(Path(args.model_save_path) / "final"))
    if trainer.is_world_process_zero():
        trainer.save_state()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--task", choices=["pretrain", "sft", "prediction"], required=True)
    parser.add_argument("--model_dir", required=True, help="Base model for pretrain; packaged GraphAdapter for SFT")
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--dataset_dir", default=None, help="Full load_from_disk dataset; defaults by task")
    parser.add_argument("--dataset_names", nargs="+", default=None, help="Optional dataset_name subset; space/comma separated")
    parser.add_argument("--model_save_path", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--wandb_project", default="NodeConceptLLM")
    parser.add_argument("--wandb_entity", default=None)

    parser.add_argument("--connector_dim", type=int, default=768)
    parser.add_argument("--connector_path", "--resume_path", dest="connector_path", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--save_resume_state", action="store_true", help="Also save optimizer/scheduler/RNG for exact resume")
    parser.add_argument(
        "--mask_eos",
        action="store_true",
        help="Mask EOS tokens from the SFT loss (default: learn EOS)",
    )
    parser.add_argument("--use_LoRA", action="store_true")
    parser.add_argument("--load_local_adapter", default=None)
    parser.add_argument("--combine_training", action="store_true", help="Train/save connector together with LoRA")
    parser.add_argument("--r_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", nargs="+", default=None)

    parser.add_argument("--train_size", "--per_device_train_batch_size", dest="train_size", type=int, default=2)
    parser.add_argument("--eval_size", "--per_device_eval_batch_size", dest="eval_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--epoch", type=float, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument(
        "--eval_strategy",
        choices=["no", "steps", "epoch"],
        default="steps",
        help="When to run evaluation",
    )
    parser.add_argument(
        "--save_strategy",
        choices=["no", "steps", "epoch"],
        default="steps",
        help="When to save checkpoints",
    )
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument(
        "--load_best_model_at_end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restore the checkpoint with the best selected evaluation metric after training",
    )
    parser.add_argument(
        "--metric_for_best_model",
        choices=["token_accuracy", "label_in_output_accuracy"],
        default="token_accuracy",
    )
    parser.add_argument(
        "--greater_is_better",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether a larger best-model metric is better",
    )
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument(
        "--label_in_output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also report whether the reference answer occurs in the teacher-forced eval output",
    )
    parser.add_argument("--eval_accumulation_steps", type=int, default=8)

    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--use_fp16", action="store_true", help="Deprecated alias for --precision fp16")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--group_by_length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_length", type=int, default=1500)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    parser.add_argument("--preprocessing_num_workers", type=int, default=16)
    parser.add_argument("--preprocessing_cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.task == "prediction":
        args.task = "sft"
    if args.use_fp16:
        args.precision = "fp16"
    if args.dataset_dir is None:
        args.dataset_dir = DEFAULT_PRETRAIN_DATASET if args.task == "pretrain" else DEFAULT_SFT_DATASET
    if args.preprocessing_cache_dir is None:
        args.preprocessing_cache_dir = str(Path(args.model_save_path) / "dataset_cache")
    if args.eval_samples < 1:
        raise ValueError("--eval_samples must be at least 1")
    if args.load_best_model_at_end:
        if args.eval_strategy == "no" or args.save_strategy == "no":
            raise ValueError("--load_best_model_at_end requires both eval and save strategies")
        if args.eval_strategy != args.save_strategy:
            raise ValueError("--eval_strategy and --save_strategy must match when loading the best model")
        if args.eval_strategy == "steps":
            if args.eval_steps < 1 or args.save_steps < 1:
                raise ValueError("Best-model selection with step strategies requires positive eval/save steps")
            if args.save_steps % args.eval_steps != 0:
                raise ValueError("--save_steps must be a multiple of --eval_steps when loading the best model")
        if args.metric_for_best_model == "label_in_output_accuracy" and not args.label_in_output:
            raise ValueError("--metric_for_best_model label_in_output_accuracy requires --label_in_output")
    return args


if __name__ == "__main__":
    train(parse_args())
