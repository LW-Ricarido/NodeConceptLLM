"""Trainer that checkpoints trainable adapters instead of the frozen base LLM."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from trl import SFTTrainer
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import has_length

from models.graphAdapter import HEAD_SAFE


def graph_adapter_from_model(model):
    candidate = model
    if isinstance(candidate, PeftModel):
        candidate = candidate.get_base_model()
    # PEFT may leave one or more BaseTuner wrappers in the chain.
    for _ in range(4):
        if hasattr(candidate, "node_embedding_connect"):
            return candidate
        candidate = getattr(candidate, "model", getattr(candidate, "base_model", None))
        if candidate is None:
            break
    raise TypeError("Could not find GraphAdapter4CausalLM inside the training model")


class LightweightSFTTrainer(SFTTrainer):
    """Save connector-only pretrain checkpoints or PEFT-only SFT checkpoints.

    Trainer still writes ``trainer_state.json``. Optimizer/scheduler/RNG states are
    written only when TrainingArguments.save_only_model is false.
    """

    def _get_train_sampler(self, train_dataset=None):
        """Build the length sampler from a materialized length list.

        Hugging Face datasets expose columns as Arrow ``Column`` objects.  The
        default Trainer passes that object directly to ``LengthGroupedSampler``,
        whose sorting path repeatedly performs random column access.  Materialize
        the column once so sampler construction is linear in Python-list access.
        """
        if train_dataset is None:
            train_dataset = self.train_dataset

        if not self.args.group_by_length:
            return super()._get_train_sampler(train_dataset)

        if train_dataset is None or not has_length(train_dataset):
            return None

        try:
            lengths = train_dataset[self.args.length_column_name][:]
        except (KeyError, TypeError):
            # Keep Trainer's fallback for datasets without the configured
            # length column (it derives lengths from model inputs instead).
            return super()._get_train_sampler(train_dataset)

        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            dataset=train_dataset,
            lengths=lengths,
            model_input_name=(
                self.processing_class.model_input_names[0]
                if getattr(self, "processing_class", None) is not None
                else self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            ),
        )

    def __init__(self, *args, training_task: str, save_connector_with_lora: bool = False, **kwargs):
        self.training_task = training_task
        self.save_connector_with_lora = save_connector_with_lora
        super().__init__(*args, **kwargs)

    def _save_connector(self, output_dir: str) -> None:
        wrapper = graph_adapter_from_model(self.model)
        state = {key: value.detach().cpu().contiguous() for key, value in wrapper.node_embedding_connect.state_dict().items()}
        save_file(state, os.path.join(output_dir, HEAD_SAFE))
        metadata = {
            "embedding_mask_id": wrapper.embedding_mask_id,
            "connector_dim": wrapper.node_embedding_connect[0].in_features,
            "hidden_size": wrapper.node_embedding_connect[0].out_features,
            "base_model_name_or_path": getattr(wrapper.base_model.config, "_name_or_path", None),
        }
        with open(os.path.join(output_dir, "connector_config.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.training_task == "pretrain":
            self._save_connector(output_dir)
        else:
            if not isinstance(self.model, PeftModel):
                raise TypeError("SFT lightweight checkpointing requires a PeftModel (--use_LoRA)")
            # The base model directory already contains the resized token
            # embeddings. PEFT's "auto" mode would otherwise detect the
            # resized vocabulary and copy embed_tokens/lm_head into every
            # adapter checkpoint, turning a few-MB LoRA into a ~GB artifact.
            self.model.save_pretrained(
                output_dir,
                safe_serialization=True,
                save_embedding_layers=False,
            )
            if self.save_connector_with_lora:
                self._save_connector(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        connector_file = Path(resume_from_checkpoint) / HEAD_SAFE
        if self.training_task == "pretrain" and connector_file.exists():
            wrapper = graph_adapter_from_model(model or self.model)
            wrapper.node_embedding_connect.load_state_dict(load_file(str(connector_file), device="cpu"))
            return
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
