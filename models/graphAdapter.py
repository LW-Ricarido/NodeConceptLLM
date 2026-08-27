from typing import Optional, Any, Dict, List, Set, Union
import torch, json
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import PeftModel
from pathlib import Path


CONFIG_NAME = "config.json"
HEAD_SAFE = "node_embedding_connect.safetensors"
HEAD_BIN = "node_embedding_connect.bin"
BASE_SUBDIR = "base_model" 

try:
    from safetensors.torch import save_file as safe_save_file, load_file as safe_load_file
    _HAVE_SAFETENSORS = True
except Exception:
    _HAVE_SAFETENSORS = False

class GraphAdapter4CausalLM(nn.Module):

    def __init__(self, base_model:PreTrainedModel,embedding_mask_id:int,connector_dim=768):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.get_input_embeddings().embedding_dim
        self.node_embedding_connect = nn.Sequential(
            nn.Linear(connector_dim, hidden_size)
        )
        self.embedding_mask_id = embedding_mask_id
        self.config = base_model.config
    
    def forward(self, *args, **kwargs):
        if 'unaligned_inputs_embeds' in kwargs.keys():
            kwargs = self.align_embeddings(**kwargs)
        return self.base_model(**kwargs)
    
    def generate(self, *args, **kwargs):
        if 'unaligned_inputs_embeds' in kwargs.keys():
            prompt_ids = kwargs['input_ids']
            
            kwargs = self.align_embeddings(**kwargs)
            completion_ids = self.base_model.generate(**kwargs)
            prompt_completion_ids = torch.cat([prompt_ids,completion_ids],dim=1)
            return prompt_completion_ids
        else:
            return self.base_model.generate(**kwargs)
    
    def align_embeddings(self, *args, **kwargs):
        unaligned_inputs_embeds = kwargs.pop('unaligned_inputs_embeds')
        embedding_positions = kwargs.pop("embedding_positions", None)
        input_ids = kwargs.pop('input_ids')
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        if embedding_positions is None:
            embedding_positions = [
                torch.nonzero(input_ids[i] == self.embedding_mask_id).flatten()
                for i in range(input_ids.shape[0])
            ]
        embed_chunks = []
        position_chunks = []
        batch_chunks = []
        for batch_idx, (embeds, positions) in enumerate(zip(unaligned_inputs_embeds, embedding_positions)):
            embeds = embeds.to(inputs_embeds)
            if embeds.ndim == 3 and embeds.shape[1] == 1:
                embeds = embeds.squeeze(1)
            positions = positions.to(device=input_ids.device, dtype=torch.long)
            if len(positions) != len(embeds):
                raise ValueError(
                    f"sample {batch_idx}: {len(positions)} embedding tokens but {len(embeds)} graph embeddings"
                )
            embed_chunks.append(embeds)
            position_chunks.append(positions)
            batch_chunks.append(torch.full_like(positions, batch_idx))
        if embed_chunks:
            aligned = self.node_embedding_connect(torch.cat(embed_chunks, dim=0)).to(inputs_embeds.dtype)
            inputs_embeds[torch.cat(batch_chunks), torch.cat(position_chunks)] = aligned
        kwargs['inputs_embeds'] = inputs_embeds
        return kwargs

    
    
    # def prepare_inputs_for_generation(self,*args, **kwargs):
    #     return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
    # @property
    # def config(self):
    #     return getattr(self.base_model, "config", None)
    
    def __getattr__(self, name):
        """
        Delegate attributes/methods not defined in this wrapper
        to the underlying base model.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def __dir__(self):
        """
        Make autocompletion & hasattr() work properly.
        """
        return list(super().__dir__()) + list(dir(self.base_model))
    
    
    @classmethod
    def from_pretrained(cls,base_model_dir, embedding_mask_id, connect_dir=None, LoRA_dir=None):
        base_model = AutoModelForCausalLM.from_pretrained(base_model_dir)
        # if LoRA_dir is not None:
        #     base_model = PeftModel.from_pretrained(base_model, LoRA_dir)
        model = GraphAdapter4CausalLM(base_model,embedding_mask_id)
        if LoRA_dir is not None:
            peft_model = PeftModel.from_pretrained(model, LoRA_dir)
            model = peft_model.base_model.merge_and_unload()
        if connect_dir is not None:
            model.node_embedding_connect.load_state_dict(torch.load(connect_dir,weights_only=False,map_location='cpu').state_dict())
        return model
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if 'model_dir' in kwargs.keys():
            connector_dim = kwargs['connector_dim'] if 'connector_dim' in kwargs.keys() else 768
            path = Path(kwargs['model_dir'])
            # 1) Load wrapper config
            cfg_path = path / CONFIG_NAME
            if not cfg_path.exists():
                raise FileNotFoundError(f"Expected {CONFIG_NAME} at {cfg_path}")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # cfg.update(override_config or {})
            embedding_mask_id = cfg['embedding_mask_id']
            # 2) Recreate a *structurally empty* wrapper to infer shapes
            #    Important: we pass base_model_name_or_path pointing to *the base subdir* we’re about to load.
            base_dir = path / BASE_SUBDIR
            if not base_dir.exists():
                # Back-compat: allow loading directly from a remote/base id in config if someone moved folders
                base_dir = cfg["base_model_name_or_path"]

            # model = cls(
            #     base_model_name_or_path=str(base_dir),
            #     num_labels=cfg["num_labels"],
            #     head_bias=cfg.get("head_bias", True),
            #     **{k: v for k, v in cfg.items() if k not in {"model_type", "version", "base_model_name_or_path", "num_labels", "head_bias"}}
            # )

            # 3) Load the base model weights via HF
            #    Use standard HF loader on the base directory. This fills 'model.base_model' in-place.
            #    We respect base_map_location if the base is huge.
            #    Note: we must rebuild base_model using .from_pretrained, not from_config:
            load_kwargs = {}
            for key in ("torch_dtype", "attn_implementation"):
                if key in kwargs and kwargs[key] is not None:
                    load_kwargs[key] = kwargs[key]
            base_model = AutoModelForCausalLM.from_pretrained(base_dir, **load_kwargs)
            # model.base_model = AutoModel.from_pretrained(
            #     str(base_dir),
            #     revision=base_revision,
            #     torch_dtype=None,  # keep original; we’ll cast dtype at the end for the whole module
            #     device_map=None,   # leave placement to our own `.to(device)` below
            # )
            model = cls(
                base_model,
                embedding_mask_id,
                connector_dim
            )

            # 4) Load head weights
            head_safe = path / HEAD_SAFE
            head_bin = path / HEAD_BIN
            if head_safe.exists():
                if not _HAVE_SAFETENSORS:
                    raise RuntimeError("Head checkpoint is safetensors but the package is not installed.")
                head_state = safe_load_file(str(head_safe), device= "cpu")
            elif head_bin.exists():
                head_state = torch.load(head_bin, map_location= "cpu")
            else:
                raise FileNotFoundError(f"Neither {HEAD_SAFE} nor {HEAD_BIN} found at {path}")

            missing, unexpected = model.node_embedding_connect.load_state_dict(head_state, strict=True)
            print("missing state: ",missing)
            print("unexpected state:", unexpected)
            # # 5) Final dtype/device placement
            # if dtype is not None:
            #     model = model.to(dtype=dtype)
            # model = model.to(device or "cpu")
            model.eval()
            return model
        elif 'base_model_dir' in kwargs.keys():
            base_model_dir = kwargs['base_model_dir']
            embedding_mask_id = kwargs['embedding_mask_id']
            connect_dir = kwargs['connect_dir'] if 'connect_dir' in kwargs.keys() else None
            LoRA_dir = kwargs['LoRA_dir'] if 'LoRA_dir' in kwargs.keys() else None
            connector_dim = kwargs['connector_dim'] if 'connector_dim' in kwargs.keys() else 768
            
            load_kwargs = {}
            for key in ("torch_dtype", "attn_implementation"):
                if key in kwargs and kwargs[key] is not None:
                    load_kwargs[key] = kwargs[key]
            base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, **load_kwargs)
            model = GraphAdapter4CausalLM(base_model,embedding_mask_id,connector_dim=connector_dim)
            if LoRA_dir is not None:
                peft_model = PeftModel.from_pretrained(model, LoRA_dir)
                model = peft_model.base_model.merge_and_unload()
            if connect_dir is not None:
                model.node_embedding_connect.load_state_dict(torch.load(connect_dir,weights_only=False,map_location='cpu').state_dict())
            return model
        else:
            raise NotImplementedError
    
    
    
    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        safe_serialization: Optional[bool] = None,
        push_config: Optional[Dict[str, Any]] = None,
    ):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1) Save your wrapper config
        cfg = self.config.to_dict()
        cfg['embedding_mask_id'] = self.embedding_mask_id
        cfg['connector_dim'] = self.node_embedding_connect[0].in_features
        if push_config:
            cfg.update(push_config)
        with open(save_dir / CONFIG_NAME, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)

        # 2) Delegate the base model saving to Hugging Face
        base_dir = save_dir / BASE_SUBDIR
        base_dir.mkdir(exist_ok=True)
        # Prefer safetensors if available unless caller forces otherwise
        use_safe = safe_serialization if safe_serialization is not None else True
        self.base_model.save_pretrained(base_dir, safe_serialization=use_safe)

        # 3) Save your head weights
        head_state = {k: v.cpu() for k, v in self.node_embedding_connect.state_dict().items()}
        if (use_safe and _HAVE_SAFETENSORS):
            safe_save_file(head_state, str(save_dir / HEAD_SAFE))
            if (save_dir / HEAD_BIN).exists():
                (save_dir / HEAD_BIN).unlink()
        else:
            torch.save(head_state, save_dir / HEAD_BIN)
            if (save_dir / HEAD_SAFE).exists():
                (save_dir / HEAD_SAFE).unlink()
