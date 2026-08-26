from transformers import TrainerCallback
import torch
import os,json
from models.LLaMa import LLama4GraphWithValueHead
from peft import PeftModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from models.graphAdapter import HEAD_BIN, HEAD_SAFE
try:
    from safetensors.torch import save_file as safe_save_file, load_file as safe_load_file
    _HAVE_SAFETENSORS = True
except Exception:
    _HAVE_SAFETENSORS = False
class SaveTrainerCallBack(TrainerCallback):
    
    def __init__(self):
        self.best_eval_metric = -100
        super().__init__()
    
    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            current_steps = state.global_step
            model = kwargs['model']
            current_eval_metric = kwargs['metrics']['eval_loss']
            output_dir = args.output_dir
            save_dir = os.path.join(output_dir,'checkpoint-{}'.format(current_steps))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(model.node_embedding_connect,save_dir+'/node_embedding_connect.pt')
            if current_eval_metric > self.best_eval_metric:
                
                self.best_eval_metric = current_eval_metric
                best_checkpoint_dir =  os.path.join(output_dir,'best-checkpoint')
                if not os.path.exists(best_checkpoint_dir):
                    os.mkdir(best_checkpoint_dir)
                torch.save(model.node_embedding_connect.state_dict(),best_checkpoint_dir+'/node_embedding_connect.pt')  
                best_infos = {"best_eval_metric":current_eval_metric,"step":current_steps}
                with open(best_checkpoint_dir+'/best_infos.json', 'w') as fp:
                    json.dump(best_infos,fp)      
        return super().on_evaluate(args, state, control, **kwargs)

class PredictionSaveTrainerCallBack(TrainerCallback):
    
    def __init__(self,combine_training=False):
        self.best_eval_metric = float('inf')
        self.combine_training=combine_training
        super().__init__()
        
    def _get_wrapper(self, model):
        # 如果外面包了一层 PeftModel，真正的 GraphAdapter4CausalLM 在 base_model 里
        if isinstance(model, PeftModel):
            return model.base_model
        return model

    def on_save(self, args, state, control, **kwargs):
        model = kwargs['model']

        # 1) 推断当前 checkpoint 目录
        # HuggingFace 默认是 output_dir/checkpoint-XXXX
        ckpt_dir = os.path.join(
            args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}",
        )
        os.makedirs(ckpt_dir, exist_ok=True)

        # 2) 拿到 wrapper 和 node_embedding_connect
        wrapper = self._get_wrapper(model)
        connect = wrapper.node_embedding_connect

        # 3) 根据 requires_grad 决定要不要存
        should_save_head = any(p.requires_grad for p in connect.parameters())

        if not should_save_head:
            print("[SaveConnectHeadCallback] node_embedding_connect frozen, skip saving head.")
            return control  # ❗ 不改 control.should_save，Trainer 继续执行默认 save

        print("[SaveConnectHeadCallback] node_embedding_connect is trainable, saving head...")
        head_state = {k: v.detach().cpu() for k, v in connect.state_dict().items()}

        # 你可以沿用之前的 HEAD_SAFE/HEAD_BIN 约定，这里用 bin 举例
        # torch.save(head_state, os.path.join(ckpt_dir, HEAD_BIN))
        # 或者 safetensors:
        if _HAVE_SAFETENSORS:
            safe_save_file(head_state, os.path.join(ckpt_dir, HEAD_SAFE))
        else:
            torch.save(head_state, os.path.join(ckpt_dir, HEAD_BIN))

        # 4) 不动 control.should_save，Trainer 原始 save_model 逻辑照常执行
        return control
    
    
    # def on_evaluate(self, args, state, control, **kwargs):
    #     if state.is_world_process_zero:
    #         current_steps = state.global_step
    #         model = kwargs['model']
    #         current_eval_metric = kwargs['metrics']['eval_loss']
    #         output_dir = args.output_dir
    #         save_dir = os.path.join(output_dir,'checkpoint-{}'.format(current_steps))
    #         if not os.path.exists(save_dir):
    #             os.mkdir(save_dir)
    #         if isinstance(model,LLama4GraphWithValueHead):
    #             model.save_pretrained(save_dir,save_embedding_layers=True)
    #         else:
    #             model.save_pretrained(save_dir+'/lora_adapter',save_embedding_layers=True)
    #         if self.combine_training:
    #             torch.save(model.node_embedding_connect,save_dir+'/node_embedding_connect.pt')
    #         if current_eval_metric < self.best_eval_metric:
                
    #             self.best_eval_metric = current_eval_metric
    #             best_checkpoint_dir =  os.path.join(output_dir,'best-checkpoint')
    #             if not os.path.exists(best_checkpoint_dir):
    #                 os.mkdir(best_checkpoint_dir)
    #             if isinstance(model,LLama4GraphWithValueHead):
    #                 model.save_pretrained(best_checkpoint_dir, save_embedding_layers=True)
    #             else:
    #                 model.save_pretrained(best_checkpoint_dir+'/lora_adapter',save_embedding_layers=True)
    #             best_infos = {"best_eval_metric":current_eval_metric,"step":current_steps}
    #             with open(best_checkpoint_dir+'/best_infos.json', 'w') as fp:
    #                 json.dump(best_infos,fp)      
    #     return super().on_evaluate(args, state, control, **kwargs)
    
    # def on_log(self, args, state, control, **kwargs):
    #     if state.is_world_process_zero:
    #         print(kwargs.keys())
    #         current_steps = state.global_step
    #         model = kwargs['model']
    #         output_dir = args.output_dir
    #         save_dir = os.path.join(output_dir,'checkpoint-{}'.format(current_steps))
    #         if not os.path.exists(save_dir):
    #             os.mkdir(save_dir)
    #         if isinstance(model,LLama4GraphWithValueHead):
    #             model.save_pretrained(save_dir,save_embedding_layers=True)
    #         else:
    #             model.policy.save_pretrained(save_dir+'/lora_adapter',save_embedding_layers=False)
    #     return super().on_log(args, state, control, **kwargs)