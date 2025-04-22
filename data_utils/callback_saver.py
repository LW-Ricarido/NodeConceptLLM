from transformers import TrainerCallback
import torch
import os,json
from models.LLaMa import LLama4GraphWithValueHead
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
    def __init__(self):
        self.best_eval_metric = -100
        super().__init__()
    
    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            current_steps = state.global_step
            model = kwargs['model']
            current_eval_metric = kwargs['metrics']['eval_prediction_acc']
            output_dir = args.output_dir
            save_dir = os.path.join(output_dir,'checkpoint-{}'.format(current_steps))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if isinstance(model,LLama4GraphWithValueHead):
                model.save_pretrained(save_dir,save_embedding_layers=True)
            else:
                model.save_pretrained(save_dir+'/lora_adapter',save_embedding_layers=True)
            if current_eval_metric > self.best_eval_metric:
                
                self.best_eval_metric = current_eval_metric
                best_checkpoint_dir =  os.path.join(output_dir,'best-checkpoint')
                if not os.path.exists(best_checkpoint_dir):
                    os.mkdir(best_checkpoint_dir)
                if isinstance(model,LLama4GraphWithValueHead):
                    model.save_pretrained(best_checkpoint_dir, save_embedding_layers=True)
                else:
                    model.save_pretrained(best_checkpoint_dir+'/lora_adapter',save_embedding_layers=True)
                best_infos = {"best_eval_metric":current_eval_metric,"step":current_steps}
                with open(best_checkpoint_dir+'/best_infos.json', 'w') as fp:
                    json.dump(best_infos,fp)      
        return super().on_evaluate(args, state, control, **kwargs)