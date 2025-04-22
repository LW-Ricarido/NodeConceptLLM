import wandb
import torch
from functools import wraps

def add_wandb_generation_logging(trainer_instance):
    
    original_log_method = trainer_instance.log
    
    @wraps(original_log_method)
    def wrapped_log(logs,start_time=None):
        original_log_method(logs,start_time)
        import ipdb; ipdb.set_trace()
    trainer_instance.log = wrapped_log
    return trainer_instance