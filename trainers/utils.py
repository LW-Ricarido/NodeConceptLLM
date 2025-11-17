import types
import torch
def wrap_trainer_no_eval_loss(trainer):
    
    """
    DDP + accelerate 场景下的低显存 eval 补丁：

      - 训练阶段：完全不改
      - eval / predict：
          * 不算 loss（不把 labels 传进 model）
          * logits 在每个 step 后搬到 CPU
          * labels 留在 CUDA，上层的 pad_across_processes 正常跑
          * pad_across_processes 对 CPU tensor（logits）直接原样返回
          * gather_function 是恒等映射，不再 all_gather logits
    """

    # 保存原始方法
    orig_prediction_step = trainer.prediction_step
    orig_pad_across_processes = trainer.accelerator.pad_across_processes

    def prediction_step_no_loss_cpu(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # ===== 训练阶段：保持原样 =====
        if model.training:
            return orig_prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        # ===== eval / predict 阶段 =====

        # 1. 用原来的逻辑准备输入（包括把东西搬到 GPU）
        inputs = self._prepare_inputs(inputs)

        # 2. 从 inputs 里拿出 labels（此时在 CUDA 上）
        labels = None
        if getattr(self, "label_names", None):
            collected = []
            for name in self.label_names:
                if name in inputs:
                    collected.append(inputs.pop(name))
            if len(collected) == 1:
                labels = collected[0]
            elif len(collected) > 1:
                labels = tuple(collected)

        # 3. 只做前向，不算 loss（因为没 labels 了）
        with torch.no_grad():
            outputs = model(**inputs)

        # 4. 拿 logits
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs[0]

        # 5. logits 搬到 CPU，释放 GPU 显存
        logits = logits.detach().cpu()

        # ⚠ labels 不要 .cpu()，保持 CUDA，
        #   让上层的 pad_across_processes(labels, ...) 正常执行
        return (None, logits, labels)

    # 6. 覆盖 prediction_step
    trainer.prediction_step = types.MethodType(prediction_step_no_loss_cpu, trainer)

    # 7. 包一层 pad_across_processes：
    #    - CUDA tensor：调用原实现（比如 labels）
    #    - 非 CUDA（CPU logits）：直接原样返回，避免报错 & 避免多余操作
    def pad_across_processes_safe(tensor, *args, **kwargs):
        if isinstance(tensor, torch.Tensor) and tensor.device.type != "cuda":
            return tensor
        return orig_pad_across_processes(tensor, *args, **kwargs)

    trainer.accelerator.pad_across_processes = pad_across_processes_safe

    # 8. logits 不再 all_gather，避免占显存
    trainer.gather_function = lambda x: x

    # 9. 防止一次性在 CPU 堆太多 logits，可以适当设置 eval_accumulation_steps
    if getattr(trainer, "args", None) is not None:
        if not trainer.args.eval_accumulation_steps:
            trainer.args.eval_accumulation_steps = 16

    return trainer
    