import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import AutoModelForCausalLMWithValueHead, SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# 🔹 Load Base Model (GPT-2 or Your Custom Model)
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Change this to your custom model path if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = "<|finetune_right_pad_id|>"

base_model = AutoModelForCausalLM.from_pretrained(model_name)

# 🔹 Configure LoRA
peft_config = LoraConfig(
    r=8,                  # Rank
    lora_alpha=16,        # Scaling factor
    lora_dropout=0.1,     # Dropout probability
    bias="none",          # No bias tuning
    task_type="CAUSAL_LM" # For autoregressive models like GPT
)
peft_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model, peft_config=peft_config)
# 🔹 Convert to LoRA Model
# peft_model = get_peft_model(model, peft_config)
peft_model.pretrained_model.print_trainable_parameters()  # Check trainable params

# 🔹 Load Dataset (Example: IMDB Reviews)
dataset = load_dataset("imdb", split="train[:100]")  # Small subset for quick testing

# 🔹 Format Dataset for SFT
def format_data(example):
    return {
        "prompt": example["text"],  # Input query (prompt)
        "chosen": "Positive review." if example["label"] == 1 else "Negative review.",  # Correct response
    }

dataset = dataset.map(format_data)
peft_model.save_pretrained("sft_trainer_output/test_2")  # Save the model
# 🔹 Training Arguments
training_args = TrainingArguments(
    output_dir="./sft_trainer_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",  # Use "wandb" for logging if needed
    fp16=True,  # Enable mixed precision for faster training
    eval_strategy='steps',
    eval_steps=20,
)

# 🔹 Initialize `SFTTrainer`
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    eval_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    tokenizer=tokenizer,
)

# 🔹 Start Training
trainer.train()

print("✅ Supervised Fine-Tuning with LoRA Complete!")
