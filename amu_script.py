from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import wandb

dsn = "UjjD/tts_mini_data"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

num_add_tokens = 4096*7 + 2

model.resize_token_embeddings(model.config.vocab_size + num_add_tokens)

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset(dsn, split="train")
dataset = dataset.shuffle(seed=42)

# Initialize wandb with project name and run name
wandb.init(
    project="motiontrainlr",  # Project name in wandb
    name="llama-finetuning",  # Name of this training run
    config={  # Track hyperparameters
        "model_name": model_name,
        "dataset": dsn,
        "learning_rate": 9e-4,
        "epochs": 1,
        "batch_size": 1
    }
)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=1,
    output_dir="./output",
    per_device_train_batch_size=1,
    logging_steps=1,
    bf16=True,
    report_to="wandb",  # Enable wandb logging
    save_steps=8374,
    remove_unused_columns=True,
    lr_scheduler_type="cosine"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Close wandb run when training is complete
wandb.finish()