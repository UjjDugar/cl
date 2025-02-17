from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import wandb

dsn = "UjjD/tts_dataset_1.05M_padded_text_labels_on"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")

num_add_tokens = 4096*7 + 2

model.resize_token_embeddings(model.config.vocab_size + num_add_tokens)

tokenizer = AutoTokenizer.from_pretrained(model_name)

learning_rate = 5e-5
epochs = 1
batch_size = 4

dataset = load_dataset(dsn, split="train")
dataset = dataset.shuffle(seed=42)

# Initialize wandb with project name and run name
wandb.init(
    project="tts-amu",  # Project name in wandb
    name="ujj-original",  # Name of this training run
    config={  # Track hyperparameters
        "model_name": model_name,
        "dataset": dsn,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size
    }
)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    output_dir="./output",
    per_device_train_batch_size=batch_size,
    logging_steps=1,
    bf16=True,
    learning_rate=learning_rate,
    fsdp="auto_wrap",
    # fsdp="auto_wrap",
    report_to="wandb",  # Enable wandb logging
    save_steps=1000,
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