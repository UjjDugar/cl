from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import wandb
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import yaml
import wandb
from huggingface_hub import HfApi

config_file = "zero_shot_voice_cloning.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["training_dataset"]
model_name = config["model_name"]
tokenizer_name = config["tokenizer_name"]

run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]

epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
# pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]

model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)

num_add_tokens = 4096*7 + 10
model.resize_token_embeddings(model.config.vocab_size + num_add_tokens)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

dataset = load_dataset(dsn, split="train")
dataset = dataset.shuffle(seed=42)

# Initialize wandb with project name and run name
wandb.init(project=project_name,name=run_name)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    logging_steps=1,
    fp16=True,
    output_dir=f"./{base_repo_id}",
    fsdp="auto_wrap",
    report_to="wandb",
    save_steps=save_steps,
    remove_unused_columns=True,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine"
)

# trainer = FSDPTrainer(
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

wandb.finish()