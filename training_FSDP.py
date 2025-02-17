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

class AlternatingDistributedSampler(DistributedSampler):
    def _init_(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super()._init_(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.shuffle = shuffle

    def _iter_(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


class FSDPTrainer(Trainer):
    def _init_(self, *args, **kwargs):
        super()._init_(*args, **kwargs)
        self.repo_id = base_repo_id
        self.api = HfApi()

    def get_train_dataloader(self):

        # Sampler is used to split the data across multiple GPUs
        # By default, shuffle is true, which means indices are shuffled. This needs to turned off for the alternating dataset
        # Not important for the TTS dataset
        sampler = AlternatingDistributedSampler(
            self.train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=False,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )

    # def log(self, logs, start_time=None):
    #     super().log(logs, start_time)
    #     if self.is_world_process_zero():
    #         global_step = self.state.global_step
    #         if global_step % 2 == 0:
    #             wandb.log({"text_loss": logs["loss"], "step": global_step})
    #         else:
    #             wandb.log({"audio_loss": logs["loss"], "step": global_step})

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        self.save_and_push_model(output_dir)

    def save_and_push_model(self, output_dir):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state_dict = self.model.state_dict()
        self.model.save_pretrained(output_dir, state_dict=cpu_state_dict)

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
# dataset = dataset.shuffle(seed=42)

# Initialize wandb with project name and run name
wandb.init(
    project="tts-amu",  # Project name in wandb
    name="ujj-original-unshuffled",  # Name of this training run
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