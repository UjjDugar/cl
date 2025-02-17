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

config_file = "PRETRAIN_ARGS-3b-10m.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

# dsn1 = config["text_QA_dataset"]
dsn2 = config["TTS_dataset_ujj"]

model_name = config["model_name"]
tokenizer_name = config["tokenizer_name"]

run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]

epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]

base_repo_id = 'checkpoints'

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


def data_collator(features):
    # max_length = 6144
    input_ids = [f["input_ids"] for f in features]

    if any("attention_mask" not in f for f in features):
        attention_mask = [[1]*len(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"] for f in features]

    if any("labels" not in f for f in features):
        labels = input_ids
    else:
        labels = [f["labels"] for f in features]


    # input_ids = [ids[:max_length] for ids in input_ids]
    # attention_mask = [m[:max_length] for m in attention_mask]
    # labels = [l[:max_length] for l in labels]

    # Convert all lists to tensors and pad
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        i, dtype=torch.long) for i in input_ids], batch_first=True, padding_value=pad_token)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        m, dtype=torch.long) for m in attention_mask], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        l, dtype=torch.long) for l in labels], batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}



dsn = "UjjD/tts_dataset_1.05M_padded_text_labels_on"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")

num_add_tokens = 4096*7 + 2

model.resize_token_embeddings(model.config.vocab_size + num_add_tokens)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# learning_rate = 5e-5
# epochs = 1
# batch_size = 4

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
    fsdp="auto_wrap", # This is a way of splitting the model into multiple GPUs. Data efficient
    report_to="wandb",
    save_steps=save_steps,
    remove_unused_columns=True,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine"
)

trainer = FSDPTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

# Close wandb run when training is complete
wandb.finish()