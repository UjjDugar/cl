import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print('Length of tokenizer before adding new special token: ', len(tokenizer))

# Define the special split token
split_token = "|<split>|"
new_special_tokens = [split_token]
tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})

# Load model and resize embeddings for the new token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to accommodate new tokens

# Save the updated tokenizer
tokenizer.save_pretrained("updated_tokenizer")

print('Length of tokenizer after adding new special token: ', len(tokenizer))

# Get the token ID of the split token
id_of_split_token = tokenizer.convert_tokens_to_ids(split_token)
print("ID of split token: ", id_of_split_token)

# Load the CSV file
df = pd.read_csv("kafka_qa_pairs.csv")

# Function to tokenize the data and mask question tokens + split token
def tokenize_function(example):
    """Tokenizes question-answer pairs with a split token and masks question & split token labels."""
    
    # Create the formatted prompt
    question = f"Question: {example['Question']}\n"
    split = f" {split_token} "  # Space before and after to ensure tokenization works correctly
    answer = f"Answer: {example['Answer']}"
    full_text = question + split + answer  # Insert split token between question and answer

    # Tokenize full text
    tokenized = tokenizer(full_text, truncation=True, return_tensors="pt")

    # Tokenize only the question to determine length
    question_tokenized = tokenizer(question, truncation=True, return_tensors="pt")

    # Find the position of the split token
    split_token_pos = question_tokenized["input_ids"].shape[1]  # Split token appears right after the question

    # Create labels with -100 for question and split token
    labels = tokenized["input_ids"].clone()
    labels[:, :split_token_pos] = -100  # Mask question tokens
    labels[:, split_token_pos] = -100  # Mask split token

    return {
        "input_ids": tokenized["input_ids"].squeeze(),
        "attention_mask": tokenized["attention_mask"].squeeze(),
        "labels": labels.squeeze(),
    }

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, remove_columns=["Question", "Answer"])

test_token_ids = [ 128000, 14924, 25, 8595, 656, 499, 1093, 279, 21562, 14, 29588, 388, 430, 499, 1093, 5380, 220, 128256, 22559, 25, 77273, 25, 578, 52909, 315, 34453, 323, 94603, 27121, 430, 2633, 856, 18273, 19882, 527, 439, 9662, 73555, 481, 439, 279, 8002, 12922, 315, 279, 80495, 5780, 13, 358, 1505, 7182, 34678, 416, 2915, 15107, 311, 3738, 21562, 323, 45518, 11, 1790, 1093, 1268, 832, 374, 34678, 416, 2915, 4675, 77, 1636, 304, 279, 3566, 315, 459, 682, 21430, 31929, 287, 64931, 13, 1102, 374, 264, 99810, 483, 11879, 315, 20356, 323, 21063, 11, 1405, 279, 14906, 315, 5222, 323, 37392, 68608, 3871, 311, 1893, 264, 8903, 430, 374, 2225, 77754, 323, 6366, 13, 4702, 439, 832, 374, 12153, 311, 12731, 279, 60135, 34477, 315, 64931, 11, 779, 2288, 1097, 358, 12153, 311, 22884, 279, 274, 47435, 1650, 315, 3738, 21562, 323, 45518, 13, 578, 3249, 315, 433, 682, 8625, 264, 23347, 11, 28016, 5655, 2949, 279, 47862, 288, 315, 856, 84951, 11, 8748, 311, 387, 75648, 555, 279, 9439, 3177, 315, 26609, 16463, 13 ]
decoded_text = tokenizer.decode(test_token_ids, skip_special_tokens=True)
print("Decoded text:", decoded_text)

# Save locally and push to Hugging Face
dataset_name = "UjjD/kafka-qa-dataset"
# tokenized_dataset.push_to_hub(dataset_name)
