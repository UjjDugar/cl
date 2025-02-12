from datasets import load_dataset

dsn = "UjjD/tts_mini_data_1.05M"
dataset = load_dataset(dsn, split="train")

# Pad sequences to length N
N = 1800  # Example max length, adjust as needed
eos_token_id = 128256 + 4096*7 + 1  # Example EOS token ID, adjust based on your tokenizer

def pad_sequence(example):
    input_ids = example['input_ids']
    attention_mask = example['attention_mask']
    labels = example['labels']
    
    # Calculate padding length
    pad_len = N - len(input_ids)
    
    if pad_len > 0:
        # Pad input_ids with eos_token_id
        input_ids = input_ids + [eos_token_id] * pad_len
        
        # Pad attention_mask with 0s
        attention_mask = attention_mask + [0] * pad_len
        
        # Pad labels with -100
        labels = labels + [-100] * pad_len
        
    # Truncate if longer than N
    input_ids = input_ids[:N]
    attention_mask = attention_mask[:N]
    labels = labels[:N]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# Apply padding to dataset
dataset = dataset.map(pad_sequence)

from huggingface_hub import login
login(token="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")

# Push padded dataset to Hub
dataset.push_to_hub("UjjD/tts_dataset_1.5M_padded", private=False)

print(f"Padded dataset with {len(dataset)} examples uploaded successfully")
print("Dataset columns:", dataset.column_names)
print("Sample from padded dataset:", dataset[0])


