from datasets import load_dataset
from huggingface_hub import HfApi
import numpy as np

# Load the dataset
dataset = load_dataset("UjjD/tts_minidataset_padded_text_labels_on", split="train")

# EOS token
eos_token = 128256 + 4096*7 + 1
target_size = 1000  # Setting a fixed size for new rows

def remove_padding_and_concatenate(examples, batch_size=1000):
    # Initialize lists to store processed sequences
    all_input_ids = []
    all_labels = []
    all_attention_mask = []
    
    # Temporary buffers for concatenation
    current_input_ids = []
    current_labels = []
    current_attention_mask = []
    
    # Process each example in the batch
    for i in range(len(examples['input_ids'])):
        input_ids = examples['input_ids'][i]
        labels = examples['labels'][i]
        attention_mask = examples['attention_mask'][i]
        
        # Find last non-padding token (keeping one EOS token)
        last_non_pad = len(input_ids) - 1
        while last_non_pad > 0 and input_ids[last_non_pad-1] == eos_token:
            last_non_pad -= 1
            
        # Add trimmed sequences to current buffers
        current_input_ids.extend(input_ids[:last_non_pad+1])
        current_labels.extend(labels[:last_non_pad+1])
        current_attention_mask.extend(attention_mask[:last_non_pad+1])
        
        # Create new rows when we have enough tokens
        while len(current_input_ids) >= target_size:
            all_input_ids.append(current_input_ids[:target_size])
            all_labels.append(current_labels[:target_size])
            all_attention_mask.append(current_attention_mask[:target_size])
            
            current_input_ids = current_input_ids[target_size:]
            current_labels = current_labels[target_size:]
            current_attention_mask = current_attention_mask[target_size:]
    
    # Handle remaining tokens if any
    if current_input_ids:
        # Pad the last row to target_size if needed
        if len(current_input_ids) < target_size:
            padding_length = target_size - len(current_input_ids)
            current_input_ids.extend([eos_token] * padding_length)
            current_labels.extend([-100] * padding_length)
            current_attention_mask.extend([0] * padding_length)
        all_input_ids.append(current_input_ids)
        all_labels.append(current_labels)
        all_attention_mask.append(current_attention_mask)
    
    return {
        'input_ids': all_input_ids,
        'labels': all_labels,
        'attention_mask': all_attention_mask
    }

# Process the dataset
processed_dataset = dataset.map(
    remove_padding_and_concatenate,
    batched=True,
    batch_size=1000,
    remove_columns=dataset.column_names
)

# Push to hub
processed_dataset.push_to_hub("UjjD/tts_minidataset_unpadded_text_labels_on_amu_style")

print("Dataset processed and uploaded successfully!")
