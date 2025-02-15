from datasets import load_dataset
from huggingface_hub import HfApi
import numpy as np

# Load the dataset
dataset = load_dataset("UjjD/tts_minidataset_padded", split="train")

def process_labels(example):
    # Convert to numpy arrays for easier manipulation
    input_ids = np.array(example['input_ids'])
    labels = np.array(example['labels'])
    
    # Find the first non -100 index
    first_valid_idx = np.where(labels != -100)[0][0]
    
    # Replace leading -100s with corresponding input_ids
    labels[:first_valid_idx] = input_ids[:first_valid_idx]
    
    # Convert back to list
    example['labels'] = labels.tolist()
    return example

# Apply the processing
processed_dataset = dataset.map(process_labels)

# Push to hub
processed_dataset.push_to_hub("UjjD/tts_minidataset_padded_text_labels_on")

print("Dataset processed and uploaded successfully!")
