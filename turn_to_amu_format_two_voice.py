input_speech_to_input_text_token_id = 128256 + 4096*7
input_text_to_output_speech_token_id = 128256 + 4096*7 + 1
end_of_output_speech_token_id = 128256 + 4096*7 + 2

from datasets import load_dataset
from huggingface_hub import HfApi
import numpy as np

# Download dataset
dataset = load_dataset("UjjD/zuck_and_luna_zero_shot", split="train")

# Initialize lists to store processed data
all_input_ids = []
all_labels = []
all_attention_masks = []

# Process each row in dataset
for row in dataset:
    input_ids = np.array(row['input_ids'])
    
    # Find indices of special tokens
    input_speech_end = np.where(input_ids == input_speech_to_input_text_token_id)[0][0]
    input_text_end = np.where(input_ids == input_text_to_output_speech_token_id)[0][0]
    output_speech_end = np.where(input_ids == end_of_output_speech_token_id)[0][0]

    # Extract segments
    input_speech = input_ids[:input_speech_end]
    input_text = input_ids[input_speech_end + 1:input_text_end]
    output_speech = input_ids[input_text_end + 1:output_speech_end]

    # Add 10 to input_speech tokens
    input_speech_updated = input_speech + 10
    output_speech_updated = output_speech + 10

    # Recombine with special tokens
    input_ids_updated = np.concatenate([
        input_speech_updated,
        [128256],
        input_text,
        [128257],
        output_speech_updated,
        [128258]
    ])

    # Create labels (-100 for input tokens, actual values for output tokens)
    labels = np.full_like(input_ids_updated, -100)
    labels[input_text_end+1:] = input_ids_updated[input_text_end+1:] # Set output speech and end token

    # Create attention mask (all 1s)
    attention_mask = np.ones_like(input_ids_updated)

    # Append to lists
    all_input_ids.append(input_ids_updated)
    all_labels.append(labels)
    all_attention_masks.append(attention_mask)

# Create new dataset with all processed rows
new_dataset = {
    'input_ids': all_input_ids,
    'labels': all_labels,
    'attention_mask': all_attention_masks
}

# Convert to HuggingFace Dataset
from datasets import Dataset
hf_dataset = Dataset.from_dict(new_dataset)

# Push to hub
hf_dataset.push_to_hub("UjjD/zuck_and_luna_zero_shot.amu_format")


