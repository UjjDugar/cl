from datasets import load_dataset
from huggingface_hub import HfApi

# Load the original dataset
dataset = load_dataset("UjjD/tts_dataset_1.05M_padded", split="train")

# Take a small subset (e.g., first 1000 examples)
mini_dataset = dataset.select(range(10))

# Push the mini dataset to hub
mini_dataset.push_to_hub("UjjD/tts_minidataset_padded")

print("Mini dataset created and uploaded successfully!")
