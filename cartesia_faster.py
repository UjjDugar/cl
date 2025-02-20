import os
from tqdm import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets
import soundfile as sf
import concurrent.futures

# Create speech_data directory if it doesn't exist
os.makedirs('speech_data', exist_ok=True)

# Load sentences
with open('sentences_clean.txt', 'r') as f:
    sentences = [s.strip() for s in f.readlines()]

from cartesia import Cartesia
import os

client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))

# Get all available voices
voices = client.voices.list()
#print(voices)

# Extract voice IDs into a list
voice_ids = [voice["id"] for voice in voices]

import os
import subprocess
from cartesia import Cartesia
from dotenv import load_dotenv

def text_to_speech_cartesia(voice_id, text, output_filename):

    load_dotenv()

    if os.environ.get("CARTESIA_API_KEY") is None:
        raise ValueError("CARTESIA_API_KEY is not set")

    client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))

    data = client.tts.bytes(
        model_id="sonic",
        transcript=text,
        voice_id=voice_id,
        output_format={
            "container": "wav",
            "encoding": "pcm_f32le",
            "sample_rate": 44100,
        },
    )

    with open(output_filename, "wb") as f:
        f.write(data)

def process_voice(i, voice_id):
    try:
        # Calculate indices for sentences
        long_sentence = sentences[50 + i]
        short_sentence = sentences[-(51 + i)]
        
        # Set up file paths
        long_path = os.path.join('speech_data', f"voice{50+i}_long.wav")
        short_path = os.path.join('speech_data', f"voice{50+i}_short.wav")
        
        # Generate speech for both sentences
        text_to_speech_cartesia(voice_id, long_sentence, long_path)
        text_to_speech_cartesia(voice_id, short_sentence, short_path)
        
        # Read audio files
        long_audio, sr_long = sf.read(long_path)
        short_audio, sr_short = sf.read(short_path)
        
        # Return a dict with the processed data
        return {
            'voice_id': 50 + i + 1,  # Start voice IDs from 51
            'long_sentence': long_sentence,
            'long_sentence_audio': long_audio,
            'short_sentence': short_sentence,
            'short_sentence_audio': short_audio,
            'sr': sr_long  # Assuming both have the same sample rate
        }
    except Exception as e:
        print(f"Error processing voice {voice_id}: {e}")
        return None

# Use ThreadPoolExecutor to process voices in parallel
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    # Submit tasks for all voices
    futures = {executor.submit(process_voice, i, voice_id): voice_id for i, voice_id in enumerate(voice_ids)}
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(voice_ids), desc="Generating speech"):
        result = future.result()
        if result is not None:
            results.append(result)

# Separate the collected data into lists for dataset creation
voice_ids_list = [r['voice_id'] for r in results]
long_sentences = [r['long_sentence'] for r in results]
long_sentence_audios = [r['long_sentence_audio'] for r in results]
short_sentences = [r['short_sentence'] for r in results]
short_sentence_audios = [r['short_sentence_audio'] for r in results]
sample_rates = [r['sr'] for r in results]

# Create dataset with the collected data
new_dataset_dict = {
    'voice_id': voice_ids_list,
    'long_sentence': long_sentences,
    'long_sentence_audio': long_sentence_audios,
    'short_sentence': short_sentences, 
    'short_sentence_audio': short_sentence_audios,
    'sr': sample_rates
}

new_dataset = Dataset.from_dict(new_dataset_dict)

# Load the existing dataset and concatenate
existing_dataset = load_dataset("UjjD/zero_shot")["train"]
combined_dataset = concatenate_datasets([existing_dataset, new_dataset])
combined_dataset.push_to_hub("UjjD/zero_shot")
