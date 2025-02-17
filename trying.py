# Main tokenization loop and dataset creation
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import numpy as np
from tqdm import tqdm 
import torch, torchaudio
from datasets import load_dataset
from snac import SNAC

def convert_waveform_to_tokens(waveform, verbose=False):
    # waveform is 24k and the output of torchaudio.functional.resample(waveform, sr, 24000)
    waveform = waveform.cuda()
    with torch.inference_mode():
        codes = tokenizing_model.encode(waveform.unsqueeze(0))
        audio_hat = tokenizing_model.decode(codes)

    codes = [codes[i].cpu().numpy().tolist()[0] for i in range(len(codes))]

    all_frames_tokens_list = []
    for i in range(len(codes[0])):
        # Note the order
        frame_tokens = [codes[0][i],]
        frame_tokens.append(codes[1][2*i])
        frame_tokens.extend(codes[2][4*i:4*i+2])
        frame_tokens.append(codes[1][2*i+1])
        frame_tokens.extend(codes[2][4*i+2:4*i+4])

        all_frames_tokens_list.append(frame_tokens)

    if verbose:
        print("First 3 frames of tokens, without offset: ", all_frames_tokens_list[:3])

    offsetted_tokens_list = [ [128256+l[i]+4096*i for i in range(len(l)) ] for l in all_frames_tokens_list]
    offsetted_tokens_list = np.array(offsetted_tokens_list).flatten().tolist()

    return offsetted_tokens_list, audio_hat

def get_hf_formatted_data_for_tts(speech_tokens, transcript_tokens, tts_token_id, eos_token_id):
    input_ids = transcript_tokens['input_ids'] + [tts_token_id] + speech_tokens + [eos_token_id]
    labels = [-100] * (len(transcript_tokens['input_ids']) + 1) + speech_tokens + [eos_token_id]
    attention_mask = [1] * len(input_ids)
    return input_ids, labels, attention_mask

tokenizing_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

model_name = "meta-llama/Llama-3.2-3B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("amuvarma/all_zuck_audio", split="train")

tts_token_id = 128256 + 4096*7 
eos_token_id = 128256 + 4096*7 + 1

# Lists to store processed data
all_input_ids = []
all_labels = []
all_attention_masks = []

# Sorting the dataset by audio length (longest first)
audio_lengths = [len(row['audio']['array']) for row in dataset]
sorted_indices = sorted(range(len(audio_lengths)), key=lambda k: audio_lengths[k], reverse=True)
dataset = dataset.select(sorted_indices)

# Calculate duration in seconds for the first audio file
first_row = dataset[0]
first_audio = first_row['audio']
duration_seconds = len(first_audio['array']) / first_audio['sampling_rate']
print(f"Duration of first audio file: {duration_seconds:.2f} seconds")

'''
for i in tqdm(range(len(dataset))):
    row = dataset[i]
    
    transcript = row['transcript']
    transcript_tokens = llama_tokenizer(transcript)
    
    audio_data = row['audio']
    waveform, sr = audio_data['array'], audio_data['sampling_rate']

    # Resample to 24kHz
    waveform = torch.from_numpy(waveform).float()
    waveform = waveform.unsqueeze(0)  # Add channel dimension [1, samples]
    waveform = torchaudio.functional.resample(waveform, sr, 24000)

    speech_tokens, audio_hat = convert_waveform_to_tokens(waveform)

    input_ids, labels, attention_mask = get_hf_formatted_data(speech_tokens, transcript_tokens, tts_token_id, eos_token_id)
    
    all_input_ids.append(input_ids)
    all_labels.append(labels)
    all_attention_masks.append(attention_mask)

# Create dataset dictionary
dataset_dict = {
    'input_ids': all_input_ids,
    'labels': all_labels,
    'attention_mask': all_attention_masks
}
'''

# # Convert to HuggingFace Dataset
# hf_dataset = Dataset.from_dict(dataset_dict)

# # Push to hub (uncomment and modify repo_id as needed)
# hf_dataset.push_to_hub(f"UjjD/tts_mini_data")
