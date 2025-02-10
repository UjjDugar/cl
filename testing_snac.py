import torch
from snac import SNAC
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

# Load the .wav file
file_path = "speech.wav"
sample_rate, audio_data = wavfile.read(file_path)

# Convert to torch tensor and resample to 24kHz
audio_tensor = torch.from_numpy(audio_data).float()
if sample_rate != 24000:
    resampler = torch.nn.functional.interpolate
    # Reshape for resampling (batch, channels, time)
    audio_tensor = audio_tensor.view(1, 1, -1)
    # Calculate new length for 24kHz
    new_length = int(len(audio_data) * 24000 / sample_rate)
    audio_data = resampler(audio_tensor, size=new_length, mode='linear').squeeze().numpy()
    sample_rate = 24000
wavfile.write('24k_speech.wav', sample_rate, audio_data.astype(np.int16))


duration = len(audio_data) / sample_rate
print(f'sample rate: {sample_rate}, duration: {duration}, length of audio data: {len(audio_data)}')
time = np.linspace(0., duration, len(audio_data))
# plt.figure(figsize=(10, 4)); plt.plot(time, audio_data, label="Audio Waveform", color='blue'); plt.xlabel("Time [s]"); plt.ylabel("Amplitude"); plt.title("Waveform of the Audio File"); plt.legend(); plt.grid(); plt.savefig('waveform.png'); plt.close()

audio = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0).float().cuda()
with torch.inference_mode():
    codes = model.encode(audio)
    audio_hat = model.decode(codes)

print(audio.shape)
print(audio_hat.shape)

# Convert the reconstructed audio tensor to numpy and save as wav file
reconstructed_audio = audio_hat.cpu().squeeze().numpy()
reconstructed_audio = reconstructed_audio * (32767 / np.max(np.abs(reconstructed_audio)))
wavfile.write('reconst_speech.wav', sample_rate, reconstructed_audio.astype(np.int16))

# Convert tensors back to numpy arrays and remove extra dimensions
audio_np = audio.cpu().numpy().squeeze()
audio_hat_np = audio_hat.cpu().numpy().squeeze()
# Create time arrays for both signals
time_orig = np.linspace(0, duration, len(audio_np))
time_hat = np.linspace(0, duration * (len(audio_hat_np)/len(audio_np)), len(audio_hat_np))
# Create subplot for comparison
plt.figure(figsize=(15, 6)); plt.subplot(2, 1, 1); plt.plot(time_orig, audio_np, label='Original Audio', color='blue'); plt.xlabel('Time [s]'); plt.ylabel('Amplitude'); plt.title('Original Audio Waveform'); plt.grid(True); plt.legend()
plt.subplot(2, 1, 2); plt.plot(time_hat, audio_hat_np, label='Reconstructed Audio', color='red'); plt.xlabel('Time [s]'); plt.ylabel('Amplitude'); plt.title('Reconstructed Audio Waveform'); plt.grid(True); plt.legend()
plt.tight_layout(); plt.savefig('audio_comparison.png'); plt.close()




