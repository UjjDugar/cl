# These lines are needed for nebius for some reason
sudo apt update
sudo apt install python3-pip
export PATH=$HOME/.local/bin:$PATH 

pip install huggingface wandb 
huggingface-cli login --token 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
wandb login 0a6c8de726025aca725a2bb24178045a0b08cc25
pip install datasets librosa soundfile accelerate snac torchaudio
pip install torch==2.5.1 
pip install transformers==4.46.3
pip install flash_attn==2.7.3
pip install trl==0.11.3

apt update 
apt install screen 
