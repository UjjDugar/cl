# These lines are needed for nebius for some reason
sudo apt update
sudo apt install python3-pip
export PATH=$HOME/.local/bin:$PATH 

pip install huggingface wandb 
export PATH=$HOME/.local/bin:$PATH 

huggingface-cli login --token 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
wandb login 0a6c8de726025aca725a2bb24178045a0b08cc25
pip install datasets librosa soundfile accelerate snac torchaudio
pip install torch==2.5.1 
pip install transformers==4.46.3
pip install flash_attn==2.7.3
pip install trl==0.11.3

export ELEVENLABS_API_KEY="sk_af7dc420ea22e0fe192d799fb6142bd14a731a0ea3e852d6"      
export CARTESIA_API_KEY="sk_car_xWj62bk0TW3ZBvDRTF3F9"           

apt update 
apt install screen 
