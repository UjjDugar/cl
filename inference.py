from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

# Load model and tokenizer
og_model_path = "meta-llama/Llama-3.2-3B-Instruct"
finetuned_model_path = "output/checkpoint-16748"
model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(og_model_path)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Format the question
question = "do you like chocolate?"
input_text = f"Question: {question}\nAnswer:"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt").to(device)
inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.tensor([[156928]]).to(device)], dim=1)
inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones((1,1)).to(device)], dim=1)


print('Inputs:', inputs)

# Generate response
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=10000,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=156929,
    )

print('Output:', output)

# Extract tokens after 156928
output_sequence = output[0].tolist()
split_index = output_sequence.index(156928) + 1
response_tokens = output_sequence[split_index:]

print('Response tokens:', response_tokens)
print('Number of response tokens:', len(response_tokens))




# print(tokenizer.decode([16533])); print(tokenizer.decode([25]))

# # Decode and print response
# response = tokenizer.decode(output[0], skip_special_tokens=True)
# # print("\nGenerated Answer:\n", response)


