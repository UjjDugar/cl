from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
import torch
from datasets import load_dataset
from playing_with_forward_function import MyModel

dataset_name = "UjjD/kafka-qa-dataset"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = MyModel.from_pretrained(model_name)

model.resize_token_embeddings(128257)
dataset = load_dataset(dataset_name, split="train")

tokenizer = AutoTokenizer.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./llama-finetuned-kafka",  # Directory for model checkpoints
    overwrite_output_dir=True,
    per_device_train_batch_size=1,  # Batch size of 1
    logging_steps=1,
    fp16=True,
    remove_unused_columns=True,
    report_to="none",  # Disable Weights & Biases logging
)

trainer = Trainer(
    model = model, 
    args = training_args, 
    train_dataset = dataset
)

trainer.train()

# Load the fine-tuned model
model.eval()  # Set model to evaluation mode
model = model.to("cuda" if torch.cuda.is_available() else "cpu")  # Move to GPU if available

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_answer(question, max_new_tokens=100):
    """Generates an answer from the fine-tuned model given a question."""
    
    # Format input with special split token if used in training
    input_text = f"Question: {question}\n|<split>| Answer:"
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,  # Limit response length
            do_sample=True,  # Enable sampling for more natural responses
            temperature=0.7,  # Adjust creativity
            top_p=0.9,  # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id  # Avoid issues with missing pad token
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example Question
question = "Do you like butterflies?"
generated_response = generate_answer(question)

print("\nGenerated Answer:\n", generated_response)
