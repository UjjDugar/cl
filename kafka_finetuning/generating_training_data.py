from ChatGPT import GPT
import csv
import time
from tqdm import tqdm

model_name = 'gpt-3.5-turbo'
LLM = GPT(model_name)

# Read questions from file
with open('questions.txt', 'r') as file:
    questions = [line.strip() for line in file if line.strip()]


# Create CSV with questions and answers
with open('qa_responses.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Question', 'Answer'])

    
    for question in tqdm(questions):
        try:
            response = LLM.generate(question, logging=False)
            response = 'RESPONSE: ' + response.replace('\n', '')
            writer.writerow([question, response])
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"Error processing question: {question}")
            print(f"Error: {e}")