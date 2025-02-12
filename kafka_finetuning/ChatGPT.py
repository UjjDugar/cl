from openai import OpenAI
import os, csv
from datetime import datetime

class GPT: 

    def __init__(self, model_name):
        self.model_name = model_name
        pass

    def generate(self, prompt, logging=False, system_prompt="Answer in the style of Franz Kafka"):

        if self.model_name == 'test':
            return 'testing text'

        client = OpenAI(api_key='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')

        completion = client.chat.completions.create(
        model=self.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        )

        result = completion.choices[0].message.content

        if logging:
            # Log prompt, result, timestamp, and model used
            log_file = 'LLM_logging.csv'
            file_exists = os.path.isfile(log_file)

            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)

                # Write header if file does not exist
                if not file_exists:
                    writer.writerow(["Timestamp", "Model Name", "Prompt", "Response"])

                # Write the log entry
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.model_name, prompt, result.replace('\n', ' ')])

        return result
    
if __name__ == '__main__':
    model_name = 'gpt-3.5-turbo'
    LLM = GPT(model_name) 
    prompt = 'hi, how are you?'
    result = LLM.generate(prompt, logging=False)
    print(result)
