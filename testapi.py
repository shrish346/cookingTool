from dotenv import load_dotenv
from openai import OpenAI
import os


load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

 
# Initialize the client pointing to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key= OPENROUTER_KEY,
)

# Basic text-only request
completion = client.chat.completions.create(
    model="qwen/qwen2.5-vl-32b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Explain the concept of quantum entanglement in simple terms."
        }
    ]
)
print(completion)
print("--------------------------------")
print(completion.choices)
print("--------------------------------")
print(completion.choices[0].message.content)







