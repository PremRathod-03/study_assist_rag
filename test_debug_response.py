import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Say hello in exactly 3 words."}],
    max_tokens=100,
)

print("Full response object:")
print(response)
print()
print("Finish reason:", response.choices[0].finish_reason)
print("Content:", repr(response.choices[0].message.content))
