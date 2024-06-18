# See tutorial here: https://platform.openai.com/docs/quickstart

from access_token import OPENAI_TOKEN
from openai import OpenAI

client = OpenAI(api_key=OPENAI_TOKEN)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are an empathetic therapist."},
    {"role": "user", "content": "I am feeling sad."}
  ]
)

print(completion.choices[0].message)