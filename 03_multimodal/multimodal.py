import requests
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from base64 import b64encode

load_dotenv()

model = init_chat_model(
    "meta-llama/llama-4-scout-17b-16e-instruct", model_provider="groq", temperature=0.1
)

message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the contents of this image."},
        {
            "type": "image",
            "base64": b64encode(open("miguel.jpg", "rb").read()).decode(),
            "mime_type": "image/jpg",
        },
    ],
}

response = model.invoke([message])

print(response.content)
