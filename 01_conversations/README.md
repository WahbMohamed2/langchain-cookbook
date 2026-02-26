# Conversations

This example shows how to simulate a multi-turn conversation with an LLM using LangChain message types. Instead of sending a single message, you build a list of messages that represents the full conversation history, then pass it to the model at once.

## How it works

LangChain provides three message types to structure a conversation:

- `SystemMessage` — sets the behavior or role of the assistant
- `HumanMessage` — represents what the user said
- `AIMessage` — represents what the assistant previously replied

You build the conversation as a list, then call `model.invoke(conversation)`. The model reads the full history and responds accordingly.

## Code

```python
import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    "llama-3.3-70b-versatile", model_provider="groq", temperature=0.1
)

conversation = [
    SystemMessage("You are a helpful assistant for questions regarding programming"),
    HumanMessage("What is Python?"),
    AIMessage("Python is an interpreted programming language."),
    HumanMessage("When was it released?"),
]

response = model.invoke(conversation)

print(response.content)
```

## What the model receives

The model sees the full conversation in order:

```
System:    You are a helpful assistant for questions regarding programming
Human:     What is Python?
AI:        Python is an interpreted programming language.
Human:     When was it released?
```

It then replies to the last human message with the context of everything above it.

## Example output

```
Python was first released in 1991 by Guido van Rossum.
```

## Why this matters

Without passing the conversation history, the model has no memory. Each call is stateless. By manually building and passing the message list, you give the model the context it needs to understand follow-up questions like "When was it released?" — which only makes sense if the model knows what "it" refers to.

## Requirements

```
langchain
langchain-groq
python-dotenv
```

## .env

```
GROQ_API_KEY=your_key_here
```