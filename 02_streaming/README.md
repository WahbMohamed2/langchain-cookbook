# Streaming

This example shows how to stream a model's response token by token instead of waiting for the full reply. This is useful when you want output to appear in real time, like a typing effect.

## How it works

Instead of `model.invoke()` which waits for the complete response, you use `model.stream()` which returns chunks as they are generated. You then print each chunk immediately as it arrives.

## Code

```python
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "llama-3.3-70b-versatile", model_provider="groq", temperature=0.1
)

for chunk in model.stream("Hello, What is python"):
    print(chunk.text, end="", flush=True)
```

## What each part does

- `model.stream("...")` — sends the prompt and returns an iterator of chunks
- `chunk.text` — the text content of each chunk as it arrives
- `end=""` — prevents `print` from adding a newline after each chunk so the output flows as one continuous line
- `flush=True` — forces the output to print immediately without buffering

## Example output

Without streaming, nothing appears until the model finishes. With streaming, you see the response build word by word:

```
Python is a high-level, interpreted programming language known for its clean syntax...
```

## invoke vs stream

| | `invoke` | `stream` |
|---|---|---|
| Returns | Full response at once | Chunks one by one |
| Wait time | Waits until done | Output appears immediately |
| Use case | Background processing | Real-time display |

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