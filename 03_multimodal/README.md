# Multimodal

This example shows how to send both text and an image to a model in a single request. This is called multimodal input — the model can read and reason about images alongside text.

## How it works

Instead of passing a plain string, you build a message manually as a dictionary with a `content` list. Each item in the list is either a text block or an image block. The image is read from disk and encoded to base64 before being sent.

## Code

```python
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
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
```

## What each part does

- `b64encode(open("miguel.jpg", "rb").read()).decode()` — reads the image file as binary, encodes it to base64, then converts it to a plain string the API can accept
- `"type": "text"` — the text instruction you want to send alongside the image
- `"type": "image"` — the image block containing the encoded image and its mime type
- `model.invoke([message])` — sends the full message (text + image) to the model

## Message structure

```
content: [
    { type: "text",  text: "your question here" },
    { type: "image", base64: "...", mime_type: "image/jpg" }
]
```

You can include multiple text or image blocks in the same content list.

## Example output

```
The image shows a person standing outdoors near a tree. They are wearing a casual outfit and appear to be smiling at the camera.
```

## Requirements

```
langchain
langchain-groq
python-dotenv
```

## Note

Not all models support image input. Make sure the model you use is a vision-capable model. In this example, `llama-4-scout` is used because it supports multimodal input, unlike `llama-3.3-70b` used in the other examples.

## .env

```
GROQ_API_KEY=your_key_here
```