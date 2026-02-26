# Dynamic Models

This example shows how to switch between models at runtime based on the state of the conversation. Instead of committing to one model for the entire session, you can route simple requests to a cheaper/faster model and only escalate to a more powerful model when needed.

## How it works

You use `@wrap_model_call` to intercept every model call before it happens. Inside the function you inspect the current state, decide which model to use, assign it to `request.model`, then call `handler(request)` to proceed with the chosen model.

## Code

```python
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

basic_model = init_chat_model(
    "meta-llama/llama-4-scout-17b-16e-instruct", model_provider="groq", temperature=0.3
)
advanced_model = init_chat_model(
    "meta-llama/llama-4-maverick-17b-128e-instruct", model_provider="groq", temperature=0.3
)

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])
    if message_count > 3:
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)

agent = create_agent(model=basic_model, middleware=[dynamic_model_selection])

response = agent.invoke({
    "messages": [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("What's 1+1?"),
    ]
})

print(response["messages"][-1].content)
print(response["messages"][-1].response_metadata["model_name"])
```

## What each part does

- `@wrap_model_call` — marks the function as middleware that wraps every model call
- `request.state["messages"]` — the current list of messages in the conversation
- `request.model = model` — swaps the model before the call is made
- `handler(request)` — executes the model call with whatever model is now set on the request

## Routing logic in this example

```
messages <= 3  →  basic_model   (fast, lighter)
messages  > 3  →  advanced_model (more capable)
```

You can replace this logic with anything: question complexity, user tier, detected language, keyword presence, time of day, etc.

## Example output

```
2
meta-llama/llama-4-scout-17b-16e-instruct
```

Since the conversation only has 2 messages here, the basic model is used. If the conversation grew beyond 3 messages, the advanced model would take over and you would see its name printed instead.

## Difference from other middleware

| Middleware type | What it controls |
|---|---|
| `@dynamic_prompt` | The system prompt sent to the model |
| `AgentMiddleware` | Lifecycle hooks (before/after agent and model) |
| `@wrap_model_call` | Which model is actually called |

## Why this is useful

Running a powerful model on every request is wasteful and expensive. With `@wrap_model_call` you can serve most requests with a fast, cheap model and only reach for the stronger one when the conversation or task complexity justifies it.

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