# Middleware (Dynamic Prompts)

This example shows how to use middleware to dynamically change the system prompt based on context. Instead of hardcoding a single system prompt, the agent adjusts its tone and depth depending on who is asking.

## How it works

1. You define a `Context` dataclass that holds information about the current request (in this case, the user's role)
2. You write a `@dynamic_prompt` function that reads that context and returns the appropriate system prompt
3. The agent is created with that middleware attached, so every request passes through it before reaching the model

## Code

```python
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.chat_models import init_chat_model

load_dotenv()

@dataclass
class Context:
    user_role: str

model = init_chat_model(
    "meta-llama/llama-4-scout-17b-16e-instruct", model_provider="groq", temperature=0.3
)

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role
    base_prompt = "You are a helpful and very concise assistant."
    match user_role:
        case "expert":
            return f"{base_prompt} Provide detailed technical responses."
        case "beginner":
            return f"{base_prompt} Keep your explanations simple and basic."
        case "child":
            return f"{base_prompt} Explain everything as if you were talking to a five-year-old."
        case _:
            return base_prompt

agent = create_agent(model=model, middleware=[user_role_prompt], context_schema=Context)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is PCA?"}]},
    context=Context(user_role="child"),
)

print(response["messages"][-1].content)
```

## What changes based on role

The same question "What is PCA?" gets a completely different answer depending on the role passed in:

| Role | Behavior |
|---|---|
| `expert` | Technical, detailed explanation with math and terminology |
| `beginner` | Simple explanation, avoids jargon |
| `child` | Very basic analogy-driven explanation |
| anything else | Default helpful assistant |

## Example output

With `user_role="child"`:
```
Imagine you have a big box of toys but your shelf is small. PCA helps you pick
the most important toys that tell you the most about your collection, so you can
fit them neatly on the shelf without losing much information.
```

With `user_role="expert"`:
```
PCA (Principal Component Analysis) is a dimensionality reduction technique that
projects data onto a lower-dimensional subspace by computing the eigenvectors of
the covariance matrix, ordered by descending eigenvalues...
```

## Why this is useful

Without middleware, you would need to manually build a different system prompt for every request and pass it each time. With `@dynamic_prompt`, the logic lives in one place and is applied automatically on every agent call. You just pass the context and the middleware handles the rest.

## Requirements

```
langchain
langchain-groq
langgraph
python-dotenv
```

## .env

```
GROQ_API_KEY=your_key_here
```