# Custom Agent Middleware

This example shows how to create a custom middleware class by extending `AgentMiddleware`. This gives you lifecycle hooks that run at specific points during the agent's execution, which is useful for logging, timing, debugging, or any side effect you want to attach to the agent without changing its core logic.

## How it works

You subclass `AgentMiddleware` and override any of the four hooks:

| Hook | When it runs |
|---|---|
| `before_agent` | Once, before the agent starts processing |
| `before_model` | Before each call to the model |
| `after_model` | After each call to the model |
| `after_agent` | Once, after the agent finishes |

## Code

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import time

load_dotenv()

class HooksDemo(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.start_time = 0.0

    def before_agent(self, state: AgentState, runtime):
        self.start_time = time.time()
        print("before_agent triggered")

    def before_model(self, state: AgentState, runtime):
        print("before_model")

    def after_model(self, state: AgentState, runtime):
        print("after_model")

    def after_agent(self, state: AgentState, runtime):
        elapsed = time.time() - self.start_time
        print(f"after_agent: {elapsed:.2f}s")

model = init_chat_model(
    "meta-llama/llama-4-scout-17b-16e-instruct", model_provider="groq", temperature=0.3
)

agent = create_agent(model=model, middleware=[HooksDemo()])

response = agent.invoke({
    "messages": [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("What's 1+1?"),
    ]
})

print(response["messages"][-1].content)
print(response["messages"][-1].response_metadata["model_name"])
```

## Example output

```
before_agent triggered
before_model
after_model
after_agent: 0.87s
2
meta-llama/llama-4-scout-17b-16e-instruct
```

## What you can do with each hook

- `before_agent` — start a timer, log the incoming request, set up resources
- `before_model` — inspect or modify the state before it reaches the model
- `after_model` — log the model's raw response, count tokens
- `after_agent` — measure total elapsed time, send metrics, clean up resources

## Difference from dynamic_prompt middleware

The `middleware.py` example used `@dynamic_prompt` which only handles the system prompt. `AgentMiddleware` is lower level and gives you access to the full agent lifecycle. Use `@dynamic_prompt` when you just need to adjust the prompt, and `AgentMiddleware` when you need to hook into execution flow.

## Accessing model metadata

The last message in the response contains metadata about the model that was used:

```python
response["messages"][-1].response_metadata["model_name"]
# meta-llama/llama-4-scout-17b-16e-instruct
```

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