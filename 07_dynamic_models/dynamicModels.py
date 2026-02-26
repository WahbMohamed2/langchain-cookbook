import requests
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, AIMessage, SystemMessage


load_dotenv()

basic_model = init_chat_model(
    "meta-llama/llama-4-scout-17b-16e-instruct", model_provider="groq", temperature=0.3
)
advanced_model = init_chat_model(
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    model_provider="groq",
    temperature=0.3,
)


@dataclass
class Context:
    user_role: str


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


response = agent.invoke(
    {
        "messages": [
            SystemMessage("You are helpful assistant."),
            HumanMessage("What's 1+1?"),
        ]
    }
)

print(response["messages"][-1].content)
print(response["messages"][-1].response_metadata["model_name"])
