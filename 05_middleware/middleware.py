import requests
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

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
            return f"{base_prompt} Provide detail technical responses."
        case "beginner":
            return f" {base_prompt} Keep your explanations simple and basic."
        case "child":
            return f" {base_prompt} Explain everything as if you were literally talking to a five-year old."
        case _:
            return base_prompt


agent = create_agent(model=model, middleware=[user_role_prompt], context_schema=Context)


response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is PCA?"}]},
    context=Context(user_role="child"),
)

print(response)
