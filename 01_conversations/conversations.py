import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
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

print(response)
print(response.content)
