from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import SystemMessage, HumanMessage, AIMessage
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
        print("after_agent:", time.time() - self.start_time)


model = init_chat_model(
    "meta-llama/llama-4-scout-17b-16e-instruct", model_provider="groq", temperature=0.3
)

agent = create_agent(model=model, middleware=[HooksDemo()])


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
