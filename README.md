<p align="center">
  <img src="https://python.langchain.com/img/brand/wordmark.png" alt="LangChain" width="400"/>
</p>

# LangChain Examples

A collection of practical examples covering the core features of LangChain. Each folder contains a focused script and its own README explaining what it does and how it works.

---

## What is LangChain?

LangChain is an open-source framework for building applications powered by large language models (LLMs). It provides a set of tools and abstractions that handle the repetitive parts of working with AI models — things like structuring conversations, connecting models to external data, building agents that can use tools, and managing the flow of information between components.

Without LangChain, you would write a lot of boilerplate code every time you wanted to do something beyond a simple API call. LangChain gives you that infrastructure out of the box so you can focus on what your application actually does.

## When to use LangChain

LangChain is a good fit when you need to:

- Build a chatbot or assistant that maintains conversation history
- Connect a model to your own data (documents, databases, files)
- Give a model access to tools like search, calculators, or APIs
- Control how and when a model is called in a larger workflow
- Switch between different models or providers without rewriting your code
- Add logging, timing, or other logic around model calls

It is not necessary for simple one-off prompts where you just need a single response. But as soon as your use case involves memory, retrieval, agents, or multi-step logic, LangChain saves significant time.

---

## Examples

| # | Topic | What it covers |
|---|-------|----------------|
| [01](./01_conversations/) | Conversations | Building multi-turn conversations using message history |
| [02](./02_streaming/) | Streaming | Printing model output token by token in real time |
| [03](./03_multimodal/) | Multimodal | Sending both text and images to a vision model |
| [04](./04_RAG/) | RAG | Storing custom data in a vector database and retrieving it to answer questions |
| [05](./05_middleware/) | Middleware | Dynamically changing the system prompt based on user context |
| [06](./06_custom_agent/) | Custom Agent Middleware | Lifecycle hooks to run logic before and after model calls |
| [07](./07_dynamic_models/) | Dynamic Models | Switching between models at runtime based on conversation state |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/LangChainSHIT.git
cd LangChainSHIT
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS / Linux
```

**3. Install dependencies**
```bash
pip install langchain langchain-groq langchain-huggingface langchain-chroma chromadb sentence-transformers python-dotenv
```

**4. Set up your environment variables**

Copy `.env.example` to `.env` and fill in your API key:
```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_key_here
```

You can get a free Groq API key at [console.groq.com](https://console.groq.com).

---

## Project structure

```
LangChainSHIT/
├── 01_conversations/
│   ├── conversations.py
│   └── README.md
├── 02_streaming/
│   ├── streaming.py
│   └── README.md
├── 03_multimodal/
│   ├── multimodal.py
│   └── README.md
├── 04_RAG/
│   ├── RAG.py
│   └── README.md
├── 05_middleware/
│   ├── middleware.py
│   └── README.md
├── 06_custom_agent/
│   ├── customAgentMiddleware.py
│   └── README.md
├── 07_dynamic_models/
│   ├── dynamicModels.py
│   └── README.md
├── .env.example
└── README.md
```