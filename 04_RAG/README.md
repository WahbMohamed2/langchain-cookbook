# RAG (Retrieval-Augmented Generation)

This example shows how to build a RAG pipeline. Instead of relying on the model's training data, you store your own documents in a vector database and have the model search and answer from them. This is useful when you want the model to answer based on specific, custom, or up-to-date information.

## How it works

1. Your texts are converted into embeddings (numerical representations) and stored in ChromaDB
2. When a question comes in, the retriever searches the database for the most relevant texts
3. The agent calls the retriever tool, gets the results, and uses them to form an answer

## Code

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
model = init_chat_model(
    "llama-3.3-70b-versatile", model_provider="groq", temperature=0.1
)

# Your custom knowledge base
texts = [
    "I love apples.",
    "I enjoy oranges.",
    "I think pears taste very good.",
    "I hate bananas.",
    "I dislike raspberries.",
    "I despise mangos.",
    "I love Linux.",
    "I hate Windows.",
]

# Store texts in ChromaDB
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)
vectorstore.reset_collection()
vectorstore.add_texts(texts)

# Direct similarity search (no agent, just raw retrieval)
results = vectorstore.similarity_search("fruits the person loves or enjoys", k=3)
for doc in results:
    print(doc.page_content)

# Wrap the vectorstore as a tool for the agent
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
    retriever,
    name="kb_search",
    description="Search the knowledge base for information",
)

# Create an agent that uses the retriever tool
agent = create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt=(
        "You are a helpful assistant with access to a knowledge base. "
        "You MUST ALWAYS call the kb_search tool before answering ANY question. "
        "Never answer from your own knowledge. Always search first, then answer based only on the retrieved results."
    ),
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "What three fruits does the person like and what three fruits does the person dislike?",
    }]
})

print(result["messages"][-1].content)
```

## Key concepts

**Embeddings** — text is converted into vectors (lists of numbers) that capture meaning. Similar sentences end up with similar vectors, which is what makes semantic search possible.

**ChromaDB** — a local vector database that stores your embeddings and lets you search them by similarity. The `persist_directory` saves the database to disk so it survives between runs.

**Similarity search** — instead of keyword matching, it finds texts that are semantically close to your query. "fruits the person loves" will match "I love apples" even though the words don't overlap.

**Retriever tool** — wraps the vectorstore so an agent can call it like a function during a conversation.

**Agent** — a model that can decide to use tools. Here it is instructed to always search the knowledge base before answering, so it never guesses from its own training data.

## Example output

```
=== Likes ===
I love apples.
---
I enjoy oranges.
---
I think pears taste very good.
---

The person likes apples, oranges, and pears. They dislike bananas, raspberries, and mangos.
```

## Adding documents instead of plain text

If you want to store structured documents with metadata, use `Document` objects instead:

```python
from langchain_core.documents import Document

docs = [
    Document(page_content="LangChain is a framework for building LLM applications."),
    Document(page_content="ChromaDB is a vector database for storing embeddings."),
]
vectorstore.add_documents(docs)
```

## Requirements

```
langchain
langchain-groq
langchain-huggingface
langchain-chroma
chromadb
sentence-transformers
python-dotenv
```

## .env

```
GROQ_API_KEY=your_key_here
```