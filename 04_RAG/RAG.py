from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv


load_dotenv()  # ← this must be called before init_chat_model

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
model = init_chat_model(
    "llama-3.3-70b-versatile", model_provider="groq", temperature=0.1
)

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
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)
vectorstore.reset_collection()  # wipe it clean
vectorstore.add_texts(texts)  # add fresh

# docs = [
#     Document(
#         page_content="Python is a programming language created by Guido van Rossum."
#     ),
#     Document(page_content="LangChain is a framework for building LLM applications."),
#     Document(page_content="ChromaDB is a vector database for storing embeddings."),
#     Document(page_content="The Eiffel Tower is located in Paris, France."),
# ]

# vectorstore.add_documents(docs)

results1 = vectorstore.similarity_search("fruits the person loves or enjoys", k=3)
results2 = vectorstore.similarity_search("fruits the person hates or dislikes", k=3)


print("=== Likes ===")
for doc in results1:
    print(doc.page_content)
    print("---")

print("=== Hates ===")
for doc in results2:
    print(doc.page_content)
    print("---")


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
    retriever,
    name="kb_search",
    description="Search the small product / fruit knowledge base for information",
)


agent = create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt=(
        "You are a helpful assistant with access to a knowledge base. "
        "You MUST ALWAYS call the kb_search tool before answering ANY question. "
        "Never answer from your own knowledge. Always search first, then answer based only on the retrieved results."
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What three fruits does the person like and what three fruits does the person dislike?",
            }
        ]
    }
)

print(result)
print(result["messages"][-1].content)
