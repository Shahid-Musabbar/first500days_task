import os
import shutil
import faiss
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
# from langchain.schema import Document

# -------------------------------------------------------------------
# 0. ENV + FASTAPI
# -------------------------------------------------------------------
load_dotenv()

app = FastAPI()
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 1. LLM + AGENT (UNCHANGED LOGIC)
# -------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

tool = {"type": "web_search_preview"}
toolbox = [
    tool,] # retrieve_context
agent = create_agent(
    llm,
    tools=toolbox,
    checkpointer=InMemorySaver(),
    system_prompt="You are a helpful assistant that answers using provided context."
)

# -------------------------------------------------------------------
# 2. EMBEDDINGS + FAISS (IN-MEMORY)
# -------------------------------------------------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

embedding_dim = len(embeddings.embed_query("hello"))
faiss_index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=faiss_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# -------------------------------------------------------------------
# 3. HELPERS
# -------------------------------------------------------------------
def add_pdf_pagewise(pdf_path: str):
    """
    Each PDF page becomes ONE document with metadata.
    """
    loader = PyPDFLoader(pdf_path, mode="page")
    docs = loader.load()

    # documents: List[Document] = []
    documents=[]
    ids: List[str] = []

    file_name = os.path.basename(pdf_path)

    for i, doc in enumerate(docs, start=1):
        doc.metadata.update({
            "source": "pdf",
            "file_name": file_name,
            "page": i
        })

        doc_id = f"{file_name}_page_{i}"
        documents.append(doc)
        ids.append(doc_id)

    vector_store.add_documents(documents=documents, ids=ids)


def retrieve_context(query: str, k: int = 4) -> str:
    """
    Similarity search → formatted context
    """
    docs = vector_store.similarity_search(query=query, k=k)

    if not docs:
        return "No relevant documents found."

    return "\n\n".join(
        f"[{d.metadata['file_name']} - page {d.metadata['page']}]\n{d.page_content}"
        for d in docs
    ) , docs

# -------------------------------------------------------------------
# 4. API SCHEMAS
# -------------------------------------------------------------------
class AskRequest(BaseModel):
    query: str
    session_id: str | None = "default"


class AskResponse(BaseModel):
    answer: str
    source: List[str] | None = None

# -------------------------------------------------------------------
# 5. ROUTES
# -------------------------------------------------------------------
@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    """
    Upload PDF → each page becomes a vector
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    add_pdf_pagewise(file_path)

    return {
        "status": "success",
        "message": f"{file.filename} indexed page-wise"
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Query → similarity search → agent response
    """
    context, docs = retrieve_context(request.query)

    final_prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{request.query}
"""

    result = agent.invoke(
        {"messages": {"role": "user", "content": final_prompt}},
        {"configurable": {"thread_id": request.session_id}}
    )

    answer = result["messages"][-1].content

    sources = [
    f"{d.metadata.get('file_name')}_page_{d.metadata.get('page')}"
    for d in docs
]

    return AskResponse(answer=answer, source=sources)

# -------------------------------------------------------------------
# 6. OPTIONAL CLI (SAME FILE)
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("RAG Agent CLI — type 'exit' to quit\n")

    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break

        ctx = retrieve_context(q)

        prompt = f"""
Context:
{ctx}

Question:
{q}
"""

        res = agent.invoke(
            {"messages": {"role": "user", "content": prompt}},
            {"configurable": {"thread_id": "cli"}}
        )

        print("AI:", res["messages"][-1].content)


#uvicorn app:app --reload
