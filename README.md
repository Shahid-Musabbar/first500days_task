# RAG Agent — FastAPI + FAISS + OpenAI

This repository implements a Retrieval-Augmented Generation (RAG) application and AI agent using FastAPI, FAISS, and OpenAI (or Azure OpenAI). It demonstrates an agent that decides whether to answer directly or fetch context from documents, includes simple session memory, and exposes a POST /ask endpoint.

**Contents**
- `app.py`: main FastAPI app implementing the agent, RAG pipeline, and /ask endpoint.
- `Dockerfile`: optional containerization support.
- `requirements.txt`: Python dependencies.
- `data/`: sample documents and upload target.
- `faiss_index/`: (optional) persisted FAISS indexes if exported.

## Architecture Overview

- User → POST /ask → Backend (FastAPI)
- Backend performs: similarity search (FAISS) → builds context → agent prompt engineering → calls LLM (OpenAI / Azure OpenAI) → returns structured response
- Components:
  - Embeddings: `OpenAIEmbeddings` (text-embedding-3-small or Azure equivalent)
  - Vector store: FAISS in-memory index
  - LLM: `ChatOpenAI` wrapper (app currently uses OpenAI-style API)
  - Agent: simple agent with a checkpointer (session-based memory) to maintain short threads

## How this meets the Task Requirements

Task 1: AI Agent Development (Core)
- Accepts user query via `POST /ask`.
- Agent decision logic: the app performs similarity search first (RAG). If relevant documents are found, they are assembled into the prompt and passed to the LLM; otherwise a direct LLM response is possible. This satisfies the requirement to decide whether to answer directly or fetch documents.
- Prompt engineering: the app builds a clear prompt that includes retrieved context and an explicit instruction: "Use the context below to answer the question." You can extend or refine prompt templates in `app.py`.
- Tool calling: the app demonstrates a simple tool-like behavior by calling the similarity-search + vector store as an external capability. If you want actual external tool-call agents (e.g., search, calculator), the agent supports adding tools via `create_agent(..., tools=[...])`.
- Basic memory: the agent uses session-based `thread_id` (via `session_id` in requests) and an in-memory checkpointer `InMemorySaver()` to preserve simple conversation context per session.

Task 2: RAG (Retrieval-Augmented Generation)
- Sample documents included in `data/` (3 text docs). These are indexed page-wise by `add_pdf_pagewise` or can be added as simple text docs.
- Embeddings are generated via `OpenAIEmbeddings` and stored in a FAISS index (`faiss.IndexFlatL2`).
- Retrieval: `vector_store.similarity_search(query, k)` returns documents which are assembled into the prompt.

Task 3: Backend API
- Implemented in `app.py` using FastAPI.
- Endpoint: `POST /ask` accepts JSON: `{ "query": "string", "session_id": "optional" }` and returns JSON: `{ "answer": "string", "source": ["doc1", "doc2"] }`.

Task 4: Azure Deployment (Guidance)
- App is compatible with Azure App Service or Azure Functions. Two main adaptation points:
  1. Use Azure OpenAI client or configure `ChatOpenAI` / `OpenAIEmbeddings` to use Azure endpoints and keys.
  2. Configure environment variables in Azure (see below).

Recommended steps (high level):
1. Create an Azure App Service (Linux) or Azure Function App.
2. Set required environment variables in the App Service configuration:
   - `OPENAI_API_KEY` (for OpenAI) OR
   - `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME` (if using Azure-specific client/config). Adapt `app.py` to use Azure client settings.
3. Deploy via Git, ZIP deploy, or Docker (this repo has `Dockerfile` for containerized deployment).
4. Ensure `requirements.txt` is installed in the environment and set `WEBSITES_PORT` if needed.

Environment variables to set (examples):
- `OPENAI_API_KEY` (OpenAI API key) or
- `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` (for Azure)

Bonus: Docker
- The included `Dockerfile` can be used to build and deploy the container to Azure App Service (Web App for Containers) or Azure Container Instances.

## Setup — Local

1. Create a Python 3.10+ virtual environment and activate it.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Install
pip install -r requirements.txt
```

2. Set environment variables (example for local OpenAI):

```bash
set OPENAI_API_KEY=sk-xxxx
```

3. Run the app locally:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

4. Upload sample docs (optional): use the `/upload-pdf` endpoint or drop text files into `data/` and adapt loader logic.

5. Ask the API:

```bash
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{ \"query\": \"What is the leave policy?\", \"session_id\": \"user-1\" }"
```

## Setup — Azure (concise)

1. Create an Azure App Service or Function App.
2. In the Azure Portal, go to Configuration → Application settings and add keys as described above.
3. Deploy via Git or Docker.
4. Configure logging/monitoring via Azure Monitor (Application Insights).

Notes on Azure OpenAI integration:
- If you use Azure OpenAI, you will likely need to adapt the `langchain_openai` constructor to use the Azure-specific parameters. Example env vars: `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, and pass these to your LLM/Embeddings constructors. See LangChain docs for `AzureOpenAI` + `AzureOpenAIEmbeddings`.

## Design Decisions

- FAISS chosen for local, fast similarity search with low operational overhead.
- In-memory docstore + checkpointer for simplicity and demo-readiness — swap to persistent storage for production (e.g., Redis, Pinecone, or Azure Cognitive Search).
- Agent threading via `session_id` offers basic conversational memory without a DB.
- Prompt template is explicit and minimal to keep behavior predictable; advanced prompt engineering or chain-of-thought toggles can be layered on later.

## Limitations & Future Improvements

- Current memory is ephemeral (in-memory). For persistence across restarts, integrate a DB (Redis, Postgres) for conversation state.
- FAISS index is in-memory — add persistence (FAISS on-disk or vector DB like Pinecone, Milvus, or Azure Cognitive Search) for scaling.
- No strict tool-call orchestration: to support multi-tool agents (e.g., calling calculators, web searches), add explicit tools in `create_agent(..., tools=[...])` with safe wrappers.
- Authentication and RBAC are not implemented — secure endpoints before production.

## Files Added (for demo) 
- [README.md](README.md)
- [data/company_policies.txt](data/company_policies.txt)
- [data/product_faq.txt](data/product_faq.txt)
- [data/technical_documentation.txt](data/technical_documentation.txt)

## Quick Development Notes
- Indexing: the repo includes `add_pdf_pagewise` to load PDFs page-by-page using `PyPDFLoader` and then add them to FAISS. For text files, adapt loader to create Document objects.
- Embeddings: `OpenAIEmbeddings` is used. For Azure, use the Azure equivalent or configure LangChain accordingly.
- Agent: `create_agent(...)` can accept `tools=` for richer tool calling. The checkpointer is `InMemorySaver()`.

## Next Steps I Can Do
- Add a small deployment script for Azure App Service or an `azure-pipelines.yml` for CI/CD (optional).
- Persist FAISS index to disk or add Pinecone/Milvus integration (optional).
- Add tracing and logging either using Azure Monitor or use opensource LLM Tracers like Langfuse.

---

If you want, I can now (A) update `app.py` to support Azure OpenAI environment variables explicitly, (B) add Azure deployment scripts, or (C) implement persistence for FAISS. Which should I do next?