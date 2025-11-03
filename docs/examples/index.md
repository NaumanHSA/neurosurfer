# Examples

Welcome to the living gallery of Neurosurfer examples. This page showcases practical, copy‑pasteable snippets to help you get productive quickly. Browse the curated example sets below, then dive into the **Basic Examples** section for a one‑screen tour of the most common patterns (model, agent, RAG, tools).

## 📚 Available Examples

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } **Models**

    ---

    Initialize and use different chat models and embedders

    [:octicons-arrow-right-24: View](./models-examples.md)

-   :material-robot:{ .lg .middle } **Agents**

    ---

    Initialize and use different agents

    [:octicons-arrow-right-24: View](./agents-examples.md)

-   :material-database:{ .lg .middle } **Server App**

    ---

    A Neurosurfer server app, explained step‑by‑step—app init, startup wiring, RAG, chat handler, shutdown, cooperative stop, and running the server.

    [:octicons-arrow-right-24: View](./server-app-example.md)

-   :material-book:{ .lg .middle } **RAG Examples**

    ---

    Document Q&A and knowledge retrieval

    [:octicons-arrow-right-24: View](./rag-examples.md)

-   :material-toolbox:{ .lg .middle } **Custom Tools**

    ---

    Creating your own agent tools

    [:octicons-arrow-right-24: View](./custom-tools-examples.md)

</div>

---

## ⚡ Basic Examples

Below are minimal, end‑to‑end snippets showing a single way to do four core tasks. Keep them simple first; you can mix, match, and scale later.

### Model — direct chat completion

```python
from neurosurfer.models.chat_models.openai import OpenAIModel

# Create a small, fast model
model = OpenAIModel(model_name="gpt-4o-mini")

# One-shot request
reply = model.ask(user_prompt="Say hi in one short sentence.", stream=False)
print(reply.choices[0].message.content)
```

### Agent — ReAct with no tools

```python
from neurosurfer import ReActAgent
from neurosurfer.models.chat_models.openai import OpenAIModel

model = OpenAIModel(model_name="gpt-4o-mini")
agent = ReActAgent(llm=model, tools=[])

answer = agent.run("What is machine learning, in one sentence?")
for chunk in answer:
    print(chunk.choices[0].message.content)
```

### RAG — Files ingestion and retrieval

```python
from neurosurfer.models.chat_models.openai import OpenAIModel
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder
from neurosurfer.vectorstores import ChromaVectorStore
from neurosurfer.rag import RAGIngestor
from neurosurfer import RAGRetrieverAgent

# Components
model = OpenAIModel(model_name="gpt-4o-mini")
embedder = SentenceTransformerEmbedder(model_name="intfloat/e5-large-v2")
vectorstore = ChromaVectorStore(collection_name="docs")

# Ingest a single PDF (adjust the path)
ingestor = RAGIngestor(embedder=embedder, vectorstore=vectorstore)
ingestor.ingest_file("document.pdf")

# Ask a question grounded in your file
rag_agent = RAGRetrieverAgent(llm=model, vectorstore=vectorstore, embedder=embedder)
content = rag_agent.retrieve("Summarize the document in two bullet points.")
print(content)
```

### Tools — built‑in SQL tools

```python
from neurosurfer import ReActAgent
from neurosurfer.models.chat_models.openai import OpenAIModel
from neurosurfer.tools import Toolkit
from neurosurfer.tools.sql import RelevantTableSchemaFinderLLM
from neurosurfer.db.sql_schema_store import SQLSchemaStore

model = OpenAIModel(model_name="gpt-4o-mini")
toolkit = Toolkit()
sql_schema_store = SQLSchemaStore(llm=model, db_uri="sqlite:///example.db")
toolkit.register_tool(RelevantTableSchemaFinderLLM(llm=model, sql_schema_store=sql_schema_store))

# Attach a single built-in tool
agent = ReActAgent(llm=model, toolkit=toolkit)

print(agent.run("What is the stock price of Apples this week?"))
```

---

## 📖 Where to go next

- Try the [Server App Example](./server-app-example.md) for a complete, production-ready app
- Explore and use more [Models Examples](./models-examples.md)
- Build agentic solutions with [Agents Examples](./agents-examples.md)
- Build document Q&A with [RAG Examples](./rag-examples.md)
- Extend functionality via [Custom Tools](./custom-tools-examples.md)