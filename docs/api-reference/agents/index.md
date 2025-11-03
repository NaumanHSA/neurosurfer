# Agents API

AI agents that can reason, act, and solve complex tasks using tools and knowledge retrieval.

## 📚 Available Agents

<div class="grid cards" markdown>

-   :material-robot-outline:{ .lg .middle } **ReActAgent**

    ---

    General-purpose agent using ReAct (Reasoning + Acting) paradigm

    [:octicons-arrow-right-24: Documentation](react-agent.md)

-   :material-database-search:{ .lg .middle } **SQLAgent**

    ---

    Query databases using natural language

    [:octicons-arrow-right-24: Documentation](sql-agent.md)

-   :material-book-search:{ .lg .middle } **RAGRetrieverAgent**

    ---

    Answer questions using document knowledge bases

    [:octicons-arrow-right-24: Documentation](rag-agent.md)

</div>

---

## 🎯 Agent Comparison

| Agent | Best For | Tools Required | Knowledge Source |
|-------|----------|----------------|------------------|
| **ReActAgent** | General tasks, multi-step reasoning | Any tools | None / Optional RAG |
| **SQLAgent** | Data analysis, business intelligence | Database | SQL schemas |
| **RAGRetrieverAgent** | Document Q&A, knowledge retrieval | None | Vector database |

---

## 🤔 Which Agent Should I Use?

### Use ReActAgent When...

- ✅ You need general-purpose reasoning
- ✅ Your task requires multiple tools
- ✅ You want maximum flexibility
- ✅ You need web search, calculations, file operations

**Example Use Cases:**
- Customer support chatbot
- Research assistant
- Task automation
- Multi-step problem solving

[View ReActAgent Documentation →](react-agent.md)

### Use SQLAgent When...

- ✅ You need to query databases
- ✅ You want natural language to SQL
- ✅ You're doing data analysis
- ✅ You need business intelligence

**Example Use Cases:**
- Business analytics
- Report generation
- Data exploration
- Dashboard queries

[View SQLAgent Documentation →](sql-agent.md)

### Use RAGRetrieverAgent When...

- ✅ You have a document knowledge base
- ✅ You need accurate, cited answers
- ✅ You want to query your own documents
- ✅ You need context-aware responses

**Example Use Cases:**
- Document Q&A
- Knowledge base search
- Technical documentation assistant
- Research paper analysis

[View RAGRetrieverAgent Documentation →](rag-agent.md)

---

## 🚀 Quick Start

### ReAct agent with a Toolkit

```python
from neurosurfer.agents.react_agent import ReActAgent
from neurosurfer.models.chat_models.openai import OpenAIModel
from neurosurfer.tools import Toolkit
from neurosurfer.tools.common.general_query_assistant import GeneralQueryAssistantTool

llm = OpenAIModel(model_name="gpt-4o-mini")

toolkit = Toolkit()
toolkit.register_tool(GeneralQueryAssistantTool(llm=llm))

agent = ReActAgent(toolkit=toolkit, llm=llm, verbose=True)

for chunk in agent.run("What is 15% of 250?"):
    print(chunk, end="")
```

### SQL agent streaming

```python
from neurosurfer.agents.sql_agent import SQLAgent

sql_agent = SQLAgent(
    llm=llm,
    db_uri="sqlite:///examples/chinook.db",
    sample_rows_in_table_info=5,
)

transcript = []
for chunk in sql_agent.run("List the top 5 artists by total sales."):
    transcript.append(chunk)

answer = "".join(transcript).split("<__final_answer__>")[-1].split("</__final_answer__>")[0].strip()
print(answer)
```

### Retrieval-only pipeline

```python
from neurosurfer.agents.rag_retriever_agent import RAGRetrieverAgent

# chroma_store and embedder should already be configured
rag = RAGRetrieverAgent(llm=llm, vectorstore=chroma_store, embedder=embedder)
result = rag.retrieve(
    user_query="Summarise the onboarding guide.",
    base_system_prompt="You are a helpful assistant.",
    base_user_prompt="{context}\n\nQuestion: {query}",
)

response = llm.ask(
    system_prompt=result.base_system_prompt,
    user_prompt=result.base_user_prompt.format(
        context=result.context,
        query="Summarise the onboarding guide.",
    ),
    max_new_tokens=result.max_new_tokens,
)
```

---

## 🔧 Tips & Best Practices

- **Register descriptive tools**: rich `ToolSpec` metadata dramatically improves the quality of the ReAct loop.
- **Watch for final answer markers**: streamed output wraps the conclusion inside `<__final_answer__>` tags—strip them before returning to end-users.
- **Cache schemas**: call `SQLAgent.train()` during startup so schema summaries are ready when the first query arrives.
- **Mind the token budget**: `RAGRetrieverAgent` exposes the remaining generation budget—pass it straight into your LLM call to avoid truncation.
- **Forward runtime context**: any keyword arguments you pass to `agent.run(...)` are merged into tool executions, making it easy to inject connections or feature flags.

---

## 📚 Learn More

- [ReActAgent](react-agent.md) — Full reasoning loop with tools
- [SQLAgent](sql-agent.md) — ReAct agent tailored for databases
- [RAGRetrieverAgent](rag-agent.md) — Retrieval-only building block
- [Tools API](../tools/index.md) — Build and document custom tools
- [Models API](../models/index.md) — Chat and embedding backends
- [Examples](../../examples/index.md) — End-to-end notebooks and scripts

---

Ready to dive in? Start with [ReActAgent →](react-agent.md)
