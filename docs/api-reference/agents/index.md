# Agents API

AI agents that can **reason**, **use tools**, and **solve complex tasks**—from database analytics to document Q&A to general multi-step workflows.

## 📚 Available Agents

<div class="grid cards" markdown>

-   :material-robot-outline:{ .lg .middle } **ReActAgent**

    ---

    General-purpose agent using the ReAct (Reasoning + Acting) loop with robust tool routing, validation, and self-repair.

    [:octicons-arrow-right-24: Documentation](react-agent.md)

-   :material-database-search:{ .lg .middle } **SQLAgent**

    ---

    Domain-tuned ReAct agent for SQL: discover schema, generate queries, execute safely, and explain results.

    [:octicons-arrow-right-24: Documentation](sql-agent.md)

-   :material-book-search:{ .lg .middle } **RAGRetrieverAgent**

    ---

    Retrieval core for document Q&A: fetch context from a vector store and return safe `max_new_tokens` budgeting.

    [:octicons-arrow-right-24: Documentation](rag-agent.md)

</div>

---


## 🤔 Which Agent Should I Use?

### Use ReActAgent When…
- You need flexible multi-step reasoning and tool use
- Tasks span file operations, search, code generation, lint/test, and similar workflows
- You want self-repair on parse/tool errors and input pruning

**Typical use cases:** research assistant, support bots, automation flows  
[View ReActAgent Documentation →](react-agent.md)

### Use SQLAgent When…
- You want natural language to SQL with schema discovery and safe execution
- You need analytics, reporting, and BI dashboards
- You value clear, natural-language explanations of results

**Typical use cases:** ad-hoc analytics, KPI reports, data exploration  
[View SQLAgent Documentation →](sql-agent.md)

### Use RAGAgent When…
- You have a document knowledge base in a vector store
- You need a retrieval core that returns prompts and safe budgets
- You want to plug into any LLM for final generation

**Typical use cases:** doc Q&A, handbook/KB assistants, code search helpers  
[View RAGAgent Documentation →](rag-agent.md)

---

## 🚀 Quick Start

### ReAct agent with a Toolkit

```python
from neurosurfer.agents.react import ReActAgent, ReActConfig
from neurosurfer.models.chat_models.openai import OpenAIModel
from neurosurfer.tools import Toolkit
from neurosurfer.tools.common.general_query_assistant import GeneralQueryAssistantTool

llm = OpenAIModel(model_name="gpt-4o-mini")
toolkit = Toolkit()
toolkit.register_tool(GeneralQueryAssistantTool(llm=llm))

agent = ReActAgent(toolkit=toolkit, llm=llm, config=ReActConfig(repair_with_llm=True))

for chunk in agent.run("What is 15% of 250?"):
    print(chunk, end="")
```

### SQL agent streaming

```python
from neurosurfer.agents.sql_agent import SQLAgent
from neurosurfer.models.chat_models.openai import OpenAIModel

llm = OpenAIModel(model_name="gpt-4o-mini")
sql_agent = SQLAgent(
    llm=llm,
    db_uri="sqlite:///examples/chinook.db",
    sample_rows_in_table_info=5,
)

transcript = []
for chunk in sql_agent.run("List the top 5 artists by total sales."):
    transcript.append(chunk)

# If your UI doesn’t suppress special tokens, strip final-answer markers:
answer = "".join(transcript).split("<__final_answer__>")[-1].split("</__final_answer__>")[0].strip()
print(answer)
```

### Retrieval-only pipeline

```python
from neurosurfer.agents.rag import RAGAgent, RAGAgentConfig
from neurosurfer.models.chat_models.openai import OpenAIModel

llm = OpenAIModel(model_name="gpt-4o-mini")
rag = RAGAgent(llm=llm, vectorstore=chroma_store, embedder=embedder,
                        config=RAGAgentConfig(top_k=8, normalize_embeddings=True))

result = rag.retrieve(
    user_query="Summarize the onboarding guide.",
    base_system_prompt="You are a helpful assistant.",
    base_user_prompt="{context}\n\nQuestion: {query}",
)

response = llm.ask(
    system_prompt=result.base_system_prompt,
    user_prompt=result.base_user_prompt.format(
        context=result.context,
        query="Summarize the onboarding guide.",
    ),
    max_new_tokens=result.max_new_tokens,
)
```

> **Note on markers:** ReAct/SQL stream final answers wrapped in `<__final_answer__>…</__final_answer__>`.  
> Set `ReActConfig(skip_special_tokens=True)` if your UI handles finalization without markers.

---

## 🔧 Tips & Best Practices

- Describe tools well: Rich `ToolSpec` metadata (inputs/returns, when-to-use) improves tool routing and reduces retries.  
- Self-repair & pruning: Enable `repair_with_llm=True` and/or `allow_input_pruning=True` in `ReActConfig` to make agents resilient to malformed Actions.  
- Cache SQL schemas: Pre-warm with `SQLAgent.train()` at startup for low-latency first answers.  
- Respect token budgets: Use `RAGRetrieverAgent`’s `max_new_tokens` and `generation_budget` to avoid truncation.  
- Pass runtime context: Extra kwargs to `agent.run(...)` get merged into tool calls—inject DB handles, feature flags, etc., without exposing them to the LLM (via `ToolResponse.extras`).

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