# Agent Examples

Practical, copy‑pasteable examples for Neurosurfer’s agents:
- **ReActAgent** — tool‑using, step‑by‑step reasoning
- **RAGRetrieverAgent** — retrieval‑augmented prompting with token‑aware context budgeting
- **SQLAgent** — database Q&A (schema → query → execution → NL format)
- **ToolsRouterAgent** — lightweight, single‑step tool picker

Each example uses the same BaseModel interface shown in [model examples](models-examples.md).

---

## 🤖 ReActAgent — quick start

The ReAct agent thinks → acts (uses tools) → observes → repeats, until it emits a final answer.
> If you enabled `ReActAgentConfig.skip_special_tokens=True`, the final‑answer delimiters are not emitted in the stream.

### Hello, tools (UI‑friendly)

```python
from neurosurfer.agents.react import ReActAgent, ReActAgentConfig
from neurosurfer.tools import Toolkit
from neurosurfer.tools.common import CalculatorTool
from neurosurfer.models.chat_models.openai import OpenAIModel

# 1) Model
llm = OpenAIModel(model_name="gpt-4o-mini")

# 2) Tools (use a built-in calculator for simplicity)
tk = Toolkit()
tk.register_tool(CalculatorTool())

# 3) Agent
agent = ReActAgent(
    toolkit=tk,
    llm=llm,
    specific_instructions="Be concise.",
    config=ReActAgentConfig(temperature=0.2, max_new_tokens=256)
)

# 4) Run (stream to your UI)
for chunk in agent.run("Solve: (42 * 7) - 5^2"):
    # chunk is either plain text or ChatCompletionChunk content
    print(chunk, end="", flush=True)
print()
```

### Custom tool with strict input validation

Assume your `Toolkit` validates inputs based on a spec. Here’s a tiny echo tool:

```python
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools import Toolkit

class EchoTool(BaseTool):
    name = "echo"
    description = "Repeat a message. Inputs: message (str)."
    def __call__(self, *, message: str, **_):
        return ToolResponse(results=f"[echo] {message}")

tk = Toolkit()
tk.register_tool(EchoTool())

agent = ReActAgent(toolkit=tk, llm=llm, specific_instructions="Use tools when helpful.")
for chunk in agent.run("Say hello using the echo tool. Message='Hello world!'", temperature=0.1):
    print(chunk, end="")
```

---

## 📄 RAGAgent — retrieval + token‑aware prompts

Use any `BaseVectorDB` and `BaseEmbedder`. The agent retrieves, builds a formatted context block, **trims it to fit** the model’s window, and returns budgets you can pass into your generation call.

### Retrieve then generate

```python
from neurosurfer.agents.rag import RAGAgent, RAGAgentConfig
from neurosurfer.vectorstores.chroma import ChromaVectorStore
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder
from neurosurfer.models.chat_models.openai import OpenAIModel

# Components
store = ChromaVectorStore(collection_name="docs")
embedder = SentenceTransformerEmbedder("intfloat/e5-small-v2")
llm = OpenAIModel(model_name="gpt-4o-mini")

# Agent
rag = RAGAgent(llm=llm, vectorstore=store, embedder=embedder, config=RAGAgentConfig(top_k=6))

# Retrieve + build context
res = rag.retrieve(
    user_query="Summarize how the ingestion pipeline works.",
    base_system_prompt="You are a precise technical writer.",
    base_user_prompt="Use the following context to answer succinctly:\n\n{context}\n\nQuestion: Summarize the ingestion pipeline.",
)

# Now call the model with trimmed budgets
answer = llm.ask(
    user_prompt=res.base_user_prompt.replace("{context}", res.context),
    system_prompt=res.base_system_prompt,
    max_new_tokens=res.max_new_tokens,     # dynamically chosen to fit
    temperature=0.2,
)
print(answer.choices[0].message.content)
```

### Pick top files by grouped chunk hits

```python
files = rag.pick_files_by_grouped_chunk_hits(
    section_query="vector store indexing rules",
    candidate_pool_size=300,
    n_files=5
)
print(files)  # top relevant file paths from your corpus
```

> The retriever exposes: `context`, `max_new_tokens`, `generation_budget`, and the retrieved `docs` & `distances` for transparency/debugging.

---

## 🗄️ SQLAgent — natural‑language database Q&A

The SQL agent wires the ReAct loop with SQL‑specific tools: schema discovery, query generation, execution, and result formatting.

### Connect and query (SQLite example)

```python
from neurosurfer.agents.sql_agent import SQLAgent
from neurosurfer.models.chat_models.openai import OpenAIModel

llm = OpenAIModel(model_name="gpt-4o-mini")
agent = SQLAgent(
    llm=llm,
    db_uri="sqlite:///my.db",
    sample_rows_in_table_info=3,
    verbose=True,
)

# (Optional) Ensure schema store is ready (train/summarize once and cache)
for chunk in agent.train(summarize=True, force=False):
    print(chunk, end="")

# Ask the database
for piece in agent.run("Top 5 products by revenue last quarter, with totals."):
    print(piece, end="")
```

### Insights and error recovery

```python
# The agent can also compute insights / stats via tools (registered in get_toolkit())
# If a generated SQL has an error, the agent refines and retries using results.
for x in agent.run("Average order value per month for 2024; format as a small table."):
    print(x, end="")
```

---

## 🔀 ToolsRouterAgent — let the LLM choose the tool

Use this when you want a light router that picks a single tool and runs it (no multi‑step planning).

```python
from neurosurfer.agents.tools_router import ToolsRouterAgent, ToolsRouterConfig, RouterRetryPolicy
from neurosurfer.tools import Toolkit
from neurosurfer.tools.common import CalculatorTool
from neurosurfer.models.chat_models.openai import OpenAIModel

llm = OpenAIModel(model_name="gpt-4o-mini")

tk = Toolkit()
tk.register_tool(CalculatorTool())          # add your tools
# tk.register_tool(WeatherTool())
# tk.register_tool(WebSearchTool())

router = ToolsRouterAgent(
    toolkit=tk,
    llm=llm,
    verbose=True,
    config=ToolsRouterConfig(
        allow_input_pruning=True,
        repair_with_llm=True,
        return_stream_by_default=True,
        retry=RouterRetryPolicy(max_route_retries=2, max_tool_retries=1, backoff_sec=0.7),
    ),
)

# Streaming
for chunk in router.run("Compute 3.5% of 12000 and explain briefly.", stream=True):
    print(chunk, end="")

# Non-stream
text = router.run("Quickly add 17 + 29", stream=False)
print("\nRESULT:", text)
```

---

## Tips

- **Keep tools small and deterministic.** The ReAct loop improves when each tool has clear inputs/outputs.
- **Use streaming for UX.** All agents integrate nicely with token/line streaming UIs.
- **Pass runtime context** (e.g., `db_engine`, `vectorstore`, `embedder`) through the agent/tool call as shown.
- **Budget tokens.** With RAGAgent, always use `res.max_new_tokens` to avoid overruns.
- **Cache SQL schema summaries.** `SQLAgent.train(...)` will speed up later queries.
- **Router retries.** Tune `RouterRetryPolicy` for your latency/error profile; set strict mode by turning off `allow_input_pruning`.