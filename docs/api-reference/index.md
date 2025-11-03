# API Reference

Complete API documentation for all Neurosurfer modules and classes.

## 📚 Quick Navigation

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } **Agents**

    ---

    ReAct, SQL, and RAG agents for different use cases

    [:octicons-arrow-right-24: View agents](agents/index.md)

-   :material-brain:{ .lg .middle } **Models**

    ---

    LLM and embedding models from various providers

    [:octicons-arrow-right-24: View models](models/index.md)

-   :material-book-open-variant:{ .lg .middle } **RAG System**

    ---

    Document ingestion, chunking, and retrieval

    [:octicons-arrow-right-24: View RAG](rag/index.md)

-   :material-toolbox:{ .lg .middle } **Tools**

    ---

    Built-in and custom tools for agents

    [:octicons-arrow-right-24: View tools](tools/index.md)

-   :material-database:{ .lg .middle } **Vector Stores**

    ---

    Vector database integrations

    [:octicons-arrow-right-24: View vector stores](vectorstores/index.md)

-   :material-table:{ .lg .middle } **Database**

    ---

    SQL database utilities

    [:octicons-arrow-right-24: View database](database/index.md)

-   :material-server:{ .lg .middle } **Server**

    ---

    FastAPI server and API endpoints

    [:octicons-arrow-right-24: View server](../server/index.md)

</div>

---

## 🔍 Module Overview

### Core Modules

| Module | Description | Key Classes |
|--------|-------------|-------------|
| [Agents](agents/index.md) | AI agents for various tasks | `ReActAgent`, `SQLAgent`, `RAGRetrieverAgent` |
| [Models](models/index.md) | LLM and embedding models | `OpenAIModel`, `AnthropicModel`, `OllamaModel` |

### Data & Retrieval

| Module | Description | Key Classes |
|--------|-------------|-------------|
| [RAG](rag/index.md) | Retrieval-augmented generation | `Chunker`, `FileReader`, `RAGIngestor` |
| [Vector Stores](vectorstores/index.md) | Vector databases | `ChromaVectorStore`, `BaseVectorDB` |
| [Database](database/index.md) | SQL database utilities | `SQLDatabase` |

### Tools & Extensions

| Module | Description | Key Classes |
|--------|-------------|-------------|
| [Tools](tools/index.md) | Agent tools and toolkit | `BaseTool`, `Toolkit`, `ToolSpec` |
| [Server](../server/index.md) | Production API server | `NeurosurferApp`, API endpoints |

---

## 🎯 Common Use Cases

### Building Agents

```python
from neurosurfer import ReActAgent
from neurosurfer.models.chat_models.openai import OpenAIModel

model = OpenAIModel(model_name="gpt-4")
agent = ReActAgent(model=model, tools=tools)
```

**See:** [ReActAgent](agents/react-agent.md) | [Models](models/index.md) | [Tools](tools/index.md)

### RAG Document Q&A

```python
from neurosurfer import RAGRetrieverAgent
from neurosurfer.rag import RAGIngestor
from neurosurfer.vectorstores import ChromaVectorStore

ingestor = RAGIngestor(embedder=embedder, vectorstore=chroma)
agent = RAGRetrieverAgent(model=model, vectorstore=chroma)
```

**See:** [RAGIngestor](rag/ingestor.md) | [ChromaVectorStore](vectorstores/chroma.md)

### SQL Database Queries

```python
from neurosurfer import SQLAgent
from neurosurfer.db import SQLDatabase

db = SQLDatabase("postgresql://...")
agent = SQLAgent(model=model, database=db)
```

**See:** [SQLAgent](agents/sql-agent.md) | [SQLDatabase](database/sql_database.md)

---

## 📖 Documentation Conventions

### Type Hints

All parameters and return values include type hints:

```python
def method(param: str, count: int = 10) -> List[str]:
    ...
```

### Optional Parameters

Parameters with defaults are marked:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `required_param` | `str` | *required* | Must be provided |
| `optional_param` | `int` | `10` | Has default value |

### Examples

All methods include working examples:

```python
# Always tested and runnable
result = agent.run("example query")
```

---

## 🔗 External References

- [GitHub Repository](https://github.com/yourusername/neurosurfer)
- [PyPI Package](https://pypi.org/project/neurosurfer/)
- [Issue Tracker](https://github.com/yourusername/neurosurfer/issues)

---

## 💡 Tips for Using API Docs

!!! tip "Start with Overview"
    Each section has an overview page explaining concepts before diving into specific classes.

!!! info "Follow Examples"
    All code examples are tested and ready to copy-paste.

!!! success "Check Related Classes"
    Use "See Also" sections to discover related functionality.

---

Ready to explore? Start with [Agents →](agents/index.md) or [Models →](models/index.md)
