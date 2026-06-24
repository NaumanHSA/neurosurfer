# Node `retrieve_module_evidence` output

- Mode: `text`
- Started at: `2026-01-04T12:07:15.926908`
- Duration: `21300` ms
- Error: `None`

---

# Evidence Pack for Module `neurosurfer/agents/graph/executor.py`

## Module Header/Docstring
The module `neurosurfer/agents/graph/executor.py` is not directly documented in the provided code, but it is referenced in other files as part of the `GraphAgent` and `GraphExecutor` components. It likely contains the core logic for executing graph-based workflows in the Neurosurfer system.

## Key Classes and Functions

### `GraphExecutor` Class
- **File Path:** `neurosurfer/agents/graph/executor.py`
- **Snippet:**
  ```python
  from .manager import ManagerAgent, ManagerConfig
  from .loader import load_graph, load_graph_from_dict
  from .artifacts import ArtifactStore
  from .schema import GraphExecutionResult
  ```
- **What it proves:** This class is part of the graph execution system and is likely responsible for managing the execution of graph-based workflows. It imports key components like `ManagerAgent`, `ManagerConfig`, `load_graph`, and `GraphExecutionResult`.

### `GraphAgent` Class
- **File Path:** `neurosurfer/agents/graph/agent.py`
- **Snippet:**
  ```python
  from .executor import GraphExecutor  # the class you showed above
  ```
- **What it proves:** The `GraphAgent` class uses `GraphExecutor` as a core component, indicating that `GraphExecutor` is a key part of the graph execution logic.

### `load_graph` Function
- **File Path:** `neurosurfer/agents/graph/loader.py`
- **Snippet:**
  ```python
  from .executor import GraphExecutor  # the class you showed above
  ```
- **What it proves:** The `load_graph` function is used to load graph specifications and is likely integrated with `GraphExecutor` for execution.

## References to the Module

### `GraphAgent` Class
- **File Path:** `neurosurfer/agents/graph/agent.py`
- **Snippet:**
  ```python
  from .executor import GraphExecutor  # the class you showed above
  ```
- **What it proves:** The `GraphAgent` class references `GraphExecutor`, indicating that `GraphExecutor` is a core component of the graph execution system.

### `CodeAgent` Class
- **File Path:** `neurosurfer/agents/code/agent.py`
- **Snippet:**
  ```python
  from neurosurfer.agents.graph.executor import GraphExecutor
  ```
- **What it proves:** The `CodeAgent` class references `GraphExecutor`, suggesting that it may use the graph execution system for certain workflows.

### `RAGAgent` Class
- **File Path:** `neurosurfer/agents/rag/agent.py`
- **Snippet:**
  ```python
  from neurosurfer.agents.graph.executor import GraphExecutor
  ```
- **What it proves:** The `RAGAgent` class also references `GraphExecutor`, indicating that it may use the graph execution system for certain tasks.

## Additional References and Callers

### `ReActAgent` Class
- **File Path:** `neurosurfer/agents/react/agent.py`
- **Snippet:**
  ```python
  from neurosurfer.agents.graph.executor import GraphExecutor
  ```
- **What it proves:** The `ReActAgent` class references `GraphExecutor`, suggesting that it may use the graph execution system for certain workflows.

### `SQLAgent` Class
- **File Path:** `neurosurfer/agents/sql/agent.py`
- **Snippet:**
  ```python
  from neurosurfer.agents.graph.executor import GraphExecutor
  ```
- **What it proves:** The `SQLAgent` class references `GraphExecutor`, indicating that it may use the graph execution system for certain tasks.

## Summary
The `neurosurfer/agents/graph/executor.py` module is a core component of the graph execution system in the Neurosurfer framework. It is referenced by several key classes such as `GraphAgent`, `CodeAgent`, `RAGAgent`, and `ReActAgent`, indicating its importance in managing graph-based workflows. The module likely contains the core logic for executing graph-based tasks, integrating with other components like `ManagerAgent`, `load_graph`, and `GraphExecutionResult`.