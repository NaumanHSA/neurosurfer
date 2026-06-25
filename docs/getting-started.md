# Getting Started

Welcome to **Neurosurfer** — a production‑grade AI framework with multi‑LLM, RAG, agents, tools, and a FastAPI server. This page walks you through installation, the CLI, and basic Python usage. It’s written in a hybrid style: short explanatory paragraphs with bullets where they help.

## 🚀 Quick Navigation

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install via pip or from source, and learn about CPU/GPU (CUDA/MPS) options.

    [:octicons-arrow-right-24: Go to Installation](#installation)

-   :material-console:{ .lg .middle } **CLI Usage**

    ---

    Serve the backend gateway with one command.

    [:octicons-arrow-right-24: Go to CLI](#cli-usage)

-   :material-robot-happy-outline:{ .lg .middle } **Basic Usage**

    ---

    Import models & agents, call an LLM, and plug in RAG.

    [:octicons-arrow-right-24: Go to Basic Usage](#basic-usage)

-   :material-cog:{ .lg .middle } **Configuration**

    ---

    Configure API keys, models, server options, and environment variables.

    [:octicons-arrow-right-24: Open Configuration](./api-reference/configuration.md)

</div>

---

## 📋 Prerequisites

Neurosurfer installs as a lightweight core so you can get started quickly. A typical setup uses Python 3.9+ and a virtual environment. Hardware acceleration is optional but recommended when working with larger models.

- **Python** `>= 3.9` and **pip** (or **uv/pipx/poetry**)
- **GPU** (optional): NVIDIA CUDA on Linux x86_64, or Apple Silicon (MPS) on macOS arm64. CPU‑only works fine for smaller models or demos.

!!! tip "Keep installs lightweight"
    Neurosurfer’s core installs **without** heavy model deps. Add the full LLM stack only when you need it via the `torch` extra.

---

## 🧰 Installation

Installation is flexible: minimal core for APIs, or the full LLM stack for local inference/finetuning. If you’re unsure, start minimal; you can always add the extra later.

### Minimal core

```bash
pip install -U neurosurfer
```

This keeps dependencies light — ideal for API servers, docs, or CI.

### Full LLM stack (recommended for model work)

```bash
pip install -U 'neurosurfer[torch]'
```

This extra adds `torch`, `transformers`, `sentence-transformers`, `accelerate`, `bitsandbytes`, and `unsloth`.

!!! info "OS Platform"
    Neurosurfer has only been tested on Ubuntu Linux x86_64. It is expected to work on Windows and macOS, but you may need to verify CUDA related dependencies.

### GPU / CUDA notes (PyTorch)

If you want GPU acceleration, the **pip wheel already bundles CUDA**; you do **not** need to install the system CUDA toolkit. You *do* need a compatible NVIDIA driver. Use these commands to verify and install appropriately:

1. **Check your GPU and driver**  
   ```bash
   nvidia-smi
   ```  
   Confirm the driver is active and note the CUDA version the driver supports.

2. **Install a CUDA‑enabled torch wheel** (Linux x86_64):  
   ```bash
   # Example for CUDA 12.4 wheels:
   pip install -U torch --index-url https://download.pytorch.org/whl/cu124
   ```

3. **Install bitsandbytes** (optional, Linux x86_64):  
   ```bash
   pip install -U bitsandbytes
   ```  
   On macOS/Windows, bitsandbytes wheels are limited; prefer CPU or other quantization strategies.

4. **Apple Silicon (MPS)**:  
   ```bash
   pip install -U torch
   ```  
   PyTorch includes MPS support on macOS arm64. No extra toolkit required.

5. **CPU‑only** (portable):  
   ```bash
   pip install -U torch --index-url https://download.pytorch.org/whl/cpu
   ```

!!! warning "Driver/toolkit alignment"
    If `torch.cuda.is_available()` is `False` despite a GPU, your **driver may be outdated**. Update the NVIDIA driver to a version compatible with the wheel you installed (e.g., CU124) and restart. You generally do **not** need the full CUDA toolkit for pip wheels.

!!! tip "Quick GPU sanity check"
    ```python
    import torch; print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
    ```

### Build from source

```bash
git clone https://github.com/NaumanHSA/neurosurfer.git
cd neurosurfer
pip install -U pip wheel build
pip install -e .            # editable
# or
pip install -e '.[torch]'   # editable + full LLM stack
```

### Quick import check

```bash
python -c "import neurosurfer; print('ok')"
```

You’ll see a banner; if optional deps are missing, you’ll get a single consolidated warning with install hints.

!!! tip "Useful environment flags"
    - `NEUROSURFER_SILENCE=1` — hide banner & warnings  
    - `NEUROSURFER_EAGER_RUNTIME_ASSERT=1` — fail fast at import if LLM deps missing (opt‑in)

---

## 🖥️ CLI Usage

The CLI runs the backend gateway API. Start simple with `neurosurfer serve`, then refine with flags as you scale.

### Help

```bash
neurosurfer --help
neurosurfer serve --help
```

### Start the backend (dev)

```bash
neurosurfer serve
```

The backend binds to `NEUROSURFER_BACKEND_HOST` / `NEUROSURFER_BACKEND_PORT` (from config).

### Common options

- **Backend app** (`--backend-app`): module path like `pkg.module:app_or_factory()` or a Python file with a `NeurosurferServer` instance. The default is `neurosurfer.examples.quickstart_app:ns`.
- **Backend**: `--backend-host`, `--backend-port`, `--backend-log-level`, `--backend-reload`, `--backend-workers`, `--backend-worker-timeout`.

### Examples

**Custom host/port**
```bash
neurosurfer serve --backend-host 0.0.0.0 --backend-port 8000
```

**Serve your own file**
```bash
neurosurfer serve --backend-app ./app.py --backend-reload
```

**Serve module**
```bash
neurosurfer serve --backend-app mypkg.myapp:ns
```

**Public backend URL (when binding 0.0.0.0)**
```bash
export NEUROSURFER_PUBLIC_HOST=your.ip.addr
neurosurfer serve
```

---

## 🤖 Basic Usage

A minimal model call shows the shape of the API; agents and RAG come next. If LLM dependencies are missing, you’ll get clear, actionable errors instructing you to install the extra.

### Minimal model call

```python
from neurosurfer.models.chat_models.transformers import TransformersModel

llm = TransformersModel(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
    enable_thinking=False,
    stop_words=[],
)

resp = llm.ask(user_prompt="Say hi!", system_prompt="You are helpful.", stream=False)
print(resp.choices[0].message.content)
```

### Agents

```python
from neurosurfer.agents import ReActAgent
from neurosurfer.tools import Toolkit
from neurosurfer.models.chat_models.transformers import TransformersModel

llm = TransformersModel(model_name="meta-llama/Llama-3.2-3B-Instruct")
agent = ReActAgent(toolkit=Toolkit(), llm=llm)

print(agent.run("What is the capital of France?"))
```

### RAG Basics

```python
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder
from neurosurfer.rag.chunker import Chunker
from neurosurfer.rag.filereader import FileReader
from neurosurfer.app.server.services.rag_orchestrator import RAGOrchestrator

embedder = SentenceTransformerEmbedder(model_name="intfloat/e5-large-v2")
rag = RAGOrchestrator(
    embedder=embedder,
    chunker=Chunker(),
    file_reader=FileReader(),
    persist_dir="./rag_store",
    max_context_tokens=2000,
    top_k=15,
    min_top_sim_default=0.35,
    min_top_sim_when_explicit=0.15,
    min_sim_to_keep=0.20,
)

aug = rag.apply(
    actor_id=1,
    thread_id="demo",
    user_query="Summarize the attached report",
    files=[{"name":"report.pdf","content":"...base64...","type":"application/pdf"}],
)
print("Augmented query:", aug.augmented_query)
```

---

## ⚙️ Configuration

The full configuration guide covers API keys, model providers, server settings, logging, and environment variables.
Read it here → **[Configuration](./api-reference/configuration.md)**.

---

## ➡️ Next Steps & Help

Explore the broader documentation and examples to deepen your setup:

- **API Reference** — classes, methods, schemas [:octicons-arrow-up-right-24](./api-reference/index.md)
- **Examples** — runnable code samples [:octicons-arrow-up-right-24](./examples/index.md)
- **CLI** — command line interface [:octicons-arrow-up-right-24](./cli.md)

Installation issues? Try the full stack extra:
```bash
pip install -U 'neurosurfer[torch]'
```

GPU problems? Verify with `nvidia-smi`, confirm driver versions, and test `torch.cuda.is_available()`. For quantization on non‑Linux platforms, consider CPU or alternative strategies.

---

**Ready?** Jump to [:material-download: Installation](#installation) or try the [:material-console: CLI](#cli-usage).