<div align="center">
  <img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/banner/neurosurfer_banner_black.png" alt="Neurosurfer — AI Agent Framework" width="50%"/>
  <img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/neurosurfer_water_wave.svg" alt="Neurosurfer — AI Agent Framework" width="100%"/>
  
  <a href="https://naumanhsa.github.io/neurosurfer/#quick-start" target="_blank"><img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/quick_start_button.png" height="40" alt="Quick Start"></a>
  <a href="https://naumanhsa.github.io/neurosurfer/examples/" target="_blank"><img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/examples_button.png" height="40" alt="Examples"></a>
  <a href="https://naumanhsa.github.io/neurosurfer/" target="_blank"><img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/documentation_button.png" height="40" alt="Documentation"></a>
  <a href="https://pypi.org/project/neurosurfer/" target="_blank"><img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/pypi_button.png" height="40" alt="PyPI"></a>
  <a href="https://discord.gg/naumanhsa" target="_blank"><img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/discord_button.png" height="40" alt="Discord"></a>


</div>

**Neurosurfer** helps you build intelligent apps that blend **LLM reasoning**, **tools**, and **retrieval**, with a ready-to-run **OpenAI-compatible FastAPI gateway**. Start lean, add power as you go — CPU-only or GPU-accelerated.

---

## 🚀 What’s in the box

- 🤖 **Agents**: Production-ready patterns for ReAct, SQL, RAG, Router etc. think → act → observe → answer
- 🧠 **Models**: Unified interface for OpenAI-style and local backends like Transformers/Unsloth, vLLM, Llama.cpp etc.
- 📚 **RAG**: Simple, swappable retrieval core: embed → search → format → **token-aware trimming**
- ⚙️ **OpenAI-Compatible Gateway**: `/v1/models` + `/v1/chat/completions` — proxy to upstream backends (vLLM/OpenAI) or route to your own agents, with hooks for request/response customization
- 🧪 **CLI**: `neurosurfer serve` to run the backend — custom backend app support

---

<h2>🎓 Tutorials</h2>

<table style="width:100%; border-collapse: collapse; text-align: left;">
  <thead>
    <tr style="border-bottom: 2px solid #ccc;">
      <th style="width:5%;">#</th>
      <th style="width:20%;">Tutorial</th>
      <th style="width:15%;">Link</th>
      <th style="width:60%;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td><strong>Neurosurfer Quickstart</strong></td>
      <td>
        <a href="https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/00_neurosurfer_quickstart.ipynb" target="_blank">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" style="vertical-align: middle;"/>
        </a>
      </td>
      <td>
        Learn how to load local and OpenAI models, stream responses, and build your first RAG and tool-based agents directly in Jupyter or Colab.
      </td>
    </tr>
  </tbody>
</table>

<p style="margin-top: 10px;">
  <em>More tutorials coming soon — covering <strong>RAG</strong>, <strong>Custom Tools</strong>, <strong>More on Agents</strong>, <strong>FastAPI integration</strong> and more.</em>
</p>

---

## 🗞️ News

- **Agents**: ReAct & SQLAgent upgraded with bounded retries, spec-aware input validation, and better final-answer streaming; new **ToolsRouterAgent** for quick one-shot tool picks.
- **Models**: Cleaner OpenAI-style responses across backends; smarter token budgeting + fallbacks when tokenizer isn’t available.
- **Server**: Rebuilt as a lean OpenAI-compatible gateway — proxy to upstream backends (vLLM/OpenAI) or register agents as models, with hooks for request/response customization.
- **CLI**: `serve` now runs the backend gateway only (the bundled NeurowebUI has been removed).

> Looking for older updates? Check the repo **Releases** and **Changelog**.

---


## ⚡ Quick Start

A 60-second path from install → dev server → your first inference.

**Install (minimal core):**
```bash
pip install -U neurosurfer
```

**Or full LLM stack (torch, transformers, bnb, unsloth):**
```bash
pip install -U "neurosurfer[torch]"
```

**Run the dev server:**
```bash
neurosurfer serve
```
- Backend binds to config defaults; override with flags or envs.

**Hello LLM Example:**
```python
from neurosurfer.models.chat_models.transformers import TransformersModel

llm = TransformersModel(
  model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
  load_in_4bit=True
)
res = llm.ask(user_prompt="Say hi!", system_prompt="Be concise.", stream=False)
print(res.choices[0].message.content)
```

## 🏗️ High-Level Architecture
<div align="center">
  <img alt="Neurosurfer Architecture" src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/architecture/neurosurfer_architecture_light.png" width="100%"/>
  <p><strong>Neurosurfer Architecture</strong></p>
</div>

## ✨ Key Features

- **Production API** — FastAPI backend with auth, chat APIs, and OpenAI-compatible endpoints → [Server setup](https://naumanhsa.github.io/neurosurfer/server/)

- **Intelligent Agents** — Build ReAct, SQL, and RAG agents with minimal code, optimized for specific tasks → [Learn about agents](https://naumanhsa.github.io/neurosurfer/api-reference/agents/)

- **Rich Tool Ecosystem** — Built-in tools (calculator, web calls, files) plus easy custom tools → [Explore tools](https://naumanhsa.github.io/neurosurfer/api-reference/tools/)

- **RAG System** — Ingest, chunk, and retrieve relevant context for your LLMs → [RAG System](https://naumanhsa.github.io/neurosurfer/api-reference/rag/)

- **Vector Databases** — Built-in ChromaDB with an extensible interface for other stores → [Vector stores](https://naumanhsa.github.io/neurosurfer/api-reference/vectorstores/)

- **Multi-LLM Support** — OpenAI, Transformers/Unsloth, vLLM, Llama.cpp, and OpenAI-compatible APIs → [Model docs](https://naumanhsa.github.io/neurosurfer/api-reference/models/)

## 📦 Install Options

**pip (recommended)**
```bash
pip install -U neurosurfer
```

**pip + full LLM stack**
```bash
pip install -U "neurosurfer[torch]"
```

**From source**
```bash
git clone https://github.com/NaumanHSA/neurosurfer.git
cd neurosurfer && pip install -e ".[torch]"
```

CUDA notes (Linux x86_64):
```bash
# Wheels bundle CUDA; you just need a compatible NVIDIA driver.
pip install -U torch --index-url https://download.pytorch.org/whl/cu124
# or CPU-only:
pip install -U torch --index-url https://download.pytorch.org/whl/cpu
```

## 📝 License

Licensed under **Apache-2.0**. See [`LICENSE`](https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/LICENSE).

## 🌟 Support

- ⭐ Star the project on [GitHub](https://github.com/NaumanHSA/neurosurfer).
- 💬 Ask & share in **Discussions**: [Discussions](https://github.com/NaumanHSA/neurosurfer/discussions).
- 🧠 Read the [Docs](https://naumanhsa.github.io/neurosurfer/).
- 🐛 File [Issues](https://github.com/NaumanHSA/neurosurfer/issues).
- 🔒 Security: report privately to **naumanhsa965@gmail.com**.

## 📚 Citation

If you use **Neurosurfer** in your work, please cite:

```bibtex
@software{neurosurfer,
  author       = {Nouman Ahsan and Neurosurfer contributors},
  title        = {Neurosurfer: A Production-Ready AI Agent Framework},
  year         = {2025},
  url          = {https://github.com/NaumanHSA/neurosurfer},
  version      = {0.1.0},
  license      = {Apache-2.0}
}
```

---

<div align="center">
  <sub>Built with ❤️ by the Neurosurfer team
</div>