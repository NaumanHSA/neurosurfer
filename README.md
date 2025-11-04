<div align="center">
  <img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/banner/neurosurfer_logo_text_white.png" alt="Neurosurfer — AI Agent Framework" width="50%"/>
  <img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/banner/neurosurfer_banner.svg" alt="Neurosurfer — AI Agent Framework" width="100%"/>
  
  <a href="https://naumanhsa.github.io/neurosurfer/#quick-start"><img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/quick_start_button.png" height="40" alt="Quick Start"></a>
  <a href="https://naumanhsa.github.io/neurosurfer/examples/"><img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/examples_button.png" height="40" alt="Examples"></a>
  <a href="https://naumanhsa.github.io/neurosurfer/"><img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/documentation_button.png" height="40" alt="Documentation"></a>
  <a href="https://pypi.org/project/neurosurfer/"><img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/pypi_button.png" height="40" alt="PyPI"></a>
  <a href=""><img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/discord_button.png" height="40" alt="Discord"></a>


</div>

**Neurosurfer** helps you build intelligent apps that blend **LLM reasoning**, **tools**, and **retrieval** with a ready-to-run **FastAPI** backend and a **React** dev UI. Start lean, add power as you go — CPU-only or GPU-accelerated.

- 🧩 **OpenAI-style API** with streaming & tool-calling  
- 📚 **RAG-ready**: ingest → chunk → retrieve → augment  
- 🤖 **Agents** (ReAct, SQL, RAG) + 🔧 **Tools** (calc, web, custom)  
- 🧠 **Multi-LLM**: OpenAI, Transformers/Unsloth, vLLM, Llama.cpp, more  
- 🖥️ **NeurowebUI** (React) for chat UX, threads, uploads

## 🗞️ News

- **CLI `serve` improvements** — run backend-only or UI-only, inject `VITE_BACKEND_URL` automatically. See [:link: CLI guide](https://naumanhsa.github.io/neurosurfer/cli/).  
- **Model registry & RAG hooks** — easier wiring for multi-model setups. See [:link: Example App](https://naumanhsa.github.io/neurosurfer/server/example-app/).  
- **Optional LLM stack** — install heavy deps only when you need them:  
  ```bash
  pip install "neurosurfer[torch]"
  ```

> Looking for older updates? Check the repo **Releases** and **Changelog**.


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

**Run the dev server (backend + UI):**
```bash
neurosurfer serve
```
- Auto-detects UI; pass `--ui-root` if needed. First run may `npm install`.  
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
  <img alt="Neurosurfer Architecture" src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/architecture/neurosurfer_architecture_dark.png" width="100%"/>
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