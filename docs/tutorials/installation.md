# Installation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/00_installation.ipynb)

Get Neurosurfer installed and confirm it can reach a model.

## Minimal walkthrough

Install the core, plus any extras you need (imports are lazy, so the base install stays light):

```bash
pip install -U neurosurfer
# common combo: web search + gateway
pip install -U "neurosurfer[search,serve]"
```

Point it at a model via the environment (here, a local OpenAI-compatible server):

```bash
export LLM_PROVIDER=openai
export OPENAI_BASE_URL=http://localhost:1234/v1
export MODEL=qwen/qwen3.5-9b
```

Verify the setup — `doctor` reports the resolved provider and whether the endpoint answers:

```bash
neurosurfer doctor
```

## Full notebook

The [Colab notebook](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/00_installation.ipynb)
walks through every extra and both cloud (Anthropic/OpenAI) and local setups.

**Next:** [Installation reference](../getting-started/installation.md) ·
[Configuration](../reference/configuration.md) · [Tutorial 1 →](providers-and-agents.md)
