import os
import time
import logging
from typing import Any, Dict

from neurosurfer.server import NeurosurferServer
from neurosurfer.server.backends import UpstreamBackend
from neurosurfer.server.hooks import Hook, HookContext, StripReasoningHook, SystemPromptInjectorHook

from neurosurfer.models.chat_models.openai import OpenAIModel

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("neurosurfer.quickstart")


# -------------------------
# Custom hook example
# -------------------------
class AddMetadataHook(Hook):
    """
    Example customization:
    - Add a metadata field to the request (Open-WebUI usually passes extra fields; we allow them)
    - Add a header-ish flag inside metadata so you can route behavior later
    """

    async def before_chat(self, ctx: HookContext, req: dict) -> dict:
        meta = req.get("metadata") or {}
        meta["ns_gateway"] = True
        meta["request_id"] = ctx.request_id
        req["metadata"] = meta
        return req

    async def after_chat(self, ctx: HookContext, resp: dict) -> dict:
        # Example: attach request_id for debugging
        resp.setdefault("metadata", {})
        resp["metadata"]["request_id"] = ctx.request_id
        return resp

    async def stream_chunk(self, ctx: HookContext, chunk: dict) -> dict:
        # Example: no-op, but you can edit chunk["choices"][0]["delta"]["content"]
        return chunk


# -------------------------
# Config
# -------------------------
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "abc")  # vLLM usually ignores but OpenAI clients expect it
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3-8B-unsloth-bnb-4bit")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8081"))

# -------------------------
# 1) Create OpenAIModel that points to vLLM
#    (useful for agent execution, warmups, internal calls)
# -------------------------
llm = OpenAIModel(
    model_name=MODEL_NAME,
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
    strip_reasoning=False,
    max_seq_length=8192,
    logger=LOGGER,
)

# -------------------------
# 2) Create gateway server
#    Module-level `ns` so the CLI (`neurosurfer serve`) can import and run it directly.
# -------------------------
ns = NeurosurferServer(
    app_name="Neurosurfer OpenAI Gateway",
    host=HOST,
    port=PORT,
    enable_docs=True,
    api_keys=[],  # put ["token1"] if you want Open-WebUI to require it
    cors_origins=["*"],
    cors_allow_credentials=False,
    workers=int(os.getenv("WORKERS", "1")),
    log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
)

# -------------------------
# 3) Register hooks (customization layer)
# -------------------------
ns.add_hook(SystemPromptInjectorHook("You are a helpful assistant."))
ns.add_hook(AddMetadataHook())
# If your model outputs <think>...</think>, enable:
# ns.add_hook(StripReasoningHook())

# -------------------------
# 4) Register upstream backend (vLLM)
#    This makes /v1/models include vLLM models and routes requests to vLLM.
# -------------------------
ns.register_backend(
    UpstreamBackend(
        name="vllm",
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        models_mode="proxy",  # forward /models from vLLM
    ),
    default=True,
)


def main():
    # Optional warmup (prove vLLM is reachable) — done here, not at import time.
    try:
        resp = llm.ask(user_prompt="Say hi in 3 words.", system_prompt="", stream=False)
        LOGGER.info("Warmup OK: %s", resp.choices[0].message.content)
    except Exception as e:
        LOGGER.warning("Warmup failed (is vLLM running at %s?): %s", VLLM_BASE_URL, e)

    LOGGER.info("Gateway starting on http://%s:%d", HOST, PORT)
    LOGGER.info("Point Open-WebUI to: http://%s:%d/v1", HOST, PORT)
    ns.run()


if __name__ == "__main__":
    main()
