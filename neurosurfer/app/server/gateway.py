from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .api.router import build_router
from .backends.agent import AgentBackend, AgentSpec
from .backends.upstream import UpstreamBackend
from .config import ServerSettings
from .hooks.base import Hook
from .middleware import OpenAIErrorMiddleware
from .registry import ModelRouter, RouteTarget


class NeurosurferServer:
    """OpenAI-compatible FastAPI gateway.

    Usage::

        server = NeurosurferServer(port=8000)

        # Optional: proxy to an upstream LLM backend
        server.register_backend(UpstreamBackend(name="vllm", base_url="http://localhost:8001/v1"))

        # Optional: expose a native agent as a model
        loop = AgenticLoop(provider=..., tools=[...])
        server.register_agent(loop, model_id="my-agent")

        server.run()
    """

    def __init__(
        self,
        *,
        app_name: str | None = None,
        api_keys: list[str] | None = None,
        enable_docs: bool | None = None,
        cors_origins: list[str] | None = None,
        cors_allow_credentials: bool | None = None,
        host: str | None = None,
        port: int | None = None,
        reload: bool | None = None,
        log_level: str | None = None,
        workers: int | None = None,
        sse_ping_interval_s: float | None = None,
        logger: logging.Logger | None = None,
    ):
        # Load from env/.env first, then apply explicit overrides.
        base = ServerSettings()
        overrides: dict = {}
        if app_name is not None:
            overrides["app_name"] = app_name
        if host is not None:
            overrides["host"] = host
        if port is not None:
            overrides["port"] = port
        if enable_docs is not None:
            overrides["enable_docs"] = enable_docs
        if reload is not None:
            overrides["reload"] = reload
        if log_level is not None:
            overrides["log_level"] = log_level
        if workers is not None:
            overrides["workers"] = workers
        if api_keys is not None:
            overrides["api_keys"] = api_keys
        if cors_origins is not None:
            overrides["cors_origins"] = cors_origins
        if cors_allow_credentials is not None:
            overrides["cors_allow_credentials"] = cors_allow_credentials
        if sse_ping_interval_s is not None:
            overrides["sse_ping_interval_s"] = sse_ping_interval_s

        self.settings = base.model_copy(update=overrides)
        self.logger = logger or logging.getLogger("neurosurfer.gateway")

        self.hooks: list[Hook] = []
        self.router = ModelRouter()
        self._upstream_backend: UpstreamBackend | None = None
        # Workflow execution API (Phase 2). Inject a preconfigured RunManager here
        # (tests / embedders), or leave None and one is built lazily on first use
        # from the configured LLM provider.
        self.run_manager: Any = None

        self.app = FastAPI(
            title=self.settings.app_name,
            docs_url="/docs" if self.settings.enable_docs else None,
            redoc_url=None,
        )

        self.app.add_middleware(OpenAIErrorMiddleware)

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.cors_origins,
            allow_credentials=self.settings.cors_allow_credentials,
            allow_methods=self.settings.cors_allow_methods,
            allow_headers=self.settings.cors_allow_headers,
        )

        @self.app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            if self.settings.api_keys:
                auth = request.headers.get("authorization") or ""
                token = (
                    auth.split(" ", 1)[1]
                    if auth.startswith("Bearer ") and " " in auth
                    else None
                )
                if token not in self.settings.api_keys:
                    from fastapi.responses import JSONResponse

                    return JSONResponse(
                        status_code=401,
                        content={
                            "error": {
                                "message": "Unauthorized",
                                "type": "invalid_request_error",
                                "param": None,
                                "code": None,
                            }
                        },
                    )
            return await call_next(request)

        self.app.include_router(build_router(self))

    # ── Registration ──────────────────────────────────────────────────────────

    def add_hook(self, hook: Hook) -> None:
        self.hooks.append(hook)

    def register_backend(self, backend: UpstreamBackend, *, default: bool = True) -> None:
        self._upstream_backend = backend
        if default:
            self.router.set_default_backend(backend)

    def register_agent(
        self,
        agent: Any,
        *,
        model_id: str,
        description: str = "Neurosurfer agent",
        owned_by: str = "neurosurfer",
        max_model_len: int = 8192,
        run_fn=None,
        result_to_text=None,
    ) -> None:
        """Register a native agent as a model endpoint.

        ``agent`` may be an :class:`~neurosurfer.agents.AgenticLoop`,
        :class:`~neurosurfer.agents.ReactAgent`, or
        :class:`~neurosurfer.agents.Agent` instance — or any object with a
        ``run(prompt)`` method.  Pass ``run_fn`` to override the invocation logic.
        """
        spec = AgentSpec(
            agent=agent,
            model_id=model_id,
            description=description,
            owned_by=owned_by,
            max_model_len=max_model_len,
            run_fn=run_fn,
            result_to_text=result_to_text or AgentSpec.result_to_text,
        )
        backend = AgentBackend(spec)
        self.router.register_model(model_id, RouteTarget(backend=backend))

    # ── Serving ───────────────────────────────────────────────────────────────

    def create_app(self) -> FastAPI:
        """Return the underlying FastAPI app (for ASGI deployment)."""
        return self.app

    def run(self, host: str | None = None, port: int | None = None, **kwargs: Any) -> None:
        """Start the uvicorn server (blocking)."""
        try:
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "uvicorn is required to run the server. "
                "Install it with: pip install 'neurosurfer[serve]'"
            ) from e

        uvicorn.run(
            self.app,
            host=host or self.settings.host,
            port=port or self.settings.port,
            reload=kwargs.get("reload", self.settings.reload),
            log_level=kwargs.get("log_level", self.settings.log_level),
            workers=kwargs.get("workers", self.settings.workers),
        )
