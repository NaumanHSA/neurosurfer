from __future__ import annotations
import logging
from typing import Any, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from .config import ServerSettings
from .middleware import OpenAIErrorMiddleware
from .api.router import build_router
from .registry import ModelRouter, RouteTarget
from .backends.upstream import UpstreamBackend
from .backends.agent import AgentBackend, AgentSpec
from .hooks.base import Hook


class NeurosurferServer:
    def __init__(
        self,
        *,
        # IMPORTANT: Treat these as optional overrides. Env/.env should be the source of truth.
        app_name: Optional[str] = None,
        api_keys: Optional[list[str]] = None,
        enable_docs: Optional[bool] = None,
        cors_origins: Optional[list[str]] = None,
        cors_allow_credentials: Optional[bool] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        reload: Optional[bool] = None,
        log_level: Optional[str] = None,
        workers: Optional[int] = None,
        sse_ping_interval_s: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ):
        # 1) Load settings from env/.env first, then apply overrides explicitly provided.
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

        # 2) Logger
        self.logger = logger or logging.getLogger("neurosurfer.gateway")

        # 3) Runtime state (per-process; fine with uvicorn workers)
        self.hooks: list[Hook] = []
        self.router = ModelRouter()
        self._upstream_backend: Optional[UpstreamBackend] = None

        # 4) FastAPI app
        self.app = FastAPI(
            title=self.settings.app_name,
            docs_url="/docs" if self.settings.enable_docs else None,
            redoc_url=None,
        )

        # 5) Error normalization first (so everything becomes OpenAI-style errors)
        self.app.add_middleware(OpenAIErrorMiddleware)

        # 6) CORS
        # NOTE: ServerSettings already validates credentials + wildcard origin safety.
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.cors_origins,
            allow_credentials=self.settings.cors_allow_credentials,
            allow_methods=self.settings.cors_allow_methods,
            allow_headers=self.settings.cors_allow_headers,
        )

        # 7) Lightweight bearer-token auth (optional)
        @self.app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            if self.settings.api_keys:
                auth = request.headers.get("authorization") or ""
                token = auth.split(" ", 1)[1] if auth.startswith("Bearer ") and " " in auth else None
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

        # 8) Mount API routes
        self.app.include_router(build_router(self))


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

    def create_app(self) -> FastAPI:
        return self.app

    def run(self, host: Optional[str] = None, port: Optional[int] = None, **kwargs: Any) -> None:
        import uvicorn
        uvicorn.run(
            self.app,
            host=host or self.settings.host,
            port=port or self.settings.port,
            reload=kwargs.get("reload", self.settings.reload),
            log_level=kwargs.get("log_level", self.settings.log_level),
            workers=kwargs.get("workers", self.settings.workers),
        )
