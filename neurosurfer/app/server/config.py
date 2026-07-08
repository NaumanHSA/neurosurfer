from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_json_loads(value: str):
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        return json.loads(s)
    return [x.strip() for x in s.split(",") if x.strip()]


def _split_csv(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in s.split(",") if x.strip()]


class ServerSettings(BaseSettings):
    """Gateway settings driven by environment variables (prefix ``NS_``).

    Override any field via env var, e.g. ``NS_PORT=9000``, or a ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="NS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_json_loads=_env_json_loads,
    )

    # Server identity
    app_name: str = Field(default="Neurosurfer Gateway")
    environment: Literal["dev", "staging", "prod"] = Field(default="dev")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)

    enable_docs: bool = Field(default=True)
    reload: bool = Field(default=False)
    workers: int = Field(default=1, ge=1)
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = Field(
        default="info"
    )

    proxy_headers: bool = Field(default=True)
    forwarded_allow_ips: str = Field(default="*")
    access_log: bool = Field(default=True)

    # Auth — optional bearer tokens (e.g. Open-WebUI)
    api_keys: list[str] = Field(default_factory=list)

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = Field(default=False)
    cors_allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_headers: list[str] = Field(default_factory=lambda: ["*"])

    # SSE / streaming
    sse_ping_interval_s: float = Field(default=15.0, ge=1.0, le=120.0)
    sse_max_seconds: float | None = Field(default=None, ge=1.0)

    # Request safety
    max_request_body_mb: int = Field(default=25, ge=1, le=1024)
    max_messages: int = Field(default=200, ge=1)
    max_input_chars: int = Field(default=2_000_000, ge=1)

    # HTTP client (gateway → upstream)
    upstream_timeout_s: float = Field(default=120.0, ge=1.0, le=3600.0)
    upstream_connect_timeout_s: float = Field(default=10.0, ge=1.0, le=120.0)
    upstream_read_timeout_s: float | None = Field(default=None)
    upstream_max_keepalive: int = Field(default=20, ge=1, le=500)
    upstream_max_connections: int = Field(default=200, ge=1, le=5000)

    # OpenAI compat knobs
    strict_openai_compat: bool = Field(default=True)
    passthrough_unknown_fields: bool = Field(default=True)
    strip_reasoning: bool = Field(default=False)
    default_model: str | None = Field(default=None)

    # Upstream backend (optional shorthand — configure in code or via env)
    upstream_base_url: str | None = Field(default=None)
    upstream_api_key: str | None = Field(default=None)
    upstream_models_mode: Literal["proxy", "static"] = Field(default="proxy")
    upstream_static_models: list[str] = Field(default_factory=list)

    @field_validator(
        "api_keys",
        "cors_origins",
        "cors_allow_methods",
        "cors_allow_headers",
        "upstream_static_models",
        mode="before",
    )
    @classmethod
    def _parse_lists(cls, v: Any) -> list[str]:
        return _split_csv(v)

    @model_validator(mode="after")
    def _validate_cors(self) -> ServerSettings:
        if self.cors_allow_credentials and "*" in self.cors_origins:
            raise ValueError(
                "Invalid CORS: cors_allow_credentials=true cannot be combined with "
                "cors_origins=['*']. Set explicit origins, e.g. "
                "NS_CORS_ORIGINS='http://localhost:3000'"
            )
        return self
