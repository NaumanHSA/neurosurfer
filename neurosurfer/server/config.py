from __future__ import annotations

from typing import Any, Literal, Optional

import json
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_json_loads(value: str):
    """
    Robust env decoder for "complex" fields (list/dict) in pydantic-settings.

    Accepts:
      - JSON lists: '["a","b"]'
      - CSV: 'a,b'
      - empty: '' -> []
    """
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    # JSON list/dict
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        return json.loads(s)
    # CSV fallback
    return [x.strip() for x in s.split(",") if x.strip()]

def _split_csv(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    # allow JSON-like list strings too
    if s.startswith("[") and s.endswith("]"):
        try:
            import json
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in s.split(",") if x.strip()]


class ServerSettings(BaseSettings):
    """
    Server + Gateway settings (env-driven).

    Env prefix: NS_
      e.g. NS_HOST, NS_PORT, NS_CORS_ORIGINS="http://localhost:3000,http://localhost:8080"

    Supports .env automatically if present.
    """
    model_config = SettingsConfigDict(
        env_prefix="NS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_json_loads=_env_json_loads,
    )

    # -------------------------
    # Server identity
    # -------------------------
    app_name: str = Field(default="Neurosurfer Gateway")
    environment: Literal["dev", "staging", "prod"] = Field(default="dev")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)

    enable_docs: bool = Field(default=True)
    reload: bool = Field(default=False)
    workers: int = Field(default=1, ge=1)
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = Field(default="info")

    # Uvicorn related
    proxy_headers: bool = Field(default=True)          # respect X-Forwarded-For/Proto
    forwarded_allow_ips: str = Field(default="*")      # or "127.0.0.1"
    access_log: bool = Field(default=True)

    # -------------------------
    # Auth
    # -------------------------
    # Optional bearer tokens accepted by gateway (Open-WebUI can send one)
    api_keys: list[str] = Field(default_factory=list)

    # -------------------------
    # CORS
    # -------------------------
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = Field(default=False)
    cors_allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_headers: list[str] = Field(default_factory=lambda: ["*"])

    # -------------------------
    # Streaming / SSE
    # -------------------------
    sse_ping_interval_s: float = Field(default=15.0, ge=1.0, le=120.0)
    sse_max_seconds: Optional[float] = Field(default=None, ge=1.0)  # safety cap, None = unlimited

    # -------------------------
    # Request safety limits
    # -------------------------
    max_request_body_mb: int = Field(default=25, ge=1, le=1024)
    max_messages: int = Field(default=200, ge=1)
    max_input_chars: int = Field(default=2_000_000, ge=1)  # protect from insane payloads

    # -------------------------
    # HTTP client defaults (gateway -> upstream)
    # -------------------------
    upstream_timeout_s: float = Field(default=120.0, ge=1.0, le=3600.0)
    upstream_connect_timeout_s: float = Field(default=10.0, ge=1.0, le=120.0)
    upstream_read_timeout_s: Optional[float] = Field(default=None)  # None = no read timeout
    upstream_max_keepalive: int = Field(default=20, ge=1, le=500)
    upstream_max_connections: int = Field(default=200, ge=1, le=5000)

    # -------------------------
    # OpenAI compatibility knobs
    # -------------------------
    # If Open-WebUI expects exactly OpenAI objects, keep True.
    strict_openai_compat: bool = Field(default=True)

    # Some UIs send unknown fields; allow passthrough to upstream
    passthrough_unknown_fields: bool = Field(default=True)

    # If true, drop / strip <think> blocks server-side (hook can do it too)
    strip_reasoning: bool = Field(default=False)

    # Default model selection behavior
    default_model: Optional[str] = Field(default=None)

    # -------------------------
    # Upstream backend config (optional but common)
    # -------------------------
    upstream_base_url: Optional[str] = Field(default=None)  # e.g. http://localhost:8001/v1
    upstream_api_key: Optional[str] = Field(default=None)
    upstream_models_mode: Literal["proxy", "static"] = Field(default="proxy")

    # If using "static", provide:
    upstream_static_models: list[str] = Field(default_factory=list)

    # -------------------------
    # Parsing env: accept CSV strings for lists
    # -------------------------
    @field_validator("api_keys", "cors_origins", "cors_allow_methods", "cors_allow_headers", "upstream_static_models", mode="before")
    @classmethod
    def _parse_lists(cls, v: Any) -> list[str]:
        return _split_csv(v)

    @model_validator(mode="after")
    def _validate_cors(self) -> "ServerSettings":
        # Browsers disallow wildcard origins with credentials
        if self.cors_allow_credentials and ("*" in self.cors_origins):
            raise ValueError(
                "Invalid CORS: cors_allow_credentials=true cannot be used with cors_origins=['*']. "
                "Set explicit origins, e.g. NS_CORS_ORIGINS='http://localhost:3000'"
            )
        return self
