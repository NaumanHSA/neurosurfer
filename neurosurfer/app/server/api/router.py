from __future__ import annotations

from fastapi import APIRouter

from .routes_architect import mount_architect_routes
from .routes_chat import mount_chat_routes
from .routes_health import mount_health_routes
from .routes_models import mount_models_routes
from .routes_workflows import mount_workflow_routes


def build_router(server) -> APIRouter:
    router = APIRouter()
    mount_health_routes(router, server)
    mount_models_routes(router, server)
    mount_chat_routes(router, server)
    mount_workflow_routes(router, server)
    mount_architect_routes(router, server)
    return router
