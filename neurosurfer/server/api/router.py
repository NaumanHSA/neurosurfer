from __future__ import annotations
from fastapi import APIRouter
from .routes_models import mount_models_routes
from .routes_chat import mount_chat_routes
from .routes_health import mount_health_routes

def build_router(server) -> APIRouter:
    router = APIRouter()
    mount_health_routes(router, server)
    mount_models_routes(router, server)
    mount_chat_routes(router, server)
    return router
