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
    # The execution and Architect surfaces. Plan 01 §3.1 originally declined
    # these; the Architect's own tests are written at the HTTP boundary, so
    # declining them meant declining the coverage of "a build that parks and asks
    # a person" — which is the behaviour, not the transport. The studio, accounts
    # and uploads stay behind; these do not.
    mount_workflow_routes(router, server)
    mount_architect_routes(router, server)
    return router
