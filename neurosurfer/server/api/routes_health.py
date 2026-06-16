from __future__ import annotations
from fastapi import APIRouter

def mount_health_routes(router: APIRouter, server) -> None:
    @router.get("/")
    async def index():
        return "Neurosurfer Gateway is running"

    @router.get("/health")
    async def health():
        return {"status": "ok"}
