from __future__ import annotations

from fastapi import APIRouter


def mount_models_routes(router: APIRouter, server) -> None:
    @router.get("/v1/models")
    async def list_models():
        data = []
        seen: set[str] = set()

        if server._upstream_backend is not None:
            payload = await server._upstream_backend.list_models()
            for m in payload.get("data") or []:
                mid = m.get("id")
                if not mid or mid in seen:
                    continue
                seen.add(mid)
                data.append(m)

        for mid, target in server.router.all_models().items():
            if mid in seen:
                continue
            try:
                payload = await target.backend.list_models()
                for m in payload.get("data") or []:
                    mmid = m.get("id")
                    if not mmid or mmid in seen:
                        continue
                    seen.add(mmid)
                    data.append(m)
            except Exception:
                continue

        return {"object": "list", "data": data}
