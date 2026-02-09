from __future__ import annotations

from fastapi import Request

from app.services.runtime import AppRuntime


def get_runtime(request: Request) -> AppRuntime:
    return request.app.state.runtime
