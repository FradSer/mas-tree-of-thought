"""Named workflow registry — resolve workflow(name) to a registered script."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

_REGISTRY: dict[str, Callable[[], Awaitable[Any]]] = {}


def register_workflow(name: str, script: Callable[[], Awaitable[Any]]) -> None:
    """Register a zero-arg async workflow script under ``name``."""
    _REGISTRY[name] = script


def get_workflow(name: str) -> Callable[[], Awaitable[Any]]:
    """Look up a registered workflow script."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"unknown workflow {name!r} — register it first. Available: {available}"
        )
    return _REGISTRY[name]


def list_workflows() -> list[str]:
    return sorted(_REGISTRY)
