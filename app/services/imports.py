from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any


def import_callable(path: str) -> Callable[..., Any]:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"Callable path must use 'module:function' format: {path}")

    module = import_module(module_name)
    imported = getattr(module, attribute)
    if not callable(imported):
        raise TypeError(f"Imported object is not callable: {path}")
    return imported
