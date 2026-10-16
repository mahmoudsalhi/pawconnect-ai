from functools import lru_cache
from importlib import import_module

from ..config import get_settings
from .base import AIProvider
from .demo import DemoProvider


@lru_cache
def get_provider() -> AIProvider:
    """Load the provider named by ``AI_PROVIDER`` ("demo" or "package.module:ClassName")."""
    spec = get_settings().ai_provider
    if spec == "demo":
        return DemoProvider()
    module_name, _, class_name = spec.partition(":")
    provider = getattr(import_module(module_name), class_name)()
    if not isinstance(provider, AIProvider):
        raise TypeError(f"{spec} must subclass app.ai.base.AIProvider")
    return provider
