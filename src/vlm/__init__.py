"""Vision Language Model adapters."""

from .base import VLMAdapter
from .openrouter import OpenRouterAdapter

__all__ = [
    'VLMAdapter',
    'OpenRouterAdapter',
]

