"""Бэкенды генерации: http (продакшен), llamacpp (dev), mock (тесты)."""

from engine.llm.backends.base import Backend, GenJob, GenResult, PreparedJob, TurnBatch
from engine.llm.backends.mock import MockBackend

__all__ = ["Backend", "GenJob", "GenResult", "PreparedJob", "TurnBatch", "MockBackend"]
