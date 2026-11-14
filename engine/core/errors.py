"""Типы ошибок движка: всё, что должно быть видно в логе, а не проглатываться."""

from __future__ import annotations


class EngineError(Exception):
    """Базовая ошибка движка."""


class BudgetError(EngineError):
    """Промпт/выход не укладывается в закреплённый бюджет."""


class BackendError(EngineError):
    """Транспорт или рантайм модели ответил ошибкой."""


class ManifestMismatch(EngineError):
    """Сейв загружен на другом манифесте — партия считается новой веткой."""
