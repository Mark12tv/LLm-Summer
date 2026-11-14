"""Агенты: реестр и (далее) сборка промптов/валидаторы."""

from engine.agents.registry import AGENTS, BY_ID, AgentSpec, get, validate

__all__ = ["AGENTS", "BY_ID", "AgentSpec", "get", "validate"]
