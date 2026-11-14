"""Контракт бэкенда генерации.

Бэкенд умеет ровно одно: по заданию выдать поток токенов. Всё остальное
(слоты, приоритеты, дедлайны, склейка) — в планировщике. Благодаря этому
ядро тестируется без модели, а замена llama-server на другой рантайм
не трогает ни память, ни ход.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

from engine.core.config import SAMPLING, STOP_SEQUENCES


@dataclass(frozen=True)
class PreparedJob:
    """Что хотим сгенерировать, до маршрутизации по слотам."""

    agent_id: str
    prompt: str
    max_tokens: int
    seed: int
    priority: int = 50          # меньше = раньше (фокус ~10, фон ~40)
    grammar: str | None = None
    deadline_s: float | None = None
    stop: tuple[str, ...] = ()
    temperature: float = SAMPLING.temperature
    top_p: float = SAMPLING.top_p
    top_k: int = SAMPLING.top_k
    repeat_penalty: float = SAMPLING.repeat_penalty
    default_stop: tuple[str, ...] = STOP_SEQUENCES


@dataclass(frozen=True)
class GenJob(PreparedJob):
    """Задание после маршрутизации: закреплённый слот + состояние префикса."""

    slot: int = 0
    cache_hit: bool = False
    reused_tokens: int = 0


@dataclass
class GenResult:
    agent_id: str
    text: str = ""
    tokens: int = 0
    ok: bool = True
    error: str | None = None
    cancelled: bool = False
    slot: int = 0
    cache_hit: bool = False
    reused_tokens: int = 0
    elapsed_ms: float = 0.0
    first_token_ms: float | None = None


@dataclass
class TurnBatch:
    """Результат одного хода: ровно то, что применяет движок."""

    results: dict[str, GenResult] = field(default_factory=dict)
    admission: list[str] = field(default_factory=list)
    wall_ms: float = 0.0
    stats: dict[str, float] = field(default_factory=dict)

    def text(self, agent_id: str) -> str:
        res = self.results.get(agent_id)
        return res.text if res else ""

    def failed(self) -> list[str]:
        return [aid for aid, r in self.results.items() if not r.ok]


@runtime_checkable
class Backend(Protocol):
    name: str
    parallel: int

    def stream(self, job: GenJob) -> AsyncIterator[str]:
        """Асинхронный генератор токенов задания."""
        ...

    async def health(self) -> bool: ...

    async def close(self) -> None: ...
