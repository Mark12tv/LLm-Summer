"""Mock-бэкенд: детерминированная генерация без модели.

Нужен для тестов планировщика (переключение контекстов, дедлайны, ошибки)
и для прогонов логики хода на машине без GPU.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator

from engine.core.errors import BackendError
from engine.llm.backends.base import GenJob


@dataclass
class MockBackend:
    parallel: int = 14
    name: str = "mock"
    token_delay_s: float = 0.0
    tokens_per_job: int = 6
    slow_agents: dict[str, float] = field(default_factory=dict)
    fail_agents: set[str] = field(default_factory=set)
    fail_once: set[str] = field(default_factory=set)
    hang_agents: set[str] = field(default_factory=set)
    calls: list[GenJob] = field(default_factory=list)
    _failed: set[str] = field(default_factory=set)

    def _tokens(self, agent_id: str) -> list[str]:
        base = f"[{agent_id}]"
        return [f"{base}{i} " for i in range(self.tokens_per_job)]

    async def stream(self, job: GenJob) -> AsyncIterator[str]:
        self.calls.append(job)
        if job.agent_id in self.fail_agents:
            raise BackendError(f"mock fail: {job.agent_id}")
        if job.agent_id in self.fail_once and job.agent_id not in self._failed:
            self._failed.add(job.agent_id)
            raise BackendError(f"mock transient fail: {job.agent_id}")

        delay = self.slow_agents.get(job.agent_id, self.token_delay_s)
        for token in self._tokens(job.agent_id):
            if job.agent_id in self.hang_agents:
                await asyncio.Event().wait()
            if delay:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)
            yield token

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None
