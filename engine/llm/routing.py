"""Учёт префиксов промптов: сколько prefill реально сэкономлено.

llama-server кэширует префикс в рамках слота. Мы храним последний промпт
агента и считаем длину общего начала со следующим — это и есть объём
переиспользованного KV. Числа идут в метрики хода и в лог.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from engine.core.tokens import estimate_tokens
from engine.llm.backends.base import GenJob, PreparedJob
from engine.llm.slots import SlotPool


@dataclass(frozen=True)
class PrefixPlan:
    cache_hit: bool
    reused_chars: int
    reused_tokens: int
    new_chars: int


class PrefixLedger:
    """Последний промпт каждого агента (в памяти процесса, не в сейве)."""

    def __init__(self) -> None:
        self._last: dict[str, str] = {}

    def observe(self, agent_id: str, prompt: str) -> PrefixPlan:
        prev = self._last.get(agent_id)
        self._last[agent_id] = prompt
        if prev is None:
            return PrefixPlan(False, 0, 0, len(prompt))

        n = min(len(prev), len(prompt))
        i = 0
        while i < n and prev[i] == prompt[i]:
            i += 1

        # Сравнение посимвольное; для оценки токенов достаточно пропорции.
        reused_tokens = estimate_tokens(prompt[:i]) if i else 0
        return PrefixPlan(i > 0, i, reused_tokens, len(prompt) - i)

    def reset(self, agent_id: str | None = None) -> None:
        if agent_id is None:
            self._last.clear()
        else:
            self._last.pop(agent_id, None)


@dataclass
class RouteStats:
    cache_hits: int = 0
    cache_misses: int = 0
    prefill_saved_tokens: int = 0


@dataclass
class ContextRouter:
    """Связывает агента, его слот и историю префикса.

    Вызывается один раз на ход до генерации: проставляет `slot`, `cache_hit`
    и `reused_tokens` — то есть сообщает планировщику, сколько prefill
    уже лежит в KV и сколько реально придётся считать.
    """

    slots: SlotPool
    ledger: PrefixLedger = field(default_factory=PrefixLedger)
    stats: RouteStats = field(default_factory=RouteStats)

    def route(self, prepared: list[PreparedJob]) -> list[GenJob]:
        jobs: list[GenJob] = []
        for item in prepared:
            slot = self.slots.slot_for(item.agent_id)
            plan = self.ledger.observe(item.agent_id, item.prompt)
            if plan.cache_hit:
                self.stats.cache_hits += 1
                self.stats.prefill_saved_tokens += plan.reused_tokens
            else:
                self.stats.cache_misses += 1
            jobs.append(
                GenJob(
                    **asdict(item),
                    slot=slot,
                    cache_hit=plan.cache_hit,
                    reused_tokens=plan.reused_tokens,
                )
            )
        return jobs

