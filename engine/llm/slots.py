"""Пул слотов: агент ↔ слот llama.cpp.

Эффективность переключения контекстов держится на двух вещах:
1. агент всегда возвращается в свой слот — KV-кэш и его префикс остаются горячими;
2. слот не переиспользуется под другого агента, пока есть свободные.

Если агентов больше, чем слотов, вытесняется самый давно использованный.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class SlotStats:
    hits: int = 0
    evictions: int = 0


@dataclass
class SlotPool:
    parallel: int
    _slots: dict[str, int] = field(default_factory=dict)
    _lru: "OrderedDict[str, None]" = field(default_factory=OrderedDict)
    stats: SlotStats = field(default_factory=SlotStats)

    def __post_init__(self) -> None:
        if self.parallel < 1:
            raise ValueError("parallel должен быть >= 1")

    def slot_for(self, agent_id: str) -> int:
        """Закрепляет слот за агентом (идемпотентно)."""
        if agent_id in self._slots:
            self._lru.move_to_end(agent_id)
            self.stats.hits += 1
            return self._slots[agent_id]

        if len(self._slots) >= self.parallel:
            victim, _ = self._lru.popitem(last=False)
            freed = self._slots.pop(victim)
            self.stats.evictions += 1
        else:
            freed = self._free_index()

        self._slots[agent_id] = freed
        self._lru[agent_id] = None
        return freed

    def _free_index(self) -> int:
        used = set(self._slots.values())
        for idx in range(self.parallel):
            if idx not in used:
                return idx
        raise RuntimeError("нет свободных слотов — ошибка учёта")

    def release(self, agent_id: str) -> None:
        self._slots.pop(agent_id, None)
        self._lru.pop(agent_id, None)

    def preassign(self, agent_ids: Iterable[str]) -> None:
        """Закрепить порядок заранее: порядок слотов = порядок агентов."""
        for agent_id in agent_ids:
            self.slot_for(agent_id)

    def snapshot(self) -> dict[str, int]:
        return dict(self._slots)
