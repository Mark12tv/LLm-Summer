"""Детерминированные потоки случайности.

Глобальный random не используется: иначе порядок вызовов начинает влиять
на исход. Каждая задача получает свой поток, выведенный из seed партии.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


def derive_seed(game_seed: int, *parts: str | int) -> int:
    """Стабильный 31-битный seed из партии и произвольных меток."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(game_seed).encode("utf-8"))
    for part in parts:
        h.update(b"\x1f")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest(), "little") & 0x7FFF_FFFF


def turn_seed(game_seed: int, turn: int, agent_id: str, purpose: str = "gen") -> int:
    """Seed конкретного прогона: не зависит от порядка запуска агентов."""
    return derive_seed(game_seed, turn, agent_id, purpose)


@dataclass
class RngStreams:
    """Фабрика независимых потоков на партию."""

    game_seed: int

    def agent(self, agent_id: str, turn: int, purpose: str = "gen") -> random.Random:
        return random.Random(turn_seed(self.game_seed, turn, agent_id, purpose))

    def world(self, turn: int) -> random.Random:
        return random.Random(derive_seed(self.game_seed, "world", turn))

    def noise(self, *parts: str | int) -> float:
        """Детерминированное число [0, 1) без состояния."""
        return random.Random(derive_seed(self.game_seed, *parts)).random()
