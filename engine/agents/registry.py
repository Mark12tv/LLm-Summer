"""Реестр агентов: единственное место, где описан состав мира.

14 прогонов одной модели. Отличия между агентами — только данные:
бюджет контекста, бюджет выхода, профиль памяти, набор «голосов».

Особые случаи:
* `boys` — Шурик и Электроник как **один** агент: два тела, общая память,
  речь с префиксом говорящего;
* `zhenya` — очень лёгкий фоновый персонаж: минимальный промпт и память.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from engine.core.config import (
    N_CTX_PER_SLOT,
    OUT_BACKGROUND,
    OUT_FULL,
    OUT_LIGHT,
    PARALLEL_SLOTS,
    PROMPT_CONVERTER,
    PROMPT_FULL,
    PROMPT_LIGHT,
    PROMPT_SHORT,
)

Kind = Literal["world", "girl", "boys", "staff", "crowd", "counselor", "light", "converter"]
MemoryProfile = Literal["full", "short", "minimal", "none"]

# Приоритеты входа в батч: меньше = раньше (ТЗ §3.2).
PRIORITY_CAMP = 0
PRIORITY_FOCAL = 10
PRIORITY_PAIR = 20
PRIORITY_CROWD = 30
PRIORITY_SINGLE = 40
PRIORITY_CONVERTER = 90


@dataclass(frozen=True)
class AgentSpec:
    id: str
    name: str
    kind: Kind
    prompt_budget: int
    out_budget: int
    memory: MemoryProfile
    priority: int
    speakers: tuple[str, ...] = ()
    persona: str | None = None

    @property
    def ctx_needed(self) -> int:
        return self.prompt_budget + self.out_budget

    @property
    def full_context(self) -> bool:
        return self.memory == "full"


AGENTS: tuple[AgentSpec, ...] = (
    AgentSpec("camp", "Лагерь", "world", PROMPT_SHORT, OUT_BACKGROUND, "none", PRIORITY_CAMP),
    AgentSpec("lena", "Лена", "girl", PROMPT_FULL, OUT_FULL, "full", PRIORITY_SINGLE, persona="lena.json"),
    AgentSpec("ulyana", "Ульяна", "girl", PROMPT_FULL, OUT_FULL, "full", PRIORITY_SINGLE, persona="ulyana.json"),
    AgentSpec("alisa", "Алиса", "girl", PROMPT_FULL, OUT_FULL, "full", PRIORITY_SINGLE, persona="alisa.json"),
    AgentSpec("slavya", "Славя", "girl", PROMPT_FULL, OUT_FULL, "full", PRIORITY_SINGLE, persona="slavya.json"),
    AgentSpec("miku", "Мику", "girl", PROMPT_FULL, OUT_FULL, "full", PRIORITY_SINGLE, persona="miku.json"),
    AgentSpec(
        "boys", "Шурик и Электроник", "boys", PROMPT_SHORT, OUT_BACKGROUND, "short",
        PRIORITY_SINGLE, speakers=("Шурик", "Электроник"), persona="boys.json",
    ),
    AgentSpec("od", "Ольга Дмитриевна", "staff", PROMPT_FULL, OUT_FULL, "full", PRIORITY_SINGLE, persona="od.json"),
    AgentSpec("nurse", "Медсестра", "staff", PROMPT_FULL, OUT_FULL, "full", PRIORITY_SINGLE, persona="nurse.json"),
    AgentSpec("crowd", "Толпа", "crowd", PROMPT_FULL, OUT_FULL, "full", PRIORITY_CROWD, persona="crowd.json"),
    AgentSpec(
        "counselor_jr", "Вожатая младшего отряда", "counselor", PROMPT_SHORT, OUT_BACKGROUND,
        "short", PRIORITY_SINGLE, persona="counselor_jr.json",
    ),
    AgentSpec(
        "counselor_sr", "Вожатая старшего отряда", "counselor", PROMPT_SHORT, OUT_BACKGROUND,
        "short", PRIORITY_SINGLE, persona="counselor_sr.json",
    ),
    AgentSpec(
        "zhenya", "Женя", "light", PROMPT_LIGHT, OUT_LIGHT, "minimal",
        PRIORITY_SINGLE, persona="zhenya.json",
    ),
    AgentSpec("converter", "Преобразователь", "converter", PROMPT_CONVERTER, OUT_LIGHT, "none", PRIORITY_CONVERTER),
)

BY_ID: dict[str, AgentSpec] = {spec.id: spec for spec in AGENTS}

FOCUSABLE = frozenset({"lena", "ulyana", "alisa", "slavya", "miku", "boys", "zhenya", "od", "nurse"})
STANDALONE = frozenset({"camp", "crowd", "converter"})


def get(agent_id: str) -> AgentSpec:
    try:
        return BY_ID[agent_id]
    except KeyError:
        raise KeyError(f"неизвестный агент: {agent_id!r}") from None


def validate() -> list[str]:
    """Проверка инвариантов реестра. Пустой список — всё в порядке."""
    problems: list[str] = []
    if len(AGENTS) != len(BY_ID):
        problems.append("дублирующиеся id агентов")
    if len(AGENTS) > PARALLEL_SLOTS:
        problems.append(f"агентов {len(AGENTS)} > слотов {PARALLEL_SLOTS}")
    for spec in AGENTS:
        if spec.ctx_needed > N_CTX_PER_SLOT:
            problems.append(f"{spec.id}: промпт+выход {spec.ctx_needed} > {N_CTX_PER_SLOT}")
        if spec.kind == "boys" and not spec.speakers:
            problems.append("boys: не заданы говорящие")
        if spec.kind != "boys" and spec.speakers:
            problems.append(f"{spec.id}: speakers допустимы только для boys")
        if spec.memory == "minimal" and spec.prompt_budget > PROMPT_LIGHT:
            problems.append(f"{spec.id}: лёгкий агент с большим бюджетом")
    return problems
