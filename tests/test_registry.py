from __future__ import annotations

import json
from pathlib import Path

from engine.agents.registry import AGENTS, BY_ID, get, validate
from engine.core.config import N_CTX_PER_SLOT, PARALLEL_SLOTS

DATA = Path(__file__).resolve().parents[1] / "data" / "agents"


def test_exactly_fourteen_agents() -> None:
    assert len(AGENTS) == 14 == PARALLEL_SLOTS
    assert len(BY_ID) == 14


def test_registry_is_valid() -> None:
    assert validate() == []


def test_boys_are_one_agent_with_two_speakers() -> None:
    boys = get("boys")
    assert boys.kind == "boys"
    assert boys.speakers == ("Шурик", "Электроник")
    assert boys.prompt_budget <= 8000
    assert "shurik" not in BY_ID and "elektronik" not in BY_ID


def test_zhenya_is_light() -> None:
    zhenya = get("zhenya")
    assert zhenya.kind == "light"
    assert zhenya.prompt_budget <= 2500
    assert zhenya.out_budget <= 120
    assert zhenya.memory == "minimal"


def test_every_agent_fits_its_slot() -> None:
    for spec in AGENTS:
        assert spec.ctx_needed <= N_CTX_PER_SLOT, spec.id


def test_full_context_agents_use_full_budget() -> None:
    for spec in AGENTS:
        if spec.memory == "full":
            assert spec.out_budget == 300, spec.id
            assert spec.prompt_budget >= 10000, spec.id


def test_persona_files_for_new_agents_are_valid_json() -> None:
    for name in ("boys.json", "zhenya.json"):
        payload = json.loads((DATA / name).read_text(encoding="utf-8"))
        assert payload["id"] == name.removesuffix(".json")
        assert payload["age"] >= 18
        assert payload["hard_limits"]
