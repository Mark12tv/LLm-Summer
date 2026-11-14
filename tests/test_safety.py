from __future__ import annotations

from engine.safety import (
    SAFETY_PREAMBLE,
    needs_extra_guard,
    sexual_content_allowed,
)


def test_preamble_mentions_adults_and_ban() -> None:
    assert "18+" in SAFETY_PREAMBLE
    assert "Запрещено" in SAFETY_PREAMBLE


def test_no_sexual_content_in_camp() -> None:
    for agent in ("ulyana", "lena", "miku", "zhenya"):
        assert sexual_content_allowed(agent, "camp") is False


def test_epilogue_does_not_open_the_ban() -> None:
    assert sexual_content_allowed("ulyana", "epilogue") is False


def test_ulyana_has_extra_guard_only_in_camp() -> None:
    assert needs_extra_guard("ulyana", "camp") is True
    assert needs_extra_guard("ulyana", "epilogue") is False
    assert needs_extra_guard("lena", "camp") is False
