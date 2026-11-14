from __future__ import annotations

from engine.core.tokens import estimate_tokens, fit_to_budget, fits


def test_estimate_scales_with_length() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("привет") >= 1
    assert estimate_tokens("а" * 300) > estimate_tokens("а" * 30)


def test_fits_respects_budget() -> None:
    assert fits("короткий текст", 100)
    assert not fits("а" * 10_000, 100)


def test_fit_keeps_head_and_tail() -> None:
    text = "НАЧАЛО\n" + "середина " * 500 + "\nКОНЕЦ"
    trimmed = fit_to_budget(text, 100)
    assert estimate_tokens(trimmed) <= 100
    assert trimmed.startswith("НАЧАЛО")
    assert trimmed.endswith("КОНЕЦ")


def test_fit_noop_when_fits() -> None:
    assert fit_to_budget("мало текста", 1000) == "мало текста"


def test_budget_of_one_token_does_not_crash() -> None:
    assert isinstance(fit_to_budget("длинный текст " * 100, 1), str)
