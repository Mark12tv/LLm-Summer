"""Оценка и подгонка токенного бюджета.

Точный токенайзер доступен только у бэкенда (llama.cpp). Здесь — консервативная
оценка для ранних проверок, чтобы не отправлять заведомо переполненный промпт.
Точные значения (prompt_n) приходят в timings ответа и пишутся в лог.
"""

from __future__ import annotations

import math

# Llama-3 на смешанном рус/англ тексте даёт ~2.5-3.2 символа на токен.
DEFAULT_CHARS_PER_TOKEN = 3.0
SAFETY_MARGIN = 0.92  # держим 8 % запаса


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / chars_per_token))


def fits(text: str, budget: int, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> bool:
    return estimate_tokens(text, chars_per_token) <= budget


def fit_to_budget(
    text: str,
    budget: int,
    *,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    head_ratio: float = 0.3,
    marker: str = "\n[…сжато…]\n",
) -> str:
    """Урезает текст до бюджета, сохраняя начало и конец.

    Важное в промпте живёт по краям (правила сверху, якорь снизу), поэтому
    середина — первое, что уходит.
    """
    budget = max(1, int(budget))
    if estimate_tokens(text, chars_per_token) <= budget:
        return text

    allowed_chars = max(1, int(budget * chars_per_token * SAFETY_MARGIN) - len(marker))
    if allowed_chars >= len(text):
        return text

    head_chars = max(0, int(allowed_chars * head_ratio))
    tail_chars = max(0, allowed_chars - head_chars)
    head = text[:head_chars].rstrip()
    tail = text[len(text) - tail_chars :].lstrip() if tail_chars else ""
    if head and tail:
        return f"{head}{marker}{tail}"
    return (head or tail)[:allowed_chars]
