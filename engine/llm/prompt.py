"""Рендер промпта в формате Llama-3.1.

Шаблон — часть замороженного контракта (train == inference для конкретной
модели): любое изменение здесь ломает golden-тесты и требует новой ветки.
"""

from __future__ import annotations

CHAT_FORMAT_VERSION = 1

BOS = "<|begin_of_text|>"
HEADER = "<|start_header_id|>{role}<|end_header_id|>\n\n"
EOT = "<|eot_id|>"


def render_llama3(system: str, user: str, *, prefill: str = "") -> str:
    """Собирает один ход диалога. prefill — начало ответа ассистента (опц.).

    BOS здесь **не** ставится: его добавляет рантайм (`add_bos`/`add_special`
    у llama.cpp). Если продублировать BOS в тексте, модель получает два
    подряд — llama.cpp об этом предупреждает и качество падает.
    """
    return (
        f"{HEADER.format(role='system')}{system}{EOT}"
        f"{HEADER.format(role='user')}{user}{EOT}"
        f"{HEADER.format(role='assistant')}{prefill}"
    )


def render_multi(system: str, turns: list[tuple[str, str]], *, prefill: str = "") -> str:
    """Многоходовый вариант: turns — список (role, content). BOS — на рантайме."""
    parts = [HEADER.format(role="system"), system, EOT]
    for role, content in turns:
        parts += [HEADER.format(role=role), content, EOT]
    parts += [HEADER.format(role="assistant"), prefill]
    return "".join(parts)
