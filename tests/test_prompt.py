from __future__ import annotations

from engine.llm.prompt import BOS, EOT, render_llama3, render_multi


def test_render_has_all_roles_and_no_double_bos() -> None:
    text = render_llama3("СИСТЕМА", "ПОЛЬЗОВАТЕЛЬ")
    assert BOS not in text, "BOS добавляет рантайм, в тексте его быть не должно"
    assert "<|start_header_id|>system<|end_header_id|>" in text
    assert "<|start_header_id|>user<|end_header_id|>" in text
    assert text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert text.count(EOT) == 2


def test_prefill_is_appended_verbatim() -> None:
    text = render_llama3("S", "U", prefill="[МЫСЛЬ]")
    assert text.endswith("[МЫСЛЬ]")


def test_render_multi_keeps_order() -> None:
    text = render_multi("S", [("user", "U1"), ("assistant", "A1"), ("user", "U2")])
    assert text.index("U1") < text.index("A1") < text.index("U2")
    assert text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert BOS not in text
