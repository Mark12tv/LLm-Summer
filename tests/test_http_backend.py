from __future__ import annotations

from engine.llm.backends.base import GenJob
from engine.llm.backends.http import HttpBackend, parse_sse_line


def test_parse_sse_content_and_stop() -> None:
    assert parse_sse_line('data: {"content": "при", "stop": false}') == {
        "content": "при",
        "stop": False,
    }
    assert parse_sse_line('data: {"stop": true}') == {"stop": True}
    assert parse_sse_line("data: [DONE]") == {"stop": True}


def test_parse_sse_ignores_garbage() -> None:
    assert parse_sse_line("") is None
    assert parse_sse_line("event: ping") is None
    assert parse_sse_line("data: {broken") is None


def test_payload_pins_slot_and_reuses_prefix() -> None:
    backend = HttpBackend()
    job = GenJob(
        agent_id="lena",
        prompt="<|begin_of_text|>...",
        max_tokens=300,
        seed=12345,
        slot=3,
        cache_hit=True,
    )
    body = backend.payload(job)
    assert body["id_slot"] == 3
    assert body["n_predict"] == 300
    assert body["cache_prompt"] is True
    assert body["seed"] == 12345
    assert body["stream"] is True
    assert "<|eot_id|>" in body["stop"]


def test_payload_includes_grammar_when_set() -> None:
    backend = HttpBackend()
    job = GenJob("converter", "p", 40, 1, grammar='root ::= "1"')
    assert backend.payload(job)["grammar"] == 'root ::= "1"'
