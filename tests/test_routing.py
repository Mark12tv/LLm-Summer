from __future__ import annotations

from engine.llm.backends.base import PreparedJob
from engine.llm.routing import ContextRouter, PrefixLedger
from engine.llm.slots import SlotPool


def test_prefix_grows_with_shared_start() -> None:
    ledger = PrefixLedger()
    first = ledger.observe("lena", "СИСТЕМНЫЙ ПРОМПТ\nмомент 1")
    assert not first.cache_hit
    second = ledger.observe("lena", "СИСТЕМНЫЙ ПРОМПТ\nмомент 2")
    assert second.cache_hit
    assert second.reused_chars == len("СИСТЕМНЫЙ ПРОМПТ\nмомент ")
    assert second.reused_tokens > 0
    assert second.new_chars > 0


def test_prefix_miss_on_changed_system_block() -> None:
    ledger = PrefixLedger()
    ledger.observe("lena", "A" * 100 + "хвост")
    plan = ledger.observe("lena", "B" * 100 + "хвост")
    assert not plan.cache_hit
    assert plan.reused_chars == 0


def test_router_assigns_slots_and_counts_savings() -> None:
    router = ContextRouter(slots=SlotPool(parallel=14))
    jobs = [
        PreparedJob("lena", "общий префикс\nход 1", 300, 1),
        PreparedJob("ulyana", "общий префикс\nход 1", 300, 2),
    ]
    routed = router.route(jobs)
    assert {j.agent_id: j.slot for j in routed} == {"lena": 0, "ulyana": 1}
    assert all(not j.cache_hit for j in routed)

    again = router.route(
        [
            PreparedJob("lena", "общий префикс\nход 2", 300, 3),
            PreparedJob("ulyana", "общий префикс\nход 2", 300, 4),
        ]
    )
    assert all(j.cache_hit and j.reused_tokens > 0 for j in again)
    assert router.stats.cache_hits == 2
    assert router.stats.prefill_saved_tokens > 0
