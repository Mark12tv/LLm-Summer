from __future__ import annotations

from engine.llm.slots import SlotPool


def test_slots_are_stable_per_agent() -> None:
    pool = SlotPool(parallel=14)
    first = pool.slot_for("lena")
    assert pool.slot_for("lena") == first
    assert pool.slot_for("ulyana") != first


def test_all_slots_are_unique() -> None:
    pool = SlotPool(parallel=14)
    ids = [f"a{i}" for i in range(14)]
    slots = [pool.slot_for(i) for i in ids]
    assert sorted(slots) == list(range(14))


def test_eviction_is_lru() -> None:
    pool = SlotPool(parallel=2)
    a = pool.slot_for("a")
    b = pool.slot_for("b")
    pool.slot_for("a")            # a становится свежим
    c = pool.slot_for("c")        # вытесняется b
    assert a == pool.slot_for("a")
    assert c == a or c == b
    assert pool.stats.evictions == 1
    assert "b" not in pool.snapshot()


def test_release_frees_the_slot() -> None:
    pool = SlotPool(parallel=2)
    slot = pool.slot_for("lena")
    pool.release("lena")
    assert pool.slot_for("ulyana") == slot


def test_preassign_keeps_registry_order() -> None:
    pool = SlotPool(parallel=4)
    pool.preassign(["camp", "lena", "ulyana", "alisa"])
    assert pool.snapshot() == {"camp": 0, "lena": 1, "ulyana": 2, "alisa": 3}
