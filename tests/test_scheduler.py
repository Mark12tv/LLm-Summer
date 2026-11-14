from __future__ import annotations

import asyncio

from engine.llm.backends.base import GenJob
from engine.llm.backends.mock import MockBackend
from engine.llm.scheduler import RoundRobinScheduler, SchedulerConfig, TokenMux


def run(coro):
    return asyncio.run(coro)


def make_jobs(*agents: str, max_tokens: int = 6, priority: int = 50) -> list[GenJob]:
    return [
        GenJob(agent_id=a, prompt=f"prompt-{a}", max_tokens=max_tokens, seed=i, priority=priority)
        for i, a in enumerate(agents)
    ]


def test_mux_is_strictly_round_robin_when_buffers_are_full() -> None:
    async def scenario() -> list[tuple[str, str]]:
        mux = TokenMux(["a", "b", "c"], max_buffered=16)
        for agent in ("a", "b", "c"):
            for i in range(3):
                await mux.push(agent, f"{agent}{i}")
        for agent in ("a", "b", "c"):
            await mux.close(agent)
        return [item async for item in mux.stream()]

    emitted = run(scenario())
    assert [agent for agent, _ in emitted] == ["a", "b", "c"] * 3
    assert [token for _, token in emitted][:3] == ["a0", "b0", "c0"]


def test_scheduler_runs_all_agents_and_merges_output() -> None:
    async def scenario():
        backend = MockBackend(tokens_per_job=5)
        scheduler = RoundRobinScheduler(backend, SchedulerConfig(admission_gap_s=0))
        batch = await scheduler.run(make_jobs("lena", "ulyana", "alisa", max_tokens=5))
        return batch

    batch = run(scenario())
    assert set(batch.results) == {"lena", "ulyana", "alisa"}
    assert all(r.ok for r in batch.results.values())
    assert all(batch.results[a].tokens == 5 for a in ("lena", "ulyana", "alisa"))
    assert batch.stats["tokens"] == 15.0
    assert batch.failed() == []


def test_budget_is_enforced_even_if_backend_is_greedy() -> None:
    async def scenario():
        backend = MockBackend(tokens_per_job=50)
        scheduler = RoundRobinScheduler(backend, SchedulerConfig(admission_gap_s=0))
        return await scheduler.run(make_jobs("lena", max_tokens=3))

    batch = run(scenario())
    assert batch.results["lena"].tokens == 3
    assert batch.text("lena").count("[lena]") == 3


def test_round_robin_does_not_starve_agents() -> None:
    async def scenario():
        backend = MockBackend(tokens_per_job=8)
        scheduler = RoundRobinScheduler(backend, SchedulerConfig(admission_gap_s=0))
        seen: list[str] = []

        def on_token(agent_id: str, _token: str) -> None:
            seen.append(agent_id)

        await scheduler.run(make_jobs("a", "b", "c", max_tokens=8), on_token=on_token)
        return seen

    seen = run(scenario())
    assert len(seen) == 24
    # никто не уходит в конец: в каждом окне из 6 подряд идут минимум двое разных
    for start in range(len(seen) - 6):
        window = seen[start : start + 6]
        assert len(set(window)) >= 2
    for agent in ("a", "b", "c"):
        assert seen.index(agent) < 6


def test_priority_orders_admission() -> None:
    jobs = [
        GenJob("slow", "p", 4, 1, priority=50, slot=2),
        GenJob("focal", "p", 4, 2, priority=10, slot=1),
        GenJob("pair", "p", 4, 3, priority=10, slot=0),
    ]

    async def scenario():
        backend = MockBackend(tokens_per_job=2)
        scheduler = RoundRobinScheduler(backend, SchedulerConfig(admission_gap_s=0))
        return await scheduler.run(jobs)

    batch = run(scenario())
    # при равном приоритете решает номер слота; camp/focal входят раньше фона
    assert batch.admission[:2] == ["pair", "focal"]


def test_deadline_cancels_only_the_stuck_agent() -> None:
    async def scenario():
        backend = MockBackend(tokens_per_job=5, hang_agents={"stuck"})
        scheduler = RoundRobinScheduler(
            backend, SchedulerConfig(admission_gap_s=0, deadline_s=0.05)
        )
        return await scheduler.run(make_jobs("stuck", "ok", max_tokens=5))

    batch = run(scenario())
    assert not batch.results["stuck"].ok
    assert batch.results["stuck"].cancelled is True
    assert batch.results["stuck"].error == "deadline"
    assert batch.results["ok"].ok is True
    assert batch.failed() == ["stuck"]


def test_transient_error_is_retried_once() -> None:
    async def scenario():
        backend = MockBackend(tokens_per_job=3, fail_once={"lena"})
        scheduler = RoundRobinScheduler(backend, SchedulerConfig(admission_gap_s=0))
        return await scheduler.run(make_jobs("lena", max_tokens=3))

    batch = run(scenario())
    assert batch.results["lena"].ok is True
    assert batch.results["lena"].tokens == 3


def test_permanent_error_is_reported_without_crash() -> None:
    async def scenario():
        backend = MockBackend(tokens_per_job=3, fail_agents={"bad"})
        scheduler = RoundRobinScheduler(backend, SchedulerConfig(admission_gap_s=0))
        return await scheduler.run(make_jobs("bad", "good", max_tokens=3))

    batch = run(scenario())
    assert batch.results["bad"].ok is False
    assert "mock fail" in (batch.results["bad"].error or "")
    assert batch.results["good"].ok is True


def test_async_callback_is_awaited() -> None:
    order: list[str] = []

    async def on_token(agent_id: str, _token: str) -> None:
        await asyncio.sleep(0)
        order.append(agent_id)

    async def scenario():
        backend = MockBackend(tokens_per_job=2)
        scheduler = RoundRobinScheduler(backend, SchedulerConfig(admission_gap_s=0))
        await scheduler.run(make_jobs("a", "b", max_tokens=2), on_token=on_token)

    run(scenario())
    assert order == ["a", "b", "a", "b"]


def test_empty_batch_is_safe() -> None:
    batch = run(RoundRobinScheduler(MockBackend()).run([]))
    assert batch.results == {}
    assert batch.admission == []
