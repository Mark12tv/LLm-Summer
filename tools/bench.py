"""Замеры рантайма: сколько реально тянет железо.

Режимы:
    python -m tools.bench --mode server   # llama-server: 1/4/8/14 слотов
    python -m tools.bench --mode inproc   # llama-cpp-python: один поток

Пишет отчёт в runs/bench/<stamp>.json. Без бинарника llama-server
режим server честно скажет, чего не хватает, и не притворится, что замерил.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from engine.core.config import (
    MODEL_PATH,
    RUNS_DIR,
    check_profile,
    profile,
)
from engine.llm.backends.base import PreparedJob
from engine.llm.backends.http import HttpBackend
from engine.llm.prompt import render_llama3
from engine.llm.routing import ContextRouter
from engine.llm.scheduler import RoundRobinScheduler, SchedulerConfig
from engine.llm.server import LlamaServer, find_server_binary
from engine.llm.slots import SlotPool
from engine.turn.order import assign_priorities, build_call_order

FILLER = "Правило поведения: не выходи из роли. " * 200


def synthetic_prompt(agent_id: str, approx_tokens: int = 800) -> str:
    system = FILLER[: approx_tokens * 3]
    return render_llama3(system, f"[ВОСПРИЯТИЕ] тестовый ход для {agent_id}")


async def bench_server(profile_name: str, model: Path, concurrent: list[int]) -> dict:
    p = profile(profile_name)
    problems = check_profile(p, model)
    if problems:
        raise SystemExit("профиль несовместим с моделью:\n  " + "\n  ".join(problems))

    binary = find_server_binary()
    server = LlamaServer(p, model_path=model, binary=binary).start()
    backend = HttpBackend(base_url=server.base_url, parallel=p.parallel)
    try:
        if not await backend.health():
            raise SystemExit("llama-server поднялся, но /health не отвечает")
        rows = []
        for slots in concurrent:
            agents = [f"a{i}" for i in range(slots)]
            order = build_call_order(focus=set(agents[:1]), singles=set(agents[1:]))
            priorities = assign_priorities(order, focus={agents[0]})
            router = ContextRouter(slots=SlotPool(parallel=max(slots, 1)))
            jobs = router.route(
                [
                    PreparedJob(
                        agent_id=a,
                        prompt=synthetic_prompt(a),
                        max_tokens=96,
                        seed=1000 + i,
                        priority=priorities.get(a, 40),
                    )
                    for i, a in enumerate(agents)
                ]
            )
            scheduler = RoundRobinScheduler(backend, SchedulerConfig(admission_gap_s=0.01))
            started = time.perf_counter()
            batch = await scheduler.run(jobs)
            wall = time.perf_counter() - started
            rows.append(
                {
                    "slots": slots,
                    "wall_s": round(wall, 2),
                    "tokens": int(batch.stats.get("tokens", 0)),
                    "tokens_per_s": round(batch.stats.get("tokens", 0) / wall, 1),
                    "failed": batch.failed(),
                }
            )
            print(rows[-1])
        return {"mode": "server", "profile": profile_name, "rows": rows}
    finally:
        await backend.close()
        server.stop()


def bench_inproc(profile_name: str, model: Path) -> dict:
    from engine.llm.backends.llamacpp import LlamaCppBackend

    p = profile(profile_name)
    backend = LlamaCppBackend(str(model), p)
    router = ContextRouter(slots=SlotPool(parallel=1))
    job = router.route([PreparedJob("lena", synthetic_prompt("lena"), 64, 1)])[0]

    async def run() -> dict:
        scheduler = RoundRobinScheduler(backend, SchedulerConfig(admission_gap_s=0))
        started = time.perf_counter()
        batch = await scheduler.run([job])
        wall = time.perf_counter() - started
        res = batch.results["lena"]
        return {
            "mode": "inproc",
            "profile": profile_name,
            "wall_s": round(wall, 2),
            "tokens": res.tokens,
            "tokens_per_s": round(res.tokens / wall, 2) if wall else 0,
            "first_token_ms": res.first_token_ms,
            "ok": res.ok,
            "error": res.error,
        }

    try:
        return asyncio.run(run())
    finally:
        asyncio.run(backend.close())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["server", "inproc"], default="server")
    parser.add_argument("--profile", choices=["dev", "player"], default="dev")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--slots", default="1,4,8,14")
    args = parser.parse_args()

    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"нет файла модели: {model}")

    if args.mode == "server":
        concurrent = [int(x) for x in args.slots.split(",") if x.strip()]
        report = asyncio.run(bench_server(args.profile, model, concurrent))
    else:
        report = bench_inproc(args.profile, model)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / "bench" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nотчёт: {out}")


if __name__ == "__main__":
    main()
