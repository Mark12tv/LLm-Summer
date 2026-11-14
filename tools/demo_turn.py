"""Демонстрация ядра без модели: один ход всех 14 агентов на MockBackend.

Показывает три вещи, которые нельзя увидеть на глаз в коде:
* порядок входа в батч (Лагерь → фокус → пары → толпа → одиночки → преобразователь);
* честное переключение по токену между контекстами;
* экономию prefill на втором и третьем ходу (префикс-кэш слота).

Запуск:
    python -m tools.demo_turn --turns 3
"""

from __future__ import annotations

import argparse
import asyncio
import time

from engine.agents.registry import AGENTS, get
from engine.core.rng import turn_seed
from engine.llm.backends.base import PreparedJob
from engine.llm.backends.mock import MockBackend
from engine.llm.prompt import render_llama3
from engine.llm.routing import ContextRouter
from engine.llm.scheduler import RoundRobinScheduler, SchedulerConfig
from engine.llm.slots import SlotPool
from engine.turn.order import assign_priorities, build_call_order

FOCUS = {"lena"}
SINGLES = {spec.id for spec in AGENTS if spec.id not in {"camp", "crowd", "converter"}} - FOCUS

SYSTEM_TEMPLATE = (
    "Ты — {name}. Ты живёшь в пионерлагере «Совёнок», лето 1985 года. "
    "Отвечай только структурой: [МЫСЛЬ], [ВЗГЛЯД], [ДЕЙСТВИЕ], [РЕЧЬ], [ЭМОЦИЯ], [ПАМЯТЬ]. "
    "Никаких мета-комментариев. " * 40
)
MEMORY_TEMPLATE = "[ПАМЯТЬ] День 3. Утро. Разговоры у сцены. " * 30


def fake_prompt(agent_id: str, turn: int) -> str:
    spec = get(agent_id)
    system = SYSTEM_TEMPLATE.format(name=spec.name)
    moment = (
        f"{MEMORY_TEMPLATE}\n"
        f"[ВОСПРИЯТИЕ] ход {turn}: площадь, солнечно, вижу Лену у сцены.\n"
        f"[ФЛАГИ] mood=60, trust_sem=40, focus=lena\n"
        f"[ПОСЛЕДНЕЕ] Лена: «Ты чего встал?»"
    )
    return render_llama3(system, moment)


def build_jobs(turn: int, game_seed: int) -> list[PreparedJob]:
    order = build_call_order(focus=FOCUS, singles=SINGLES)
    priorities = assign_priorities(order, FOCUS)
    jobs: list[PreparedJob] = []
    for agent_id in order:
        spec = get(agent_id)
        jobs.append(
            PreparedJob(
                agent_id=agent_id,
                prompt=fake_prompt(agent_id, turn),
                max_tokens=min(spec.out_budget, 24),  # демо-бюджет, не боевой
                seed=turn_seed(game_seed, turn, agent_id),
                priority=priorities[agent_id],
            )
        )
    return jobs


async def run(turns: int, seed: int) -> None:
    backend = MockBackend(tokens_per_job=12, token_delay_s=0.001)
    router = ContextRouter(slots=SlotPool(parallel=len(AGENTS)))
    router.slots.preassign([spec.id for spec in AGENTS])
    scheduler = RoundRobinScheduler(backend, SchedulerConfig(admission_gap_s=0.0))

    for turn in range(1, turns + 1):
        jobs = router.route(build_jobs(turn, seed))
        seen: list[str] = []

        def on_token(agent_id: str, _token: str) -> None:
            seen.append(agent_id)

        started = time.perf_counter()
        batch = await scheduler.run(jobs, on_token=on_token)
        wall = (time.perf_counter() - started) * 1000

        print(f"\n=== ХОД {turn} ===")
        print(f"вход в батч: {', '.join(batch.admission[:6])}…")
        print(f"первая шестёрка токенов: {seen[:6]}")
        print(f"стена: {wall:.0f} мс | токенов: {int(batch.stats['tokens'])} "
              f"| prefill сэкономлено: {int(batch.stats['prefill_saved_tokens'])} ток.")
        for agent_id in batch.admission:
            res = batch.results[agent_id]
            hit = "HIT " if res.cache_hit else "miss"
            print(
                f"  {agent_id:<12} slot={res.slot:<2} {hit} tok={res.tokens:<3} "
                f"first={0 if res.first_token_ms is None else res.first_token_ms:>5.0f}ms "
                f"total={res.elapsed_ms:>5.0f}ms"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1985)
    args = parser.parse_args()
    asyncio.run(run(args.turns, args.seed))
    print(
        "\nГотово. Это ядро на mock-бэкенде: без модели, но с реальным "
        "планировщиком, слотами и учётом префиксов."
    )


if __name__ == "__main__":
    main()
