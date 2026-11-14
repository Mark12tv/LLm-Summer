"""Дымовой тест in-process бэкенда на реальной модели (маленький контекст).

Проверяет, что цепочка «промпт → слот → KV-снимок → токены» жива,
не нагружая машину 12k контекстом:

    python -m tools.smoke --ctx 1024 --tokens 6
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import replace

from engine.core.config import DEV_PROFILE, MODEL_PATH
from engine.llm.backends.base import GenJob
from engine.llm.backends.llamacpp import LlamaCppBackend
from engine.llm.prompt import render_llama3


async def run(ctx: int, tokens: int, agent: str, second: bool) -> None:
    profile = replace(
        DEV_PROFILE,
        n_ctx_per_slot=ctx,
        n_gpu_layers=0,
        n_batch=128,
        n_ubatch=64,
        state_cache_mb=512,
    )
    started = time.perf_counter()
    backend = LlamaCppBackend(str(MODEL_PATH), profile)
    print(f"модель загружена за {time.perf_counter() - started:.1f} с")
    try:
        for attempt in (1, 2) if second else (1,):
            prompt = render_llama3(
                "Ты — Лена, тихая девочка из пионерлагеря. Отвечай одной фразой.",
                f"Скажи ровно одно короткое предложение. Попытка {attempt}.",
            )
            job = GenJob(agent_id=agent, prompt=prompt, max_tokens=tokens, seed=100 + attempt)
            t0 = time.perf_counter()
            chunks: list[str] = []
            async for token in backend.stream(job):
                chunks.append(token)
            dt = time.perf_counter() - t0
            print(f"\n--- прогон {attempt} ---")
            print("текст:", "".join(chunks).strip()[:300])
            print(f"токенов: {len(chunks)}, время: {dt:.1f} с, "
                  f"{len(chunks) / dt:.1f} ток/с")
        cache = backend._state_cache
        print(f"\nKV-кэш: hits={cache.hits} misses={cache.misses} "
              f"evictions={cache.evictions} used={cache.used_bytes / 2**20:.0f} МиБ")
    finally:
        await backend.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--agent", default="lena")
    parser.add_argument("--second", action="store_true", help="второй прогон: проверить KV-снимок")
    args = parser.parse_args()
    asyncio.run(run(args.ctx, args.tokens, args.agent, args.second))


if __name__ == "__main__":
    main()
