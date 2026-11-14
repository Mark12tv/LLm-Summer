"""Планировщик хода: генерация всех агентов «одновременно» с честным
переключением по токену.

Как это устроено:

* каждое задание получает свой слот (см. `routing.ContextRouter`);
* задания стартуют в фиксированном порядке (Лагерь → фокус → пары → толпа
  → одиночки → преобразователь) — это и есть приоритет входа в батч;
* дальше все потоки живут параллельно, а `TokenMux` выдаёт токены
  по кругу: один токен первому живому агенту, один — следующему и т.д.
  Никто не ждёт полного ответа другого;
* у каждого задания свой дедлайн; просроченное отменяется, его частичный
  текст уходит в валидатор как «неполный», остальные агенты не страдают.

В серверном режиме настоящий батчинг делает llama.cpp, а `TokenMux`
обеспечивает справедливый порядок уже готовых токенов (нужен для
стриминга игроку и для детерминированного лога).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Sequence

from engine.core.config import ADMISSION_GAP_S, TURN_DEADLINE_S
from engine.core.errors import BackendError
from engine.llm.backends.base import Backend, GenJob, GenResult, TurnBatch

TokenCallback = Callable[[str, str], Awaitable[None] | None]


class TokenMux:
    """Справедливая выдача токенов из N параллельных потоков."""

    def __init__(self, ids: Sequence[str], max_buffered: int = 64) -> None:
        self._order = list(dict.fromkeys(ids))
        self._pos = {aid: i for i, aid in enumerate(self._order)}
        self._buf: dict[str, deque[str]] = {aid: deque() for aid in self._order}
        self._closed: set[str] = set()
        self._cv = asyncio.Condition()
        self._max = max(1, max_buffered)

    async def push(self, agent_id: str, token: str) -> None:
        async with self._cv:
            while len(self._buf[agent_id]) >= self._max and agent_id not in self._closed:
                await self._cv.wait()
            if agent_id in self._closed:
                return
            self._buf[agent_id].append(token)
            self._cv.notify_all()

    async def close(self, agent_id: str) -> None:
        async with self._cv:
            self._closed.add(agent_id)
            self._cv.notify_all()

    @property
    def closed_count(self) -> int:
        return len(self._closed)

    def _next_ready(self, start: int) -> str | None:
        n = len(self._order)
        for step in range(n):
            aid = self._order[(start + step) % n]
            if self._buf[aid]:
                return aid
        return None

    def _done(self) -> bool:
        return all(not d for d in self._buf.values()) and len(self._closed) >= len(self._order)

    async def stream(self):
        """Round-robin: один токен за шаг, курсор двигается вперёд."""
        cursor = 0
        while True:
            async with self._cv:
                aid = self._next_ready(cursor)
                if aid is None:
                    if self._done():
                        return
                    await self._cv.wait()
                    continue
                token = self._buf[aid].popleft()
                cursor = (self._pos[aid] + 1) % len(self._order)
                self._cv.notify_all()
            yield aid, token


@dataclass
class SchedulerConfig:
    deadline_s: float = TURN_DEADLINE_S
    admission_gap_s: float = ADMISSION_GAP_S
    max_buffered_tokens: int = 64
    retry_once: bool = True


@dataclass
class RoundRobinScheduler:
    backend: Backend
    config: SchedulerConfig = field(default_factory=SchedulerConfig)

    async def run(
        self,
        jobs: Sequence[GenJob],
        on_token: TokenCallback | None = None,
    ) -> TurnBatch:
        if not jobs:
            return TurnBatch()

        started = time.perf_counter()
        ordered = sorted(jobs, key=lambda j: (j.priority, j.slot, j.agent_id))
        mux = TokenMux([j.agent_id for j in ordered], self.config.max_buffered_tokens)
        results: dict[str, GenResult] = {}
        admission: list[str] = []

        consumer = asyncio.create_task(self._drain(mux, on_token))
        producers: list[asyncio.Task[None]] = []
        for job in ordered:
            admission.append(job.agent_id)
            producers.append(asyncio.create_task(self._produce(job, mux, results)))
            if self.config.admission_gap_s > 0:
                await asyncio.sleep(self.config.admission_gap_s)

        await asyncio.gather(*producers)
        await consumer

        wall_ms = (time.perf_counter() - started) * 1000.0
        failed = [aid for aid, r in results.items() if not r.ok]
        total_tokens = sum(r.tokens for r in results.values())
        saved = sum(r.reused_tokens for r in results.values() if r.cache_hit)
        return TurnBatch(
            results=results,
            admission=admission,
            wall_ms=wall_ms,
            stats={
                "agents": float(len(results)),
                "failed": float(len(failed)),
                "tokens": float(total_tokens),
                "prefill_saved_tokens": float(saved),
                "cache_hits": float(sum(1 for r in results.values() if r.cache_hit)),
            },
        )

    async def _drain(self, mux: TokenMux, on_token: TokenCallback | None) -> None:
        async for agent_id, token in mux.stream():
            if on_token is None:
                continue
            result = on_token(agent_id, token)
            if inspect.isawaitable(result):
                await result

    async def _produce(
        self,
        job: GenJob,
        mux: TokenMux,
        results: dict[str, GenResult],
    ) -> None:
        t0 = time.perf_counter()
        first_token_ms: float | None = None
        text: list[str] = []
        ok, error, cancelled = False, None, False
        deadline = job.deadline_s or self.config.deadline_s
        retries_left = 1 if self.config.retry_once else 0

        while True:
            text.clear()
            count = 0
            agen = self.backend.stream(job)
            try:
                async with asyncio.timeout(deadline):
                    async for token in agen:
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - t0) * 1000.0
                        text.append(token)
                        count += 1
                        await mux.push(job.agent_id, token)
                        if count >= job.max_tokens:
                            break
                ok, error = True, None
                break
            except TimeoutError:
                cancelled, error = True, "deadline"
                break
            except asyncio.CancelledError:
                cancelled = True
                error = "cancelled"
                raise
            except Exception as exc:  # noqa: BLE001 - любой сбой рантайма
                error = (
                    str(exc) if isinstance(exc, BackendError)
                    else f"{type(exc).__name__}: {exc}"
                )
                # Повторяем только «пустой» прогон: частичный текст уже ушёл
                # в mux, дублировать его нельзя.
                if retries_left > 0 and count == 0:
                    retries_left -= 1
                    await asyncio.sleep(0.05)
                    continue
                break
            finally:
                try:
                    await agen.aclose()
                except Exception:  # pragma: no cover - закрытие потока
                    pass

        await mux.close(job.agent_id)
        results[job.agent_id] = GenResult(
            agent_id=job.agent_id,
            text="".join(text),
            tokens=len(text),
            ok=ok,
            error=error,
            cancelled=cancelled,
            slot=job.slot,
            cache_hit=job.cache_hit,
            reused_tokens=job.reused_tokens,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            first_token_ms=first_token_ms,
        )
