"""In-process бэкенд на `llama-cpp-python` (dev-режим без сервера).

Особенности:
* одна последовательность за раз — настоящего батчинга тут нет;
* переключение контекстов идёт через **снимки KV-состояния** (`StateCache`):
  после прогона состояние агента уходит в RAM, перед следующим — грузится
  обратно. Для 12k это ~670 МиБ f16 на агента, поэтому кэш ограничен
  бюджетом и вытесняет давно неиспользованных (LRU).

Для продакшена используется HttpBackend: там переключение дешевле и
бесплатное (слоты llama.cpp).
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from typing import Any, AsyncIterator

from engine.core.config import (
    SAMPLING,
    STOP_SEQUENCES,
    RuntimeProfile,
    model_quant_bits,
)
from engine.core.errors import BackendError
from engine.llm.backends.base import GenJob


class StateCache:
    """LRU-кэш KV-состояний агентов с бюджетом в мегабайтах.

    В llama-cpp-python `save_state()` возвращает объект `LlamaState`
    (не bytes), поэтому размер берётся из `llama_state_size`.
    """

    def __init__(self, max_mb: int) -> None:
        self.max_bytes = max(0, max_mb) * 1024 * 1024
        self._store: "OrderedDict[str, object]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def blob_size(blob: object) -> int:
        size = getattr(blob, "llama_state_size", None)
        if isinstance(size, int):
            return int(size)
        try:
            return len(blob)  # type: ignore[arg-type]
        except TypeError:
            return 0

    @property
    def used_bytes(self) -> int:
        return sum(self.blob_size(blob) for blob in self._store.values())

    def get(self, agent_id: str) -> object | None:
        blob = self._store.get(agent_id)
        if blob is None:
            self.misses += 1
            return None
        self._store.move_to_end(agent_id)
        self.hits += 1
        return blob

    def put(self, agent_id: str, blob: object) -> bool:
        size = self.blob_size(blob)
        if self.max_bytes <= 0 or size == 0 or size > self.max_bytes:
            return False
        self._store[agent_id] = blob
        self._store.move_to_end(agent_id)
        while self.used_bytes > self.max_bytes and self._store:
            self._store.popitem(last=False)
            self.evictions += 1
        return True

    def clear(self) -> None:
        self._store.clear()


class LlamaCppBackend:
    name = "llamacpp"

    def __init__(
        self,
        model_path: str,
        profile: RuntimeProfile,
        *,
        state_cache: StateCache | None = None,
        verbose: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise BackendError("llama-cpp-python не установлен") from exc

        bits = model_quant_bits(model_path)
        if profile.is_player() and bits and bits > 4.5:
            raise BackendError(
                f"player: модель {bits:.1f} бит не поместится в VRAM; нужен 4-битный билд"
            )

        self.profile = profile
        self.parallel = 1  # одна последовательность
        self._lock = threading.Lock()
        self._state_cache = state_cache or StateCache(profile.state_cache_mb)
        self._llm: Any = Llama(
            model_path=model_path,
            n_ctx=profile.n_ctx_per_slot,
            n_gpu_layers=profile.n_gpu_layers,
            n_batch=profile.n_batch,
            n_threads=profile.n_threads,
            use_mmap=True,
            use_mlock=False,
            offload_kqv=True,
            flash_attn=profile.flash_attn,
            verbose=verbose,
        )

    def _restore(self, agent_id: str) -> bool:
        blob = self._state_cache.get(agent_id)
        if blob is None:
            return False
        self._llm.load_state(blob)  # type: ignore[arg-type]
        return True

    def _save(self, agent_id: str) -> None:
        try:
            blob = self._llm.save_state()
        except Exception:  # pragma: no cover - зависит от сборки
            return
        self._state_cache.put(agent_id, blob)

    async def stream(self, job: GenJob) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        error: list[BaseException] = []

        def worker() -> None:
            try:
                with self._lock:
                    self._restore(job.agent_id)
                    iterator = self._llm.create_completion(
                        prompt=job.prompt,
                        max_tokens=job.max_tokens,
                        temperature=SAMPLING.temperature,
                        top_p=SAMPLING.top_p,
                        top_k=SAMPLING.top_k,
                        repeat_penalty=SAMPLING.repeat_penalty,
                        seed=job.seed,
                        stop=list(job.stop or STOP_SEQUENCES),
                        stream=True,
                    )
                    for chunk in iterator:
                        text = chunk["choices"][0].get("text", "")
                        if text:
                            loop.call_soon_threadsafe(queue.put_nowait, text)
                    self._save(job.agent_id)
            except BaseException as exc:  # noqa: BLE001 - пробрасываем в async
                error.append(exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=worker, name=f"llm-{job.agent_id}", daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
            if len(error):
                break
        if error:
            raise BackendError(f"llama-cpp: {type(error[0]).__name__}: {error[0]}")

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        self._state_cache.clear()
