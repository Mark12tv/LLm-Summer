"""Бэкенд llama-server (HTTP, SSE).

Почему сервер, а не python-биндинг: только `llama-server --parallel N`
умеет непрерывный батчинг — генерация всех агентов идёт одновременно,
и переключение между контекстами стоит почти ничего (общие веса,
разные слоты KV).
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from engine.core.errors import BackendError
from engine.llm.backends.base import GenJob

DEFAULT_BASE_URL = "http://127.0.0.1:8080"


def parse_sse_line(line: str) -> dict[str, Any] | None:
    """Разбирает одну строку SSE. Возвращает dict события или None."""
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return {"stop": True}
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class HttpBackend:
    name = "http"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        parallel: int = 14,
        timeout_s: float = 120.0,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.parallel = parallel
        self._timeout = httpx.Timeout(connect=10.0, read=timeout_s, write=30.0, pool=30.0)
        self._client = client or httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)

    def payload(self, job: GenJob) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": job.prompt,
            "id_slot": job.slot,
            "n_predict": job.max_tokens,
            "seed": job.seed,
            "temperature": job.temperature,
            "top_p": job.top_p,
            "top_k": job.top_k,
            "repeat_penalty": job.repeat_penalty,
            "cache_prompt": True,      # переиспользовать префикс слота
            "stream": True,
            "stop": list(job.stop or job.default_stop),
        }
        if job.grammar:
            body["grammar"] = job.grammar
        return body

    async def stream(self, job: GenJob) -> AsyncIterator[str]:
        if not await self.health():
            raise BackendError(f"llama-server недоступен: {self.base_url}")

        try:
            async with self._client.stream("POST", "/completion", json=self.payload(job)) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:500]
                    raise BackendError(f"HTTP {resp.status_code}: {detail}")

                async for line in resp.aiter_lines():
                    data = parse_sse_line(line)
                    if data is None:
                        continue
                    if "error" in data:
                        raise BackendError(str(data["error"]))
                    token = data.get("content")
                    if token:
                        yield str(token)
                    if data.get("stop"):
                        break
        except httpx.HTTPError as exc:
            raise BackendError(f"транспорт: {type(exc).__name__}: {exc}") from exc

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=3.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def tokenize(self, text: str) -> int | None:
        try:
            resp = await self._client.post("/tokenize", json={"content": text}, timeout=10.0)
            if resp.status_code != 200:
                return None
            data = resp.json()
            tokens = data.get("tokens")
            return len(tokens) if isinstance(tokens, list) else None
        except httpx.HTTPError:
            return None

    async def close(self) -> None:
        await self._client.aclose()
