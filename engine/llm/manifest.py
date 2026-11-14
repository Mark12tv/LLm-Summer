"""Манифест рантайма: что именно мы заморозили.

Сейв хранит манифест. Несовпадение любого поля = «⚠ новая ветка»:
партия не продолжается молча (ТЗ §8).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from engine import __version__
from engine.core.config import MODEL_PATH, RUNS_DIR, SAMPLING, RuntimeProfile, model_quant_bits
from engine.llm.prompt import CHAT_FORMAT_VERSION

HASH_CACHE: Path = RUNS_DIR / ".model_hash.json"


def sha256_file(path: Path | str, cache_path: Path = HASH_CACHE) -> str:
    """SHA-256 файла с кэшем по (размер, mtime): 6 ГБ не хэшируем каждый запуск."""
    path = Path(path)
    stat = path.stat()
    key = str(path.resolve())
    cache: dict[str, dict[str, float | str]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}

    entry = cache.get(key)
    if entry and entry.get("size") == stat.st_size and entry.get("mtime") == stat.st_mtime:
        return str(entry["sha256"])

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()

    cache[key] = {"size": stat.st_size, "mtime": stat.st_mtime, "sha256": value}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, cache_path)
    return value


@dataclass(frozen=True)
class Manifest:
    engine: str
    model_sha256: str
    model_quant_bits: float | None
    profile: str
    kv_type: str
    n_ctx_per_slot: int
    parallel: int
    n_gpu_layers: int
    flash_attn: bool
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    chat_format: int
    prompt_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def collect(
    profile: RuntimeProfile,
    *,
    model_path: Path = MODEL_PATH,
    prompt_version: int = 1,
    compute_hash: bool = True,
) -> Manifest:
    return Manifest(
        engine=__version__,
        model_sha256=sha256_file(model_path) if compute_hash else "unhashed",
        model_quant_bits=model_quant_bits(model_path),
        profile=profile.name,
        kv_type=profile.kv_type,
        n_ctx_per_slot=profile.n_ctx_per_slot,
        parallel=profile.parallel,
        n_gpu_layers=profile.n_gpu_layers,
        flash_attn=profile.flash_attn,
        temperature=SAMPLING.temperature,
        top_p=SAMPLING.top_p,
        top_k=SAMPLING.top_k,
        repeat_penalty=SAMPLING.repeat_penalty,
        chat_format=CHAT_FORMAT_VERSION,
        prompt_version=prompt_version,
    )


def diff(saved: dict[str, object], current: dict[str, object]) -> list[str]:
    """Список расхождений «поле: было → стало»."""
    keys = sorted(set(saved) | set(current))
    out: list[str] = []
    for key in keys:
        old, new = saved.get(key), current.get(key)
        if old != new:
            out.append(f"{key}: {old!r} → {new!r}")
    return out


def requires_new_branch(saved: dict[str, object], current: dict[str, object]) -> bool:
    return bool(diff(saved, current))


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
