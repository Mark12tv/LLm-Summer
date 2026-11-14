"""Базовые константы, профили рантайма и математика бюджета памяти.

Правило проекта: ни одно число, влияющее на вход/выход модели, не живёт
в коде модулей — только здесь. Изменение любого значения = «⚠ новая ветка»
партии (см. forLLM.md §8).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
RUNS_DIR: Final = PROJECT_ROOT / "runs"

# --- модель -----------------------------------------------------------------

MODEL_PATH: Final = Path("D:/AImodels/llama3.1-heretic-unsensored.Q6_K.gguf")

# Игроку нужен билд, целиком помещающийся в 8 ГБ VRAM:
# веса <= 4 бит + KV q4_0. Для dev-прогонов допускается Q6_K с выгрузкой в RAM.
PLAYER_MAX_QUANT_BITS: Final = 4.5

MODEL_ARCH_LAYERS: Final = 28
MODEL_KV_HEADS: Final = 4
MODEL_HEAD_DIM: Final = 128
MODEL_CTX_TRAIN: Final = 131_072

KV_ELEMS_PER_TOKEN: Final = 2 * MODEL_ARCH_LAYERS * MODEL_KV_HEADS * MODEL_HEAD_DIM
KV_BYTES_PER_ELEM: Final[dict[str, float]] = {
    "f16": 2.0,
    "q8_0": 34 / 32,      # 1.0625
    "q4_0": 18 / 32,      # 0.5625
}

# --- контекст и бюджеты выхода ----------------------------------------------

N_CTX_PER_SLOT: Final = 12_288      # 12k промпта + запас на выход
PARALLEL_SLOTS: Final = 14          # по числу агентов

PROMPT_FULL: Final = 11_800         # девочки, персонал, толпа
PROMPT_SHORT: Final = 8_000         # мальчики, вожатые, «Лагерь»
PROMPT_LIGHT: Final = 2_500         # Женя
PROMPT_CONVERTER: Final = 2_000     # числовой слой

OUT_FULL: Final = 300
OUT_BACKGROUND: Final = 200
OUT_LIGHT: Final = 120

TURN_MINUTES: Final = 12            # игровых минут на один ход
TURN_DEADLINE_S: Final = 55.0       # жёсткий потолок хода (ТЗ: <= 60 c)
ADMISSION_GAP_S: Final = 0.015      # разнос старта запросов: детерминизм порядка

# --- сэмплирование -----------------------------------------------------------

@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.05


SAMPLING: Final = Sampling()

STOP_SEQUENCES: Final = ("<|eot_id|>", "<|end_of_text|>")

# --- профили рантайма --------------------------------------------------------

@dataclass(frozen=True)
class RuntimeProfile:
    """Набор флагов llama.cpp.

    dev    — тесты: большая часть слоёв в RAM, f16 KV, без FA.
    player — целевой: все слои на GPU, q4_0 KV, FA включён.
    """

    name: str
    n_gpu_layers: int              # -1 = все слои
    kv_type: str                   # f16 | q8_0 | q4_0
    flash_attn: bool
    n_ctx_per_slot: int = N_CTX_PER_SLOT
    parallel: int = PARALLEL_SLOTS
    n_batch: int = 512
    n_ubatch: int = 256
    n_threads: int = 8
    state_cache_mb: int = 0        # кэш KV-состояний агентов (только in-process)

    @property
    def total_ctx(self) -> int:
        return self.n_ctx_per_slot * self.parallel

    def kv_bytes(self) -> int:
        per_token = KV_ELEMS_PER_TOKEN * KV_BYTES_PER_ELEM[self.kv_type]
        return int(per_token * self.total_ctx)

    def kv_gib(self) -> float:
        return self.kv_bytes() / 2**30

    def is_player(self) -> bool:
        return self.name == "player"


DEV_PROFILE: Final = RuntimeProfile(
    name="dev",
    n_gpu_layers=12,
    kv_type="f16",
    flash_attn=False,
    state_cache_mb=2048,
)

PLAYER_PROFILE: Final = RuntimeProfile(
    name="player",
    n_gpu_layers=-1,
    kv_type="q4_0",
    flash_attn=True,
    state_cache_mb=0,
)

PROFILES: Final = {"dev": DEV_PROFILE, "player": PLAYER_PROFILE}


def profile(name: str) -> RuntimeProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"неизвестный профиль: {name!r}; есть: {sorted(PROFILES)}") from None


# --- проверка совместимости профиля и модели --------------------------------

_QUANT_RE = re.compile(r"(F32|F16|BF16|I?Q\d(?:_[A-Z0-9]+)*)\.gguf$", re.IGNORECASE)


def model_quant_bits(model_path: Path | str) -> float | None:
    """Грубая оценка бит на вес по имени файла. None — если непонятно."""
    m = _QUANT_RE.search(Path(model_path).name)
    if not m:
        return None
    q = m.group(1).upper()
    if q.startswith("F32"):
        return 32.0
    if q.startswith(("F16", "BF16")):
        return 16.0
    if q.startswith("I"):
        q = q.lstrip("I")
    digits = re.match(r"Q(\d)", q)
    if not digits:
        return None
    bits = float(digits.group(1))
    # Q4_K_M / Q6_K и т.п. — фактически чуть больше номинала
    return bits + 0.5 if "_" in q else bits


def check_profile(p: RuntimeProfile, model_path: Path | str = MODEL_PATH) -> list[str]:
    """Возвращает список проблем. Пустой список — профиль пригоден."""
    problems: list[str] = []
    bits = model_quant_bits(model_path)
    if p.is_player():
        if p.n_gpu_layers != -1:
            problems.append("player: все слои должны быть на GPU (n_gpu_layers=-1)")
        if bits is None:
            problems.append("player: не удалось определить квантование модели")
        elif bits > PLAYER_MAX_QUANT_BITS:
            problems.append(
                f"player: модель {bits:.1f} бит > {PLAYER_MAX_QUANT_BITS} бит — не влезет в VRAM"
            )
        if not p.flash_attn and p.kv_type == "q4_0":
            problems.append("player: q4_0 KV требует flash_attn")
    return problems
