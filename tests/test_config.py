from __future__ import annotations

import pytest

from engine.core.config import (
    DEV_PROFILE,
    N_CTX_PER_SLOT,
    PARALLEL_SLOTS,
    PLAYER_PROFILE,
    check_profile,
    model_quant_bits,
    profile,
)


def test_total_context_is_product_of_slots() -> None:
    assert DEV_PROFILE.total_ctx == N_CTX_PER_SLOT * PARALLEL_SLOTS
    assert DEV_PROFILE.total_ctx == 172_032


def test_quantized_kv_is_cheaper_than_f16() -> None:
    dev_kv = DEV_PROFILE.kv_gib()      # f16
    player_kv = PLAYER_PROFILE.kv_gib()  # q4_0
    assert dev_kv > player_kv * 3
    assert 0.5 < player_kv < 4.0


def test_quant_parser() -> None:
    assert model_quant_bits("llama3.1-heretic-unsensored.Q6_K.gguf") == 6.5
    assert model_quant_bits("model.Q4_K_M.gguf") == 4.5
    assert model_quant_bits("model.Q2_K.gguf") == 2.5
    assert model_quant_bits("Qwen2.5-7B-Instruct-Uncensored.i1-Q4_K_M.gguf") == 4.5
    assert model_quant_bits("model.Q8_0.gguf") == 8.5
    assert model_quant_bits("model.gguf") is None


def test_player_profile_rejects_six_bit_model() -> None:
    problems = check_profile(PLAYER_PROFILE, "llama3.1-heretic-unsensored.Q6_K.gguf")
    assert problems, "Q6_K не должен проходить проверку player-профиля"
    assert any("бит" in item or "VRAM" in item for item in problems)


def test_player_profile_accepts_four_bit_model() -> None:
    assert check_profile(PLAYER_PROFILE, "Qwen2.5-7B-Instruct-Uncensored.i1-Q4_K_M.gguf") == []


def test_dev_profile_accepts_big_model() -> None:
    assert check_profile(DEV_PROFILE, "llama3.1-heretic-unsensored.Q6_K.gguf") == []


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError):
        profile("gpu")


def test_player_requires_flash_attn_for_q4_kv() -> None:
    from dataclasses import replace

    broken = replace(PLAYER_PROFILE, flash_attn=False)
    assert any("flash_attn" in item for item in check_profile(broken, "model.Q4_K_M.gguf"))
