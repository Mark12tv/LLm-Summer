from __future__ import annotations

import hashlib

from engine.core.config import PLAYER_PROFILE
from engine.llm.manifest import collect, diff, requires_new_branch, sha256_file


def test_sha256_is_cached_by_size_and_mtime(tmp_path) -> None:
    target = tmp_path / "model.gguf"
    target.write_bytes(b"weights")
    cache = tmp_path / "hash.json"
    first = sha256_file(target, cache)
    second = sha256_file(target, cache)
    assert first == second == hashlib.sha256(b"weights").hexdigest()
    assert cache.exists()


def test_manifest_collect_and_diff(tmp_path) -> None:
    target = tmp_path / "model.Q4_K_M.gguf"
    target.write_bytes(b"x")
    manifest = collect(PLAYER_PROFILE, model_path=target)
    current = manifest.to_dict()
    assert current["profile"] == "player"
    assert current["kv_type"] == "q4_0"
    assert current["model_quant_bits"] == 4.5
    assert diff(current, current) == []
    assert not requires_new_branch(current, current)


def test_any_parameter_change_requires_new_branch(tmp_path) -> None:
    target = tmp_path / "model.Q4_K_M.gguf"
    target.write_bytes(b"x")
    saved = collect(PLAYER_PROFILE, model_path=target).to_dict()
    changed = dict(saved, temperature=0.2)
    assert requires_new_branch(saved, changed)
    assert any("temperature" in item for item in diff(saved, changed))


def test_model_change_requires_new_branch(tmp_path) -> None:
    first = tmp_path / "a.Q4_K_M.gguf"
    second = tmp_path / "b.Q4_K_M.gguf"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    m1 = collect(PLAYER_PROFILE, model_path=first).to_dict()
    m2 = collect(PLAYER_PROFILE, model_path=second).to_dict()
    assert requires_new_branch(m1, m2)
