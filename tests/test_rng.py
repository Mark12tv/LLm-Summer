from __future__ import annotations

import random

from engine.core.rng import RngStreams, derive_seed, turn_seed


def test_seed_is_stable() -> None:
    assert derive_seed(7, "lena", 3) == derive_seed(7, "lena", 3)


def test_seed_separates_parts() -> None:
    assert derive_seed(7, "lena", 3) != derive_seed(7, "ulyana", 3)
    assert derive_seed(7, "lena", 3) != derive_seed(7, "lena", 4)
    assert derive_seed(8, "lena", 3) != derive_seed(7, "lena", 3)


def test_seed_fits_sampler_range() -> None:
    for i in range(100):
        assert 0 <= derive_seed(i, "x", i) < 2**31


def test_agent_stream_is_reproducible_and_independent() -> None:
    streams = RngStreams(game_seed=42)
    a1 = [streams.agent("lena", 1).random() for _ in range(3)]
    a2 = [streams.agent("lena", 1).random() for _ in range(3)]
    b1 = [streams.agent("ulyana", 1).random() for _ in range(3)]
    assert a1 == a2
    assert a1 != b1


def test_turn_seed_matches_agent_stream() -> None:
    assert turn_seed(42, 5, "lena") == derive_seed(42, 5, "lena", "gen")


def test_noise_is_deterministic_float() -> None:
    streams = RngStreams(1)
    value = streams.noise("map", "lena", 3)
    assert 0.0 <= value < 1.0
    assert value == RngStreams(1).noise("map", "lena", 3)
    assert isinstance(random.Random(1).random(), float)
