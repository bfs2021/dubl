from __future__ import annotations

import numpy as np
import pytest

from dubl_engine.audio.io import write_wav
from dubl_engine.audio.rooms import ROOM_TITLES, ROOMS, apply_room, estimate_rt60, guess_room
from dubl_engine.util import EngineError

from .synth_voice import SR, voice_track


def test_keys():
    assert set(ROOMS) == {"dry", "room", "kitchen", "hall", "street", "cave", "car", "bathroom", "phone",
                          "behind_door"}
    assert set(ROOM_TITLES) == set(ROOMS)


@pytest.mark.parametrize("name", sorted(ROOMS))
def test_apply_room_length_and_level(name):
    x = voice_track([(0.2, 1.2)], 2.0, seed=3)
    y = apply_room(x, SR, name)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()
    if name == "dry":
        assert np.array_equal(x, y)
    elif name != "behind_door":
        r = np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x ** 2))
        assert 0.7 < r < 1.4


def test_reverb_tail_after_speech_ends():
    x = voice_track([(0.2, 0.8)], 3.0, seed=3)
    tail = slice(int(1.2 * SR), int(1.7 * SR))
    e_dry = np.sqrt(np.mean(apply_room(x, SR, "dry")[tail] ** 2))
    e_cave = np.sqrt(np.mean(apply_room(x, SR, "cave")[tail] ** 2))
    e_room = np.sqrt(np.mean(apply_room(x, SR, "room")[tail] ** 2))
    assert e_dry == 0.0 and e_cave > 10 * e_room > 0


def test_behind_door_darker_and_quieter():
    x = voice_track([(0.2, 1.2)], 2.0, seed=3)
    y = apply_room(x, SR, "behind_door")
    assert np.sqrt(np.mean(y ** 2)) < 0.6 * np.sqrt(np.mean(x ** 2))


def test_stereo_input():
    x = voice_track([(0.2, 1.0)], 1.5, seed=3)
    y = apply_room(np.stack([x, x], 1), SR, "room")
    assert y.shape == (len(x), 2)


def test_unknown_room():
    with pytest.raises(EngineError):
        apply_room(np.zeros(100, dtype=np.float32), SR, "moon")


def _bursts_with_decay(rt60: float, total=12.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(int(total * SR))
    t = np.arange(int(3 * SR)) / SR
    for s in np.arange(0.5, total - 3, 2.0):
        n = int(0.15 * SR)
        burst = rng.standard_normal(n)
        tail = rng.standard_normal(len(t)) * np.exp(-6.9078 * np.clip(t - 0.15, 0, None) / rt60)
        tail[:n] = burst
        i = int(s * SR)
        x[i:i + len(t)] += tail
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def test_estimate_and_guess_room(tmp_path):
    long_ = write_wav(tmp_path / "long.wav", _bursts_with_decay(2.5), SR)
    short = write_wav(tmp_path / "short.wav", _bursts_with_decay(0.25, seed=1), SR)
    rt_long = estimate_rt60(_bursts_with_decay(2.5), SR)
    rt_short = estimate_rt60(_bursts_with_decay(0.25, seed=1), SR)
    assert rt_long is not None and rt_short is not None and rt_long > 3 * rt_short
    assert guess_room(long_) in {"cave", "hall"}
    assert guess_room(short) in {"dry", "room", "kitchen"}


def test_guess_room_default(tmp_path):
    noise = (0.1 * np.random.default_rng(0).standard_normal(SR * 3)).astype(np.float32)
    assert guess_room(write_wav(tmp_path / "n.wav", noise, SR)) == "room"
    assert guess_room(tmp_path / "missing.wav") == "room"
