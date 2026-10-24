from __future__ import annotations

import numpy as np
import pytest

from dubl_engine.audio.fit import FitResult, detect_speech, fit_take, render_fitted, time_stretch
from dubl_engine.audio.io import read_wav, write_wav

from .synth_voice import SR, speech_burst, stretch_signal, voice_track


def _make_take(guide, *, seg, shift_s, factor, take_start_rel, preroll, total=8.0, noise=0.0):
    """Дубль = кусок оригинала seg, сдвинутый на shift_s и растянутый в factor раз (вокруг начала речи)."""
    a, b = seg
    piece = guide[int(a * SR):int(b * SR)]
    piece = stretch_signal(piece, factor)
    scene_sig = np.zeros(int(total * SR), dtype=np.float32)
    i0 = int((a + shift_s) * SR)
    scene_sig[i0:i0 + len(piece)] = piece
    # файл: сцена с take_start_rel, перед ней preroll секунд тишины
    from_i = int(take_start_rel * SR)
    body = scene_sig[from_i:]
    take = np.concatenate([np.zeros(int(preroll * SR), dtype=np.float32), body])
    if noise:
        take = take + noise * np.random.default_rng(1).standard_normal(len(take)).astype(np.float32)
    return take


def test_detect_speech_basic():
    x = voice_track([(1.0, 2.0), (3.0, 3.6)], 5.0)
    x += 0.001 * np.random.default_rng(0).standard_normal(len(x)).astype(np.float32)
    iv = detect_speech(x, SR)
    assert len(iv) == 2
    assert abs(iv[0][0] - 1.0) < 0.06 and abs(iv[0][1] - 2.0) < 0.08
    assert abs(iv[1][0] - 3.0) < 0.06


def test_detect_speech_silence_and_merge():
    assert detect_speech(np.zeros(SR, dtype=np.float32), SR) == []
    t = np.arange(SR * 3) / SR
    x = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    x[:SR] = 0
    x[int(1.5 * SR):int(1.6 * SR)] = 0       # пауза 100 мс < min_gap
    x[2 * SR:] = 0
    assert len(detect_speech(x, SR, min_gap=0.25)) == 1
    assert len(detect_speech(x, SR, min_gap=0.05)) == 2


def test_fit_finds_offset_and_stretch(tmp_path):
    guide = voice_track([(2.0, 3.5), (6.0, 7.0)], 8.0, seed=10)
    take = _make_take(guide, seg=(2.0, 3.5), shift_s=0.18, factor=1.08, take_start_rel=0.5, preroll=1.0)
    gp, tp = write_wav(tmp_path / "guide.wav", guide, SR), write_wav(tmp_path / "take.wav", take, SR)
    r = fit_take(tp, gp, line_start_rel=1.9, line_end_rel=3.6, take_start_rel=0.5, preroll=1.0, latency_ms=0.0)
    assert isinstance(r, FitResult)
    assert abs(r.offset_ms - (-180)) <= 25, r
    assert abs(r.stretch - 1 / 1.08) <= 0.02, r
    assert r.confidence > 0.6, r


def test_fit_with_noise_and_late_speaker(tmp_path):
    guide = voice_track([(2.0, 3.5)], 8.0, seed=11)
    take = _make_take(guide, seg=(2.0, 3.5), shift_s=0.4, factor=1.0, take_start_rel=1.0, preroll=2.0, noise=0.003)
    gp, tp = write_wav(tmp_path / "g.wav", guide, SR), write_wav(tmp_path / "t.wav", take, SR)
    r = fit_take(tp, gp, line_start_rel=2.0, line_end_rel=3.5, take_start_rel=1.0, preroll=2.0, latency_ms=80)
    assert abs(r.offset_ms - (-400)) <= 25, r
    assert r.stretch == 1.0, r


def test_fit_clamps_stretch(tmp_path):
    guide = voice_track([(2.0, 3.0)], 6.0, seed=12)
    take = _make_take(guide, seg=(2.0, 3.0), shift_s=0.0, factor=1.5, take_start_rel=0.0, preroll=0.5, total=6.0)
    gp, tp = write_wav(tmp_path / "g.wav", guide, SR), write_wav(tmp_path / "t.wav", take, SR)
    r = fit_take(tp, gp, line_start_rel=2.0, line_end_rel=3.0, take_start_rel=0.0, preroll=0.5)
    assert r.stretch == pytest.approx(0.85, abs=1e-6)
    assert r.confidence < 0.7


def test_fit_no_speech_returns_zero_confidence(tmp_path):
    guide = voice_track([(2.0, 3.0)], 6.0)
    gp = write_wav(tmp_path / "g.wav", guide, SR)
    tp = write_wav(tmp_path / "t.wav", np.zeros(SR * 5, dtype=np.float32), SR)
    r = fit_take(tp, gp, line_start_rel=2.0, line_end_rel=3.0, take_start_rel=0.0, preroll=0.5)
    assert r.confidence == 0.0 and r.stretch == 1.0


def test_time_stretch_length():
    x = speech_burst(2.0, seed=3)
    y = time_stretch(x, SR, 0.9)
    assert abs(len(y) / len(x) - 0.9) < 0.01


def test_render_fitted_aligns(tmp_path):
    guide = voice_track([(2.0, 3.5)], 8.0, seed=10)
    take = _make_take(guide, seg=(2.0, 3.5), shift_s=0.18, factor=1.08, take_start_rel=0.5, preroll=1.0)
    gp, tp = write_wav(tmp_path / "guide.wav", guide, SR), write_wav(tmp_path / "take.wav", take, SR)
    r = fit_take(tp, gp, line_start_rel=1.9, line_end_rel=3.6, take_start_rel=0.5, preroll=1.0)
    out = render_fitted(tp, r, tmp_path / "fitted.wav")
    y, sr = read_wav(out)
    # после подгонки файл лежит на «наивной» позиции: сцена t = take_start_rel − preroll + t_файл
    iv = detect_speech(y, sr)
    start_scene = 0.5 - 1.0 + iv[0][0]
    end_scene = 0.5 - 1.0 + iv[-1][1]
    assert abs(start_scene - 2.0) < 0.05
    assert abs(end_scene - 3.5) < 0.1


def test_clip_intervals_drops_short_tail_of_previous_sound():
    """Хвост звука, начавшегося до окна и заходящего в него на 40 мс, — не начало реплики (разделённые голоса дают такие «щелчки»)."""
    from dubl_engine.audio.fit import _clip_intervals
    iv = [(1.0, 1.84), (2.12, 3.06), (3.7, 4.2)]
    assert _clip_intervals(iv, 1.8, 4.5) == [(2.12, 3.06), (3.7, 4.2)]
    assert _clip_intervals([(1.0, 2.2)], 1.8, 4.5) == [(1.8, 2.2)]          # длинный заход — это речь, она просто началась раньше окна
