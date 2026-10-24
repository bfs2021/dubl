"""Готовый ролик «Для YouTube»: true peak по mp4 (ffmpeg ebur128, как меряют снаружи) не выше цели + 0.2 дБ.

Слепой тест 7: мастер WAV −1.15 dBTP, а в mp4 −0.4 dBTP (цель −1.0). Выброс давал сам кодер AAC в первом кадре —
сцена начинается сразу громко; проверка после AAC при этом последний пересвод отдавала в ролик непроверенным."""
from __future__ import annotations

import re

import numpy as np
import pytest

from dubl_engine.audio.io import read_wav, write_wav
from dubl_engine.audio.render import AAC_FADE_IN_SAMPLES, master_for_aac, render_video
from dubl_engine.models import RenderSettings
from dubl_engine.util import run_ffmpeg

from .synth_voice import SR, voice_track


def _loud_mix(seed: int, seconds: float = 2.0) -> np.ndarray:
    """Громкий микс «голос + бас + удары», самый громкий кусок — в первых 20 мс (сцена вырезана посреди звука)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    v = voice_track([(0.0, seconds)], seconds, seed=seed, level=0.5)
    bass = np.sin(2 * np.pi * rng.uniform(40, 90) * t + rng.uniform(0, 6))
    kick = rng.standard_normal(len(t)) * np.exp(-(((t - rng.uniform(0, 0.02)) % 0.4) * 30))
    s = 0.5 * v + 0.3 * bass + 0.3 * kick
    x = np.stack([s, 0.9 * np.roll(s, int(rng.integers(1, 20)))], axis=1)
    return (x / np.abs(x[:1024]).max() * 0.3).astype(np.float32)


def _ebur128_true_peak(mp4) -> float:
    p = run_ffmpeg(["-loglevel", "info", "-nostats", "-i", mp4, "-map", "0:a", "-af", "ebur128=peak=true",
                    "-f", "null", "-"])
    return float(re.findall(r"Peak:\s+(-?[\d.]+)", p.stderr)[-1])


def test_master_for_aac_fades_in_and_checks_last_attempt(tmp_path, monkeypatch):
    """Начало мастера — плавное (против выброса AAC в первом кадре); последний вариант тоже проверяется: не уложился
    пересвод — мастер тише постоянным гейном."""
    src = write_wav(tmp_path / "in.wav", _loud_mix(1), SR)
    readings = iter([-0.2, -0.3, -0.4, -1.0])     # три пересвода «не помогли», после гейна −1.0
    monkeypatch.setattr("dubl_engine.audio.render.aac_true_peak", lambda *_a, **_k: next(readings))
    res = master_for_aac(src, tmp_path / "o.wav", target_lufs=-14.0, true_peak_db=-1.0)
    y, _ = read_wav(tmp_path / "o.wav")
    assert next(readings, None) is None                         # проверены все варианты, включая последний
    assert np.abs(y[:8]).max() < 1e-3                           # первые отсчёты — почти тишина (нарастание)
    assert np.abs(y[AAC_FADE_IN_SAMPLES:]).max() > 0.1
    assert res.true_peak_db < -1.0                              # гейн −0.65 дБ поверх последнего пересвода


@pytest.mark.parametrize("seed", [19, 30, 10, 15, 7, 17])
def test_youtube_render_true_peak_holds_in_mp4(tmp_path, seed):
    """Синтетический громкий микс → master_for_aac → render_video → ebur128 по mp4: пик ≤ −1.0 + 0.2.
    Быстрый (~3 с на все сиды), поэтому не slow. На сидах 19/30/15 старый код давал −0.7 / −0.8 / −0.9
    при мастере −1.15 (выброс в первом кадре AAC)."""
    clip = tmp_path / "v.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=25", "-t", "2", "-pix_fmt", "yuv420p",
                "-c:v", "libx264", "-preset", "ultrafast", clip])
    src = write_wav(tmp_path / "mix.wav", _loud_mix(seed), SR)
    res = master_for_aac(src, tmp_path / "master.wav", target_lufs=-14.0, true_peak_db=-1.0)
    out = render_video(clip, tmp_path / "master.wav", tmp_path / "yt.mp4",
                       RenderSettings(output_path=str(tmp_path / "yt.mp4")))
    tp = _ebur128_true_peak(out)
    assert tp <= -1.0 + 0.2, (seed, tp, res)
    assert res.lufs >= -15.5, res                               # громкость не «утопили» ради пика
