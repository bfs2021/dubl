"""Медиа на настоящем фильме: релизный MKV, probe, звук, центр 5.1, субтитры, склейки, Sintel.
Пороги — из замеров (docs/QUALITY.md) с запасом."""
from __future__ import annotations

import time

import pytest
import soundfile as sf

import helpers as H

pytestmark = pytest.mark.slow


def test_release_mkv_probe(release_mkv):
    from dubl_engine.media.probe import pick_original_audio, probe
    info = probe(release_mkv)
    assert abs(info.duration - 734.17) < 1.0
    assert (info.width, info.height) == (1280, 534)
    assert len(info.audio_streams) == 2 and len(info.subtitle_streams) == 2
    assert {s.language for s in info.subtitle_streams} == {"eng", "rus"}
    assert all(s.is_text for s in info.subtitle_streams)
    assert len(info.chapters) == 3 and info.chapters[0].title == "Пролог"
    a = pick_original_audio(info)
    assert a.channels == 6 and a.language == "eng" and a.default


def test_ac3_is_in_sync_with_film(release_mkv):
    """5.1 из ac3 по длительности и огибающей совпадает с фильмом (иначе вся идея с центром неверна)."""
    import numpy as np
    from scipy.signal import fftconvolve

    from dubl_engine.media.probe import probe
    assert abs(probe(H.TOS_51).duration - probe(H.TOS_MOV).duration) < 1.0
    d = H.WORK / "e2e_sync"
    d.mkdir(exist_ok=True)
    H.ff("-t", 300, "-i", H.TOS_MOV, "-vn", "-ac", 1, "-ar", 8000, d / "film.wav")
    H.ff("-t", 300, "-i", H.TOS_51, "-vn", "-ac", 1, "-ar", 8000, d / "ac3.wav")

    def env(p):
        x, _ = sf.read(str(p))
        k = 80
        m = len(x) // k
        e = np.sqrt((x[: m * k].reshape(m, k) ** 2).mean(1))
        return e - e.mean()
    a, b = env(d / "film.wav"), env(d / "ac3.wav")
    c = fftconvolve(b, a[::-1], mode="full")
    lag = (int(np.argmax(c)) - (len(a) - 1)) / 100
    assert abs(lag) < 0.1                       # замер: 0.00 с
    assert c.max() / np.sqrt((a ** 2).sum() * (b ** 2).sum()) > 0.5   # замер: 0.69


def test_extract_center_and_audio(tos_audio):
    import numpy as np

    from dubl_engine.audio.io import read_wav
    mix, sr = read_wav(tos_audio["mix16"])
    assert sr == 16000 and mix.ndim == 1 and abs(len(mix) / sr - 734.0) < 1.5
    c, csr = read_wav(tos_audio["center16"])
    assert csr == 16000 and abs(len(c) / csr - 734.0) < 1.5
    # центр 5.1 в ToS — почти чистый диалог: вне речи он на ~15 дБ тише, чем в речи
    rms = lambda x: 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)  # noqa: E731
    seg_speech = c[int(318 * csr): int(323 * csr)]          # «We have to follow our passions… you have your robotics»
    seg_quiet = c[int(100 * csr): int(105 * csr)]
    assert rms(seg_speech) - rms(seg_quiet) > 10


def test_extract_subtitles_and_parse(release_mkv, en_lines, ru_lines):
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from dubl_engine.media.extract import extract_subtitles
    from dubl_engine.media.probe import probe
    info = probe(release_mkv)
    eng = next(s for s in info.subtitle_streams if s.language == "eng")
    p = extract_subtitles(release_mkv, eng.index, H.WORK / "e2e_subs" / "eng")
    lines = parse_subtitle_file(p)
    assert len(lines) == len(en_lines) == 76 == len(ru_lines)
    assert lines[0].text == "You're a jerk, Thom."
    assert all(a.start <= b.start for a, b in zip(lines, lines[1:]))
    assert abs(sum(l.end - l.start for l in lines) - 136.2) < 1.0
    # многострочные реплики склеены, переносов нет
    assert all("\n" not in l.text for l in lines)


def test_shot_cuts_whole_film(release_mkv):
    from dubl_engine.media.shots import detect_shot_cuts
    t = time.time()
    cuts = detect_shot_cuts(release_mkv)
    dt = time.time() - t
    # замер: 136 склеек за 4.2 с (порог 0.3)
    assert 100 <= len(cuts) <= 180
    assert dt < 60
    assert cuts == sorted(cuts) and cuts[0] > 0.05
    assert 8.0 < cuts[0] < 10.0                  # первая склейка в фильме — 8.9 с (замер)


def test_sintel_release_probe():
    if H.need(H.SINTEL_MKV):
        pytest.skip("нет Sintel.2010.720p.mkv (python3 samples/fetch.py sintel)")
    from dubl_engine.media.probe import pick_original_audio, probe
    info = probe(H.SINTEL_MKV)
    assert abs(info.duration - 888.03) < 1
    assert len(info.subtitle_streams) == 10 and len(info.chapters) == 8
    a = pick_original_audio(info)
    assert a.channels == 6 and a.language == "eng"
