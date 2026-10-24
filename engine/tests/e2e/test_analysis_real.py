"""Разбор настоящей речи: whisper (small), субтитры + align, диаризация, snap. Пороги — с запасом от замеров QUALITY.md."""
from __future__ import annotations

import json

import numpy as np
import pytest

import helpers as H

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def whisper_small_center(tos_audio):
    from dubl_engine.analysis.transcribe import transcribe
    lines, lang = transcribe(tos_audio["center16"], language="en", model="small")
    return lines, lang


@pytest.fixture(scope="module")
def whisper_small_mix(tos_audio):
    from dubl_engine.analysis.transcribe import transcribe
    lines, lang = transcribe(tos_audio["mix16"], language="en", model="small")
    return lines, lang


def test_whisper_small_on_dialogue_center(whisper_small_center, en_lines):
    """Центр 5.1 — почти чистый диалог. Замер: WER 0.22, 61/76 реплик найдено, сдвиг начал ≈ −0.07 с (медиана)."""
    lines, lang = whisper_small_center
    assert lang == "en"
    st = H.compare_to_reference(en_lines, lines)
    assert st.wer_total < 0.35
    assert st.matched_lines >= 50
    assert abs(float(np.median(st.start_offsets))) < 0.5
    # слова с таймингами есть, тайминги монотонны
    words = [w for l in lines for w in l.words]
    assert len(words) > 200
    assert all(w.end >= w.start for w in words)


def test_whisper_small_on_full_mix_is_worse_than_center(whisper_small_mix, whisper_small_center, en_lines):
    """Микс с музыкой хуже центра (замер: WER 0.42 против 0.22) — поэтому подавать на whisper надо центр, если он есть."""
    wm = H.compare_to_reference(en_lines, whisper_small_mix[0]).wer_total
    wc = H.compare_to_reference(en_lines, whisper_small_center[0]).wer_total
    assert wm < 0.65          # хоть что-то слышит
    assert wc < wm            # центр точнее микса


def _outside_reference(ref, lines, pad=1.5):
    return [l for l in lines if not any(min(l.end, r.end + pad) - max(l.start, r.start - pad) > 0 for r in ref)]


@pytest.fixture(scope="module")
def whisper_turbo_mix(tos_audio):
    """Бэкенд по умолчанию: на Mac arm64 — mlx-whisper large-v3-turbo (GPU), иначе faster-whisper."""
    import time

    from dubl_engine.analysis.transcribe import transcribe
    t = time.time()
    lines, lang = transcribe(tos_audio["mix16"], language="en", model="large-v3-turbo")
    return lines, lang, time.time() - t


def test_whisper_turbo_default_backend_does_not_lose_dialogue(whisper_turbo_mix, en_lines):
    """Баг 2 (QUALITY.md): штатный VAD терял диалог поверх музыки (41/76 реплик, WER 0,42). Теперь на Mac — mlx без VAD,
    на остальных — faster-whisper с «мягким» VAD. Замер (mlx, микс): 68/76, WER 0,19, 27 с; faster CPU + VAD: 65/76, WER 0,20, 97 с."""
    lines, lang, _ = whisper_turbo_mix
    assert lang == "en"
    st = H.compare_to_reference(en_lines, lines)
    assert st.matched_lines >= 62
    assert st.wer_total < 0.30
    assert abs(float(np.median(st.start_offsets))) < 0.3
    assert all(w.end >= w.start for l in lines for w in l.words)


def test_whisper_turbo_no_hallucinations_on_music(whisper_turbo_mix, en_lines):
    """Без фильтра mlx на музыке и титрах «слышит» «Thank you.», «I'm sorry.», «We'll be right back.», «I'm going to go to the next video.» ×15."""
    lines, _, _ = whisper_turbo_mix
    fake = _outside_reference(en_lines, lines)
    texts = " ".join(l.text.lower() for l in fake)
    for bad in ("thank you", "i'm sorry", "next video", "be right back", "subscribe"):
        assert bad not in texts, [(l.start, l.text) for l in fake]
    assert len(fake) <= 4, [(l.start, l.text) for l in fake]       # замер: 3 («Go back!», обратный отсчёт в начале — настоящая речь)
    # ни одна реплика не лежит в зоне титров после 590 с, кроме настоящих
    assert not [l for l in lines if l.start > 590 and l.end - l.start > 0.2 and len(l.text.split()) > 0], \
        [(l.start, l.text) for l in lines if l.start > 590]


def test_whisper_turbo_speed_on_mac(whisper_turbo_mix):
    import platform
    import sys
    if sys.platform != "darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        pytest.skip("mlx-whisper — только Mac на Apple Silicon")
    from dubl_engine.analysis.transcribe import mlx_available
    if not mlx_available():
        pytest.skip("mlx-whisper не установлен")
    assert whisper_turbo_mix[2] < 120          # замер 27–40 с на 12-минутном фильме (0,04–0,05 × реального времени)


def test_faster_whisper_soft_vad_keeps_dialogue(tos_audio, en_lines):
    """Только faster-whisper (CPU int8) с «мягким» VAD: замер на миксе — 65/76 реплик, WER 0,20 (штатный VAD: 41/76, WER 0,42)."""
    from dubl_engine.analysis.transcribe import transcribe
    lines, _ = transcribe(tos_audio["mix16"], language="en", model="small", backend="faster", device="cpu")
    st = H.compare_to_reference(en_lines, lines)
    assert st.matched_lines >= 55
    assert st.wer_total < 0.45


def test_whisper_sees_speech_in_known_scene(whisper_small_center):
    lines, _ = whisper_small_center
    text = " ".join(l.text.lower() for l in lines if 290 <= l.start <= 330)
    for w in ("celia", "passions", "robotics"):
        assert w in text, text


def test_align_embedded_to_speech_on_real_lines(en_lines, whisper_small_center):
    from dubl_engine.analysis.align import align_embedded_to_speech
    speech, _ = whisper_small_center
    al = align_embedded_to_speech(en_lines, speech)
    assert len(al) == len(en_lines)
    assert [a.text for a in al] == [e.text for e in en_lines]
    assert all(a.end > a.start for a in al)
    shifts = np.array([abs(a.start - e.start) for a, e in zip(al, en_lines)])
    # правила выравнивания двигают не больше чем на несколько секунд; в среднем — доли секунды (замер: 0.59 с)
    assert shifts.mean() < 1.5
    assert shifts.max() < 6.0


def _saved_turbo_lines():
    import json

    from dubl_engine.models import Line
    f = H.WORK / "analysis" / "whisper_large-v3-turbo.json"
    if not f.exists():
        pytest.skip("нет samples/work/analysis/whisper_large-v3-turbo.json (tests/e2e/rb.sh tr_turbo transcribe large-v3-turbo)")
    return [Line(**d) for d in json.loads(f.read_text("utf-8"))]


def _align_metrics(en_lines, al):
    shifts = [abs(a.start - e.start) for a, e in zip(al, en_lines)]
    overlaps = sum(1 for a, b in zip(al, al[1:]) if b.start < a.end - 1e-6)
    same = sum(1 for a, b in zip(al, al[1:]) if (a.start, a.end) == (b.start, b.end))
    return shifts, overlaps, same


def test_align_keeps_lines_apart(en_lines):
    """Баг 1 (QUALITY.md) исправлен: на сохранённом whisper turbo — ни одного перекрытия соседей, ни одной пары с одинаковым
    интервалом, не больше 2 сдвигов > 0,8 с (было: 12 перекрытий, 16 сдвигов, максимум 4,8 с).
    Замер: 0 перекрытий, 1 сдвиг (0,83 с), среднее 0,2 с."""
    from dubl_engine.analysis.align import align_embedded_to_speech
    al = align_embedded_to_speech(en_lines, _saved_turbo_lines())
    assert [a.id for a in al] == [e.id for e in en_lines] and [a.text for a in al] == [e.text for e in en_lines]
    shifts, overlaps, same = _align_metrics(en_lines, al)
    assert overlaps == 0 and same == 0
    assert sum(s > 0.8 for s in shifts) <= 2
    assert max(shifts) < 1.5
    assert float(np.mean(shifts)) < 0.5
    assert all(0.1 <= a.end - a.start <= 8.0 for a in al)
    assert all(b.start >= a.start for a, b in zip(al, al[1:]))          # порядок сохранён


def test_align_keeps_lines_apart_whisper_small(en_lines, whisper_small_center):
    from dubl_engine.analysis.align import align_embedded_to_speech
    al = align_embedded_to_speech(en_lines, whisper_small_center[0])
    shifts, overlaps, same = _align_metrics(en_lines, al)
    # small по центру: SRT округлён до 0,5 с, поэтому честные сдвиги к речи бывают 0,8–1 с (замер: 7), но не больше 1,5 с
    assert overlaps == 0 and same == 0 and sum(s > 0.8 for s in shifts) <= 10 and max(shifts) < 1.5


# Диаризация на настоящих фильмах (ToS, Sintel, Cosmos Laundromat) — tests/e2e/test_diarize_real.py


def test_snap_and_dialogue_on_real_lines(en_lines):
    from dubl_engine.analysis.snap import dialogue_around, shot_around, snap_selection
    cuts = [8.917, 13.458, 18.417, 25.0, 40.25, 42.083, 45.625, 49.792, 51.125, 53.667]
    lines = en_lines
    # выделение из середины реплики → накрывает реплику целиком (замер: 100 % реплик)
    for ln in lines:
        mid = (ln.start + ln.end) / 2
        a, b = snap_selection(mid - 0.2, mid + 0.2, lines, cuts)
        assert a <= ln.start + 1e-6 and b >= ln.end - 1e-6
    # клик по реплике «Look Celia…» (№2) → вся ссора (реплики 1–9, 23–45 с)
    s, e = dialogue_around(lines[1].start + 0.1, lines)
    assert s <= lines[0].start + 1e-6 and e >= lines[8].end - 1e-6
    n = sum(1 for l in lines if l.start >= s - 1e-6 and l.end <= e + 1e-6)
    assert 9 <= n <= 11
    # кадр вокруг 30 с — от склейки 25.0 до 40.25 (замер)
    assert shot_around(30.0, cuts, 734.17) == (25.0, 40.25)


def test_snap_to_real_cut():
    """Выделение, начатое рядом со склейкой в паузе между репликами, прилипает к ней."""
    from dubl_engine.analysis.snap import snap_selection
    lines = []
    a, b = snap_selection(8.7, 12.0, lines, [8.917, 13.458])
    assert abs(a - 8.917) < 1e-6
    assert b == 12.0                      # до склейки 13.458 — 1.46 с, дальше допуска 0.6
