"""Звук на настоящем материале: разделение голоса и фона (против эталона «музыка+шумы без диалога»), 5.1,
подгонка дубля (`say -v Milena`), сведение, мастеринг. Пороги — с запасом от замеров QUALITY.md."""
from __future__ import annotations

import sys

import numpy as np
import pytest

import helpers as H

pytestmark = pytest.mark.slow


def _sep_metrics(sep_clip, bg_path, voices_path):
    from dubl_engine.audio.io import read_wav
    ref, sr = read_wav(sep_clip["ref"])
    mix, _ = read_wav(sep_clip["mix"])
    bg, _ = read_wav(bg_path)
    vo, _ = read_wav(voices_path)
    n = min(len(ref), len(mix), len(bg))
    ref, mix, bg = ref[:n], mix[:n], bg[:n]
    lag = H.align_lag(ref, mix, sr, 0.3)
    ref = H.shift(ref, -lag)
    sp = np.zeros(n, bool)
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    for l in parse_subtitle_file(H.TOS_EN):
        if l.end > sep_clip["t0"] and l.start < sep_clip["t1"]:
            sp[max(0, int((l.start - sep_clip["t0"] - 0.2) * sr)): int((l.end - sep_clip["t0"] + 0.2) * sr)] = True
    cut = int(0.2 * sr)
    speech = sp.copy()
    speech[:cut] = False
    speech[n - cut:] = False
    quiet = ~sp
    quiet[:cut] = False
    quiet[n - cut:] = False
    g = float(np.sum(mix[quiet] * ref[quiet]) / np.sum(ref[quiet] ** 2))
    r2 = ref * g
    return {
        "sdr_bg_speech": H.sdr(r2[speech], bg[speech]), "sdr_mix_speech": H.sdr(r2[speech], mix[speech]),
        "corr_bg_speech": H.correlation(r2[speech], bg[speech]), "corr_mix_speech": H.correlation(r2[speech], mix[speech]),
        "residual_db": H.rms_db(bg[speech] - r2[speech]), "dialogue_db": H.rms_db(mix[speech] - r2[speech]),
        "voices_len": len(vo), "bg_len": len(bg), "n": n,
    }


def test_separate_fast_beats_raw_mix_on_5_1(sep_clip, sep_fast):
    """demucs по полному даунмиксу 5.1 (центр не используется, см. fac0566): фон в речи ближе к «музыке без диалога», чем
    исходный микс. Замер (30 с, M5): SDR 0.4 дБ против −3.7 дБ у микса, корреляция 0.71 против 0.54, остаток диалога −32 дБ против −28 дБ."""
    m = _sep_metrics(sep_clip, sep_fast["background"], sep_fast["voices"])
    assert m["sdr_bg_speech"] - m["sdr_mix_speech"] > 2.0          # замер: +4.1 дБ
    assert m["corr_bg_speech"] > m["corr_mix_speech"] + 0.08       # замер: +0.17
    assert m["residual_db"] < m["dialogue_db"] - 2.0               # замер: −4.1 дБ
    # длины ровно как у входа (контракт)
    assert abs(m["voices_len"] - m["n"]) <= 2 and abs(m["bg_len"] - m["n"]) <= 2


def test_background_has_no_recognizable_words(sep_clip, sep_fast, tmp_path):
    """whisper по background.wav не должен находить слов (замер: 0 слов и у fast, и у best)."""
    import scipy.signal as sg

    from dubl_engine.analysis.transcribe import transcribe
    from dubl_engine.audio.io import read_wav, write_wav
    bg, sr = read_wav(sep_fast["background"])
    p = write_wav(tmp_path / "bg16.wav", sg.resample_poly(bg.mean(axis=1), 1, 3).astype(np.float32), 16000)
    lines, _ = transcribe(p, language="en", model="small")
    words = [w for l in lines for w in H.norm_words(l.text)]
    assert len(words) <= 3, words


def test_voices_contain_the_dialogue(sep_clip, sep_fast, tmp_path):
    """whisper по voices.wav слышит реплики сцены (замер: WER ≈ 0.35–0.4 на 32 словах, т.е. не хуже микса)."""
    import scipy.signal as sg

    from dubl_engine.analysis.transcribe import transcribe
    from dubl_engine.audio.io import read_wav, write_wav
    v, sr = read_wav(sep_fast["voices"])
    p = write_wav(tmp_path / "v16.wav", sg.resample_poly(v, 1, 3).astype(np.float32), 16000)
    lines, _ = transcribe(p, language="en", model="small")
    txt = " ".join(w for l in lines for w in H.norm_words(l.text))
    for w in ("celia", "passions", "robotics"):
        assert w in txt, txt


def test_split_by_speaker_on_real_voices(sep_clip, sep_fast, tmp_path):
    from dubl_engine.audio.io import read_wav
    from dubl_engine.audio.separate import split_by_speaker
    from dubl_engine.models import SceneLine
    lines = [SceneLine(line_id="l1", speaker_id="s1", start_rel=1.5, end_rel=3.0, original="a", text="a"),
             SceneLine(line_id="l2", speaker_id="s2", start_rel=3.0, end_rel=5.0, original="b", text="b")]
    out = split_by_speaker(sep_fast["voices"], lines, tmp_path / "spk")
    assert set(out) == {"s1", "s2"}
    x1, sr = read_wav(out["s1"])
    full, _ = read_wav(sep_fast["voices"])
    assert len(x1) == len(full)
    assert np.abs(x1[int(10 * sr): int(20 * sr)]).max() < 1e-6      # тишина вне реплик персонажа


def test_center_channel_carries_dialogue_in_5_1():
    """В 5.1 ToS центр несёт диалог (в речи ~26 % энергии, в паузах 1 %) — годится для РАСПОЗНАВАНИЯ речи. Но для РАЗДЕЛЕНИЯ
    не годится: диалог есть и в L/R (замер: FC добавляет в FL/FR ≈ 0.25 своего уровня), поэтому голос вычитают из полного микса."""
    import soundfile as sf
    d = H.WORK / "e2e_sync"
    d.mkdir(exist_ok=True)
    H.ff("-ss", 296, "-to", 326, "-i", H.TOS_51, "-ar", 48000, "-c:a", "pcm_f32le", d / "c51_30s.wav")
    x, sr = sf.read(str(d / "c51_30s.wav"), dtype="float32")
    assert x.shape[1] == 6
    fc = x[:, 2]
    speech = np.concatenate([fc[int(1.5 * sr): int(3.0 * sr)], fc[int(4.0 * sr): int(6.0 * sr)]])
    quiet = fc[int(11 * sr): int(19 * sr)]
    assert H.rms_db(speech) - H.rms_db(quiet) > 10


def test_fit_take_finds_synthetic_delay(tmp_path):
    """`say -v Milena` с искусственной задержкой: подгонка находит сдвиг начала речи.
    Замер: медиана |ошибка| 22 мс, 80 % в пределах 50 мс, 100 % в пределах 150 мс (40 дублей, синтетический гид)."""
    if sys.platform != "darwin":
        pytest.skip("say есть только на macOS")
    from dubl_engine.audio.fit import fit_take
    from dubl_engine.audio.io import read_wav, write_wav
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    en, ru = parse_subtitle_file(H.TOS_EN), parse_subtitle_file(H.TOS_RU)
    errs = []
    for i in (34, 35, 40, 44, 20):
        g = H.say_take(en[i - 1].text, tmp_path / f"g{i}.wav", voice="Samantha", lead_silence=2.0)
        gx, sr = read_wav(g)
        g_on = H.speech_onset(gx, sr, -35)
        gx = np.concatenate([gx, np.zeros(3 * sr, np.float32)])
        write_wav(tmp_path / f"gp{i}.wav", gx, sr)
        ls, le = g_on, len(gx) / sr - 3.0
        for dl in (-0.25, 0.3, 0.6):
            tk = H.say_take(ru[i - 1].text, tmp_path / f"t{i}_{dl}.wav", voice="Milena", lead_silence=1.0 + dl)
            x, _ = read_wav(tk)
            t_on = H.speech_onset(x, sr, -35)
            start_rel, pre = max(0.0, ls - 1.5), 1.0
            truth = (g_on - (start_rel + t_on - pre)) * 1000
            r = fit_take(tk, tmp_path / f"gp{i}.wav", line_start_rel=ls, line_end_rel=le, take_start_rel=start_rel,
                         preroll=pre, latency_ms=0.0)
            errs.append(r.offset_ms - truth)
            assert 0.85 <= r.stretch <= 1.15
    errs = np.abs(np.array(errs))
    assert np.median(errs) < 60
    assert (errs <= 150).mean() >= 0.9


def fit_errors_on_voices(voices_wav, t0, tmp_path, lines=(34, 35, 38, 39), delays=(-0.2, 0.5), fit_lines=None):
    """Ошибка (мс) подгонки say-дублей против начала живой речи в voices.wav (независимый детектор, окно реплики ±0.3 с).
    fit_lines — тайминги реплик, с которыми идёт подгонка (по умолчанию из SRT; можно подать выровненные по речи, как в проекте);
    «истинное» начало речи ищется всегда по окну реплики из SRT, чтобы сравнение не зависело от выравнивания."""
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from dubl_engine.audio.fit import fit_take
    from dubl_engine.audio.io import read_wav
    ru = parse_subtitle_file(H.TOS_RU)
    en = parse_subtitle_file(H.TOS_EN)
    fl = fit_lines if fit_lines is not None else en
    voices, vsr = read_wav(voices_wav)
    errs = []
    for k in lines:                                  # номера реплик в SRT
        ln = en[k - 1]
        ls, le = ln.start - t0, ln.end - t0
        # окно поиска «истинного» начала — вокруг реплики по SRT и по подгоняемым таймингам (SRT округлён до 0,5 с:
        # у реплики 36 речь начинается на 0,4 с раньше SRT, и окно только по SRT обрезало бы её начало)
        w0 = min(ls, fl[k - 1].start - t0)
        i0, i1 = int((w0 - 0.3) * vsr), int((max(le, fl[k - 1].end - t0) + 0.3) * vsr)
        g_on = H.speech_onset_sustained(voices[i0:i1], vsr, -30) + i0 / vsr
        for dl in delays:
            tk = H.say_take(ru[k - 1].text, tmp_path / f"r{k}_{dl}.wav", lead_silence=1.0 + dl)
            x, sr = read_wav(tk)
            start_rel, pre = max(0.0, ls - 1.5), 1.0
            truth = (g_on - (start_rel + H.speech_onset(x, sr, -35) - pre)) * 1000
            r = fit_take(tk, voices_wav, line_start_rel=fl[k - 1].start - t0, line_end_rel=fl[k - 1].end - t0,
                         take_start_rel=start_rel, preroll=pre, latency_ms=0.0)
            errs.append(abs(r.offset_ms - truth))
    return np.array(errs)


def test_fit_take_on_real_separated_voices_fast(sep_clip, sep_fast, tmp_path):
    """Гид — voices.wav от demucs (fast, по полному даунмиксу 5.1). Замер: медиана |ошибка| 17–21 мс, максимум 59–62 мс (8 дублей).
    (До исправления fac0566, когда голос брался из центра 5.1, ошибка доходила до 250 мс — тест это ловит.)"""
    if sys.platform != "darwin":
        pytest.skip("say есть только на macOS")
    errs = fit_errors_on_voices(sep_fast["voices"], sep_clip["t0"], tmp_path)
    print("fit errs (ms):", np.round(errs).tolist(), "median", np.median(errs))
    assert np.median(errs) < 50                       # замер: 17–21 мс (цель задачи — < 100)
    assert errs.max() < 120                           # замер: 59–62 мс


def test_fit_take_on_aligned_lines_of_whole_scene(sep_clip, sep_fast, tmp_path):
    """Подгонка на ВЫРОВНЕННЫХ по речи репликах (как в проекте: analyze → align → сцена), реплики 34–39 (внутри куска 296–326 с).
    До исправления align (QUALITY.md, баг 1) l0038/l0039 получали один интервал и ошибка доходила до 140–160 мс.
    Замер после (align + XCORR_RANGE_S 0,06): медиана 30 мс, максимум 61 мс (12 дублей, 6 реплик × 2 задержки)."""
    if sys.platform != "darwin":
        pytest.skip("say есть только на macOS")
    import json

    from dubl_engine.analysis.align import align_embedded_to_speech
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from dubl_engine.models import Line
    speech_json = H.WORK / "analysis" / "whisper_large-v3-turbo.json"
    if not speech_json.exists():
        pytest.skip("нет samples/work/analysis/whisper_large-v3-turbo.json (tests/e2e/rb.sh tr_turbo transcribe large-v3-turbo)")
    en = parse_subtitle_file(H.TOS_EN)
    al = align_embedded_to_speech(en, [Line(**d) for d in json.loads(speech_json.read_text("utf-8"))])
    errs = fit_errors_on_voices(sep_fast["voices"], sep_clip["t0"], tmp_path, lines=tuple(range(34, 40)), fit_lines=al)
    print("fit errs aligned (ms):", np.round(errs).tolist(), "median", np.median(errs))
    assert len(errs) == 12
    assert np.median(errs) < 60                       # цель < 100 мс; замер 29–31 мс
    assert errs.max() < 120                           # замер 59–61 мс (при XCORR_RANGE_S 0,15 было 170 мс)
