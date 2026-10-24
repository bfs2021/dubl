"""Замеры «Дубля» на настоящих фильмах. Запуск одного шага:

    cd engine && /usr/bin/time -l .venv/bin/python tests/e2e/bench.py <шаг> [аргументы]

Результат шага — JSON в samples/work/bench/<шаг>.json (его собирает run_bench.py в таблицу и docs/QUALITY.md).
Шаги тяжёлые — запускать ПО ОДНОМУ (16 ГБ памяти). Эталоны: TOS_MUSIC+FX (музыка и шумы без диалога), TOS-en.srt.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers as H  # noqa: E402

BENCH = H.WORK / "bench"
BENCH.mkdir(parents=True, exist_ok=True)
A = H.WORK / "analysis"
A.mkdir(parents=True, exist_ok=True)


def save(name: str, data: dict) -> None:
    (BENCH / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=1))


def lines_to_json(lines) -> list[dict]:
    return [ln.model_dump(mode="json") for ln in lines]


def load_lines(path: Path):
    from dubl_engine.models import Line
    return [Line.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]


# ───────────────────────── шаги ─────────────────────────
def step_prep() -> None:
    """Собрать релизный MKV; probe; извлечение звука разными способами."""
    from dubl_engine.media.extract import extract_audio, extract_center_channel, extract_subtitles
    from dubl_engine.media.probe import pick_original_audio, probe
    res: dict = {}
    t = time.time()
    H.build_release_mkv()
    res["build_release_mkv_s"] = round(time.time() - t, 2)
    res["release_size_mb"] = round(H.TOS_RELEASE.stat().st_size / 1e6, 1)
    for name, p in (("release_mkv", H.TOS_RELEASE), ("tos_mov", H.TOS_MOV), ("sintel_mkv", H.SINTEL_MKV),
                    ("ed_mov", H.ED_MOV)):
        t = time.time()
        info = probe(p)
        dt = time.time() - t
        res[f"probe_{name}"] = {"s": round(dt, 3), "duration": info.duration, "audio": len(info.audio_streams),
                                "subs": len(info.subtitle_streams), "chapters": len(info.chapters),
                                "res": f"{info.width}x{info.height}", "fps": round(info.fps, 3)}
    info = probe(H.TOS_RELEASE)
    orig = pick_original_audio(info)
    res["picked_audio"] = orig.model_dump()
    t = time.time()
    extract_audio(H.TOS_RELEASE, orig.index, A / "audio_16k.wav", sample_rate=16000, mono=True)
    res["extract_16k_mono_s"] = round(time.time() - t, 2)
    t = time.time()
    extract_audio(H.TOS_RELEASE, orig.index, A / "audio_48k_51.wav", sample_rate=48000, mono=False)
    res["extract_48k_6ch_s"] = round(time.time() - t, 2)
    t = time.time()
    extract_center_channel(H.TOS_RELEASE, orig.index, A / "center.wav", sample_rate=48000)
    res["extract_center_s"] = round(time.time() - t, 2)
    st = next(s for s in info.audio_streams if s.channels == 2)
    extract_audio(H.TOS_RELEASE, st.index, A / "audio_48k_stereo.wav", sample_rate=48000, mono=False)
    for s in info.subtitle_streams:
        t = time.time()
        extract_subtitles(H.TOS_RELEASE, s.index, A / f"subs_{s.language}")
        res[f"extract_subs_{s.language}_s"] = round(time.time() - t, 3)
    # синхронность 5.1 / стерео / MUSIC+FX с фильмом (по огибающей)
    save("prep", res)


def step_subs() -> None:
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    t = time.time()
    en = parse_subtitle_file(H.TOS_EN)
    ru = parse_subtitle_file(H.TOS_RU)
    dt = time.time() - t
    dur = sum(l.end - l.start for l in en)
    save("subs", {"parse_both_s": round(dt, 4), "en": len(en), "ru": len(ru), "en_speech_s": round(dur, 1),
                  "en_words": sum(len(H.norm_words(H.clean_ref(l.text))) for l in en)})


def step_transcribe(model: str, src: str = "full") -> None:
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from dubl_engine.analysis.transcribe import transcribe
    wav = A / "audio_16k.wav"
    audio_dur = 734.17
    # прогрев загрузки модели отдельно не делаем: время включает загрузку модели, потом печатаем отдельно
    marks = {}
    prog = {"last": 0.0}

    def on(p):
        prog["last"] = p
    t0 = time.time()
    lines, lang = transcribe(wav, language="en", model=model, on_progress=on)
    total = time.time() - t0
    ref = parse_subtitle_file(H.TOS_EN)
    st = H.compare_to_reference(ref, lines)
    offs = np.array(st.start_offsets)
    out = A / f"whisper_{model}.json"
    out.write_text(json.dumps(lines_to_json(lines), ensure_ascii=False), encoding="utf-8")
    save(f"transcribe_{model}", {
        "model": model, "audio_s": audio_dur, "total_s": round(total, 1),
        "s_per_audio_min": round(total / (audio_dur / 60), 2), "rtf": round(total / audio_dur, 3),
        "lang": lang, "lines": len(lines), "words": sum(len(l.words) for l in lines),
        "wer": round(st.wer_total, 4), "ref_words": st.n_ref,
        "matched_lines": f"{st.matched_lines}/{st.total_lines}",
        "start_offset_mean_s": round(float(offs.mean()), 3), "start_offset_median_s": round(float(np.median(offs)), 3),
        "start_offset_abs_mean_s": round(float(np.abs(offs).mean()), 3),
        "start_offset_p90_abs_s": round(float(np.percentile(np.abs(offs), 90)), 3),
        "hyp_text": " | ".join(l.text for l in lines[:12]),
    })


def step_transcribe_center(model: str = "small") -> None:
    """Whisper по центру 5.1 против микса (5.1→mono): что лучше подавать на распознавание."""
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from dubl_engine.analysis.transcribe import transcribe
    from dubl_engine.media.extract import extract_audio
    from dubl_engine.audio.io import read_wav, write_wav
    import scipy.signal as sg
    ref = parse_subtitle_file(H.TOS_EN)
    c, sr = read_wav(A / "center.wav")
    c16 = sg.resample_poly(c, 1, 3).astype(np.float32)
    write_wav(A / "center_16k.wav", c16, 16000)
    t0 = time.time()
    lines, _ = transcribe(A / "center_16k.wav", language="en", model=model)
    dt = time.time() - t0
    st = H.compare_to_reference(ref, lines)
    save(f"transcribe_center_{model}", {"model": model, "total_s": round(dt, 1), "wer": round(st.wer_total, 4),
                                        "lines": len(lines)})


def step_diarize() -> None:
    from dubl_engine.analysis.align import align_embedded_to_speech
    from dubl_engine.analysis.diarize import diarize
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    res: dict = {}
    en = parse_subtitle_file(H.TOS_EN)
    whisper = load_lines(A / "whisper_large-v3-turbo.json")
    aligned = align_embedded_to_speech(en, whisper)
    res["align_mean_shift_s"] = round(float(np.mean([abs(a.start - b.start) for a, b in zip(aligned, en)])), 3)
    for tag, lines in (("srt_aligned", aligned), ("whisper_turbo", whisper)):
        t0 = time.time()
        out, speakers, sugg = diarize(A / "audio_16k.wav", lines, hf_token=None)
        dt = time.time() - t0
        # чистота кластеров по эталону (только реплики из GT_SPEAKERS; для whisper-реплик — по времени начала)
        ref_lines = {i: en[i - 1] for i in H.GT_SPEAKERS}
        pairs = []
        for i, gt in H.GT_SPEAKERS.items():
            r = ref_lines[i]
            mid = (r.start + r.end) / 2
            cand = [l for l in out if l.start - 0.3 <= mid <= l.end + 0.3]
            if cand:
                pairs.append((gt, cand[0].speaker_id))
        # purity: для каждого эталонного персонажа — доля самого частого кластера; для кластера — доля самого частого персонажа
        from collections import Counter
        by_gt: dict = {}
        by_cl: dict = {}
        for g, c in pairs:
            by_gt.setdefault(g, []).append(c)
            by_cl.setdefault(c, []).append(g)
        pur_gt = sum(Counter(v).most_common(1)[0][1] for v in by_gt.values()) / max(1, len(pairs))
        pur_cl = sum(Counter(v).most_common(1)[0][1] for v in by_cl.values()) / max(1, len(pairs))
        res[tag] = {"s": round(dt, 1), "lines": len(out), "speakers_found": len(speakers),
                    "per_speaker_lines": {s.id: s.line_count for s in speakers},
                    "gt_chars": len(by_gt), "gt_lines_checked": len(pairs),
                    "gt_to_cluster": {g: dict(Counter(v)) for g, v in by_gt.items()},
                    "purity_by_character": round(pur_gt, 3), "purity_by_cluster": round(pur_cl, 3),
                    "merge_suggestions": [(a, b, round(s, 2)) for a, b, s in sugg][:8]}
    save("diarize", res)


def step_shots(threshold: str = "0.3") -> None:
    from dubl_engine.media.shots import detect_shot_cuts
    threshold = float(threshold)
    t0 = time.time()
    cuts = detect_shot_cuts(H.TOS_RELEASE, threshold=threshold)
    dt = time.time() - t0
    (A / f"cuts_{threshold}.json").write_text(json.dumps(cuts))
    d = np.diff([0, *cuts])
    save(f"shots_{threshold}", {"threshold": threshold, "s": round(dt, 1), "cuts": len(cuts),
                                "per_min": round(len(cuts) / (734.17 / 60), 1),
                                "min_shot_s": round(float(d.min()), 2), "median_shot_s": round(float(np.median(d)), 2),
                                "first10": cuts[:10]})


def _scene_wavs(name: str, t0: float, t1: float, *, five_one: bool) -> dict:
    """Нарезать кусок сцены: stereo-микс, центр (если 5.1), эталон MUSIC+FX."""
    from dubl_engine.audio.io import read_wav, write_wav
    d = H.WORK / "sep" / name
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    if five_one:
        # как в prepare_scene: extract_audio(..., stereo=True) → стандартный даунмикс ffmpeg (-ac 2)
        H.ff("-ss", t0, "-to", t1, "-i", H.TOS_51, "-vn", "-ac", 2, "-ar", 48000, "-c:a", "pcm_f32le", d / "mix.wav")
        H.ff("-ss", t0, "-to", t1, "-i", H.TOS_51, "-af", "pan=mono|c0=FC", "-ar", 48000, "-c:a", "pcm_f32le", d / "center.wav")
        out["center"] = d / "center.wav"
    else:
        H.ff("-ss", t0, "-to", t1, "-i", H.TOS_MOV, "-ar", 48000, "-ac", 2, "-c:a", "pcm_f32le", d / "mix.wav")
    H.ff("-ss", t0, "-to", t1, "-i", H.TOS_MFX, "-ar", 48000, "-ac", 2, "-c:a", "pcm_f32le", d / "ref_mfx.wav")
    out["mix"] = d / "mix.wav"
    out["ref"] = d / "ref_mfx.wav"
    out["dir"] = d
    return out


def step_separate(quality: str, clip: str, mode: str) -> None:
    """mode: stereo — вход стерео-микс из mov (mp3); c51 — центр из 5.1 + микс из ac3."""
    from dubl_engine.audio.separate import separate
    clips = {"A": (296.0, 326.0), "B": (190.0, 220.0), "C": (100.0, 130.0), "D": (200.0, 320.0)}
    t0, t1 = clips[clip]
    w = _scene_wavs(f"{clip}_{mode}", t0, t1, five_one=(mode == "c51"))
    out_dir = w["dir"] / quality
    t = time.time()
    v, b = separate(w["mix"], out_dir, center_wav=w.get("center"), quality=quality)
    dt = time.time() - t
    save(f"separate_{quality}_{clip}_{mode}", {"quality": quality, "clip": clip, "mode": mode, "t0": t0, "t1": t1,
                                               "s": round(dt, 1), "s_per_30s": round(dt / ((t1 - t0) / 30), 1), "clip_s": t1 - t0,
                                               "voices": str(v), "background": str(b)})


def step_sep_metrics(clip: str, mode: str, quality: str, whisper_model: str = "small") -> None:
    """SDR/корреляция фона против эталона MUSIC+FX + whisper по background.wav и voices.wav."""
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from dubl_engine.analysis.transcribe import transcribe
    from dubl_engine.audio.io import read_wav, write_wav
    import scipy.signal as sg
    clips = {"A": (296.0, 326.0), "B": (190.0, 220.0), "C": (100.0, 130.0)}
    t0, t1 = clips[clip]
    d = H.WORK / "sep" / f"{clip}_{mode}"
    ref, sr = read_wav(d / "ref_mfx.wav")
    mix, _ = read_wav(d / "mix.wav")
    bg, _ = read_wav(d / quality / "background.wav")
    vo, _ = read_wav(d / quality / "voices.wav")
    n = min(len(ref), len(bg), len(mix))
    ref, bg, mix = ref[:n], bg[:n], mix[:n]
    lag = H.align_lag(ref, mix, sr, 0.3)
    # эталон синхронизируем с миксом (вдруг есть сдвиг), одним лагом на оба
    ref_s = H.shift(ref, -lag)
    res = {"clip": clip, "mode": mode, "quality": quality, "ref_lag_ms": round(lag / sr * 1000, 1)}
    cut = int(0.2 * sr)
    sl = slice(cut, n - cut)
    # шкала: эталон и микс из разных мастеров — подгоняем усиление эталона к миксу по неречевым участкам нельзя,
    # поэтому SDR считаем и «как есть», и с оптимальным усилением (si-SDR)
    res["sdr_mix_vs_ref"] = round(H.sdr(ref_s[sl], mix[sl]), 2)
    res["sdr_bg_vs_ref"] = round(H.sdr(ref_s[sl], bg[sl]), 2)
    res["sisdr_mix_vs_ref"] = round(H.sdr(ref_s[sl], mix[sl], scale_invariant=True), 2)
    res["sisdr_bg_vs_ref"] = round(H.sdr(ref_s[sl], bg[sl], scale_invariant=True), 2)
    res["corr_bg_ref"] = round(H.correlation(ref_s[sl], bg[sl]), 4)
    res["corr_mix_ref"] = round(H.correlation(ref_s[sl], mix[sl]), 4)
    res["rms_db_ref"] = round(H.rms_db(ref_s), 1)
    res["rms_db_bg"] = round(H.rms_db(bg), 1)
    res["rms_db_mix"] = round(H.rms_db(mix), 1)
    res["rms_db_voices"] = round(H.rms_db(vo), 1)
    # остаточный диалог в фоне: (bg − ref) относительно (mix − ref) = диалог, что ушёл/остался
    dlg = (mix - ref_s)[sl]
    resid = (bg - ref_s)[sl]
    res["dialogue_reduction_db"] = round(H.rms_db(dlg) - H.rms_db(resid), 1)
    # честнее: усиление эталона берём по участкам БЕЗ реплик (там микс = музыка и шумы), дальше метрики в речи и вне её
    ref_lines = [l for l in parse_subtitle_file(H.TOS_EN) if l.end > t0 and l.start < t1]
    sp = np.zeros(n, bool)
    for l in ref_lines:
        sp[max(0, int((l.start - t0 - 0.2) * sr)): int((l.end - t0 + 0.2) * sr)] = True
    quiet = ~sp
    quiet[:cut] = False
    quiet[n - cut:] = False
    speech = sp.copy()
    speech[:cut] = False
    speech[n - cut:] = False
    res["nonspeech_seconds"] = round(float(quiet.sum() / sr), 1)
    res["speech_seconds"] = round(float(speech.sum() / sr), 1)
    if quiet.sum() > sr:
        g = float(np.sum(mix[quiet] * ref_s[quiet]) / np.sum(ref_s[quiet] ** 2))
        res["ref_gain_fit"] = round(g, 3)
        r2 = ref_s * g
        res["sdr_bg_nonspeech_db"] = round(H.sdr(r2[quiet], bg[quiet]), 2)      # сохранность музыки там, где диалога нет
        res["sdr_mix_nonspeech_db"] = round(H.sdr(r2[quiet], mix[quiet]), 2)    # (для контроля: должно быть высоким)
        res["sdr_bg_speech_db"] = round(H.sdr(r2[speech], bg[speech]), 2)       # фон в речи против чистого M+E
        res["sdr_mix_speech_db"] = round(H.sdr(r2[speech], mix[speech]), 2)     # то же для исходного микса (база)
        res["sdr_gain_in_speech_db"] = round(res["sdr_bg_speech_db"] - res["sdr_mix_speech_db"], 2)
        res["corr_bg_ref_speech"] = round(H.correlation(r2[speech], bg[speech]), 3)
        res["corr_mix_ref_speech"] = round(H.correlation(r2[speech], mix[speech]), 3)
        res["residual_dialogue_db_in_speech"] = round(H.rms_db(bg[speech] - r2[speech]), 1)
        res["original_dialogue_db_in_speech"] = round(H.rms_db(mix[speech] - r2[speech]), 1)
    # whisper по фону и по голосам
    ref_words_n = sum(len(H.norm_words(H.clean_ref(l.text))) for l in ref_lines)
    for tag, x in (("background", bg), ("voices", vo), ("mix", mix)):
        xm = x.mean(axis=1) if x.ndim == 2 else x
        x16 = sg.resample_poly(xm, 1, 3).astype(np.float32)
        p = d / quality / f"{tag}_16k.wav"
        write_wav(p, x16, 16000)
        lines, _ = transcribe(p, language="en", model=whisper_model)
        words = [H.norm_words(l.text) for l in lines]
        flat = [w for ws in words for w in ws]
        ref_flat = [w for l in ref_lines for w in H.norm_words(H.clean_ref(l.text))]
        res[f"whisper_{tag}_words"] = len(flat)
        res[f"whisper_{tag}_wer"] = round(H.wer(ref_flat, flat), 3)
        res[f"whisper_{tag}_text"] = " ".join(flat)[:200]
    res["ref_words"] = ref_words_n
    save(f"sepmetrics_{quality}_{clip}_{mode}", res)


def step_fivone() -> None:
    """Вклад центра 5.1: RMS каналов во время реплик и вне их; утечка музыки в центр (корреляция с MUSIC+FX)."""
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    import soundfile as sf
    ref = parse_subtitle_file(H.TOS_EN)
    x, sr = sf.read(str(A / "audio_48k_51.wav"), dtype="float32")
    names = ["FL", "FR", "FC", "LFE", "SL", "SR"]
    n = len(x)
    speech = np.zeros(n, bool)
    for l in ref:
        speech[int(l.start * sr): int(l.end * sr)] = True
    res = {"channels": {}}
    mfx, msr = sf.read(str(H.TOS_MFX), dtype="float32")
    for i, nm in enumerate(names):
        ch = x[:, i]
        res["channels"][nm] = {"rms_db_speech": round(H.rms_db(ch[speech]), 1),
                               "rms_db_nonspeech": round(H.rms_db(ch[~speech]), 1)}
    e = (x[speech] ** 2).mean(axis=0)
    res["energy_share_during_speech_pct"] = {nm: round(float(100 * e[i] / e.sum()), 1) for i, nm in enumerate(names)}
    e2 = (x[~speech] ** 2).mean(axis=0)
    res["energy_share_nonspeech_pct"] = {nm: round(float(100 * e2[i] / e2.sum()), 1) for i, nm in enumerate(names)}
    # коэффициент: сколько диалога лежит в FL/FR (по MUSIC+FX: FL ≈ mfx.L + g·FC?)
    mx = mfx[: n]
    nn = min(len(mx), n)
    fc = x[:nn, 2]
    for nm, idx, mch in (("FL", 0, 0), ("FR", 1, 1)):
        ch = x[:nn, idx]
        s = speech[:nn]
        # разность канала и эталона (MUSIC+FX) в речи ≈ доля диалога в этом канале
        g = float(np.dot(fc[s], ch[s] - mx[:nn, mch][s]) / (np.dot(fc[s], fc[s]) + 1e-9))
        res[f"dialogue_gain_of_FC_in_{nm}"] = round(g, 3)
    res["corr_FC_vs_mfx_nonspeech"] = round(H.correlation(fc[~speech[:nn]], ((mx[:nn, 0] + mx[:nn, 1]) / 2)[~speech[:nn]]), 3)
    save("fivone", res)


def step_werdetail() -> None:
    """Разбор ошибок whisper: сколько слов эталона найдено (recall), сколько лишних (precision), WER внутри окон реплик."""
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    ref = parse_subtitle_file(H.TOS_EN)
    res = {}
    for model in ("small", "large-v3-turbo"):
        hyp = load_lines(A / f"whisper_{model}.json")
        # слова гипотезы с временем
        hw, ht = [], []
        for ln in hyp:
            for w in ln.words or []:
                for x in H.norm_words(w.text):
                    hw.append(x)
                    ht.append(w.start)
        # окна: слова гипотезы, начало которых в [ref.start-1.0, ref.end+1.0]
        tot_err = tot_ref = 0
        used = np.zeros(len(hw), bool)
        missed_lines = []
        for k, r in enumerate(ref, 1):
            rw = H.norm_words(H.clean_ref(r.text))
            idx = [i for i, t in enumerate(ht) if r.start - 1.0 <= t <= r.end + 1.0]
            used[idx] = True
            hwin = [hw[i] for i in idx]
            e = H.edit_distance(rw, hwin)
            tot_err += e
            tot_ref += len(rw)
            if not hwin:
                missed_lines.append(k)
        extra = [hw[i] for i in range(len(hw)) if not used[i]]
        res[model] = {"wer_in_line_windows": round(tot_err / tot_ref, 3), "ref_lines_with_no_words": len(missed_lines),
                      "missed_line_ids": missed_lines, "hyp_words_outside_windows": len(extra),
                      "outside_sample": " ".join(extra[:40])}
    save("werdetail", res)


def step_whispervar(model: str) -> None:
    """Варианты входа/VAD для whisper (прямой вызов faster-whisper, чтобы менять vad_parameters, которые transcribe() не открывает)."""
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from dubl_engine.analysis.transcribe import whisper_dir
    from faster_whisper import WhisperModel
    ref = parse_subtitle_file(H.TOS_EN)
    wm = WhisperModel(model, device="cpu", compute_type="int8", download_root=str(whisper_dir()))
    res = {}
    soft = {"threshold": 0.2, "min_silence_duration_ms": 500, "speech_pad_ms": 400}
    for name, f in (("mix51_mono", "audio_16k.wav"), ("center", "center_16k.wav")):
        for vad_name, vad, vp in (("vad_default", True, {"min_silence_duration_ms": 500}), ("vad_soft", True, soft),
                                  ("no_vad", False, None)):
            t = time.time()
            segs, _ = wm.transcribe(str(A / f), language="en", word_timestamps=True, vad_filter=vad, vad_parameters=vp,
                                    beam_size=5, condition_on_previous_text=False)
            lines = []
            for sg_ in segs:
                lines.append(type("L", (), dict(text=sg_.text, start=sg_.start,
                                                words=[type("W", (), dict(text=w.word, start=w.start))() for w in sg_.words]))())
            st = H.compare_to_reference(ref, lines)
            offs = np.array(st.start_offsets)
            res[f"{name}/{vad_name}"] = {"s": round(time.time() - t, 1), "wer": round(st.wer_total, 3),
                                         "matched": f"{st.matched_lines}/{st.total_lines}",
                                         "offset_median_s": round(float(np.median(offs)), 3),
                                         "offset_abs_mean_s": round(float(np.abs(offs).mean()), 3)}
    save(f"whispervar_{model}", res)


MLX_MODELS = {"turbo": "mlx-community/whisper-large-v3-turbo", "small": "mlx-community/whisper-small-mlx"}


def step_mlxdl() -> None:
    from huggingface_hub import snapshot_download
    for k, m in MLX_MODELS.items():
        t = time.time()
        snapshot_download(m)
        print(k, m, round(time.time() - t, 1))


def step_mlx(model: str, inp: str, vad: str = "none") -> None:
    """mlx-whisper (Apple GPU). inp: mix | center. Один запуск = один процесс (чтобы честно мерить память).
    vad=silero: сначала режем по тем же VAD-интервалам, что и faster-whisper (не реализовано — mlx-whisper без VAD)."""
    import mlx_whisper
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    ref = parse_subtitle_file(H.TOS_EN)
    f = A / ("audio_16k.wav" if inp == "mix" else "center_16k.wav")
    t = time.time()
    r = mlx_whisper.transcribe(str(f), path_or_hf_repo=MLX_MODELS[model], language="en", word_timestamps=True,
                               condition_on_previous_text=False, verbose=None)
    dt = time.time() - t
    lines = []
    for sg_ in r["segments"]:
        lines.append(type("L", (), dict(text=sg_["text"], start=sg_["start"], end=sg_["end"],
                                        words=[type("W", (), dict(text=w["word"], start=w["start"]))() for w in sg_.get("words", [])]))())
    st = H.compare_to_reference(ref, lines)
    offs = np.array(st.start_offsets)
    try:
        import mlx.core as mx
        gpu_peak = round(mx.get_peak_memory() / 1e6) if hasattr(mx, "get_peak_memory") else None
    except Exception:  # noqa: BLE001
        gpu_peak = None
    (A / f"mlx_{model}_{inp}.json").write_text(json.dumps(
        [{"start": l.start, "end": l.end, "text": l.text, "words": [{"text": w.text, "start": w.start} for w in l.words]} for l in lines],
        ensure_ascii=False), encoding="utf-8")
    save(f"mlx_{model}_{inp}", {"model": MLX_MODELS[model], "input": inp, "s": round(dt, 1), "rtf": round(dt / 734.17, 3),
                                "s_per_audio_min": round(dt / (734.17 / 60), 2), "segments": len(lines),
                                "words": sum(len(l.words) for l in lines), "wer": round(st.wer_total, 3),
                                "matched": f"{st.matched_lines}/{st.total_lines}",
                                "offset_median_s": round(float(np.median(offs)), 3),
                                "offset_abs_mean_s": round(float(np.abs(offs).mean()), 3),
                                "offset_p90_abs_s": round(float(np.percentile(np.abs(offs), 90)), 3),
                                "mlx_gpu_peak_mb": gpu_peak,
                                "first_text": " | ".join(l.text.strip() for l in lines[:6])})


def step_fw(model: str, inp: str, vad: str) -> None:
    """faster-whisper (CPU int8) один прогон: inp mix|center, vad default|soft|none — для честного сравнения с mlx."""
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from dubl_engine.analysis.transcribe import whisper_dir
    from faster_whisper import WhisperModel
    ref = parse_subtitle_file(H.TOS_EN)
    vp = {"default": {"min_silence_duration_ms": 500}, "soft": {"threshold": 0.2, "min_silence_duration_ms": 500, "speech_pad_ms": 400}}.get(vad)
    wm = WhisperModel(model, device="cpu", compute_type="int8", download_root=str(whisper_dir()))
    t = time.time()
    segs, _ = wm.transcribe(str(A / ("audio_16k.wav" if inp == "mix" else "center_16k.wav")), language="en", word_timestamps=True,
                            vad_filter=vad != "none", vad_parameters=vp, beam_size=5, condition_on_previous_text=False)
    lines = [type("L", (), dict(text=x.text, start=x.start, words=[type("W", (), dict(text=w.word, start=w.start))() for w in x.words]))() for x in segs]
    dt = time.time() - t
    st = H.compare_to_reference(ref, lines)
    offs = np.array(st.start_offsets)
    save(f"fw_{model}_{inp}_{vad}", {"model": model, "input": inp, "vad": vad, "s": round(dt, 1), "rtf": round(dt / 734.17, 3),
                                     "wer": round(st.wer_total, 3), "matched": f"{st.matched_lines}/{st.total_lines}",
                                     "offset_median_s": round(float(np.median(offs)), 3),
                                     "offset_abs_mean_s": round(float(np.abs(offs).mean()), 3)})


def step_diarize2() -> None:
    """Диагностика кластеризации: какой эмбеддер, сколько кластеров до обрезки MAX_SPEAKERS, вход микс vs центр."""
    import logging
    from dubl_engine.analysis import diarize as D
    from dubl_engine.analysis.align import align_embedded_to_speech
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from collections import Counter
    logging.basicConfig(level=logging.INFO)
    en = parse_subtitle_file(H.TOS_EN)
    lines = align_embedded_to_speech(en, load_lines(A / "whisper_large-v3-turbo.json"))
    res = {"MAX_SPEAKERS": D.MAX_SPEAKERS, "CLUSTER_THRESHOLD": D.CLUSTER_THRESHOLD}
    for tag, f in (("mix", "audio_16k.wav"), ("center", "center_16k.wav")):
        x = D._LazyAudio(A / f)
        emb_cls = D._SpeechbrainEmbedder
        e = emb_cls()
        res["embedder"] = e.name
        idx = [i for i, l in enumerate(lines) if l.end - l.start >= D.MIN_EMBED]
        t = time.time()
        E = D._embed_lines(x, lines, idx, e, None, 0, 1)
        res[f"{tag}_embed_s"] = round(time.time() - t, 1)
        res[f"{tag}_lines_embedded"] = len(idx)
        per = {}
        for thr in (0.45, 0.55, 0.65, 0.75, 0.85):
            lab = D.cluster_embeddings(E, threshold=thr, max_clusters=99)
            k = len(set(lab))
            gt_pairs = []
            for i, g in H.GT_SPEAKERS.items():
                j = i - 1
                if j in idx:
                    gt_pairs.append((g, int(lab[idx.index(j)])))
            by_gt, by_cl = {}, {}
            for g, c in gt_pairs:
                by_gt.setdefault(g, []).append(c)
                by_cl.setdefault(c, []).append(g)
            pg = sum(Counter(v).most_common(1)[0][1] for v in by_gt.values()) / max(1, len(gt_pairs))
            pc = sum(Counter(v).most_common(1)[0][1] for v in by_cl.values()) / max(1, len(gt_pairs))
            per[str(thr)] = {"clusters": k, "purity_by_character": round(pg, 3), "purity_by_cluster": round(pc, 3),
                             "sizes": sorted(Counter(lab.tolist()).values(), reverse=True)}
        res[tag] = per
    save("diarize2", res)


class TimingCtx:
    """Контекст пайплайна, который запоминает время шагов."""
    job_id = "bench"
    cancelled = False

    def __init__(self) -> None:
        self.t: dict[str, float] = {}

    def step(self, key: str):
        import contextlib

        @contextlib.contextmanager
        def cm():
            t = time.time()
            try:
                yield
            finally:
                self.t[key] = round(time.time() - t, 2)
        return cm()

    def progress(self, key, value) -> None:
        pass

    def skip(self, key) -> None:
        self.t[key] = 0.0

    def check(self) -> None:
        pass

    def set_result(self, result) -> None:
        pass


def step_e2e(quality: str = "best", whisper_model: str = "large-v3-turbo") -> None:
    """Весь путь: проект → analyze_project → сцена → prepare_scene → «дубли» (say) → add_take/fit → render 9:16."""
    import shutil
    import subprocess
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from dubl_engine.audio.io import read_wav
    from dubl_engine.audio.master import measure
    from dubl_engine.media.probe import probe
    from dubl_engine.models import RenderAspect, RenderSettings, Scene, SubtitleMode
    from dubl_engine.pipeline.analyze import analyze_project
    from dubl_engine.pipeline.scene import add_take, prepare_scene, render_scene
    from dubl_engine.settings import AppSettings
    from dubl_engine.store import ProjectStore

    root = H.WORK / "store" / "projects"
    shutil.rmtree(H.WORK / "store", ignore_errors=True)
    store = ProjectStore(root)
    settings = AppSettings(whisper_model=whisper_model, separation_quality=quality)
    res: dict = {"quality": quality, "whisper_model": whisper_model}
    media = probe(H.TOS_RELEASE)
    proj = store.create(H.TOS_RELEASE, media, "Tears of Steel (release)")
    pid = proj.id
    ctx = TimingCtx()
    t = time.time()
    an = analyze_project(store, pid, ctx, settings=settings)
    res["analyze_total_s"] = round(time.time() - t, 1)
    res["analyze_steps_s"] = dict(ctx.t)
    res["analysis"] = {"language": an.language, "lines": len(an.lines), "speakers": len(an.speakers),
                       "shot_cuts": len(an.shot_cuts), "chapters": len(an.chapters),
                       "line_source": sorted({l.source for l in an.lines}), "audio_stream": an.audio_stream_index,
                       "speaker_lines": {s.id: s.line_count for s in an.speakers}}

    # сцена
    a, b = H.DEMO_SCENE
    with store.edit(pid) as p:
        scid = store.next_scene_id(p)
        p.scenes.append(Scene(id=scid, title="Память перезаписывается", start=a, end=b))
    ctx2 = TimingCtx()
    t = time.time()
    sc = prepare_scene(store, pid, scid, ctx2, settings=settings)
    res["prepare_total_s"] = round(time.time() - t, 1)
    res["prepare_steps_s"] = dict(ctx2.t)
    res["scene_lines"] = len(sc.lines)
    sd = store.scene_dir(pid, scid)

    # русские реплики из TOS-ru.srt (по совпадению времени) + «дубли» через say
    ru = parse_subtitle_file(H.TOS_RU)

    def ru_for(line):      # SRT en и ru построчно совпадают (76/76): берём ту же по номеру
        k = int(line.id[1:]) - 1
        return ru[k].text if 0 <= k < len(ru) else None
    delays = [0.35, -0.15, 0.6, 0.0, 0.25, 0.45, -0.1, 0.3]
    preroll = 0.5
    takes_res = []
    new_lines = []
    pr = store.get(pid)
    sc = store.find_scene(pr, scid)
    for k, ln in enumerate(sc.lines):
        src_line = next(l for l in an.lines if l.id == ln.line_id)
        text = ru_for(src_line)
        if not text:
            new_lines.append(ln)
            continue
        ln = ln.model_copy(update={"text": text})
        new_lines.append(ln)
    with store.edit(pid) as p:
        p.scenes[store.scene_index(p, scid)].lines = new_lines
        p.scenes[store.scene_index(p, scid)].dubbed_speakers = sorted({l.speaker_id for l in new_lines if l.speaker_id})
    sc = store.find_scene(store.get(pid), scid)
    voices_all, vsr = read_wav(sd / "voices.wav")
    for k, ln in enumerate(sc.lines):
        if ln.text == ln.original:
            continue
        d = delays[k % len(delays)]
        start_rel = max(0.0, ln.start_rel - 1.0)
        lead = preroll + (ln.start_rel - start_rel) + d      # речь «опоздала» на d секунд относительно начала реплики
        wav = H.say_take(ln.text, H.WORK / "takes" / f"take_{k}.wav", lead_silence=lead)
        x, sr = read_wav(wav)
        # эталонное начало речи оригинала в окне реплики (независимый от движка детектор по голосам сцены)
        i0, i1 = int((ln.start_rel - 0.3) * vsr), int((ln.end_rel + 0.3) * vsr)
        spk_file = sd / "voices" / f"{ln.speaker_id}.wav"      # гид — голос именно этого персонажа (как в add_take)
        gv, gsr = read_wav(spk_file) if spk_file.exists() else (voices_all, vsr)
        j0, j1 = int((ln.start_rel - 0.3) * gsr), int((ln.end_rel + 0.3) * gsr)
        g_onset = H.speech_onset(gv[max(0, j0):j1], gsr, -30) + max(0, j0) / gsr
        t_dub_onset = H.speech_onset(x, sr, -35)                  # начало речи в файле дубля
        naive = start_rel + (t_dub_onset - preroll)
        expected_offset_ms = (g_onset - naive) * 1000
        t = time.time()
        take = add_take(store, pid, scid, wav.read_bytes(), ln.line_id, start_rel, preroll, settings)
        dt = time.time() - t
        takes_res.append({"line": ln.line_id, "spk": ln.speaker_id, "delay_s": d, "orig_s": round(ln.end_rel - ln.start_rel, 2),
                          "dub_s": round(len(x) / sr - t_dub_onset, 2),
                          "expected_offset_ms": round(expected_offset_ms), "offset_ms": round(take.offset_ms),
                          "err_ms": round(take.offset_ms - expected_offset_ms), "stretch": round(take.stretch, 3),
                          "conf": take.fit_confidence, "add_take_s": round(dt, 2), "text": ln.text})
    res["takes"] = takes_res

    # рендер ролика 9:16 с вшитыми субтитрами
    OUT = H.OUT
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "tears_of_steel_dub_ru_9x16.mp4"
    rs = RenderSettings(aspect=RenderAspect.V9_16, height=1080, subtitles=SubtitleMode.BURN, output_path=str(out))
    ctx3 = TimingCtx()
    t = time.time()
    render_scene(store, pid, scid, rs, ctx3)
    res["render_total_s"] = round(time.time() - t, 1)
    res["render_steps_s"] = dict(ctx3.t)
    rendered = store.renders_dir(pid) / out.name
    if rendered.exists():
        shutil.copy(rendered, out)
    info = probe(out)
    res["output"] = {"path": str(out), "size_mb": round(out.stat().st_size / 1e6, 2), "res": f"{info.width}x{info.height}",
                     "duration": info.duration, "audio": [a.model_dump() for a in info.audio_streams]}
    # громкость итога — по звуку готового mp4 (master.wav во временной папке рендера, после рендера его уже нет)
    H.ff("-i", out, "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_f32le", H.WORK / "e2e_master.wav")
    m, msr = read_wav(H.WORK / "e2e_master.wav")
    ld = measure(m, msr)
    res["master"] = {"lufs": round(ld.lufs, 2), "true_peak_db": round(ld.true_peak_db, 2), "lra": round(ld.lra, 2)}
    sc = store.find_scene(store.get(pid), scid)
    res["mix_settings"] = sc.mix.model_dump(mode="json")
    for name in ("background", "voices"):
        shutil.copy(sd / f"{name}.wav", H.WORK / f"e2e_{name}.wav")
    res["scene_files"] = sorted(str(q.relative_to(sd)) for q in sd.rglob("*") if q.is_file())
    save("e2e", res)


def step_fit() -> None:
    """Автоподгонка: (A) чистая синтетика с точной истиной; (B) реальные голоса сцены (best-разделение) + истина по словам whisper."""
    import random
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from dubl_engine.audio.fit import detect_speech, fit_take
    from dubl_engine.audio.io import read_wav, write_wav
    en = parse_subtitle_file(H.TOS_EN)
    ru = parse_subtitle_file(H.TOS_RU)
    wl = load_lines(A / "whisper_large-v3-turbo.json")
    wwords = [(w.start, w.text) for l in wl for w in l.words]
    d = H.WORK / "fit"
    d.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(7)
    res: dict = {}
    # (A) синтетика: гид = «оригинал» (say Samantha, англ. текст), дубль = Milena (русский перевод) с задержкой
    rowsA = []
    ids = [34, 35, 36, 38, 40, 44, 47, 59, 20, 4]
    for i in ids:
        eng, rus = en[i - 1].text, ru[i - 1].text
        g = H.say_take(eng, d / f"g{i}.wav", voice="Samantha", lead_silence=2.0)
        gx, sr = read_wav(g)
        g_on = H.speech_onset(gx, sr, -35)
        g_len = len(gx) / sr - g_on
        g_iv = detect_speech(gx, sr)
        pad = np.zeros(int(3 * sr), np.float32)
        gx = np.concatenate([gx, pad])
        write_wav(d / f"gp{i}.wav", gx, sr)
        ls, le = g_on, len(gx) / sr - 3.0
        for dl in (-0.25, 0.0, 0.3, 0.6):
            lead = 1.0 + dl + rnd.random() * 0.1
            tk = H.say_take(rus, d / f"t{i}_{dl}.wav", voice="Milena", lead_silence=lead)
            x, _ = read_wav(tk)
            t_on = H.speech_onset(x, sr, -35)
            t_len = len(x) / sr - t_on
            take_start_rel, pre = max(0.0, ls - 1.5), 1.0
            naive = take_start_rel + (t_on - pre)
            truth = (g_on - naive) * 1000
            r = fit_take(tk, d / f"gp{i}.wav", line_start_rel=ls, line_end_rel=le, take_start_rel=take_start_rel,
                         preroll=pre, latency_ms=0.0)
            raw = (len(gx) / sr - 3.0 - g_on) / t_len
            rowsA.append({"line": i, "delay": dl, "guide_s": round(g_len, 2), "dub_s": round(t_len, 2),
                          "raw_stretch": round(raw, 2), "truth_ms": round(truth), "offset_ms": round(r.offset_ms),
                          "err_ms": round(r.offset_ms - truth), "stretch": round(r.stretch, 3), "conf": r.confidence})
    errs = np.array([r["err_ms"] for r in rowsA], float)
    res["A_synthetic"] = {"n": len(rowsA), "err_ms_median_abs": round(float(np.median(np.abs(errs))), 1),
                          "err_ms_mean_abs": round(float(np.abs(errs).mean()), 1),
                          "err_ms_p90_abs": round(float(np.percentile(np.abs(errs), 90)), 1),
                          "within_50ms": round(float((np.abs(errs) <= 50).mean()), 3),
                          "within_100ms": round(float((np.abs(errs) <= 100).mean()), 3),
                          "within_150ms": round(float((np.abs(errs) <= 150).mean()), 3), "rows": rowsA}
    # (B) реальные голоса, выделенные из фильма (best, куски по 30 с: A = 296–326 с); истина — начало речи в voices.wav
    #     независимым детектором (окно реплики ±0.3 с, порог −30 дБ от пика окна), т.е. цель — «попасть в живую речь Тома/Селии»
    vfile = H.WORK / "sep" / "A_stereo" / "best" / "voices.wav"
    voices, vsr = read_wav(vfile)
    t0 = 296.0
    rowsB = []
    for k in (33, 34, 35, 36, 37, 38, 39):             # индексы строк en (номера 34–40)
        ln = en[k]
        ls, le = ln.start - t0, ln.end - t0
        if ls < 0.3 or le > 29.5:
            continue
        i0, i1 = int((ls - 0.3) * vsr), int((le + 0.3) * vsr)
        win = voices[i0:i1]
        g_on = H.speech_onset(win, vsr, -30) + i0 / vsr
        rus = ru[k].text
        for dl in (-0.2, 0.3, 0.6):
            tk = H.say_take(rus, d / f"rt{k}_{dl}.wav", voice="Milena", lead_silence=1.0 + dl)
            x, sr = read_wav(tk)
            t_on = H.speech_onset(x, sr, -35)
            take_start_rel, pre = max(0.0, ls - 1.5), 1.0
            naive = take_start_rel + (t_on - pre)
            truth = (g_on - naive) * 1000
            r = fit_take(tk, vfile, line_start_rel=ls, line_end_rel=le, take_start_rel=take_start_rel,
                         preroll=pre, latency_ms=0.0)
            rowsB.append({"line": k + 1, "text": ln.text[:28], "delay": dl, "truth_ms": round(truth), "offset_ms": round(r.offset_ms),
                          "err_ms": round(r.offset_ms - truth), "stretch": round(r.stretch, 3), "conf": r.confidence})
    errs = np.array([r["err_ms"] for r in rowsB], float)
    res["B_real_voices"] = {"n": len(rowsB), "err_ms_median_abs": round(float(np.median(np.abs(errs))), 1),
                            "err_ms_mean_abs": round(float(np.abs(errs).mean()), 1),
                            "within_100ms": round(float((np.abs(errs) <= 100).mean()), 3),
                            "within_200ms": round(float((np.abs(errs) <= 200).mean()), 3), "rows": rowsB}
    save("fit", res)


def step_snap() -> None:
    """snap_selection / dialogue_around / shot_around на реальных репликах и склейках ToS."""
    from dubl_engine.analysis.snap import dialogue_around, shot_around, snap_selection
    from dubl_engine.analysis.align import align_embedded_to_speech
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    en = parse_subtitle_file(H.TOS_EN)
    lines = align_embedded_to_speech(en, load_lines(A / "whisper_large-v3-turbo.json"))
    cuts = json.loads((A / "cuts_0.3.json").read_text())
    res: dict = {"cuts": len(cuts)}
    t = time.time()
    # 1) выделение, начатое в середине реплики, расширяется до её границ и не режет слова
    cut_words = 0
    n = 0
    inside = []
    for ln in lines:
        mid = (ln.start + ln.end) / 2
        a, b = snap_selection(mid - 0.2, mid + 0.2, lines, cuts)
        n += 1
        ok = a <= ln.start + 1e-6 and b >= ln.end - 1e-6
        inside.append(ok)
        # края не должны попадать внутрь чужой реплики
        for other in lines:
            if other is ln:
                continue
            if other.start + 0.02 < a < other.end - 0.02 or other.start + 0.02 < b < other.end - 0.02:
                cut_words += 1
                break
    res["snap_mid_line_covers_line_pct"] = round(100 * float(np.mean(inside)), 1)
    res["snap_edges_inside_other_line"] = cut_words
    # 2) прилипание к склейке
    stuck = 0
    cand = 0
    for c in cuts:
        near = [l for l in lines if abs(l.start - c) < 0.5 or abs(l.end - c) < 0.5]
        if near:
            continue
        cand += 1
        a, b = snap_selection(c - 0.3, c + 2.0, lines, cuts)
        stuck += int(abs(a - c) < 1e-3)
    res["snap_to_cut_pct_of_free_cuts"] = round(100 * stuck / max(1, cand), 1)
    res["snap_free_cuts"] = cand
    # 3) dialogue_around: сколько реплик в «разговоре» (ручная проверка по сюжету: 1–9, 25–32, 34–36, 37–40, 55–60)
    dl = {}
    for i in (2, 26, 35, 38, 56):
        s_, e_ = dialogue_around(lines[i - 1].start + 0.1, lines)
        dl[str(i)] = {"start": round(s_, 2), "end": round(e_, 2), "lines": sum(1 for l in lines if l.start >= s_ - 1e-6 and l.end <= e_ + 1e-6)}
    res["dialogue_around"] = dl
    # 4) shot_around
    sh = {}
    for tt in (30.0, 150.0, 300.0, 500.0):
        s_, e_ = shot_around(tt, cuts, 734.17)
        sh[str(tt)] = [round(s_, 2), round(e_, 2)]
    res["shot_around"] = sh
    res["calls_s"] = round(time.time() - t, 3)
    save("snap", res)


def step_sintel() -> None:
    """Второй фильм: Sintel MKV (релиз с 10 дорожками субтитров, 5.1) — probe, выбор дорожки, субтитры, склейки, whisper по центру."""
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from dubl_engine.analysis.transcribe import transcribe
    from dubl_engine.media.extract import extract_audio, extract_center_channel, extract_subtitles
    from dubl_engine.media.probe import pick_original_audio, probe
    from dubl_engine.media.shots import detect_shot_cuts
    from dubl_engine.audio.io import read_wav, write_wav
    import scipy.signal as sg
    res: dict = {}
    info = probe(H.SINTEL_MKV)
    orig = pick_original_audio(info)
    res["picked"] = orig.model_dump()
    res["subs_langs"] = [s.language for s in info.subtitle_streams]
    res["chapters"] = [(c.title, round(c.start), round(c.end)) for c in info.chapters]
    d = H.WORK / "sintel"
    d.mkdir(parents=True, exist_ok=True)
    en_idx = next(s.index for s in info.subtitle_streams if s.language == "eng")
    p = extract_subtitles(H.SINTEL_MKV, en_idx, d / "subs_eng")
    ref = parse_subtitle_file(p)
    res["sub_lines"] = len(ref)
    res["sub_words"] = sum(len(H.norm_words(l.text)) for l in ref)
    t = time.time()
    extract_center_channel(H.SINTEL_MKV, orig.index, d / "center.wav")
    c, sr = read_wav(d / "center.wav")
    write_wav(d / "center_16k.wav", sg.resample_poly(c, 1, 3).astype(np.float32), 16000)
    extract_audio(H.SINTEL_MKV, orig.index, d / "mix_16k.wav", sample_rate=16000, mono=True)
    res["extract_s"] = round(time.time() - t, 1)
    t = time.time()
    cuts = detect_shot_cuts(H.SINTEL_MKV)
    res["shots"] = {"s": round(time.time() - t, 1), "cuts": len(cuts)}
    for tag in ("mix", "center"):
        t = time.time()
        lines, _ = transcribe(d / f"{tag}_16k.wav", language="en", model="small")
        dt = time.time() - t
        st = H.compare_to_reference(ref, lines)
        res[f"transcribe_small_{tag}"] = {"s": round(dt, 1), "wer": round(st.wer_total, 3), "matched": f"{st.matched_lines}/{st.total_lines}"}
    save("sintel", res)


def step_masterpeak() -> None:
    """Проверка потолка true peak: mix_scene → master(−14 LUFS, −1 dBTP) на сцене из e2e; потом после AAC-кодирования."""
    import json as _j
    from dubl_engine.audio.io import read_wav
    from dubl_engine.audio.master import master, measure
    from dubl_engine.audio.mix import mix_scene
    from dubl_engine.models import Scene
    from dubl_engine.pipeline.scene import stems_for
    from dubl_engine.store import ProjectStore
    store = ProjectStore(H.WORK / "store" / "projects")
    pid = store.list()[0].id
    proj = store.get(pid)
    sc = proj.scenes[0]
    stems = stems_for(store, pid, sc)
    d = H.WORK / "masterpeak"
    d.mkdir(exist_ok=True)
    t = time.time()
    mix_scene(sc, stems, d / "mix.wav")
    t_mix = time.time() - t
    res = {"scene_s": sc.end - sc.start, "mix_s": round(t_mix, 2)}
    x, sr = read_wav(d / "mix.wav")
    m0 = measure(x, sr)
    res["premaster"] = {"lufs": round(m0.lufs, 2), "tp": round(m0.true_peak_db, 2)}
    for tp in (-1.0, -2.0):
        t = time.time()
        m = master(d / "mix.wav", d / f"master_{tp}.wav", target_lufs=-14.0, true_peak_db=tp)
        dt = time.time() - t
        y, _ = read_wav(d / f"master_{tp}.wav")
        mm = measure(y, sr)
        # после AAC 192k (как в render_video)
        H.ff("-i", d / f"master_{tp}.wav", "-c:a", "aac", "-b:a", "192k", d / f"m_{tp}.m4a")
        H.ff("-i", d / f"m_{tp}.m4a", "-c:a", "pcm_f32le", d / f"m_{tp}_dec.wav")
        z, _ = read_wav(d / f"m_{tp}_dec.wav")
        ma = measure(z, sr)
        res[f"target_tp_{tp}"] = {"master_s": round(dt, 2), "returned": {"lufs": round(m.lufs, 2), "tp": round(m.true_peak_db, 2)},
                                  "measured_wav": {"lufs": round(mm.lufs, 2), "tp": round(mm.true_peak_db, 2), "lra": round(mm.lra, 2)},
                                  "after_aac": {"lufs": round(ma.lufs, 2), "tp": round(ma.true_peak_db, 2)}}
    save("masterpeak", res)


def step_nodub(quality: str = "fast") -> None:
    """Сведение сцены БЕЗ дублей и без дыхания: все персонажи озвучиваются, но дублей нет → в миксе остаётся только фон.
    whisper по результату не должен слышать английских реплик оригинала (регрессия «английская речь под дубляжом»)."""
    import scipy.signal as sg
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from dubl_engine.analysis.transcribe import transcribe
    from dubl_engine.audio.io import read_wav, write_wav
    from dubl_engine.audio.mix import mix_scene
    from dubl_engine.pipeline.scene import stems_for
    from dubl_engine.store import ProjectStore
    store = ProjectStore(H.WORK / "store" / "projects")
    pid = store.list()[0].id
    sc = store.get(pid).scenes[0]
    spk = sorted({l.speaker_id for l in sc.lines if l.speaker_id})
    mix = sc.mix.model_copy(update={"keep_breaths": False, "autopilot": False})
    sc2 = sc.model_copy(update={"takes": [], "dubbed_speakers": spk, "mix": mix})
    stems = stems_for(store, pid, sc2)
    d = H.WORK / "nodub"
    d.mkdir(exist_ok=True)
    mix_scene(sc2, stems, d / "mix.wav")
    x, sr = read_wav(d / "mix.wav")
    x16 = sg.resample_poly(x.mean(axis=1), 1, 3).astype(np.float32)
    write_wav(d / "mix16.wav", x16, 16000)
    res = {"quality": quality, "speakers_dubbed": spk, "rms_db": round(H.rms_db(x), 1)}
    ref = [l for l in parse_subtitle_file(H.TOS_EN) if l.end > sc.start and l.start < sc.end]
    ref_words = {w for l in ref for w in H.norm_words(H.clean_ref(l.text)) if len(w) > 3}
    for model in ("small", "large-v3-turbo"):
        lines, _ = transcribe(d / "mix16.wav", language="en", model=model)
        words = [w for l in lines for w in H.norm_words(l.text)]
        res[model] = {"words": words, "ref_content_words_heard": sorted(set(words) & ref_words)}
    save(f"nodub_{quality}", res)


def step_analyze(whisper_model: str = "large-v3-turbo", device: str = "auto") -> None:
    """Один полный analyze_project по релизному MKV (probe → звук → субтитры → whisper → align → диаризация → склейки).
    Для пика памяти продукта: запускать через rb.sh (peak_footprint_mb)."""
    import shutil
    from dubl_engine.media.probe import probe
    from dubl_engine.pipeline.analyze import analyze_project
    from dubl_engine.settings import AppSettings
    from dubl_engine.store import ProjectStore

    shutil.rmtree(H.WORK / "store_analyze", ignore_errors=True)
    store = ProjectStore(H.WORK / "store_analyze" / "projects")
    settings = AppSettings(whisper_model=whisper_model, separation_quality="fast", processing_device=device)
    proj = store.create(H.TOS_RELEASE, probe(H.TOS_RELEASE), "ToS release")
    ctx = TimingCtx()
    t = time.time()
    an = analyze_project(store, proj.id, ctx, settings=settings)
    save(f"analyze_{whisper_model}_{device}", {
        "total_s": round(time.time() - t, 1), "steps_s": dict(ctx.t), "lines": len(an.lines), "speakers": len(an.speakers),
        "language": an.language, "shot_cuts": len(an.shot_cuts), "merge_suggestions": an.merge_suggestions})


def main() -> None:
    step, *a = sys.argv[1:]
    fn = globals()[f"step_{step}"]
    fn(*a)


if __name__ == "__main__":
    main()
