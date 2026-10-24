from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from dubl_engine.audio.fit import detect_speech
from dubl_engine.audio.io import db_to_gain, read_wav, write_wav
from dubl_engine.audio.mix import SceneStems, _pick_takes, automix, mix_scene
from dubl_engine.models import MixMode, MixSettings, Scene, SceneLine, Take, TrackSettings

from .synth_voice import SR, voice_track

DUR = 8.0
DUCK = 6.0


def _noise_bg(level=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n = int(DUR * SR)
    t = np.arange(n) / SR
    mono = level * (0.5 * np.sin(2 * np.pi * 180 * t) + 0.5 * rng.standard_normal(n)).astype(np.float32)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)


def _line(lid, spk, a, b):
    return SceneLine(line_id=lid, speaker_id=spk, start_rel=a, end_rel=b, original="o", text="t")


def _take(tid, lid, spk, file, *, start_rel=0.0, preroll=1.0, offset_ms=0.0, stretch=1.0, selected=False,
          created=None):
    return Take(id=tid, line_id=lid, speaker_id=spk, file=file, duration=5.0, preroll=preroll, start_rel=start_rel,
                offset_ms=offset_ms, stretch=stretch, selected=selected,
                **({"created": created} if created else {}))


@pytest.fixture
def world(tmp_path):
    """Сцена 8 с: A (кот) говорит 1–3 с, B (мила) 4–6 с. Озвучиваем A."""
    scene_dir = tmp_path / "scene"
    (scene_dir / "takes").mkdir(parents=True)
    voices_a = voice_track([(1.0, 3.0)], DUR, seed=1, level=1.0, f0=120)
    voices_b = voice_track([(4.0, 6.0)], DUR, seed=2, level=1.0, f0=220)
    bg = _noise_bg()
    paths = {
        "voices": write_wav(scene_dir / "voices.wav", voices_a + voices_b, SR),
        "background": write_wav(scene_dir / "background.wav", bg, SR),
        "a": write_wav(scene_dir / "voices_a.wav", voices_a, SR),
        "b": write_wav(scene_dir / "voices_b.wav", voices_b, SR),
    }
    stems = SceneStems(voices=paths["voices"], background=paths["background"],
                       per_speaker={"s1": paths["a"], "s2": paths["b"]},
                       takes_dir=scene_dir / "takes", scene_dir=scene_dir)
    scene = Scene(id="sc001", title="t", start=0, end=DUR, dubbed_speakers=["s1"],
                  lines=[_line("l1", "s1", 1.0, 3.0), _line("l2", "s2", 4.0, 6.0)])
    return scene, stems, scene_dir


def _make_take_file(scene_dir, name, *, seg=(1.0, 3.0), delay=0.0, level=1.0, preroll=1.0, start_rel=0.0, seed=20):
    """Дубль: файл с preroll тишины, речь на сцене-времени seg[0]+delay."""
    n = int((DUR + preroll) * SR)
    speech = voice_track([(0, seg[1] - seg[0])], seg[1] - seg[0] + 0.1, seed=seed, level=level, f0=150)
    x = np.zeros(n, dtype=np.float32)
    i = int((preroll + seg[0] - start_rel + delay) * SR)
    x[i:i + len(speech)] = speech
    x += 0.0005 * np.random.default_rng(3).standard_normal(n).astype(np.float32)
    write_wav(scene_dir / "takes" / f"{name}.wav", x, SR)
    return f"takes/{name}.wav"


def test_pick_takes_selection():
    now = datetime.now().astimezone()
    sc = Scene(id="s", title="t", start=0, end=8, dubbed_speakers=["s1"],
               lines=[_line("l1", "s1", 1, 3), _line("l2", "s2", 4, 6)],
               takes=[_take("t1", "l1", "s1", "a", created=now), _take("t2", "l1", "s1", "b", created=now + timedelta(1)),
                      _take("t3", "l2", "s2", "c")])
    assert [t.id for t, _, _ in _pick_takes(sc)] == ["t2"]                     # последний, чужой персонаж не берётся
    sc.takes[0].selected = True
    assert [t.id for t, _, _ in _pick_takes(sc)] == ["t1"]                     # выбранный важнее свежего
    sc.takes.append(_take("w", None, "s1", "w", selected=True))
    ids = [t.id for t, _, _ in _pick_takes(sc)]
    assert ids == ["t1", "w"] or set(ids) == {"t1", "w"}


def test_mix_position_and_replacement(world, tmp_path):
    scene, stems, scene_dir = world
    # речь дубля на 0.3 с позже оригинала, offset_ms=-300 возвращает её на место
    f = _make_take_file(scene_dir, "t1", delay=0.3)
    scene.takes = [_take("t1", "l1", "s1", f, offset_ms=-300, selected=True)]
    scene.mix = MixSettings(tracks={"s1": TrackSettings(room="dry")}, keep_breaths=False)
    out = mix_scene(scene, stems, tmp_path / "o" / "mix.wav")
    y, sr = read_wav(out)
    assert sr == SR and y.ndim == 2 and y.shape[0] == int(DUR * SR)
    # новая речь начинается ~1.0 с, оригинал s1 не слышен, s2 (оригинал) на 4–6 с остался
    # (фон шумит, поэтому сравниваем с mix без голосов)
    mono = y.mean(axis=1)
    bg = stems.background
    bg_only = read_wav(bg)[0].mean(axis=1)
    diff = mono - bg_only * 1.0
    e1 = _rms_db(diff[int(1.0 * SR):int(2.8 * SR)])
    e0 = _rms_db(diff[int(0.0 * SR):int(0.8 * SR)])
    assert e1 > e0 + 12
    # начало речи дубля в смеси около 1.0 с
    v = voice_track([(1.0, 3.0)], DUR, seed=1)  # опора только для сравнения диапазона
    first = np.argmax(np.abs(diff[:int(2 * SR)]) > 0.02) / SR
    assert 0.9 < first < 1.15
    assert _rms_db(diff[int(4.2 * SR):int(5.8 * SR)]) > e0 + 12                  # оригинал B остался
    # без ducking-артефактов длина и конечность
    assert np.isfinite(y).all()


def test_mix_dub_keeps_original_until_take_recorded(world, tmp_path):
    scene, stems, scene_dir = world
    scene.mix = MixSettings(keep_breaths=False, background=TrackSettings(mute=True))
    y, _ = read_wav(mix_scene(scene, stems, tmp_path / "m.wav"))
    # дублей нет → оба персонажа звучат как в оригинале (раньше s1 пропадал — это и был баг «пропал голос»)
    assert _rms_db(y[int(1.2 * SR):int(2.8 * SR)]) > -30
    assert _rms_db(y[int(4.2 * SR):int(5.8 * SR)]) > -30


def test_mix_dub_removes_original_only_on_recorded_lines(world, tmp_path):
    scene, stems, scene_dir = world
    # второй персонаж тоже озвучивается, но дубль записан только на реплику s1
    scene.dubbed_speakers = ["s1", "s2"]
    silent = np.zeros(int((DUR + 1.0) * SR), dtype=np.float32)
    write_wav(scene_dir / "takes" / "quiet.wav", silent + 1e-4, SR)   # «дубль» без речи: проверяем только оригинал
    scene.takes = [_take("t1", "l1", "s1", "takes/quiet.wav", selected=True)]
    scene.mix = MixSettings(keep_breaths=False, background=TrackSettings(mute=True))
    y, _ = read_wav(mix_scene(scene, stems, tmp_path / "m.wav"))
    assert _rms_db(y[int(1.2 * SR):int(2.8 * SR)]) < -60          # реплика s1 заменена → оригинала s1 нет
    assert _rms_db(y[int(4.2 * SR):int(5.8 * SR)]) > -30          # у s2 дубля нет → его голос остался


def test_voices_outside_lines_return_to_background(world, tmp_path):
    scene, stems, scene_dir = world
    # нейросеть отнесла к голосам крик толпы на 6.5–7.5 с — вне реплик; в сведении он должен звучать
    crowd = voice_track([(6.5, 7.5)], DUR, seed=9, level=1.0, f0=300)
    v, _ = read_wav(stems.voices)
    write_wav(stems.voices, v + crowd, SR)
    scene.mix = MixSettings(keep_breaths=False)
    y, _ = read_wav(mix_scene(scene, stems, tmp_path / "m.wav"))
    bg_only = read_wav(stems.background)[0].mean(axis=1)
    diff = y.mean(axis=1) - bg_only * np.float32(db_to_gain(scene.mix.background.gain_db))
    assert _rms_db(diff[int(6.6 * SR):int(7.4 * SR)]) > -30


def test_voiceover_keeps_originals_quietly(world, tmp_path):
    scene, stems, scene_dir = world
    scene.mix = MixSettings(mode=MixMode.VOICEOVER, original_voices_db=-14.0, keep_breaths=False,
                            background=TrackSettings(mute=True))
    y, _ = read_wav(mix_scene(scene, stems, tmp_path / "m.wav"))
    a = read_wav(stems.per_speaker["s1"])[0]
    got = _rms_db(y[int(1.2 * SR):int(2.8 * SR), 0])
    want = _rms_db(a[int(1.2 * SR):int(2.8 * SR)]) - 14.0
    assert got == pytest.approx(want, abs=0.5)


def test_mute_solo_and_gain(world, tmp_path):
    scene, stems, scene_dir = world
    f = _make_take_file(scene_dir, "t1")
    scene.takes = [_take("t1", "l1", "s1", f, selected=True)]
    base = dict(keep_breaths=False, background=TrackSettings(mute=True))
    scene.mix = MixSettings(tracks={"s1": TrackSettings(room="dry")}, **base)
    y0, _ = read_wav(mix_scene(scene, stems, tmp_path / "a.wav"))
    scene.mix = MixSettings(tracks={"s1": TrackSettings(room="dry", gain_db=-6)}, **base)
    y1, _ = read_wav(mix_scene(scene, stems, tmp_path / "b.wav"))
    w = slice(int(1.2 * SR), int(2.8 * SR))
    assert _rms_db(y1[w]) == pytest.approx(_rms_db(y0[w]) - 6, abs=0.3)
    scene.mix = MixSettings(tracks={"s1": TrackSettings(room="dry", mute=True)}, **base)
    y2, _ = read_wav(mix_scene(scene, stems, tmp_path / "c.wav"))
    assert _rms_db(y2[w]) < -80
    # solo на s1: молчит оригинал s2
    scene.mix = MixSettings(tracks={"s1": TrackSettings(room="dry", solo=True)}, **base)
    y3, _ = read_wav(mix_scene(scene, stems, tmp_path / "d.wav"))
    assert _rms_db(y3[w]) > -40 and _rms_db(y3[int(4.2 * SR):int(5.8 * SR)]) < -80


def test_ducking_lowers_background_under_voice(world, tmp_path):
    scene, stems, scene_dir = world
    f = _make_take_file(scene_dir, "t1")
    scene.takes = [_take("t1", "l1", "s1", f, selected=True)]
    scene.mix = MixSettings(tracks={"s1": TrackSettings(room="dry", mute=True)}, keep_breaths=False,
                            duck_background=True)
    # дубль заглушён → ducking нет; проверяем, что с включённым дублем фон под речью тише
    y_nomute, _ = read_wav(mix_scene(scene, stems, tmp_path / "x.wav"))
    scene.mix = MixSettings(tracks={"s1": TrackSettings(room="dry")}, keep_breaths=False, duck_background=True)
    y_duck, _ = read_wav(mix_scene(scene, stems, tmp_path / "y.wav"))
    scene.mix = MixSettings(tracks={"s1": TrackSettings(room="dry")}, keep_breaths=False, duck_background=False)
    y_flat, _ = read_wav(mix_scene(scene, stems, tmp_path / "z.wav"))
    # между слогами речи дубля — но надёжнее: средний уровень вне диапазона речи одинаков, внутри — ниже на ~DUCK
    # сравниваем разность «со звуком дубля» и «без»: фон с duck тише, чем без
    bg = read_wav(stems.background)[0]
    resid_duck = y_duck - (y_duck - y_flat) * 0   # заглушка читаемости
    seg = slice(int(1.5 * SR), int(2.5 * SR))
    # оценим фон под речью: убираем дубль (y_flat - bg = дубль; y_duck - дубль = фон*duck)
    dub = y_flat - bg
    bg_under = y_duck - dub
    assert _rms_db(bg_under[seg]) < _rms_db(bg[seg]) - 3.0
    quiet = slice(int(5.0 * SR), int(7.0 * SR))
    assert _rms_db(bg_under[quiet]) == pytest.approx(_rms_db(bg[quiet]), abs=0.3)


def test_keep_breaths_adds_quiet_original_in_gaps(tmp_path):
    scene_dir = tmp_path / "sc"
    (scene_dir / "takes").mkdir(parents=True)
    # оригинал s1: громкая речь 1–2 с + шорох 2.2–2.8 с (внутри реплики 1–3); шорох заметно громче шума дубля (−66 дБ)
    rng = np.random.default_rng(0)
    orig = voice_track([(1.0, 2.0)], DUR, seed=1)
    orig[int(2.2 * SR):int(2.8 * SR)] += (0.01 * rng.standard_normal(int(0.6 * SR))).astype(np.float32)
    bg = np.zeros((int(DUR * SR), 2), dtype=np.float32)
    stems = SceneStems(voices=write_wav(scene_dir / "v.wav", orig, SR), background=write_wav(scene_dir / "b.wav", bg, SR),
                       per_speaker={"s1": write_wav(scene_dir / "s1.wav", orig, SR)},
                       takes_dir=scene_dir / "takes", scene_dir=scene_dir)
    f = _make_take_file(scene_dir, "t1", seg=(1.0, 2.0))
    scene = Scene(id="s", title="t", start=0, end=DUR, dubbed_speakers=["s1"], lines=[_line("l1", "s1", 1.0, 3.0)],
                  takes=[_take("t1", "l1", "s1", f, selected=True)])
    scene.mix = MixSettings(tracks={"s1": TrackSettings(room="dry")}, keep_breaths=True)
    y, _ = read_wav(mix_scene(scene, stems, tmp_path / "m.wav"))
    breath = y[int(2.3 * SR):int(2.7 * SR), 0]
    expected = _rms_db(orig[int(2.3 * SR):int(2.7 * SR)]) - 18.0
    assert _rms_db(breath) == pytest.approx(expected, abs=2.0)
    scene.mix = MixSettings(tracks={"s1": TrackSettings(room="dry")}, keep_breaths=False)
    y2, _ = read_wav(mix_scene(scene, stems, tmp_path / "m2.wav"))
    assert _rms_db(breath) > _rms_db(y2[int(2.3 * SR):int(2.7 * SR), 0]) + 6


def test_automix_levels_and_background(world, tmp_path):
    scene, stems, scene_dir = world
    # два озвучиваемых персонажа с очень разной громкостью дублей
    f1 = _make_take_file(scene_dir, "t1", seg=(1.0, 3.0), level=1.0, seed=20)
    f2 = _make_take_file(scene_dir, "t2", seg=(4.0, 6.0), level=0.1, seed=21)
    scene.dubbed_speakers = ["s1", "s2"]
    scene.takes = [_take("t1", "l1", "s1", f1, selected=True), _take("t2", "l2", "s2", f2, selected=True)]
    scene.mix = MixSettings()
    mx = automix(scene, stems)
    assert mx.tracks["s2"].gain_db - mx.tracks["s1"].gain_db > 8.0       # тихий дубль подтянут сильнее
    assert mx.tracks["s1"].room != "auto"
    assert -18 <= mx.background.gain_db <= 6
    scene.mix = mx
    y, _ = read_wav(mix_scene(scene, stems, tmp_path / "m.wav"))
    bg = read_wav(stems.background)[0] * (10 ** (mx.background.gain_db / 20))
    # речь A, речь B — одинаковой громкости (±2 дБ)
    ra = _rms_db((y - bg)[int(1.2 * SR):int(2.8 * SR)])
    rb = _rms_db((y - bg)[int(4.2 * SR):int(5.8 * SR)])
    assert abs(ra - rb) < 2.5
    # речь выше фона в моменты речи на 6–14 дБ (с учётом ducking)
    seg = slice(int(1.2 * SR), int(2.8 * SR))
    dub = y - bg
    margin = _rms_db(dub[seg]) - _rms_db(bg[seg] * 10 ** (-DUCK / 20))
    assert 6.0 <= margin <= 14.0, margin


def test_automix_autopilot_off_returns_same(world):
    scene, stems, _ = world
    scene.mix = MixSettings(autopilot=False, tracks={"s1": TrackSettings(gain_db=3)})
    assert automix(scene, stems) == scene.mix


# ─────────────────────────── шум, перегруз, тихие и пустые дубли ───────────────────────────
def _noisy_take_file(scene_dir, name, *, noise, level, seg=(1.0, 3.0), preroll=1.0, hot=False, seed=30):
    n = int((DUR + preroll) * SR)
    speech = voice_track([(0, seg[1] - seg[0])], seg[1] - seg[0] + 0.1, seed=seed, level=level, f0=150)
    x = np.zeros(n, dtype=np.float32)
    i = int((preroll + seg[0]) * SR)
    x[i:i + len(speech)] = speech
    x += (noise * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)
    if hot:
        x = np.clip(x * 8, -0.98, 0.98).astype(np.float32)
    write_wav(scene_dir / "takes" / f"{name}.wav", x, SR)
    return f"takes/{name}.wav"


def test_take_is_denoised_in_mix(world, tmp_path):
    scene, stems, scene_dir = world
    f = _noisy_take_file(scene_dir, "t1", noise=0.004, level=0.4)
    scene.takes = [_take("t1", "l1", "s1", f, selected=True)]
    base = dict(tracks={"s1": TrackSettings(room="dry")}, keep_breaths=False)
    scene.mix = MixSettings(denoise=False, **base)
    y0, _ = read_wav(mix_scene(scene, stems, tmp_path / "m0.wav"))
    scene.mix = MixSettings(denoise=True, **base)
    y1, _ = read_wav(mix_scene(scene, stems, tmp_path / "m1.wav"))
    bg = read_wav(stems.background)[0]
    pause = slice(int(0.82 * SR), int(0.95 * SR))      # дубль уже начался (запас до речи), но молчит; оригинала s1 тут нет
    assert _rms_db((y1 - bg)[pause]) < _rms_db((y0 - bg)[pause]) - 4.0


def test_clipped_take_gets_repaired_before_mix(world, tmp_path):
    from dubl_engine.audio.levels import clip_report
    scene, stems, scene_dir = world
    f = _noisy_take_file(scene_dir, "t1", noise=0.0005, level=0.5, hot=True)
    assert clip_report(read_wav(scene_dir / f)[0]).pct > 0.5
    scene.takes = [_take("t1", "l1", "s1", f, selected=True)]
    scene.mix = MixSettings(tracks={"s1": TrackSettings(room="dry")}, keep_breaths=False)
    from dubl_engine.audio.mix import _dub_tracks
    rep: dict = {}
    d = _dub_tracks(scene, stems, scene.mix, SR, int(DUR * SR), report=rep)
    assert rep["s1"]["clip_pct"] > 0.5 and np.abs(d["s1"]).max() < 1.0


def test_empty_take_is_not_picked_and_not_mixed(world, tmp_path):
    scene, stems, scene_dir = world
    quiet = _noisy_take_file(scene_dir, "t1", noise=1e-4, level=1e-4)       # пик ≈ −75 дБ: микрофон выключен
    scene.takes = [_take("t1", "l1", "s1", quiet, selected=True)]
    scene.takes[0].peak_db = -75.0
    assert _pick_takes(scene) == []
    scene.takes[0].peak_db = None
    scene.takes[0].empty = True
    assert _pick_takes(scene) == []
    scene.takes[0].empty = False                                              # без замера в Take — по самому файлу
    scene.mix = MixSettings(tracks={"s1": TrackSettings(room="dry")}, keep_breaths=False)
    from dubl_engine.audio.mix import _dub_tracks
    assert _dub_tracks(scene, stems, scene.mix, SR, int(DUR * SR)) == {}


def test_empty_take_falls_back_to_previous_take(world):
    scene, stems, scene_dir = world
    good = _make_take_file(scene_dir, "t1", seg=(1.0, 3.0))
    bad = _noisy_take_file(scene_dir, "t2", noise=1e-4, level=1e-4)
    t1 = _take("t1", "l1", "s1", good, created=datetime(2026, 1, 1), selected=False)
    t2 = _take("t2", "l1", "s1", bad, created=datetime(2026, 1, 2), selected=True)
    t2.empty = True
    scene.takes = [t1, t2]
    assert [t.id for t, _, _ in _pick_takes(scene)] == ["t1"]


def test_automix_resets_gain_when_speaker_has_no_takes(world):
    scene, stems, scene_dir = world
    scene.takes = []
    scene.mix = MixSettings(tracks={"s1": TrackSettings(gain_db=-7.8, boost_capped=True)})
    mx = automix(scene, stems)
    assert mx.tracks["s1"].gain_db == 0.0 and not mx.tracks["s1"].boost_capped
    from dubl_engine.audio.mix import drop_stale_tracks
    fixed = drop_stale_tracks(scene)
    assert fixed.tracks["s1"].gain_db == 0.0


def test_automix_quiet_noisy_take_boost_is_capped_and_flagged(world):
    scene, stems, scene_dir = world
    # речь около −50 дБ, шум ≈ −66 дБ: поднять до уровня оригинала (−20 LUFS) без шума нельзя
    f = _noisy_take_file(scene_dir, "t1", noise=0.0005, level=0.012, seed=31)
    scene.takes = [_take("t1", "l1", "s1", f, selected=True)]
    scene.mix = MixSettings()
    mx = automix(scene, stems)
    ts = mx.tracks["s1"]
    assert ts.boost_capped and 5.9 <= ts.gain_db <= 24.0


def test_automix_normal_take_not_flagged(world):
    scene, stems, scene_dir = world
    f = _make_take_file(scene_dir, "t1", seg=(1.0, 3.0), level=0.3)
    scene.takes = [_take("t1", "l1", "s1", f, selected=True)]
    scene.mix = MixSettings()
    mx = automix(scene, stems)
    assert not mx.tracks["s1"].boost_capped


def test_automix_film_target_uses_original_loudness(world, tmp_path):
    scene, stems, scene_dir = world
    orig = voice_track([(1.0, 3.0), (4.0, 6.0)], DUR, seed=7, level=0.5)
    write_wav(scene_dir / "original.wav", np.stack([orig, orig], axis=1), SR)
    f = _make_take_file(scene_dir, "t1", seg=(1.0, 3.0), level=0.3)
    scene.takes = [_take("t1", "l1", "s1", f, selected=True)]
    scene.mix = MixSettings(loudness_target="film")
    mx = automix(scene, stems)
    from dubl_engine.audio.master import measure
    want = max(measure(np.stack([orig, orig], axis=1), SR).lufs, -24.0)    # тише −24 LUFS «как в фильме» не бывает
    assert mx.target_lufs == pytest.approx(want, abs=0.1)
    scene.mix = MixSettings(loudness_target="online")
    assert automix(scene, stems).target_lufs == -14.0


def test_duck_envelope_does_not_pump_between_words():
    from dubl_engine.audio.mix import _activity
    v = voice_track([(0.5, 2.5)], 4.5, seed=2, level=0.5)            # слоги с провалами 40–80 мс
    a = _activity(v, SR, lookahead=0.08, attack=0.06, release=0.40, hold=0.30)
    inside = a[int(0.9 * SR):int(2.2 * SR)]
    assert inside.min() > 0.9                                          # между слогами фон не отпускает
    assert a[int(0.1 * SR)] < 0.05 and a[-1] < 0.05                    # вне речи — отпущен


def test_blip_at_start_is_blanked_but_speech_kept():
    from dubl_engine.audio.mix import _blank_start_blip
    x = voice_track([(1.4, 3.0)], 4.0, seed=3, level=0.5)
    x[int(0.1 * SR):int(0.2 * SR)] += 0.3 * np.sin(2 * np.pi * 1400 * np.arange(int(0.1 * SR)) / SR).astype(np.float32)
    iv = detect_speech(x, SR)
    assert len(iv) >= 2 and iv[0][1] - iv[0][0] < 0.25
    y = _blank_start_blip(x, SR, iv)
    assert np.abs(y[:int(0.3 * SR)]).max() == 0.0
    assert np.array_equal(y[int(1.3 * SR):], x[int(1.3 * SR):])
