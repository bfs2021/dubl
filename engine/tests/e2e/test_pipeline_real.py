"""Сквозной тест на Tears of Steel: релизный MKV → analyze_project → сцена → prepare_scene → дубли (say) → render 9:16.
Использует быстрые модели (whisper small по центру 5.1, demucs), чтобы идти ≈ 1–2 минуты. Готовый ролик пишет
`samples/out/e2e_test_9x16.mp4` (основной ролик-пример собирает `tests/e2e/bench.py e2e`)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import helpers as H

pytestmark = pytest.mark.slow


class Ctx:
    job_id = "e2e"
    cancelled = False

    def __init__(self):
        self.steps = []

    def step(self, key):
        import contextlib

        @contextlib.contextmanager
        def cm():
            self.steps.append(key)
            yield
        return cm()

    def progress(self, key, value):
        pass

    def skip(self, key):
        self.steps.append("skip:" + key)

    def check(self):
        pass

    def set_result(self, r):
        pass


@pytest.fixture(scope="module")
def project(release_mkv, tmp_path_factory):
    from dubl_engine.media.probe import probe
    from dubl_engine.pipeline.analyze import analyze_project
    from dubl_engine.settings import AppSettings
    from dubl_engine.store import ProjectStore
    store = ProjectStore(tmp_path_factory.mktemp("store"))
    settings = AppSettings(whisper_model="small", separation_quality="fast", processing_device="cpu")
    proj = store.create(release_mkv, probe(release_mkv), "ToS release")
    ctx = Ctx()
    an = analyze_project(store, proj.id, ctx, settings=settings)
    return store, proj.id, settings, an, ctx


def test_analyze_project_on_release_mkv(project):
    store, pid, settings, an, ctx = project
    assert an.language == "en"
    assert an.audio_stream_index == 1                       # 5.1 английская дорожка, не стерео-дубль
    assert len(an.lines) == 76                              # реплики из встроенных субтитров
    assert {l.source for l in an.lines} == {"embedded"}
    assert 100 <= len(an.shot_cuts) <= 180
    assert [c.title for c in an.chapters] == ["Пролог", "Лаборатория", "Финал"]
    assert 2 <= len(an.speakers) <= 10
    assert "transcribe" in ctx.steps and "diarize" in ctx.steps
    adir = store.analysis_dir(pid)
    assert (adir / "center.wav").exists()                   # 5.1 → центральный канал сохранён
    # проект сохранён на диск и читается
    assert store.get(pid).analysis is not None


@pytest.fixture(scope="module")
def scene(project):
    from dubl_engine.models import Scene
    from dubl_engine.pipeline.scene import prepare_scene
    store, pid, settings, an, _ = project
    a, b = H.DEMO_SCENE
    with store.edit(pid) as p:
        scid = store.next_scene_id(p)
        p.scenes.append(Scene(id=scid, title="Память перезаписывается", start=a, end=b))
    ctx = Ctx()
    sc = prepare_scene(store, pid, scid, ctx, settings=settings)
    return store, pid, scid, sc, ctx


def test_prepare_scene_makes_stems(scene):
    store, pid, scid, sc, ctx = scene
    assert sc.stems_ready and ctx.steps == ["cut", "audio", "separate", "speakers"]
    sd = store.scene_dir(pid, scid)
    for f in ("clip.mp4", "original.wav", "voices.wav", "background.wav"):
        assert (sd / f).exists(), f
    assert len(sc.lines) >= 6
    assert list((sd / "voices").glob("*.wav"))              # голос по персонажам
    from dubl_engine.audio.io import read_wav
    o, sr = read_wav(sd / "original.wav")
    v, _ = read_wav(sd / "voices.wav")
    b, _ = read_wav(sd / "background.wav")
    assert abs(len(v) - len(o)) <= 2 and abs(len(b) - len(o)) <= 2
    assert abs(len(o) / sr - (sc.end - sc.start)) < 0.1


def test_mix_without_takes_has_no_english_speech(scene, tmp_path):
    """Регрессия «английская речь под дубляжом» (исправлено в fac0566): сцена из 5.1, где диалог лежит и в центре, и в L/R.
    Все реплики всех персонажей заменены (дубли-тишина), дыхание выключено → в миксе должен остаться один фон.
    whisper (small и large-v3-turbo) по результату не должен слышать реплик оригинала (замер после правки: 0 слов).
    Заодно проверяет, что звуки, которые возвращаются в фон «вне реплик», не тащат за собой речь."""
    import scipy.signal as sg

    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from dubl_engine.analysis.transcribe import transcribe
    from dubl_engine.audio.io import read_wav, write_wav
    from dubl_engine.audio.mix import mix_scene
    from dubl_engine.models import Take
    from dubl_engine.pipeline.scene import stems_for
    store, pid, scid, sc, _ = scene
    spk = sorted({l.speaker_id for l in sc.lines if l.speaker_id})
    assert spk
    sd = store.scene_dir(pid, scid)
    (sd / "takes").mkdir(exist_ok=True)
    write_wav(sd / "takes" / "silent.wav", np.zeros(int(2 * 48000), dtype=np.float32), 48000)
    takes = [Take(id=f"z{i:03d}", line_id=l.line_id, speaker_id=l.speaker_id, file="takes/silent.wav", duration=2.0,
                  preroll=0.0, start_rel=l.start_rel, selected=True) for i, l in enumerate(sc.lines)]
    sc2 = sc.model_copy(update={"takes": takes, "dubbed_speakers": spk,
                                "mix": sc.mix.model_copy(update={"keep_breaths": False, "autopilot": False})})
    mix_scene(sc2, stems_for(store, pid, sc2), tmp_path / "mix.wav")
    x, sr = read_wav(tmp_path / "mix.wav")
    assert len(x) > 0 and np.abs(x).max() > 1e-4            # фон на месте, не тишина
    p = write_wav(tmp_path / "mix16.wav", sg.resample_poly(x.mean(axis=1), 1, 3).astype(np.float32), 16000)
    ref_words = {w for l in parse_subtitle_file(H.TOS_EN) if l.end > sc.start and l.start < sc.end
                 for w in H.norm_words(H.clean_ref(l.text)) if len(w) > 3}
    for model in ("small", "large-v3-turbo"):
        lines, _ = transcribe(p, language="en", model=model)
        words = [w for l in lines for w in H.norm_words(l.text)]
        assert not (set(words) & ref_words), (model, words)
        assert len(words) <= 3, (model, words)


def test_dub_scene_end_to_end(scene, tmp_path):
    """say-дубли с разной задержкой → add_take (подгонка) → render_scene 9:16 с вшитыми субтитрами."""
    if sys.platform != "darwin":
        pytest.skip("say есть только на macOS")
    from dubl_engine.audio.io import read_wav
    from dubl_engine.audio.master import measure
    from dubl_engine.media.probe import probe
    from dubl_engine.models import RenderAspect, RenderSettings, SubtitleMode
    from dubl_engine.pipeline.scene import add_take, render_scene
    store, pid, scid, sc, _ = scene
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    ru = parse_subtitle_file(H.TOS_RU)
    new_lines = []
    for ln in sc.lines:
        k = int(ln.line_id[1:]) - 1
        new_lines.append(ln.model_copy(update={"text": ru[k].text}))
    with store.edit(pid) as p:
        s = p.scenes[store.scene_index(p, scid)]
        s.lines = new_lines
        s.dubbed_speakers = sorted({l.speaker_id for l in new_lines if l.speaker_id})
    sc = store.find_scene(store.get(pid), scid)
    takes = 0
    for k, ln in enumerate(sc.lines):
        start_rel = max(0.0, ln.start_rel - 1.0)
        wav = H.say_take(ln.text, tmp_path / f"t{k}.wav", lead_silence=0.5 + (ln.start_rel - start_rel) + 0.25)
        t = add_take(store, pid, scid, wav.read_bytes(), ln.line_id, start_rel, 0.5, None)
        assert 0.85 <= t.stretch <= 1.15 and t.fit_confidence is not None
        assert t.loudness_lufs is not None and t.peak_db is not None
        takes += 1
    assert takes == len(sc.lines)
    out = H.OUT / "e2e_test_9x16.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    rs = RenderSettings(aspect=RenderAspect.V9_16, height=1080, subtitles=SubtitleMode.BURN, output_path=str(out))
    render_scene(store, pid, scid, rs, Ctx())
    assert out.exists()                      # абсолютный output_path render_scene пишет прямо туда
    info = probe(out)
    assert (info.width, info.height) == (1080, 1920)   # height = короткая сторона: 9:16 «1080p» = 1080×1920
    assert abs(info.duration - (H.DEMO_SCENE[1] - H.DEMO_SCENE[0])) < 0.3
    assert info.audio_streams and info.audio_streams[0].channels == 2
    from dubl_engine.util import ffprobe_json
    st = {x["codec_type"]: x for x in ffprobe_json(out, "-show_streams")["streams"]}
    end = {k: float(v.get("start_time") or 0) + float(v["duration"]) for k, v in st.items() if k in ("video", "audio")}
    assert abs(end["audio"] - end["video"]) <= 0.025, end         # звук и картинка кончаются вместе
    H.ff("-i", out, "-vn", "-ac", 2, "-ar", 48000, "-c:a", "pcm_f32le", tmp_path / "out.wav")
    x, sr = read_wav(tmp_path / "out.wav")
    ld = measure(x, sr)
    assert abs(ld.lufs - (-14.0)) < 1.0                     # замер: −14.15 LUFS
    assert ld.true_peak_db <= -0.1                          # замер: −0.43 dBTP (цель −1.0)
    sc = store.find_scene(store.get(pid), scid)
    # автопилот: дубли подтягиваются к громкости оригинала (диапазон −30…+24), фон не громче фильма
    assert sc.mix.tracks and all(-30 <= t.gain_db <= 24 for t in sc.mix.tracks.values())
    assert sc.mix.background.gain_db <= 0.0
