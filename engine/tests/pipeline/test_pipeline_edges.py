"""Граничные случаи пайплайнов: пустая речь, выбор дорожки, отмена, огрызки реплик, большие блоки."""
from __future__ import annotations

import numpy as np
import pytest

from dubl_engine.audio.io import write_wav
from dubl_engine.audio.separate import subtract_voices
from dubl_engine.jobs import JobCancelled, NullContext
from dubl_engine.media.probe import probe
from dubl_engine.models import Analysis, Line, Scene
from dubl_engine.pipeline import analyze_project
from dubl_engine.pipeline.scene import scene_lines_from_analysis
from dubl_engine.settings import AppSettings
from dubl_engine.store import ProjectStore

SR = 48000


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "projects")


@pytest.fixture
def no_speech(monkeypatch):
    monkeypatch.setattr("dubl_engine.analysis.transcribe.transcribe", lambda *a, **k: ([], "en"))

    def fake_diarize(wav, lines, **_kw):          # без нейросети: все реплики — один говорящий
        from dubl_engine.analysis.diarize import _build_result
        return _build_result(sorted(lines, key=lambda x: x.start), [0] * len(lines), None)
    monkeypatch.setattr("dubl_engine.analysis.diarize.diarize", fake_diarize)


def test_analyze_file_without_speech_and_subs(store, simple_mp4, no_speech):
    p = store.create(simple_mp4, probe(simple_mp4), title="тишина")
    an = analyze_project(store, p.id, NullContext(), settings=AppSettings())
    assert an.lines == [] and an.speakers == [] and an.language == "en"
    assert (store.thumbnail_path(p.id)).is_file()
    assert store.list()[0].thumbnail == "thumbnail.jpg"
    assert (store.analysis_dir(p.id) / "audio_16k.wav").is_file()


def test_analyze_respects_chosen_audio_stream(store, rich_mkv, no_speech):
    p = store.create(rich_mkv, probe(rich_mkv), title="t")
    an = analyze_project(store, p.id, NullContext(), settings=AppSettings(), audio_stream_index=2, language="ru")
    assert an.audio_stream_index == 2 and an.language == "en"          # язык берёт у whisper-заглушки
    assert not (store.analysis_dir(p.id) / "center.wav").exists()      # у стерео-дорожки центра нет
    # субтитры выбраны по языку ru: русский SRT, а не английский ASS с именами
    assert [ln.text for ln in an.lines][:1] == ["Кто трогал мою колбасу?"]


def test_analyze_bad_stream_is_clear_error(store, rich_mkv, no_speech):
    from dubl_engine.util import EngineError
    p = store.create(rich_mkv, probe(rich_mkv), title="t")
    with pytest.raises(EngineError, match="нет выбранной"):
        analyze_project(store, p.id, NullContext(), settings=AppSettings(), audio_stream_index=77)


def test_analyze_long_audio_is_16bit_on_disk(store, simple_mp4, no_speech):
    import soundfile as sf
    p = store.create(simple_mp4, probe(simple_mp4), title="t")
    analyze_project(store, p.id, NullContext(), settings=AppSettings())
    assert sf.info(str(store.analysis_dir(p.id) / "audio_16k.wav")).subtype == "PCM_16"


def test_cancel_stops_analysis_between_steps(store, simple_mp4, no_speech):
    class Ctx(NullContext):
        def step(self, key):
            if key == "transcribe":
                raise JobCancelled()
            return super().step(key)
    p = store.create(simple_mp4, probe(simple_mp4), title="t")
    with pytest.raises(JobCancelled):
        analyze_project(store, p.id, Ctx(), settings=AppSettings())
    assert store.get(p.id).analysis is None


def test_scene_lines_drop_slivers_and_clip_to_scene():
    an = Analysis(audio_stream_index=1, lines=[
        Line(id="l1", start=0.0, end=2.05, text="огрызок в начале"),      # в сцену попало 0.05 с
        Line(id="l2", start=2.0, end=6.0, text="целая"),
        Line(id="l3", start=7.8, end=12.0, text="обрезана справа"),
        Line(id="l4", start=20, end=21, text="вне")])
    from dubl_engine.models import MediaInfo, Project
    pr = Project(id="p", title="t", source_path="x", analysis=an,
                 media=MediaInfo(path="x", duration=30, container="m", width=2, height=2, fps=1, video_codec="h"))
    sc = Scene(id="sc001", title="s", start=2.0, end=8.0)
    got = scene_lines_from_analysis(pr, sc)
    assert [ln.line_id for ln in got] == ["l2", "l3"]
    assert got[1].start_rel == 5.8 and got[1].end_rel == 6.0


def test_scene_lines_keep_user_edits():
    from dubl_engine.models import MediaInfo, Project, SceneLine
    an = Analysis(audio_stream_index=1, lines=[Line(id="l2", start=2.0, end=6.0, text="Hello")])
    pr = Project(id="p", title="t", source_path="x", analysis=an,
                 media=MediaInfo(path="x", duration=30, container="m", width=2, height=2, fps=1, video_codec="h"))
    sc = Scene(id="sc001", title="s", start=0, end=8, lines=[
        SceneLine(line_id="l2", speaker_id=None, start_rel=2, end_rel=6, original="Hello", text="Привет")])
    assert scene_lines_from_analysis(pr, sc)[0].text == "Привет"


def test_subtract_voices_blocks_match_single_pass():
    rng = np.random.default_rng(0)
    n = SR * 7
    voice = (0.2 * np.sin(2 * np.pi * 300 * np.arange(n) / SR) * (np.arange(n) % SR < SR // 2)).astype(np.float32)
    music = (0.1 * rng.standard_normal((n, 2))).astype(np.float32)
    mix = music + voice[:, None] * 0.7
    whole = subtract_voices(mix, voice, block_s=100.0)
    parts = subtract_voices(mix, voice, block_s=1.0)
    assert whole.shape == parts.shape == (n, 2)
    err = np.sqrt(np.mean((whole - parts) ** 2)) / np.sqrt(np.mean(whole ** 2))
    assert err < 0.05
    # голос действительно вычтен
    seg = slice(0, SR // 2)
    assert np.sqrt(np.mean(parts[seg] ** 2)) < np.sqrt(np.mean(mix[seg] ** 2)) * 0.8


def test_lazy_audio_reads_slices_without_loading_all(tmp_path):
    from dubl_engine.analysis.diarize import _LazyAudio
    x = np.arange(16000 * 4, dtype=np.float32) / 1e5
    write_wav(tmp_path / "a.wav", x, 16000)
    la = _LazyAudio(tmp_path / "a.wav")
    assert len(la) == len(x)
    np.testing.assert_allclose(la[16000:16100], x[16000:16100])
    assert len(la[len(x) + 5:len(x) + 50]) == 0 and len(la[-5:]) == 5
