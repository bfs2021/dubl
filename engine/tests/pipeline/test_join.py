"""Склейка нескольких сцен в один ролик и умолчание/откат разделения."""
from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dubl_engine.api import create_app
from dubl_engine.audio.io import read_wav
from dubl_engine.audio.master import measure
from dubl_engine.audio.render import concat_videos, fit_length, join_audio, video_duration
from dubl_engine.jobs import JobManager, NullContext
from dubl_engine.models import RenderSettings, SceneLine, SubtitleMode
from dubl_engine.pipeline.join import render_project, validate_scene_ids
from dubl_engine.pipeline.scene import BEST_MAX_SCENE_S, best_memory_gb, effective_quality
from dubl_engine.settings import AppSettings
from dubl_engine.util import ConflictError, EngineError, ffprobe_json, run_ffmpeg

SR = 48000


# ───────────── умолчания ─────────────
def test_separation_default_is_fast():
    assert AppSettings().separation_quality == "fast"


def test_best_falls_back_to_fast_on_long_scene_with_explanation():
    assert effective_quality("best", 30.0) == ("best", None)
    assert effective_quality("best", BEST_MAX_SCENE_S) == ("best", None)
    q, why = effective_quality("best", 120.0)
    assert q == "fast" and why
    assert "ГБ" in why and "10 МБ" in why and "8 ГБ" in why and "быстрое" in why
    assert effective_quality("fast", 600.0) == ("fast", None)
    assert best_memory_gb(60) == pytest.approx(8.0 + 60 * 10 / 1024)


def test_prepare_scene_records_fallback_in_job_steps(tmp_path, simple_mp4, monkeypatch):
    """Длинная сцена + best → separate получает 'fast', а в шаге задачи остаётся пояснение."""
    from dubl_engine.media.probe import probe
    from dubl_engine.models import Scene
    from dubl_engine.pipeline import PREPARE_STEPS, prepare_scene
    from dubl_engine.store import ProjectStore

    store = ProjectStore(tmp_path / "projects")
    p = store.create(simple_mp4, probe(simple_mp4), title="t")
    with store.edit(p.id) as pr:
        pr.scenes.append(Scene(id="sc001", title="s", start=0.0, end=9.0))
    seen = {}

    def fake_sep(wav, out_dir, *, center_wav=None, quality="fast", **_kw):
        from dubl_engine.audio.io import read_wav, write_wav
        seen["quality"] = quality
        x, sr = read_wav(wav)
        write_wav(out_dir / "voices.wav", x.mean(axis=1), sr)
        write_wav(out_dir / "background.wav", x, sr)
        return out_dir / "voices.wav", out_dir / "background.wav"

    monkeypatch.setattr("dubl_engine.audio.separate.separate", fake_sep)
    monkeypatch.setattr("dubl_engine.pipeline.scene.BEST_MAX_SCENE_S", 5.0)       # «длинная» = дольше 5 с
    jobs = JobManager()
    import asyncio

    async def go():
        jobs.bind_loop()
        job = jobs.submit("prepare_scene", p.id, PREPARE_STEPS,
                          lambda ctx: prepare_scene(store, p.id, "sc001", ctx,
                                                    settings=AppSettings(separation_quality="best")).model_dump(mode="json"),
                          scene_id="sc001")
        return await jobs.wait(job.id, 60)

    done = asyncio.run(go())
    assert done.state.value == "done", done.error
    assert seen["quality"] == "fast"
    step = next(s for s in done.steps if s.key == "separate")
    assert step.note and "быстрое" in step.note


# ───────────── кирпичики склейки ─────────────
def test_join_audio_crossfades_seams_keeps_length():
    a = np.ones((SR, 2), dtype=np.float32)
    b = np.ones((SR, 2), dtype=np.float32)
    c = np.ones((SR, 2), dtype=np.float32)
    y = join_audio([a, b, c], SR, seam_ms=80)
    assert y.shape == (3 * SR, 2)
    assert y[SR // 2, 0] == pytest.approx(1.0)                      # середина куска нетронута
    assert y[SR - 1, 0] < 0.01 and y[SR, 0] < 0.01                  # на стыке тишина-в-ноль с двух сторон (без щелчка)
    seam = y[SR - 1920: SR + 1920, 0]
    assert np.abs(np.diff(seam)).max() < 0.01                       # плавно
    assert y[0, 0] == 1.0 and y[-1, 0] == 1.0                       # края ролика не затронуты
    assert join_audio([a], SR).shape == a.shape and np.array_equal(join_audio([a], SR), a)


def test_fit_length_pads_and_trims():
    x = np.ones((10, 2), dtype=np.float32)
    assert fit_length(x, 15).shape == (15, 2) and fit_length(x, 15)[12, 0] == 0
    assert fit_length(x, 4).shape == (4, 2)


def test_concat_videos_hard_cut(simple_mp4, tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b with ' quote.mp4"
    run_ffmpeg(["-i", simple_mp4, "-t", "2", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", a])
    run_ffmpeg(["-i", simple_mp4, "-ss", "3", "-t", "3", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", b])
    out = concat_videos([a, b], tmp_path / "o" / "j.mp4")
    assert video_duration(out) == pytest.approx(video_duration(a) + video_duration(b), abs=0.15)


# ───────────── render_project ─────────────
def _rs(tmp_path, **kw):
    return RenderSettings(height=None, target_lufs=-14.0, output_path=str(tmp_path / "ролик.mp4"), **kw)


def test_render_project_joins_scenes_in_given_order(ready_project, tmp_path):
    store, pid, ids = ready_project
    steps = []

    class Ctx(NullContext):
        def progress(self, key, value):
            steps.append(key)

    out = render_project(store, pid, [ids[1], ids[0]], _rs(tmp_path), Ctx())
    assert out == tmp_path / "ролик.mp4" and out.is_file()
    info = ffprobe_json(out, "-show_streams", "-show_format", "-show_chapters")
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert v["codec_name"] == "h264" and a["codec_name"] == "aac"
    assert (v["width"], v["height"]) == (640, 360)
    assert float(info["format"]["duration"]) == pytest.approx(8.0, abs=0.3)
    assert info["chapters"] == []
    assert f"scene:{ids[0]}" in steps and f"scene:{ids[1]}" in steps
    # итог отмастерен целиком
    wav = tmp_path / "ролик.wav"
    run_ffmpeg(["-i", out, "-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", wav])
    y, sr = read_wav(wav)
    m = measure(y, sr)
    assert abs(m.lufs + 14.0) <= 1.0 and m.true_peak_db <= -0.7
    # порядок: [sc002, sc001] → сначала «речь» 1000 Гц (сцена 2), потом 600 Гц (сцена 1)
    from scipy.signal import stft
    f, t, z = stft(y.mean(axis=1), fs=sr, nperseg=4096)
    band = lambda hz: np.abs(z[(f > hz - 40) & (f < hz + 40)]).sum(axis=0)
    first, second = t < 3.5, t > 4.5
    assert band(1000)[first].sum() > 5 * band(600)[first].sum()
    assert band(600)[second].sum() > 5 * band(1000)[second].sum()


def test_render_project_single_scene_and_srt_file(ready_project, tmp_path):
    store, pid, ids = ready_project
    with store.edit(pid) as p:
        for i, sc in enumerate(p.scenes):
            sc.lines = [SceneLine(line_id=f"l{i}", speaker_id=None, start_rel=1.0, end_rel=2.0, original="x", text=f"реплика {i}")]
    out = render_project(store, pid, ids, _rs(tmp_path, subtitles=SubtitleMode.FILE), NullContext())
    srt = out.with_suffix(".srt").read_text(encoding="utf-8")
    assert "реплика 0" in srt and "реплика 1" in srt
    # вторая реплика сдвинута на длину первой сцены (4 с): 4+1 … 4+2
    assert "00:00:05,000 --> 00:00:06,000" in srt
    one = render_project(store, pid, [ids[0]], RenderSettings(height=None, output_path=str(tmp_path / "one.mp4")), NullContext())
    assert float(ffprobe_json(one, "-show_format")["format"]["duration"]) == pytest.approx(4.0, abs=0.3)


def test_validate_scene_ids(ready_project):
    store, pid, ids = ready_project
    with pytest.raises(EngineError, match="хотя бы одну"):
        validate_scene_ids(store, pid, [])
    with pytest.raises(EngineError, match="дважды"):
        validate_scene_ids(store, pid, [ids[0], ids[0]])
    from dubl_engine.store import NotFoundError
    with pytest.raises(NotFoundError):
        validate_scene_ids(store, pid, ["sc999"])
    with store.edit(pid) as p:
        p.scenes[1].stems_ready = False
    with pytest.raises(ConflictError):
        validate_scene_ids(store, pid, ids)


def _wait(c, jid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/jobs/{jid}").json()
        if j["state"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("задача не закончилась")


def test_render_project_route(ready_project, tmp_path):
    store, pid, ids = ready_project
    with TestClient(create_app(store=store, jobs=JobManager())) as c:
        body = {"scene_ids": ids, "settings": {"height": 360, "output_path": "итог.mp4"}}
        r = c.post(f"/projects/{pid}/render", json=body)
        assert r.status_code == 200, r.text
        job = r.json()
        assert job["kind"] == "render_project" and job["project_id"] == pid
        assert [s["key"] for s in job["steps"]] == [f"scene:{ids[0]}", f"scene:{ids[1]}", "join", "master", "video"]
        done = _wait(c, job["id"])
        assert done["state"] == "done", done
        assert done["result"]["path"].endswith("итог.mp4")
        assert done["result"]["url"] == f"/files/{pid}/renders/итог.mp4"
        assert c.get(done["result"]["url"]).status_code == 200
        # ошибки сразу, без задачи
        assert c.post(f"/projects/{pid}/render", json={"scene_ids": [], "settings": {"output_path": "a.mp4"}}).status_code == 422
        assert c.post(f"/projects/{pid}/render", json={"scene_ids": [ids[0], ids[0]], "settings": {"output_path": "a.mp4"}}).status_code == 400
        assert c.post(f"/projects/{pid}/render", json={"scene_ids": ["sc404"], "settings": {"output_path": "a.mp4"}}).status_code == 404
        assert c.post(f"/projects/{pid}/render", json={"scene_ids": ids, "settings": {"output_path": ""}}).status_code == 400
        # старый роут одной сцены жив и отдаёт тот же формат результата
        r = c.post(f"/projects/{pid}/scenes/{ids[0]}/render", json={"height": 360, "output_path": "одна.mp4"})
        assert r.status_code == 200
        d = _wait(c, r.json()["id"])
        assert d["state"] == "done" and d["result"]["url"] == f"/files/{pid}/renders/одна.mp4"
