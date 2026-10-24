"""Общий прогресс задачи: взвешенный, монотонный, с оценкой «осталось»; события «done» не дублируются."""
from __future__ import annotations

import asyncio
import time

import pytest

from dubl_engine.jobs import JobManager
from dubl_engine.jobs import manager as mgr
from dubl_engine.models import Job, JobState, JobStep
from dubl_engine.pipeline import ANALYZE_STEPS


def _job(kind, steps):
    return Job(id="j", kind=kind, project_id="p", steps=[JobStep(key=k, title=k, progress=pr, state=st) for k, pr, st in steps])


def test_weights_heavy_steps_dominate():
    w = {k: mgr.step_weight("analyze", k) for k, _ in ANALYZE_STEPS}
    assert w["transcribe"] > w["diarize"] > w["audio"] > w["probe"]
    assert w["transcribe"] + w["diarize"] > 0.5 * sum(w.values())
    assert mgr.step_weight("prepare_scene", "separate") > mgr.step_weight("prepare_scene", "cut")


def test_overall_progress_is_weighted():
    done, run, q = JobState.DONE, JobState.RUNNING, JobState.QUEUED
    j = _job("analyze", [("probe", 1, done), ("audio", 1, done), ("transcribe", 0.5, run), ("diarize", 0, q)])
    total = sum(mgr.step_weight("analyze", k) for k in ("probe", "audio", "transcribe", "diarize"))
    want = (mgr.step_weight("analyze", "probe") + mgr.step_weight("analyze", "audio") +
            0.5 * mgr.step_weight("analyze", "transcribe")) / total
    assert mgr.overall_progress(j) == pytest.approx(want)
    assert 0 < mgr.overall_progress(j) < 1


@pytest.mark.asyncio
async def test_progress_monotonic_with_per_step_resets(monkeypatch):
    """Шаги идут 0→1 каждый, а общий прогресс только растёт (в отчёте было 71% → 52% → 0%)."""
    monkeypatch.setattr(mgr, "MIN_PROGRESS_INTERVAL", 0.0)
    jm = JobManager()
    jm.bind_loop()
    q = jm.subscribe()

    def fn(ctx):
        for key, _ in ANALYZE_STEPS:
            with ctx.step(key):
                for v in (0.0, 0.3, 0.71, 0.5, 1.0):         # даже откат внутри шага не роняет общий прогресс
                    ctx.progress(key, v)
    job = jm.submit("analyze", "p", list(ANALYZE_STEPS), fn)
    done = await jm.wait(job.id, timeout=10)
    await asyncio.sleep(0.05)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert done.state == JobState.DONE and done.progress == 1.0 and done.eta_s is None
    seq = [e.progress for e in events]
    assert len(seq) > 10 and seq == sorted(seq), seq
    assert all(0.0 <= p <= 1.0 for p in seq) and seq[0] == 0.0 and seq[-1] == 1.0
    # шаги при этом действительно «скачут» — общий прогресс этого не повторяет
    steps_seq = [e.steps[3].progress for e in events]
    assert steps_seq != sorted(steps_seq)
    jm.shutdown()


@pytest.mark.asyncio
async def test_eta_none_at_start_then_estimated(monkeypatch):
    monkeypatch.setattr(mgr, "MIN_PROGRESS_INTERVAL", 0.0)
    monkeypatch.setattr(mgr, "ETA_MIN_ELAPSED", 0.3)
    jm = JobManager()
    jm.bind_loop()
    q = jm.subscribe()

    def fn(ctx):
        with ctx.step("a"):
            ctx.progress("a", 0.1)
            time.sleep(0.6)                 # прошло больше ETA_MIN_ELAPSED при прогрессе 0.1/2 шага
            ctx.progress("a", 0.5)
        with ctx.step("b"):
            ctx.progress("b", 0.5)
    job = jm.submit("render", "p", [("a", "A"), ("b", "B")], fn)
    done = await jm.wait(job.id, timeout=10)
    await asyncio.sleep(0.05)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert events[0].eta_s is None                                    # первые секунды — без оценки
    etas = [e.eta_s for e in events if e.eta_s is not None]
    assert etas and all(x >= 0 for x in etas)
    mid = next(e for e in events if e.eta_s is not None)
    assert mid.eta_s == pytest.approx(0.6 * (1 - mid.progress) / mid.progress, rel=0.6)
    assert done.eta_s is None
    jm.shutdown()


@pytest.mark.asyncio
async def test_failed_and_cancelled_jobs_have_no_eta_and_keep_progress(monkeypatch):
    jm = JobManager()
    jm.bind_loop()

    def boom(ctx):
        with ctx.step("a"):
            ctx.progress("a", 0.5)
        raise RuntimeError("упало")
    job = jm.submit("render", "p", [("a", "A"), ("b", "B")], boom)
    done = await jm.wait(job.id, timeout=10)
    assert done.state == JobState.FAILED and done.eta_s is None and 0 < done.progress < 1
    jm.shutdown()


@pytest.mark.asyncio
async def test_each_job_publishes_exactly_one_terminal_event(monkeypatch):
    """Движок шлёт событие «done» ровно один раз (WS/опрос). Дубли тостов «Рендер готов» — на стороне фронта."""
    monkeypatch.setattr(mgr, "MIN_PROGRESS_INTERVAL", 0.0)
    jm = JobManager()
    jm.bind_loop()
    q = jm.subscribe()

    def fn(ctx):
        with ctx.step("a"):
            ctx.progress("a", 0.5)
    jobs = [jm.submit("render", "p", [("a", "A")], fn) for _ in range(3)]
    for j in jobs:
        await jm.wait(j.id, timeout=10)
    await asyncio.sleep(0.05)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    for j in jobs:
        terminal = [e for e in events if e.id == j.id and e.state == JobState.DONE]
        assert len(terminal) == 1
        assert events[-1 - [e.id for e in reversed(events)].index(j.id)].state == JobState.DONE   # и он последний
    jm.shutdown()
