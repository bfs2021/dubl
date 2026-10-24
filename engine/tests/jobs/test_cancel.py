"""Отмена задач: посреди ffmpeg, из `except Exception`, в очереди; секреты в ошибках."""
from __future__ import annotations

import asyncio
import time

import pytest

from dubl_engine.jobs import JobCancelled, JobManager
from dubl_engine.models import JobState
from dubl_engine.util import EngineError, redact, run_ffmpeg


def test_cancel_is_not_swallowed_by_except_exception():
    assert not issubclass(JobCancelled, Exception)


@pytest.mark.asyncio
async def test_cancel_kills_running_ffmpeg_quickly(tmp_path):
    jm = JobManager()
    jm.bind_loop()
    started = {}

    def fn(ctx):
        with ctx.step("long"):
            started["t"] = time.monotonic()
            # без отмены это работало бы минуту
            run_ffmpeg(["-re", "-f", "lavfi", "-i", "sine=f=440:d=60", "-af", "aresample=48000", tmp_path / "x.wav"])

    job = jm.submit("render", "p", [("long", "Долго")], fn)
    await asyncio.sleep(0.8)
    jm.cancel(job.id)
    done = await jm.wait(job.id, timeout=10)
    assert done.state == JobState.CANCELLED
    assert time.monotonic() - started["t"] < 8
    jm.shutdown()


@pytest.mark.asyncio
async def test_cancel_inside_broad_except_still_cancels():
    jm = JobManager()
    jm.bind_loop()

    def fn(ctx):
        for i in range(200):
            try:
                ctx.progress("s", i / 200)
            except Exception:            # noqa: BLE001 — как в нейросетевых модулях
                pass
            time.sleep(0.02)

    job = jm.submit("analyze", "p", [("s", "Шаг")], fn)
    await asyncio.sleep(0.3)
    jm.cancel(job.id)
    assert (await jm.wait(job.id, timeout=5)).state == JobState.CANCELLED
    jm.shutdown()


@pytest.mark.asyncio
async def test_failed_job_error_hides_secrets():
    jm = JobManager()
    jm.bind_loop()

    def fn(ctx):
        raise RuntimeError("401 for https://x/y?api_key=sk-ABCDEF1234567890XYZ and token=hf_abcdefghijklmnop")

    job = jm.submit("render", "p", [("s", "Шаг")], fn)
    done = await jm.wait(job.id, timeout=5)
    assert done.state == JobState.FAILED
    assert "sk-ABCDEF" not in done.error and "hf_abcdef" not in done.error
    jm.shutdown()


def test_redact():
    assert "SECRETVALUE123" not in redact("Authorization: Bearer SECRETVALUE123")
    assert "AIzaSyA1234567890123456789012345" not in redact("url AIzaSyA1234567890123456789012345 failed")
    assert redact("обычный текст про ключ и токен") == "обычный текст про ключ и токен"


def test_run_ffmpeg_timeout(tmp_path):
    with pytest.raises(EngineError, match="слишком много времени"):
        run_ffmpeg(["-re", "-f", "lavfi", "-i", "sine=f=440:d=60", tmp_path / "x.wav"], timeout=0.8)
