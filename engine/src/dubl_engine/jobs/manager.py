"""Очередь фоновых задач с прогрессом.

* Тяжёлые задачи (разбор, разделение голоса — нейросети) идут по одной: пул из 1 потока, чтобы не съесть память.
* Лёгкие (подгонка дубля, превью, рендер) — отдельный пул из 2 потоков.
* Задача — функция `fn(ctx: JobContext)`, которая работает в потоке. Шаги: `with ctx.step("key"): ...`,
  прогресс внутри шага: `ctx.progress("key", 0.4)`.
* Отмена кооперативная: при входе в шаг и при каждом `ctx.progress()` проверяется флаг и бросается JobCancelled.
* Подписчики (asyncio.Queue) получают снимок Job при каждом изменении; прогресс — не чаще 10 раз/с на задачу.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

from ..models import Job, JobResult, JobState, JobStep
from ..util import EngineError, cancel_check, redact

log = logging.getLogger(__name__)

MAX_JOBS = 200
ETA_MIN_ELAPSED = 3.0                # раньше этого времени от начала задачи оценку «осталось» не даём
ETA_MIN_PROGRESS = 0.02

# Веса шагов для общего прогресса: условные «единицы» по типичной длительности (тяжёлые нейросетевые шаги — больше).
# Ключ — (вид задачи, ключ шага); запасной вариант — по ключу шага, затем DEFAULT_WEIGHT.
STEP_WEIGHTS: dict[str, float] = {
    "download": 12, "probe": 1, "audio": 4, "subtitles": 1, "transcribe": 40, "diarize": 18, "shots": 8,   # analyze
    "cut": 3, "separate": 30, "speakers": 10,                                                            # prepare_scene
    "mix": 3, "master": 3, "video": 6, "proxy": 1, "join": 3,                                            # render / preview
}
KIND_WEIGHTS: dict[tuple[str, str], float] = {("prepare_scene", "audio"): 2, ("render_project", "video"): 4,
                                              ("render_project", "master"): 3}
SCENE_STEP_WEIGHT = 12               # шаг «scene:<id>» склейки сцен — это сведение, мастеринг и видео целой сцены
DEFAULT_WEIGHT = 2.0


def step_weight(kind: str, key: str) -> float:
    if (kind, key) in KIND_WEIGHTS:
        return KIND_WEIGHTS[(kind, key)]
    if key.startswith("scene:"):
        return SCENE_STEP_WEIGHT
    return STEP_WEIGHTS.get(key, DEFAULT_WEIGHT)


def overall_progress(job: Job) -> float:
    """Взвешенная доля выполненного по шагам, 0..1 (без защиты от отката — её даёт JobManager)."""
    total = sum(step_weight(job.kind, s.key) for s in job.steps)
    if total <= 0:
        return 0.0
    done = sum(step_weight(job.kind, s.key) * (1.0 if s.state == JobState.DONE else min(max(s.progress, 0.0), 1.0))
               for s in job.steps)
    return min(done / total, 1.0)
MIN_PROGRESS_INTERVAL = 0.1          # ≤ 10 событий прогресса в секунду на задачу
HEAVY_KINDS = frozenset({"analyze", "prepare_scene", "separate", "transcribe"})
FINISHED = (JobState.DONE, JobState.FAILED, JobState.CANCELLED)


class JobCancelled(BaseException):
    """Задача отменена пользователем (бросается из ctx.step / ctx.progress).

    Наследуется от BaseException намеренно: в модулях полно `except Exception` («что-то пошло не так →
    EngineError»), и они иначе превращали бы отмену в ошибку или запускали повтор на другом устройстве.
    """


CancelledError = JobCancelled   # имя из ТЗ


StepSpec = "tuple[str, str] | JobStep"


class JobContext:
    def __init__(self, manager: JobManager, job_id: str, project_id: str, scene_id: str | None):
        self._m = manager
        self.job_id = job_id
        self.project_id = project_id
        self.scene_id = scene_id
        self._cancel = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled()

    @contextmanager
    def step(self, key: str) -> Iterator[None]:
        """Шаг задачи: RUNNING на входе, DONE (progress=1) на выходе, FAILED/CANCELLED при исключении."""
        self.check()
        self._m._set_step(self.job_id, key, state=JobState.RUNNING, progress=0.0, force=True)
        try:
            yield
        except JobCancelled:
            self._m._set_step(self.job_id, key, state=JobState.CANCELLED, force=True)
            raise
        except BaseException:
            self._m._set_step(self.job_id, key, state=JobState.FAILED, force=True)
            raise
        else:
            self._m._set_step(self.job_id, key, state=JobState.DONE, progress=1.0, force=True)

    def progress(self, key: str, value: float) -> None:
        self.check()
        v = min(max(float(value), 0.0), 1.0)
        self._m._set_step(self.job_id, key, progress=v, force=v >= 1.0)

    def set_result(self, result: dict | JobResult) -> None:
        self._m._set_result(self.job_id, result)

    def note(self, key: str, text: str) -> None:
        """Пояснение к шагу для пользователя (показывается рядом с шагом)."""
        self._m._set_step(self.job_id, key, note=text, force=True)

    def skip(self, key: str) -> None:
        """Шаг не нужен в этой ветке — показать его выполненным."""
        self._m._set_step(self.job_id, key, state=JobState.DONE, progress=1.0, force=True)


class JobManager:
    def __init__(self, *, heavy_workers: int = 1, light_workers: int = 2, max_jobs: int = MAX_JOBS):
        self._heavy = ThreadPoolExecutor(max_workers=heavy_workers, thread_name_prefix="dubl-heavy")
        self._light = ThreadPoolExecutor(max_workers=light_workers, thread_name_prefix="dubl-light")
        self._max_jobs = max_jobs
        self._lock = threading.RLock()
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._ctx: dict[str, JobContext] = {}
        self._last_emit: dict[str, float] = {}
        self._started: dict[str, float] = {}      # job_id -> monotonic времени старта
        self._subs: set[asyncio.Queue] = set()
        self._events: dict[str, asyncio.Event] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    # ───────────── подключение к циклу событий ─────────────
    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_running_loop()

    def shutdown(self, wait: bool = False) -> None:
        with self._lock:
            for ctx in self._ctx.values():
                ctx._cancel.set()
        self._heavy.shutdown(wait=wait, cancel_futures=True)
        self._light.shutdown(wait=wait, cancel_futures=True)

    # ───────────── публичный API ─────────────
    def submit(self, kind: str, project_id: str, steps: Sequence[Any], fn: Callable[[JobContext], Any],
               scene_id: str | None = None, *, heavy: bool | None = None) -> Job:
        """Поставить задачу в очередь. steps — [(key, title), …] или [JobStep]. Можно из любого потока, если цикл привязан
        (bind_loop; приложение делает это при старте)."""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None:
                raise RuntimeError("JobManager.submit нужно вызывать из asyncio-цикла (или сделать bind_loop)") from None
        job_steps = [s if isinstance(s, JobStep) else JobStep(key=s[0], title=s[1]) for s in steps]
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, project_id=project_id, scene_id=scene_id, steps=job_steps)
        ctx = JobContext(self, job.id, project_id, scene_id)
        is_heavy = kind in HEAVY_KINDS if heavy is None else heavy
        with self._lock:
            self._jobs[job.id] = job
            self._ctx[job.id] = ctx
            self._events[job.id] = asyncio.Event()
            self._evict()
            snap = job.model_copy(deep=True)
        self._publish(snap)
        (self._heavy if is_heavy else self._light).submit(self._run, job.id, fn)
        return snap

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            j = self._jobs.get(job_id)
            return j.model_copy(deep=True) if j else None

    def list(self) -> list[Job]:
        with self._lock:
            return [j.model_copy(deep=True) for j in reversed(self._jobs.values())]

    def active(self) -> list[Job]:
        with self._lock:
            return [j.model_copy(deep=True) for j in self._jobs.values() if j.state not in FINISHED]

    def cancel(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            ctx = self._ctx.get(job_id)
            if job is None:
                return None
            if job.state in FINISHED:
                return job.model_copy(deep=True)
            if ctx:
                ctx._cancel.set()
            if job.state == JobState.QUEUED:
                self._finish(job, JobState.CANCELLED)
                snap = job.model_copy(deep=True)
            else:
                snap = None
        if snap is not None:
            self._publish(snap)
            return snap
        return self.get(job_id)

    async def wait(self, job_id: str, timeout: float | None = None) -> Job:
        """Дождаться завершения задачи (для тестов и вызовов из кода)."""
        ev = self._events.get(job_id)
        if ev is None:
            raise KeyError(job_id)
        await asyncio.wait_for(ev.wait(), timeout)
        j = self.get(job_id)
        assert j is not None
        return j

    def wait_project_idle(self, project_id: str, scene_id: str | None = None, timeout: float = 15.0) -> bool:
        """Дождаться, пока у проекта (сцены) не останется работающих задач. Для удаления: не оставлять файлы-сироты."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if not any(j.project_id == project_id and (scene_id is None or j.scene_id == scene_id
                                                       or j.kind == "render_project")      # склейка трогает любые сцены
                       for j in self.active()):
                return True
            time.sleep(0.1)
        return False

    async def run_light(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Выполнить лёгкую функцию в лёгком пуле (не блокируя цикл) и дождаться результата."""
        import functools
        return await asyncio.get_running_loop().run_in_executor(self._light, functools.partial(fn, *args))

    # ───────────── подписчики ─────────────
    def subscribe(self, maxsize: int = 1000) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    # ───────────── внутреннее ─────────────
    def _evict(self) -> None:
        while len(self._jobs) > self._max_jobs:
            victim = next((jid for jid, j in self._jobs.items() if j.state in FINISHED), None)
            if victim is None:
                break
            self._drop(victim)

    def _drop(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
        self._ctx.pop(job_id, None)
        self._last_emit.pop(job_id, None)
        self._started.pop(job_id, None)
        self._events.pop(job_id, None)

    def _update_progress(self, job: Job) -> None:
        """Под self._lock: общий прогресс (монотонный) и оценка оставшегося времени."""
        if job.state == JobState.DONE:
            job.progress, job.eta_s = 1.0, None
            return
        if job.state in FINISHED:
            job.eta_s = None
            return
        job.progress = round(max(job.progress, overall_progress(job)), 4)
        started = self._started.get(job.id)
        elapsed = time.monotonic() - started if started is not None else 0.0
        if started is None or elapsed < ETA_MIN_ELAPSED or job.progress < ETA_MIN_PROGRESS:
            job.eta_s = None
        else:
            job.eta_s = round(elapsed * (1.0 - job.progress) / job.progress, 1)

    def _finish(self, job: Job, state: JobState, error: str | None = None, error_debug: str | None = None,
                error_code: str | None = None) -> None:
        """Под self._lock."""
        job.state = state
        job.error = error
        job.error_debug = error_debug
        job.error_code = error_code
        self._update_progress(job)
        ev = self._events.get(job.id)
        if ev is not None and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(ev.set)
            except RuntimeError:     # цикл уже закрыт
                log.debug("event loop закрыт, событие завершения не доставлено")

    def _publish(self, snap: Job) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._fanout, snap)
        except RuntimeError:   # цикл уже закрыт
            pass

    def _fanout(self, snap: Job) -> None:
        for q in list(self._subs):
            try:
                q.put_nowait(snap)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()          # медленный подписчик: выкидываем самое старое
                    q.put_nowait(snap)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    log.debug("медленный подписчик пропустил событие")

    def _set_step(self, job_id: str, key: str, *, state: JobState | None = None,
                  progress: float | None = None, note: str | None = None, force: bool = False) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            step = next((s for s in job.steps if s.key == key), None)
            if step is None:
                step = JobStep(key=key, title=key)      # шаг, не заявленный при submit — добавим
                job.steps.append(step)
            if state is not None:
                step.state = state
            if progress is not None:
                step.progress = progress
            if note is not None:
                step.note = note
            self._update_progress(job)
            now = time.monotonic()
            if not force and now - self._last_emit.get(job_id, 0.0) < MIN_PROGRESS_INTERVAL:
                return
            self._last_emit[job_id] = now
            snap = job.model_copy(deep=True)
        self._publish(snap)

    @staticmethod
    def _as_result(result: Any) -> JobResult | None:
        if isinstance(result, JobResult):
            return result
        if isinstance(result, dict) and ("path" in result or "url" in result):
            return JobResult.model_validate({k: v for k, v in result.items() if k in JobResult.model_fields})
        return None

    def _set_result(self, job_id: str, result: dict | JobResult) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.result = self._as_result(result)

    def _run(self, job_id: str, fn: Callable[[JobContext], Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            ctx = self._ctx.get(job_id)
            if job is None or ctx is None or job.state != JobState.QUEUED:
                return                                   # отменена, пока стояла в очереди
            job.state = JobState.RUNNING
            self._started[job_id] = time.monotonic()
            self._update_progress(job)
            snap = job.model_copy(deep=True)
        self._publish(snap)

        state, error, error_debug, error_code, result = JobState.DONE, None, None, None, None
        try:
            ctx.check()
            with cancel_check(ctx.check):        # ffmpeg и прочие внешние процессы останавливаются сразу при отмене
                result = fn(ctx)
        except JobCancelled:
            state = JobState.CANCELLED
        except EngineError as e:
            state, error = JobState.FAILED, str(e)
            error_debug = redact(e.debug) if e.debug else None
            error_code = getattr(e, "code", None)
        except Exception as e:  # noqa: BLE001
            log.error("Задача %s упала:\n%s", job_id, redact(traceback.format_exc()))
            state, error = JobState.FAILED, f"Внутренняя ошибка движка ({type(e).__name__}). Подробности в журнале."
            error_debug = redact(f"{type(e).__name__}: {str(e)[:300]}")

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if state == JobState.DONE:
                for s in job.steps:                      # ветки, которые не понадобились, считаем пройденными
                    if s.state in (JobState.QUEUED, JobState.RUNNING):
                        s.state, s.progress = JobState.DONE, 1.0
                if job.result is None:
                    job.result = self._as_result(result)
            elif state == JobState.CANCELLED:
                for s in job.steps:
                    if s.state in (JobState.QUEUED, JobState.RUNNING):
                        s.state = JobState.CANCELLED
            self._finish(job, state, error, error_debug, error_code)
            snap = job.model_copy(deep=True)
        self._publish(snap)


class NullContext:
    """Контекст без очереди — для прямых вызовов пайплайнов (скрипты, тесты)."""
    job_id = "null"
    cancelled = False

    @contextmanager
    def step(self, key: str) -> Iterator[None]:
        yield

    def progress(self, key: str, value: float) -> None:
        pass

    def skip(self, key: str) -> None:
        pass

    def check(self) -> None:
        pass

    def set_result(self, result: dict | JobResult) -> None:
        pass

    def note(self, key: str, text: str) -> None:
        pass
