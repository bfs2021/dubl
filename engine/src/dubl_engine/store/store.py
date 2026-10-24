"""Хранилище проектов на диске: <root>/<project_id>/project.json + файлы сцен.

Потокобезопасность: у каждого проекта своя RLock. Любое «прочитал → поменял → записал» делай через
`with store.edit(pid) as project:` — внутри блока проект защищён от параллельных правок из других задач
(анализ в фоне, загрузка дубля из API и т. п.). Запись всегда атомарная (tmp + os.replace).
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, model_validator

from ..models import AnalysisState, MediaInfo, Project, Scene, Take
from ..util import EngineError, data_dir
from .atomic import atomic_write_text

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
THUMBNAIL_NAME = "thumbnail.jpg"
PROXY_NAME = "proxy.mp4"
SUMMARY_NAME = "summary.json"      # короткая карточка проекта: список проектов не читает многомегабайтный project.json
DOWNLOADS_DIR = "downloads"        # <data_dir>/downloads/<project_id>/ — видео, скачанные по ссылке (принадлежат проекту)
TRASH_PREFIX = ".trash-"           # удаляемая папка проекта сначала переименовывается сюда, потом стирается
# Файлы сцены, без которых «голоса и фон готовы» — неправда (voices/ — по персонажам, может быть пустой без реплик).
SCENE_STEM_FILES = ("clip.mp4", "original.wav", "voices.wav", "background.wav")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")

_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
    "э": "e", "ю": "yu", "я": "ya",
}


class NotFoundError(EngineError):
    """Нет такого проекта/сцены/дубля (API отвечает 404)."""


class ProjectSummary(BaseModel):
    id: str
    title: str
    duration: float
    scene_count: int
    has_analysis: bool
    updated: datetime
    created: datetime | None = None    # когда проект создан (карточки с одинаковым именем различаются по дате)
    thumbnail: str | None = None    # путь относительно папки проекта (отдаётся как /files/<id>/<thumbnail>)
    # none — разбор не запускали; running — идёт; interrupted — не дошёл до конца (закрыли программу, отмена, ошибка),
    # перезапуск: POST /projects/{pid}/analyze; done — разбор готов (has_analysis=true). Старая карточка без поля читается как none.
    analysis_state: AnalysisState = "none"

    @model_validator(mode="after")
    def _done_iff_analysis(self):
        if self.has_analysis:
            self.analysis_state = "done"
        elif self.analysis_state == "done":
            self.analysis_state = "interrupted"
        return self


def slugify(title: str, max_len: int = 24) -> str:
    out: list[str] = []
    for ch in title.lower():
        if ch in _TRANSLIT:
            out.append(_TRANSLIT[ch])
        elif ch.isascii() and ch.isalnum():
            out.append(ch)
        else:
            out.append("-")
    slug = re.sub(r"-{2,}", "-", "".join(out)).strip("-")[:max_len].strip("-")
    return slug or "project"


def _check_id(value: str, what: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value) or ".." in value:
        raise NotFoundError(f"Некорректный идентификатор ({what}): {value!r}")
    return value


class ProjectStore:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else data_dir() / "projects"
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    # ───────────── пути ─────────────
    def project_dir(self, pid: str) -> Path:
        return self.root / _check_id(pid, "проект")

    def project_file(self, pid: str) -> Path:
        return self.project_dir(pid) / "project.json"

    def analysis_dir(self, pid: str) -> Path:
        return self.project_dir(pid) / "analysis"

    def scene_dir(self, pid: str, scid: str) -> Path:
        return self.project_dir(pid) / "scenes" / _check_id(scid, "сцена")

    def takes_dir(self, pid: str, scid: str) -> Path:
        return self.scene_dir(pid, scid) / "takes"

    def take_path(self, pid: str, scid: str, take_id: str) -> Path:
        return self.takes_dir(pid, scid) / f"{_check_id(take_id, 'дубль')}.wav"

    def renders_dir(self, pid: str) -> Path:
        return self.project_dir(pid) / "renders"

    def proxy_path(self, pid: str) -> Path:
        """Лёгкая копия исходного видео для плеера (540p h264+aac, faststart)."""
        return self.project_dir(pid) / PROXY_NAME

    def thumbnail_path(self, pid: str) -> Path:
        return self.project_dir(pid) / THUMBNAIL_NAME

    def rel_to_project(self, pid: str, path: str | Path) -> str | None:
        """Путь относительно папки проекта (posix) или None, если файл лежит снаружи."""
        try:
            return Path(path).resolve().relative_to(self.project_dir(pid).resolve()).as_posix()
        except ValueError:
            return None

    # ───────────── блокировки ─────────────
    def lock(self, pid: str) -> threading.RLock:
        _check_id(pid, "проект")
        with self._locks_guard:
            lk = self._locks.get(pid)
            if lk is None:
                lk = self._locks[pid] = threading.RLock()
            return lk

    # ───────────── CRUD ─────────────
    def exists(self, pid: str) -> bool:
        try:
            return self.project_file(pid).is_file()
        except NotFoundError:
            return False

    def new_id(self, title: str) -> str:
        while True:
            pid = f"{slugify(title)}-{uuid.uuid4().hex[:6]}"
            if not self.project_dir(pid).exists():
                return pid

    def create(self, source_path: str | Path, media: MediaInfo, title: str | None = None) -> Project:
        src = Path(source_path)
        title = (title or src.stem or "Без названия").strip() or "Без названия"
        pid = self.new_id(title)
        project = Project(id=pid, title=title, source_path=str(source_path), media=media)
        (self.project_dir(pid) / "scenes").mkdir(parents=True, exist_ok=True)
        with self.lock(pid):
            return self._write(project)

    def get(self, pid: str) -> Project:
        path = self.project_file(pid)
        with self.lock(pid):
            try:
                raw = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                raise NotFoundError("Проект не найден.") from None
            try:
                return Project.model_validate(json.loads(raw))
            except (ValueError, TypeError) as e:
                raise EngineError(f"Файл проекта повреждён ({pid}): {e}") from None

    def _write(self, project: Project, *, touch: bool = True) -> Project:
        project = project.model_copy(update={"updated": datetime.now().astimezone()}) if touch else project
        project.sync_states()                 # статусы не расходятся с данными (analysis, stems_ready)
        data = project.model_dump(mode="json")
        data["schema"] = SCHEMA_VERSION
        atomic_write_text(self.project_file(project.id), json.dumps(data, ensure_ascii=False, indent=1))
        atomic_write_text(self.project_dir(project.id) / SUMMARY_NAME, self.summary_of(project).model_dump_json())
        return project

    @staticmethod
    def summary_of(project: Project, thumbnail: str | None = None) -> ProjectSummary:
        return ProjectSummary(id=project.id, title=project.title, duration=project.media.duration,
                              scene_count=len(project.scenes), has_analysis=project.analysis is not None,
                              analysis_state=project.analysis_state, updated=project.updated, created=project.created,
                              thumbnail=thumbnail)

    def save(self, project: Project) -> Project:
        """Атомарно записать проект; вернуть копию с обновлённым `updated`."""
        with self.lock(project.id):
            if not self.project_dir(project.id).exists():
                raise NotFoundError("Проект не найден.")
            return self._write(project)

    @contextmanager
    def edit(self, pid: str) -> Iterator[Project]:
        """`with store.edit(pid) as p: ...` — под блокировкой; при выходе без ошибки проект сохраняется.

        Правь объект `p` на месте (p.scenes[i] = …, p.analysis = …).
        """
        with self.lock(pid):
            project = self.get(pid)
            yield project
            self._write(project)

    def delete(self, pid: str) -> list[Path]:
        """Удалить проект: папку проекта и видео, которое программа сама скачала по ссылке (и только его).

        Папка сначала переименовывается (проект пропадает из списка мгновенно, запоздавшая задача получает «не найден»,
        а не пишет в уже стираемую папку), потом стирается. Исходный файл пользователя и всё, что лежит вне
        <data_dir>/downloads, не трогаем. Возвращает, какие скачанные файлы стёрты."""
        d = self.project_dir(pid)
        with self.lock(pid):
            if not d.exists():
                raise NotFoundError("Проект не найден.")
            try:
                source = str(json.loads((d / "project.json").read_text(encoding="utf-8")).get("source_path") or "")
            except (OSError, ValueError):
                source = ""
            trash = d.with_name(f"{TRASH_PREFIX}{pid}-{uuid.uuid4().hex[:6]}")
            d.rename(trash)
            self._rmtree(trash)
        with self._locks_guard:
            self._locks.pop(pid, None)
        return self._delete_downloads(pid, source)

    @staticmethod
    def _rmtree(path: Path) -> None:
        for attempt in range(3):             # файл мог ещё быть открыт (ffmpeg добивает запись) — стираем повторно
            shutil.rmtree(path, ignore_errors=True)
            if not path.exists():
                return
            time.sleep(0.2 * (attempt + 1))
        raise EngineError("Не удалось стереть файлы проекта: они заняты другой программой. Закрой её и повтори.")

    @staticmethod
    def downloads_root() -> Path:
        return data_dir() / DOWNLOADS_DIR

    def _delete_downloads(self, pid: str, source: str) -> list[Path]:
        """Стереть скачанное по ссылке видео проекта. Свои файлы пользователя (источник вне downloads) — никогда."""
        root = self.downloads_root().resolve()
        removed: list[Path] = []
        own = root / pid
        if own.is_dir() and not own.is_symlink():
            shutil.rmtree(own, ignore_errors=True)
            removed.append(own)
        if source:
            # старый порядок: файл лежал прямо в downloads/<id ролика>.mkv — стираем, если им больше никто не пользуется
            src = Path(source)
            try:
                inside = src.parent.resolve() == root and (src.is_file() or src.is_symlink())
            except OSError:
                inside = False
            if inside and not self._source_in_use(source):
                try:
                    src.unlink()
                    removed.append(src)
                except OSError as e:
                    log.warning("Не удалось стереть скачанный файл %s: %s", src, e)
        return removed

    def _source_in_use(self, source: str) -> bool:
        for d in self.root.iterdir() if self.root.exists() else []:
            f = d / "project.json"
            if d.name.startswith(".") or not f.is_file():
                continue
            try:
                if str(json.loads(f.read_text(encoding="utf-8")).get("source_path") or "") == source:
                    return True
            except (OSError, ValueError):
                return True              # не смогли прочитать — на всякий случай считаем, что файл нужен
        return False

    def list(self) -> list[ProjectSummary]:
        out: list[ProjectSummary] = []
        if not self.root.exists():
            return out
        for d in self.root.iterdir():
            if not d.is_dir() or d.name.startswith(".") or not (d / "project.json").is_file():
                continue
            thumb = THUMBNAIL_NAME if (d / THUMBNAIL_NAME).is_file() else None
            try:
                sm = ProjectSummary.model_validate_json((d / SUMMARY_NAME).read_text(encoding="utf-8"))
                if sm.id == d.name:
                    out.append(sm.model_copy(update={"thumbnail": thumb}))
                    continue
            except (OSError, ValueError):
                pass                        # нет карточки (старый проект) — читаем весь файл
            try:
                p = self.get(d.name)
            except EngineError as e:
                log.warning("Пропускаю проект %s: %s", d.name, e)
                continue
            out.append(self.summary_of(p, thumb))
        out.sort(key=lambda s: s.updated, reverse=True)
        return out

    # ───────────── честные статусы после перезапуска ─────────────
    def scene_files_complete(self, pid: str, scene: Scene) -> bool:
        """Все файлы подготовленной сцены на месте и целые (клип открывается, звуки читаются и одной длины)."""
        sd = self.scene_dir(pid, scene.id)
        files = [sd / n for n in SCENE_STEM_FILES]
        if not all(f.is_file() and f.stat().st_size > 0 for f in files):
            return False
        try:
            import soundfile as sf
            lengths = [sf.info(str(sd / n)).frames for n in ("original.wav", "voices.wav", "background.wav")]
            if min(lengths) <= 0 or max(lengths) - min(lengths) > 4:
                return False
            from ..util import ffprobe_json
            return any(st.get("codec_type") == "video" for st in ffprobe_json(sd / "clip.mp4", "-show_streams").get("streams", []))
        except Exception as e:  # noqa: BLE001 — оборванный/битый файл
            log.info("Сцена %s/%s: файлы неполные (%s)", pid, scene.id, e)
            return False

    def recover_interrupted(self, *, check_files: bool = True) -> list[str]:
        """При старте движка (задач ещё нет): всё, что значилось «идёт», на самом деле оборвано.

        * разбор без результата (analysis_state none/running) → interrupted: его можно перезапустить (POST …/analyze);
        * сцена не готова (running) → interrupted (POST …/scenes/{scid}/prepare);
        * stems_ready=true, а файлов нет или они обрезаны → stems_ready=false, interrupted.
        Время `updated` проектов не меняется. Возвращает id проектов, у которых что-то поправили."""
        fixed: list[str] = []
        if not self.root.exists():
            return fixed
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and d.name.startswith(TRASH_PREFIX):       # удаление оборвалось на середине — дочищаем
                shutil.rmtree(d, ignore_errors=True)
                continue
            if not d.is_dir() or d.name.startswith(".") or not (d / "project.json").is_file():
                continue
            pid = d.name
            try:
                with self.lock(pid):
                    p = self.get(pid)
                    changed = False
                    if p.analysis is None and p.analysis_state != "interrupted":
                        p.analysis_state, changed = "interrupted", True
                    for sc in p.scenes:
                        if sc.stems_ready and check_files and not self.scene_files_complete(pid, sc):
                            sc.stems_ready, sc.prepare_state, changed = False, "interrupted", True
                        elif not sc.stems_ready and sc.prepare_state != "interrupted":
                            sc.prepare_state, changed = "interrupted", True
                    summary_ok = False
                    try:
                        sm = ProjectSummary.model_validate_json((d / SUMMARY_NAME).read_text(encoding="utf-8"))
                        summary_ok = sm == self.summary_of(p, sm.thumbnail).model_copy(update={"updated": sm.updated}) \
                            and sm.updated == p.updated
                    except (OSError, ValueError):
                        pass
                    if changed or not summary_ok:
                        self._write(p, touch=False)
                        if changed:
                            fixed.append(pid)
            except EngineError as e:
                log.warning("Проект %s не проверен при старте: %s", pid, e)
        return fixed

    # ───────────── сцены и дубли ─────────────
    @staticmethod
    def find_scene(project: Project, scid: str) -> Scene:
        for sc in project.scenes:
            if sc.id == scid:
                return sc
        raise NotFoundError("Сцена не найдена.")

    @staticmethod
    def scene_index(project: Project, scid: str) -> int:
        for i, sc in enumerate(project.scenes):
            if sc.id == scid:
                return i
        raise NotFoundError("Сцена не найдена.")

    @staticmethod
    def find_take(scene: Scene, take_id: str) -> Take:
        for t in scene.takes:
            if t.id == take_id:
                return t
        raise NotFoundError("Дубль не найден.")

    @staticmethod
    def next_scene_id(project: Project) -> str:
        n = 0
        for sc in project.scenes:
            m = re.fullmatch(r"sc(\d+)", sc.id)
            if m:
                n = max(n, int(m.group(1)))
        return f"sc{n + 1:03d}"

    @staticmethod
    def next_take_id(scene: Scene) -> str:
        n = 0
        for t in scene.takes:
            m = re.fullmatch(r"t(\d+)", t.id)
            if m:
                n = max(n, int(m.group(1)))
        return f"t{n + 1:03d}"

    def delete_scene_files(self, pid: str, scid: str) -> None:
        shutil.rmtree(self.scene_dir(pid, scid), ignore_errors=True)
