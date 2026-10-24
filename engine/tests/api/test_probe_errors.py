"""Ссылки в /probe проверяются; ошибки ffmpeg — человеческие; 404 по-русски. Сеть не нужна (yt-dlp подменён)."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from dubl_engine.api import create_app
from dubl_engine.jobs import JobManager
from dubl_engine.pipeline import download
from dubl_engine.store import ProjectStore


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(store=ProjectStore(tmp_path / "projects"), jobs=JobManager())) as c:
        yield c


def _fake_ydl(monkeypatch, *, info=None, error=None, sleep=0.0):
    class FakeYDL:
        def __init__(self, opts):
            assert opts["skip_download"] is True and opts["socket_timeout"] <= 15
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download=True):
            assert download is False
            if sleep:
                time.sleep(sleep)
            if error:
                raise error
            return info
    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)


@pytest.mark.parametrize("err,fragment", [
    ("ERROR: [youtube] abcdefghijk: Video unavailable", "недоступно"),
    ("ERROR: [youtube] zzzzzzzzzzz: This video is unavailable", "недоступно"),
    ("ERROR: [youtube] x: Private video. Sign in if you've been granted access", "закрыто"),
    ("ERROR: Unsupported URL: https://example.com/page", "нет видео"),
    ("ERROR: [generic] HTTP Error 404: Not Found", "недоступно"),
])
def test_probe_url_bad_link_is_422_in_russian(client, monkeypatch, err, fragment):
    _fake_ydl(monkeypatch, error=RuntimeError(err))
    r = client.post("/probe", json={"url": "https://www.youtube.com/watch?v=abcdefghijk"})
    assert r.status_code == 422, r.text
    j = r.json()
    assert fragment in j["detail"] and "ERROR" not in j["detail"] and "youtube]" not in j["detail"]
    assert "Video unavailable" in j.get("debug", "") or "Unsupported" in j.get("debug", "") or "404" in j.get("debug", "") \
        or "Private" in j.get("debug", "") or "unavailable" in j.get("debug", "")


@pytest.mark.parametrize("err", [
    "ERROR: Unable to download webpage: <urlopen error [Errno 8] nodename nor servname provided, or not known>",
    "<urlopen error [Errno -3] Temporary failure in name resolution>",
    "The read operation timed out",
])
def test_probe_url_no_network_is_not_an_error(client, monkeypatch, err):
    _fake_ydl(monkeypatch, error=RuntimeError(err))
    r = client.post("/probe", json={"url": "https://www.youtube.com/watch?v=abcdefghijk"})
    assert r.status_code == 200
    j = r.json()
    assert j["known"] is False and "Не удалось проверить ссылку" in j["message"]


def test_probe_url_timeout(client, monkeypatch):
    _fake_ydl(monkeypatch, info={"title": "x"}, sleep=1.5)
    monkeypatch.setattr(download, "PROBE_URL_TIMEOUT_S", 0.2)
    monkeypatch.setattr(download.probe_url, "__kwdefaults__", {"timeout": 0.2})
    t0 = time.time()
    r = client.post("/probe", json={"url": "https://example.com/v"})
    assert r.status_code == 200 and r.json()["known"] is False and "Не удалось проверить" in r.json()["message"]
    assert time.time() - t0 < 1.4


def test_probe_url_ok_and_not_a_video(client, monkeypatch):
    _fake_ydl(monkeypatch, info={"title": "Ролик", "duration": 12, "subtitles": {}, "automatic_captions": {}})
    r = client.post("/probe", json={"url": "https://example.com/v"})
    assert r.status_code == 200 and r.json()["known"] is True and r.json()["title"] == "Ролик"
    _fake_ydl(monkeypatch, info={"_type": "playlist", "entries": []})
    assert client.post("/probe", json={"url": "https://example.com/v"}).status_code == 422
    _fake_ydl(monkeypatch, info=None)
    assert client.post("/probe", json={"url": "https://example.com/v"}).status_code == 422


def test_probe_file_url_becomes_path(client, simple_mp4, tmp_path):
    r = client.post("/probe", json={"url": simple_mp4.as_uri()})
    assert r.status_code == 200 and r.json()["width"] == 640
    missing = (tmp_path / "нет такого.mp4").as_uri()
    r = client.post("/probe", json={"url": missing})
    assert r.status_code == 400 and "Файл не найден" in r.json()["detail"] and "нет такого.mp4" in r.json()["detail"]
    assert client.post("/probe", json={"url": "ftp://x/y.mp4"}).status_code == 422


def test_create_project_from_file_url(client, simple_mp4):
    r = client.post("/projects", json={"url": simple_mp4.as_uri()})
    assert r.status_code == 200, r.text
    client.delete(f"/projects/{r.json()['id']}")


# ───────────── сырой ffmpeg не доходит до человека ─────────────
@pytest.fixture
def truncated_mp4(simple_mp4, tmp_path):
    p = tmp_path / "обрезанный.mp4"
    p.write_bytes(simple_mp4.read_bytes()[:40_000])          # moov в конце файла — его нет
    return p


@pytest.fixture
def fake_mkv(tmp_path):
    p = tmp_path / "заметки.mkv"
    p.write_text("это просто текст, а не видео\n" * 50, encoding="utf-8")
    return p


NOT_VIDEO = "Это не видео"
BROKEN = "Видео повреждено"


def _assert_human(r, fragment, *, status=400):
    assert r.status_code == status, r.text
    j = r.json()
    assert fragment in j["detail"], j
    other = BROKEN if fragment == NOT_VIDEO else NOT_VIDEO
    assert other not in j["detail"], j                         # .txt и обрезанное видео — разные тексты
    for raw in ("moov", "Invalid data", "ffprobe", "ffmpeg:", "atom", "Выберите", "Скачайте", "Проверьте"):
        assert raw not in j["detail"], j
    return j


def test_probe_truncated_mp4(client, truncated_mp4):
    j = _assert_human(client.post("/probe", json={"path": str(truncated_mp4)}), BROKEN)
    assert "moov atom not found" in j["debug"]          # сырой текст — отдельным полем
    assert "Скачай файл ещё раз" in j["detail"]


def test_probe_text_file_as_mkv(client, fake_mkv):
    j = _assert_human(client.post("/probe", json={"path": str(fake_mkv)}), NOT_VIDEO)
    assert "Invalid data found" in j["debug"] and "Выбери видеофайл" in j["detail"]


def test_probe_txt_and_other_formats_are_not_video(client, tmp_path):
    """.txt и случайные байты: «это не видео», а не «повреждено»."""
    txt = tmp_path / "заметки.txt"
    txt.write_text("просто текст\n" * 20, encoding="utf-8")
    junk = tmp_path / "мусор.mkv"
    junk.write_bytes(bytes(range(256)) * 20)
    for p in (txt, junk):
        _assert_human(client.post("/probe", json={"path": str(p)}), NOT_VIDEO)


def test_create_project_broken_file(client, truncated_mp4, fake_mkv):
    _assert_human(client.post("/projects", json={"path": str(truncated_mp4)}), BROKEN)
    _assert_human(client.post("/projects", json={"path": str(fake_mkv)}), NOT_VIDEO)


def test_ffmpeg_failure_mapping_and_secrets():
    from dubl_engine.util import EngineError, ffmpeg_failure
    e = ffmpeg_failure("x.mp4: Permission denied", 1)
    assert isinstance(e, EngineError) and "Нет доступа" in str(e) and "Permission" in e.debug
    e = ffmpeg_failure("[aac @ 0x1] something weird token=hf_abc...mnop", 1)
    assert "weird" not in str(e) and "ffmpeg" not in str(e).lower() and "hf_abc...mnop" not in e.debug
    assert "код 7" in ffmpeg_failure("", 7).debug
    # у обрезанного mp4 в stderr оба признака — выигрывает «повреждено»
    both = ffmpeg_failure("moov atom not found\nx.mp4: Invalid data found when processing input", 1)
    assert BROKEN in str(both)


def test_error_texts_are_on_ty(tmp_path):
    """Ни одного «вы»-обращения в текстах ошибок движка (зона pipeline/api/analysis/media/store/jobs/util)."""
    import re
    from pathlib import Path

    import dubl_engine
    root = Path(dubl_engine.__file__).parent
    bad = re.compile(r"\b(Выберите|Проверьте|Попробуйте|Установите|Скачайте|Переустановите|Создайте|Укажите|Обновите|"
                     r"Подождите|Дождитесь|Закройте|Включите|Откройте|Разрешите|Введите|Повторите|Напишите|Верните|"
                     r"Принимайте|примите)\b")
    hits = []
    for sub in ("pipeline", "api", "analysis", "media", "store", "jobs"):
        for p in (root / sub).rglob("*.py"):
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if bad.search(line) and not line.lstrip().startswith("#"):
                    hits.append(f"{p.relative_to(root)}:{n}: {line.strip()[:90]}")
    assert not hits, hits
    from dubl_engine import util
    for text in (util.NOT_VIDEO_MSG, util.BROKEN_FILE_MSG, util.ML_MISSING_MSG):
        assert not bad.search(text), text


def test_run_ffmpeg_on_garbage_has_human_text(fake_mkv, tmp_path):
    from dubl_engine.util import EngineError, run_ffmpeg
    with pytest.raises(EngineError) as ei:
        run_ffmpeg(["-i", fake_mkv, tmp_path / "o.wav"])
    assert "Это не видео" in str(ei.value) and "Invalid data" in ei.value.debug


def test_job_error_is_human_with_debug_field(client, truncated_mp4, simple_mp4, tmp_path):
    """Задача с битым файлом: Job.error — по-русски, сырое — в Job.error_debug."""
    from dubl_engine.media.extract import extract_audio
    jm = client.app.state.jobs

    def fn(ctx):
        with ctx.step("audio"):
            extract_audio(truncated_mp4, 1, tmp_path / "a.wav")
    job = jm.submit("analyze", "p", [("audio", "Звук")], fn)
    t0 = time.time()
    while time.time() - t0 < 20:
        j = client.get(f"/jobs/{job.id}").json()
        if j["state"] == "failed":
            break
        time.sleep(0.1)
    assert j["state"] == "failed" and "Видео повреждено" in j["error"] and "moov" not in j["error"]
    assert "moov" in (j["error_debug"] or "")


def test_unknown_route_and_method_are_russian(client):
    r = client.get("/net-takogo-marshruta")
    assert r.status_code == 404 and "Not Found" not in r.json()["detail"]
    assert "нет" in r.json()["detail"].lower()
    r = client.delete("/probe")
    assert r.status_code == 405 and "Method Not Allowed" not in r.json()["detail"]
    r = client.get("/projects/zzz")
    assert r.status_code == 404 and r.json()["detail"] == "Проект не найден."


def test_auth_error_stays_russian(tmp_path):
    with TestClient(create_app(store=ProjectStore(tmp_path / "p"), jobs=JobManager(), token="secret")) as c:
        r = c.get("/projects")
        assert r.status_code == 401 and r.json()["detail"] == "Нужен токен доступа."
