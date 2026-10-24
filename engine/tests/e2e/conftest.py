"""Фикстуры e2e-тестов на настоящих фильмах. Все тесты помечены slow и пропускаются, если samples не скачаны
(`python3 samples/fetch.py`). Нужен ещё `engine/.venv` с ml-зависимостями и скачанные модели (при первом запуске
качаются в общую папку моделей — см. `models_home`).

Запуск:  cd engine && .venv/bin/pytest -m slow tests/e2e -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers as H  # noqa: E402

pytestmark = pytest.mark.slow


def pytest_collection_modifyitems(config, items):
    """Всё в tests/e2e — slow (маркер уже есть в pyproject, новых не добавляем) и skip без сэмплов."""
    missing = H.need(H.TOS_MOV, H.TOS_EN, H.TOS_RU, H.TOS_51)
    for item in items:
        if "tests/e2e" not in str(item.fspath).replace("\\", "/"):
            continue
        item.add_marker(pytest.mark.slow)
        if missing:
            item.add_marker(pytest.mark.skip(reason="не скачаны samples: python3 samples/fetch.py (нет: "
                                                    + ", ".join(p.name for p in missing) + ")"))


def models_home() -> Path:
    """Папка с уже скачанными моделями (чтобы не качать их в каждый временный data_dir)."""
    from platformdirs import user_data_dir
    return Path(os.environ.get("DUBL_E2E_MODELS") or Path(user_data_dir("Dubl", "Dubl")) / "models")


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Переопределяет одноимённую фикстуру из tests/conftest.py: данные проектов — во временной папке,
    а модели — симлинк на общую папку, иначе каждый тест качал бы сотни мегабайт заново."""
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    home = models_home()
    home.mkdir(parents=True, exist_ok=True)
    link = data / "models"
    if not link.exists():
        try:
            link.symlink_to(home, target_is_directory=True)
        except OSError:
            pass
    monkeypatch.setenv("DUBL_DATA_DIR", str(data))


@pytest.fixture(scope="session")
def work() -> Path:
    H.WORK.mkdir(parents=True, exist_ok=True)
    return H.WORK


@pytest.fixture(scope="session")
def release_mkv(work) -> Path:
    if not H.TOS_RELEASE.exists():
        H.build_release_mkv()
    return H.TOS_RELEASE


@pytest.fixture(scope="session")
def en_lines():
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    return parse_subtitle_file(H.TOS_EN)


@pytest.fixture(scope="session")
def ru_lines():
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    return parse_subtitle_file(H.TOS_RU)


@pytest.fixture(scope="session")
def tos_audio(work, release_mkv):
    """Звук релизного MKV: mono 16 кГц (микс 5.1) и центральный канал 16 кГц (5.1)."""
    import numpy as np
    import scipy.signal as sg

    from dubl_engine.audio.io import read_wav, write_wav
    from dubl_engine.media.extract import extract_audio, extract_center_channel
    from dubl_engine.media.probe import pick_original_audio, probe

    d = work / "e2e_audio"
    d.mkdir(exist_ok=True)
    info = probe(release_mkv)
    st = pick_original_audio(info)
    mix16 = d / "mix_16k.wav"
    center16 = d / "center_16k.wav"
    if not mix16.exists():
        extract_audio(release_mkv, st.index, mix16, sample_rate=16000, mono=True)
    if not center16.exists():
        c = extract_center_channel(release_mkv, st.index, d / "center_48k.wav", sample_rate=48000)
        x, sr = read_wav(c)
        write_wav(center16, sg.resample_poly(x, 1, 3).astype(np.float32), 16000)
    return {"mix16": mix16, "center16": center16, "stream": st}


@pytest.fixture(scope="session")
def sep_clip(work):
    """Кусок 30 с (296–326 с) из 5.1: стерео-микс, центр и эталон «музыка+шумы без диалога»."""
    t0, t1 = 296.0, 326.0
    d = work / "e2e_sep"
    d.mkdir(exist_ok=True)
    if not (d / "mix.wav").exists():
        H.ff("-ss", t0, "-to", t1, "-i", H.TOS_51, "-vn", "-ac", 2, "-ar", 48000, "-c:a", "pcm_f32le", d / "mix.wav")   # даунмикс, как в prepare_scene
        H.ff("-ss", t0, "-to", t1, "-i", H.TOS_51, "-af", "pan=mono|c0=FC", "-ar", 48000, "-c:a", "pcm_f32le", d / "center.wav")
        H.ff("-ss", t0, "-to", t1, "-i", H.TOS_MFX, "-ar", 48000, "-ac", 2, "-c:a", "pcm_f32le", d / "ref_mfx.wav")
    return {"t0": t0, "t1": t1, "dir": d, "mix": d / "mix.wav", "center": d / "center.wav", "ref": d / "ref_mfx.wav"}


@pytest.fixture(scope="session")
def sep_fast(sep_clip):
    """Результат separate(quality='fast') на куске; считается один раз за сессию."""
    from dubl_engine.audio.separate import separate
    out = sep_clip["dir"] / "fast"
    v, b = separate(sep_clip["mix"], out, center_wav=sep_clip["center"], quality="fast", device="cpu")
    return {"voices": v, "background": b}


@pytest.fixture(autouse=True, scope="module")
def _free_models_between_modules():
    """Нейросети (mlx/torch/ctranslate2) не должны копиться в одном процессе pytest: после каждого модуля — gc и очистка
    кэшей Metal/MPS. Без этого полный прогон slow e2e доходил до 17,6 ГБ footprint на 16-гигабайтном Mac."""
    yield
    import gc
    from dubl_engine.analysis.transcribe import release_memory
    release_memory()
    gc.collect()
    try:
        import mlx.core as mx
        getattr(mx, "clear_cache", None) and mx.clear_cache()
    except Exception:  # noqa: BLE001
        pass
    if "torch" in sys.modules:
        try:
            import torch
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    gc.collect()
