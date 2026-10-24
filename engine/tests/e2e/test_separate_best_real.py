"""Тяжёлые проверки на настоящем материале: BS-RoFormer (best) на куске 30 с против эталона.
Отдельно — потому что скачивает 640 МБ модели и берёт ≈ 8,6 ГБ памяти (peak memory footprint на M5)."""
from __future__ import annotations

import pytest

import helpers as H
import sys

from test_audio_real import _sep_metrics, fit_errors_on_voices

pytestmark = pytest.mark.slow


def test_separate_best_roformer(sep_clip, sep_fast):
    from dubl_engine.audio.separate import separate
    out = sep_clip["dir"] / "best"
    v, b = separate(sep_clip["mix"], out, center_wav=sep_clip["center"], quality="best")
    m = _sep_metrics(sep_clip, b, v)
    f = _sep_metrics(sep_clip, sep_fast["background"], sep_fast["voices"])
    # замер (даунмикс 5.1, 30 с): best SDR 0.14 дБ, корр 0.70; fast SDR 0.41 дБ, корр 0.71 — на этом материале best не лучше fast
    assert m["sdr_bg_speech"] - m["sdr_mix_speech"] > 2.0
    assert m["corr_bg_speech"] > m["corr_mix_speech"] + 0.05
    assert m["sdr_bg_speech"] > f["sdr_bg_speech"] - 2.5       # не сильно хуже быстрой


def test_fit_take_on_best_voices(sep_clip, tmp_path):
    """Подгонка say-дублей по голосам от BS-RoFormer (stereo-вход). Замер: медиана |ошибка| 13 мс, макс 43 мс (8 дублей)."""
    if sys.platform != "darwin":
        pytest.skip("say есть только на macOS")
    from dubl_engine.audio.separate import separate
    import numpy as np
    out = sep_clip["dir"] / "best_voices"
    v, _ = separate(sep_clip["mix"], out, quality="best")
    errs = fit_errors_on_voices(v, sep_clip["t0"], tmp_path)
    assert np.median(errs) < 80
    assert errs.max() < 200
