"""Пометки из субтитров вместо речи: «ДИНАМИЧНАЯ МУЗЫКА», «[музыка]», «(смеётся)» — не реплики."""
import numpy as np

from dubl_engine.analysis import transcribe as T
from dubl_engine.models import Word


def _seg(s, e, text, prob=0.9):
    n = max(1, len(text.split()))
    ws = [Word(start=s + i * (e - s) / n, end=s + (i + 1) * (e - s) / n, text=" " + w, prob=prob)
          for i, w in enumerate(text.split())]
    return T.Seg(s, e, " " + text, ws, avg_logprob=-0.3, no_speech_prob=0.0, compression_ratio=1.1)


def test_caption_score():
    assert T.caption_score("ДИНАМИЧНАЯ МУЗЫКА") >= 3
    assert T.caption_score("[музыка]") >= 3
    assert T.caption_score("(смеётся)") >= 3
    assert T.caption_score("♪ la la ♪") >= 3
    assert T.caption_score("Ты осмелился бросить вызов Мегамозгу?") == 0
    assert T.caption_score("НЕТ!") == 0                          # короткий крик капсом — это речь
    assert T.caption_score("СТОЙ, ГДЕ СТОИШЬ!") < 3                # капс без слов-пометок сам по себе не выкидывается


def test_caption_lines_dropped_speech_kept():
    env = T.Envelope(np.full(4000, -25.0))
    segs = [_seg(50, 52, "Я знала, ты вернёшься."), _seg(52.2, 54, "А я не был в этом уверен."),
            _seg(54.3, 56, "ДИНАМИЧНАЯ МУЗЫКА"), _seg(56.2, 58, "Нет, только не по лицу!")]
    out = [s.text.strip() for s in T.drop_hallucinations(segs, env)]
    assert out == ["Я знала, ты вернёшься.", "А я не был в этом уверен.", "Нет, только не по лицу!"]
