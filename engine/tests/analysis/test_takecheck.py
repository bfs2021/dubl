"""Разбор дубля по тексту: полнота, хвост-перезапуск, на каком языке можно сравнивать. Без нейросетей."""
from __future__ import annotations

import pytest

from dubl_engine.analysis.takecheck import analyze_text, comparable, match_ratio, norm_words
from dubl_engine.models import Word


def _words(text: str, start: float = 0.0, step: float = 0.4, gap_before: dict[int, float] | None = None) -> list[Word]:
    out, t = [], start
    for i, w in enumerate(text.split()):
        t += (gap_before or {}).get(i, 0.0)
        out.append(Word(start=round(t, 3), end=round(t + step * 0.8, 3), text=w))
        t += step
    return out


LINE = "Кто трогал мою колбасу, пока я спал"


def test_full_take_is_complete():
    r = analyze_text(LINE, _words("Кто трогал мою колбасу пока я спал"))
    assert r.text_match == 1.0 and r.complete and r.tail_cut_s is None
    assert r.recognized == "Кто трогал мою колбасу пока я спал"


def test_endings_and_yo_do_not_count_as_errors():
    r = analyze_text("Это всё моё, ёжик", _words("Это все мое ежик"))
    assert r.complete and r.text_match == 1.0
    r = analyze_text("Он пошёл домой", _words("Он пошла домой"))          # окончание распознали иначе
    assert r.complete


def test_incomplete_take():
    r = analyze_text(LINE, _words("Кто трогал мою колбасу"))
    assert not r.complete and r.text_match == pytest.approx(4 / 7, abs=0.01) and r.tail_cut_s is None
    r = analyze_text(LINE, _words("Совсем другая фраза про кота"))
    assert not r.complete and r.text_match < 0.2


def test_silence_and_empty_text():
    r = analyze_text(LINE, [])
    assert r.text_match == 0.0 and not r.complete
    assert analyze_text("", _words("что-то")).complete                  # нечего сравнивать — не «неполный»


def test_tail_lets_try_again_phrase_is_found_and_cut_point_is_between_words():
    ws = _words("Кто трогал мою колбасу пока я спал давай ещё раз", gap_before={7: 0.5})
    r = analyze_text(LINE, ws)
    assert r.complete and r.tail_cut_s is not None
    assert r.tail_text == "давай ещё раз"
    spal_end, davai_start = ws[6].end, ws[7].start
    assert spal_end < r.tail_cut_s < davai_start                         # режем в паузе, не по слову


def test_tail_repeated_beginning_is_found():
    ws = _words("Кто трогал мою колбасу пока я спал кто трогал мою колбасу", gap_before={7: 0.6})
    r = analyze_text(LINE, ws)
    assert r.complete and r.tail_cut_s is not None and r.tail_text.startswith("кто трогал")


@pytest.mark.parametrize("tail", ["стоп", "ой ошибся", "нет нет не то", "ещё раз", "сначала"])
def test_tail_variants(tail):
    r = analyze_text(LINE, _words("Кто трогал мою колбасу пока я спал " + tail, gap_before={7: 0.4}))
    assert r.complete and r.tail_cut_s is not None


def test_honest_extra_words_are_not_a_tail():
    # лишние слова, не похожие на перезапуск, не режем: человек мог добавить своё
    r = analyze_text(LINE, _words("Кто трогал мою колбасу пока я спал вчера вечером на кухне"))
    assert r.complete and r.tail_cut_s is None
    # длинный хвост (больше 14 слов) — это уже другой текст
    long_tail = " ".join(["ещё раз"] * 9)
    assert analyze_text(LINE, _words("Кто трогал мою колбасу пока я спал " + long_tail)).tail_cut_s is None


def test_no_tail_when_take_is_incomplete():
    r = analyze_text(LINE, _words("Кто трогал мою давай ещё раз"))
    assert r.tail_cut_s is None or not r.complete


def test_match_ratio():
    assert match_ratio("раз два три", "раз два три") == 1.0
    assert match_ratio("раз два три", "раз три") == pytest.approx(2 / 3, abs=0.01)
    assert match_ratio("раз два три", "") == 0.0
    assert norm_words("Ёлка, ПАЛКА! it's") == ["елка", "палка", "it's"]


@pytest.mark.parametrize("text,original,target,source,expected", [
    ("Кто трогал мою колбасу?", "Who touched my sausage?", "ru", "en", True),
    ("Who touched my sausage?", "Who touched my sausage?", "ru", "en", False),      # перевода нет — остался оригинал
    ("Who touched my sausage?", "Who touched my sausage?", "en", "en", True),
    ("Hello, мир", "Hello", "ru", "en", False),
    ("Привет, мир!", "Hello world", "ru", None, True),
    ("...", "...", "ru", None, False),
    ("", "x", "ru", None, False),
    ("Привет", "Привет", "ru", "ru", True),
    ("Ciao a tutti", "Ciao a tutti", "it", "it", True),
    ("おはよう", "おはよう", "ja", "ja", True),
])
def test_comparable(text, original, target, source, expected):
    assert comparable(text, original, target, source) is expected
