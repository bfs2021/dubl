from __future__ import annotations

import pytest

from dubl_engine.llm.tasks import count_syllables, estimate_duration, fit_line, name_speakers, translate_lines
from dubl_engine.models import Line, Speaker
from dubl_engine.util import EngineError


# ─────────── estimate_duration ───────────
def test_estimate_sausage():
    d = estimate_duration("Кто трогал мою колбасу?")
    assert 1.6 <= d <= 2.0


def test_estimate_scales_and_pauses():
    assert count_syllables("Кто трогал мою колбасу") == 8
    assert estimate_duration("") == 0.0
    plain = estimate_duration("Я иду домой")
    assert estimate_duration("Я иду, домой") == pytest.approx(plain + 0.15)
    assert estimate_duration("Я иду домой.") == pytest.approx(plain + 0.25)
    assert estimate_duration("Я иду домой...") == pytest.approx(plain + 0.25)
    assert estimate_duration("Я иду домой, а ты нет") > estimate_duration("Я иду домой")


def test_estimate_english():
    d = estimate_duration("Who touched my sausage?", "en")
    assert 1.0 <= d <= 1.8
    assert count_syllables("make", "en") == 1 and count_syllables("sausage", "en") == 2


# ─────────── translate_lines ───────────
@pytest.mark.asyncio
async def test_translate_preserves_order_and_count(fake, scene_line):
    lines = [scene_line("a", 1.0, lid=f"l{i}", start=i * 2.0, original=f"orig {i}") for i in range(3)]
    # модель вернула в другом порядке
    p = fake({"translations": [{"id": 2, "text": "три"}, {"id": 0, "text": "один"}, {"id": 1, "text": "два"}]})
    out = await translate_lines(lines, "ru", p)
    assert out == ["один", "два", "три"]
    call = p.calls[0]
    assert call["schema"]["required"] == ["translations"]
    assert "русский" in call["system"]
    assert '"duration": 1.0' in call["user"] and "orig 1" in call["user"]


@pytest.mark.asyncio
async def test_translate_empty(fake):
    p = fake("{}")
    assert await translate_lines([], "ru", p) == []
    assert p.calls == []


@pytest.mark.asyncio
async def test_translate_retries_missing_then_fails(fake, scene_line):
    lines = [scene_line("a", lid="l1"), scene_line("b", lid="l2", start=3)]
    p = fake({"translations": [{"id": 0, "text": "один"}]}, {"translations": [{"id": 1, "text": "два"}]})
    assert await translate_lines(lines, "ru", p) == ["один", "два"]
    assert len(p.calls) == 2 and '"id": 0' not in p.calls[1]["user"]

    p = fake({"translations": [{"id": 0, "text": "один"}]})
    with pytest.raises(EngineError, match="не все реплики"):
        await translate_lines(lines, "ru", p)


@pytest.mark.asyncio
async def test_translate_batches(fake, scene_line, monkeypatch):
    from dubl_engine.llm import tasks
    monkeypatch.setattr(tasks, "BATCH_SIZE", 2)
    lines = [scene_line(f"t{i}", lid=f"l{i}", start=i) for i in range(5)]
    replies = [
        {"translations": [{"id": 0, "text": "0"}, {"id": 1, "text": "1"}]},
        {"translations": [{"id": 2, "text": "2"}, {"id": 3, "text": "3"}]},
        {"translations": [{"id": 4, "text": "4"}]},
    ]
    p = fake(*replies)
    assert await translate_lines(lines, "en", p) == ["0", "1", "2", "3", "4"]
    assert len(p.calls) == 3 and "НЕ переводить" in p.calls[1]["user"]


@pytest.mark.asyncio
async def test_translate_bad_json(fake, scene_line):
    with pytest.raises(EngineError):
        await translate_lines([scene_line()], "ru", fake("извините, не могу"))


# ─────────── fit_line ───────────
@pytest.mark.asyncio
async def test_fit_line_filters_and_sorts(fake, scene_line):
    line = scene_line("Я совершенно не понимаю, о чём ты говоришь", dur=1.8)
    variants = [
        "Я вообще не понимаю, о чём ты сейчас говоришь",     # слишком длинный
        "Не понял, о чём ты.",                              # ~1.5
        "Ты о чём?",                                          # короткий
        "Не понимаю.",
        "Не понял, о чём ты.",                              # дубль
    ]
    p = fake({"variants": variants})
    out = await fit_line(line, p, n=3)
    assert "Я вообще не понимаю, о чём ты сейчас говоришь" not in out
    assert len(out) == len(set(out)) == 3
    durs = [abs(estimate_duration(v) - 1.8) for v in out]
    assert durs == sorted(durs)                               # ближе к длине — раньше
    assert all(estimate_duration(v) <= 1.8 * 1.1 for v in out)
    assert out[0] == "Не понял, о чём ты."
    assert p.calls[0]["schema"]["required"] == ["variants"] and "1.8" in p.calls[0]["user"]


@pytest.mark.asyncio
async def test_fit_line_second_attempt_when_all_too_long(fake, scene_line):
    line = scene_line("Длинная фраза про колбасу и кота", dur=1.2)
    p = fake({"variants": ["Очень длинная фраза про колбасу и кота и ещё много слов"]}, {"variants": ["Кот, колбаса!"]})
    out = await fit_line(line, p)
    assert out == ["Кот, колбаса!"] and len(p.calls) == 2
    assert "Прошлые варианты" in p.calls[1]["user"]


@pytest.mark.asyncio
async def test_fit_line_nothing_fits_returns_empty(fake, scene_line):
    p = fake({"variants": ["Слишком длинный вариант для такой короткой реплики совсем"]})
    assert await fit_line(scene_line("раз", dur=0.3), p) == []


@pytest.mark.asyncio
async def test_fit_line_degenerate(fake, scene_line):
    p = fake({"variants": ["x"]})
    assert await fit_line(scene_line("текст", dur=0.0), p) == []
    assert p.calls == []


# ─────────── name_speakers ───────────
def _dialogue():
    lines = [
        Line(id="l1", start=0, end=1, text="Барсик, ты где?", speaker_id="s1"),
        Line(id="l2", start=2, end=3, text="Я тут, Мила!", speaker_id="s2"),
        Line(id="l3", start=4, end=5, text="Кто это?", speaker_id="s3"),
    ]
    sp = [Speaker(id=f"s{i}", label=f"Голос {i}", color="#fff") for i in (1, 2, 3)]
    return lines, sp


@pytest.mark.asyncio
async def test_name_speakers_only_confident(fake):
    lines, sp = _dialogue()
    p = fake({"names": [
        {"speaker_id": "s2", "name": "Барсик", "confidence": 0.9},
        {"speaker_id": "s1", "name": "Мила", "confidence": 0.8},
        {"speaker_id": "s3", "name": "Вася", "confidence": 0.95},          # такого имени нет в диалоге — выдумка
    ]})
    assert await name_speakers(lines, sp, p) == {"s2": "Барсик", "s1": "Мила"}


@pytest.mark.asyncio
async def test_name_speakers_filters_low_conf_unknown_and_duplicates(fake):
    lines, sp = _dialogue()
    p = fake({"names": [
        {"speaker_id": "s2", "name": "Барсик", "confidence": 0.4},         # неуверенно
        {"speaker_id": "s9", "name": "Мила", "confidence": 0.9},           # нет такого говорящего
        {"speaker_id": "s1", "name": "Мила", "confidence": 0.9},
        {"speaker_id": "s3", "name": "Мила", "confidence": 0.9},           # дубль имени → оба отбрасываются
    ]})
    assert await name_speakers(lines, sp, p) == {}


@pytest.mark.asyncio
async def test_name_speakers_skips_named_and_empty(fake):
    lines, sp = _dialogue()
    sp[1].name = "Барсик"
    p = fake({"names": [{"speaker_id": "s2", "name": "Другой", "confidence": 0.99}]})
    assert await name_speakers(lines, sp, p) == {}
    p2 = fake({"names": []})
    assert await name_speakers([], sp, p2) == {} and p2.calls == []
