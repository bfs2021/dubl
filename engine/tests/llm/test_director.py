from __future__ import annotations

import re

import pytest

from dubl_engine.llm.director import (
    MixPatch,
    apply_patches,
    available_fx,
    available_rooms,
    director_command,
    director_feedback,
    rule_based_feedback,
    take_report,
)
from dubl_engine.models import MixMode, MixSettings, Scene, SceneLine, Speaker, Take, TrackSettings
from dubl_engine.util import EngineError


def mk_take(**kw) -> Take:
    base = dict(id="t001", line_id="l1", speaker_id="s1", file="takes/t001.wav", duration=2.0, preroll=1.0,
                start_rel=0.0, offset_ms=0.0, stretch=1.0, fit_confidence=0.85, peak_db=-6.0, loudness_lufs=-20.0,
                recognized_text="Кто трогал мою колбасу")
    base.update(kw)
    return Take(**base)


LINE = SceneLine(line_id="l1", speaker_id="s1", start_rel=0, end_rel=2, original="Who", text="Кто трогал мою колбасу?")


# ─────────── take_report ───────────
def test_report_all_good():
    r = take_report(mk_take(), LINE)
    assert r["ok"] and r["problems"] == [] and r["timing"] == "ok" and r["length"] == "ok"
    assert r["text_match"] == "ok" and r["loudness"] == "ok"


def test_report_timing_and_stretch():
    r = take_report(mk_take(offset_ms=-300), LINE)
    assert r["timing"] == "late" and "late" in r["problems"] and r["timing_level"] == "noticeable"
    r = take_report(mk_take(offset_ms=500), LINE)
    assert r["timing"] == "early" and r["timing_level"] == "big"
    r = take_report(mk_take(stretch=0.92), LINE)
    assert r["length"] == "too_long" and r["length_level"] == "noticeable"
    r = take_report(mk_take(stretch=1.12), LINE)
    assert r["length"] == "too_short"


def test_report_text_match():
    r = take_report(mk_take(recognized_text="Кто трогал колбасу"), LINE)
    assert "мою" in r["missing_words"] and r["text_match"] in ("ok", "partial")
    r = take_report(mk_take(recognized_text="совсем другие слова вообще"), LINE)
    assert r["text_match"] == "mismatch" and "text_mismatch" in r["problems"]
    r = take_report(mk_take(recognized_text=""), LINE)
    assert r["text_match"] == "silent"
    r = take_report(mk_take(recognized_text=None), LINE)
    assert r["text_match"] is None and r["ok"]


def test_report_loudness():
    assert take_report(mk_take(peak_db=-0.2), LINE)["loudness"] == "clipping"
    assert take_report(mk_take(peak_db=-40, loudness_lufs=-45), LINE)["loudness"] == "too_quiet"
    assert take_report(mk_take(peak_db=None, loudness_lufs=None), LINE)["loudness"] == "ok"
    # пик нормальный, а речь тихая (−35,5 LUFS при пике −15,8 — реальный дубль) — всё равно «тихо записано»
    r = take_report(mk_take(peak_db=-15.8, loudness_lufs=-35.5), LINE)
    assert r["loudness"] == "too_quiet" and "микрофон ближе" in rule_based_feedback(r)
    assert take_report(mk_take(peak_db=-8, loudness_lufs=-22), LINE)["loudness"] == "ok"


# ─────────── rule_based_feedback ───────────
def test_rule_feedback_praise_and_advice():
    good = rule_based_feedback(take_report(mk_take(), LINE))
    assert "Отличный" in good
    for kw, word in [(dict(offset_ms=-300), "отстаёшь"), (dict(offset_ms=300), "раньше губ"), (dict(peak_db=0.0), "перегружен"),
                     (dict(peak_db=-40, loudness_lufs=-45), "Тихо записано"), (dict(stretch=0.85), "длиннее"),
                     (dict(recognized_text="что-то иное совсем"), "Текст")]:
        text = rule_based_feedback(take_report(mk_take(**kw), LINE))
        assert word in text, (kw, text)
        assert not re.search(r"\d", text)                                 # без цифр


def test_rule_feedback_max_two_sentences_and_priority():
    r = take_report(mk_take(offset_ms=-600, peak_db=0.0, stretch=0.8, loudness_lufs=-20), LINE)
    text = rule_based_feedback(r)
    assert "перегружен" in text and text.count("—") <= 4
    assert len(re.findall(r"[.!?]", text)) <= 3


def test_rule_feedback_empty_report():
    assert rule_based_feedback({}) == rule_based_feedback({"problems": []})


# ─────────── director_feedback ───────────
@pytest.mark.asyncio
async def test_feedback_uses_llm_without_numbers(fake):
    r = take_report(mk_take(offset_ms=-320), LINE)
    p = fake("Ты слегка отстаёшь от губ — начни чуть раньше.")
    assert await director_feedback(r, p) == "Ты слегка отстаёшь от губ — начни чуть раньше."
    assert not re.search(r"\d", p.calls[0]["user"])                       # модели цифры не отдаём


@pytest.mark.asyncio
async def test_feedback_fallbacks(fake):
    r = take_report(mk_take(offset_ms=-320), LINE)
    rb = rule_based_feedback(r)
    assert await director_feedback(r, None) == rb                          # без ИИ
    assert await director_feedback(r, fake(EngineError("нет сети"))) == rb  # ошибка модели
    assert await director_feedback(r, fake("Опоздание 320 мс!")) == rb      # модель назвала цифры
    assert await director_feedback(r, fake("")) == rb
    good = take_report(mk_take(), LINE)
    p = fake("не должно вызываться")
    assert "Отличный" in await director_feedback(good, p) and p.calls == []  # всё хорошо — без вызова ИИ


# ─────────── честная похвала: губы не проверялись (слепой тест 7) ───────────
def test_file_take_praise_does_not_claim_lips():
    """Дубль загружен файлом: подгонка к губам не проверялась — «попал в губы» не пишем, просим проверить на превью."""
    r = take_report(mk_take(source="file", offset_ms=-900, stretch=0.8), LINE)
    assert r["ok"] and r["source"] == "file" and r["lips_checked"] is False
    text = rule_based_feedback(r)
    assert text == "Дубль загружен файлом: текст и громкость на месте. Попадание в губы проверь на превью."
    assert "попал в губы" not in text.lower()
    no_text = rule_based_feedback(take_report(mk_take(source="file", recognized_text=None), LINE))
    assert "загружен файлом" in no_text and "текст" not in no_text and "попал" not in no_text


@pytest.mark.parametrize("conf", [None, 0.2, 0.39])
def test_unsure_fit_praise_does_not_claim_lips(conf):
    r = take_report(mk_take(fit_confidence=conf), LINE)
    assert r["ok"] and r["lips_checked"] is False
    text = rule_based_feedback(r)
    assert "попал в губы" not in text.lower() and "не уверен" in text and "превью" in text


def test_sure_mic_take_still_praised_for_lips():
    r = take_report(mk_take(fit_confidence=0.4), LINE)
    assert r["lips_checked"] is True
    assert rule_based_feedback(r).startswith("Отличный дубль: попал в губы")


def test_file_take_problem_advice_is_about_file():
    r = take_report(mk_take(source="file", recognized_text=""), LINE)
    assert "файле" in rule_based_feedback(r) and "микрофон" not in rule_based_feedback(r)


@pytest.mark.asyncio
async def test_llm_knows_lips_were_not_checked(fake):
    """Путь через ИИ: модели прямо сказано, что губы не проверялись; для файла сдвиг/длину ей не отдаём; если модель
    всё равно похвалила за губы — берём совет по правилам."""
    import json

    r = take_report(mk_take(source="file", peak_db=-40, loudness_lufs=-45, offset_ms=-800), LINE)
    assert r["problems"] == ["too_quiet"]
    p = fake("Тихо, подними громкость — и проверь губы на превью.")
    assert await director_feedback(r, p) == "Тихо, подними громкость — и проверь губы на превью."
    facts = json.loads(p.calls[0]["user"].split("\n", 1)[1])
    assert facts["source"] == "file" and facts["lips_checked"] is False
    assert "timing" not in facts and "length" not in facts
    assert "lips_checked=false" in p.calls[0]["system"]
    bad = fake("Отлично попал в губы, только тихо — подойди ближе.")
    assert await director_feedback(r, bad) == rule_based_feedback(r)
    mic = take_report(mk_take(fit_confidence=0.1, offset_ms=-320), LINE)
    p2 = fake("Начни чуть раньше.")
    await director_feedback(mic, p2)
    assert json.loads(p2.calls[0]["user"].split("\n", 1)[1])["lips_checked"] is False


# ─────────── MixPatch / apply_patches ───────────
def test_apply_patches_basic():
    mix = MixSettings()
    out = apply_patches(mix, [
        MixPatch(target="background", field="gain_db", value=-5),
        MixPatch(target="s1", field="room", value="cave"),
        MixPatch(target="s1", field="voice_fx", value="robot"),
        MixPatch(target="s1", field="mute", value=True),
        MixPatch(target="master", field="duck_background", value=False),
        MixPatch(target="master", field="target_lufs", value=-16),
        MixPatch(target="master", field="mode", value="voiceover"),
    ])
    assert out.background.gain_db == -5 and out.tracks["s1"].room == "cave" and out.tracks["s1"].voice_fx == "robot"
    assert out.tracks["s1"].mute is True and out.duck_background is False and out.target_lufs == -16
    assert out.mode == MixMode.VOICEOVER and out.autopilot is False
    assert mix.background.gain_db == 0 and mix.tracks == {} and mix.autopilot is True      # исходник не тронут


def test_apply_patches_clamp_and_reject():
    out = apply_patches(MixSettings(), [
        MixPatch(target="s1", field="gain_db", value=99),
        MixPatch(target="background", field="gain_db", value=-99),
        MixPatch(target="s1", field="room", value="moon_base"),          # нет такой комнаты
        MixPatch(target="s2", field="voice_fx", value="laser"),
        MixPatch(target="master", field="gain_db", value=3),              # у master нет громкости
        MixPatch(target="background", field="room", value="cave"),
        MixPatch(target="master", field="target_lufs", value=0),
    ])
    assert out.tracks["s1"].gain_db == 12 and out.background.gain_db == -30
    assert "s2" not in out.tracks and out.tracks["s1"].room == "auto"
    assert out.target_lufs == -6


def test_apply_patches_nothing_valid_keeps_autopilot():
    out = apply_patches(MixSettings(), [MixPatch(target="master", field="mode", value="karaoke")])
    assert out.autopilot is True and out == MixSettings()


def test_apply_patches_room_fx_mute_keep_autopilot():
    """Комнату, эффект и «выкл.» автопилот сам не трогает — значит, и выключаться ему незачем."""
    out = apply_patches(MixSettings(), [
        MixPatch(target="s1", field="room", value="street"),
        MixPatch(target="s1", field="voice_fx", value="robot"),
        MixPatch(target="s2", field="mute", value=True),
    ])
    assert out.autopilot is True and out.tracks["s1"].room == "street" and out.tracks["s2"].mute is True
    for p in (MixPatch(target="s1", field="gain_db", value=2), MixPatch(target="master", field="target_lufs", value=-16),
              MixPatch(target="master", field="duck_background", value=False),
              MixPatch(target="master", field="mode", value="voiceover")):
        assert apply_patches(MixSettings(), [p]).autopilot is False, p


def test_rooms_and_fx_have_required_keys():
    assert {"dry", "room", "kitchen", "hall", "street", "cave", "car", "bathroom", "phone", "behind_door"} <= set(available_rooms())
    assert {"none", "robot", "chipmunk", "giant", "villain", "radio", "phone", "ghost"} <= set(available_fx())


# ─────────── director_command ───────────
def scene(**mix_kw) -> Scene:
    lines = [SceneLine(line_id="l1", speaker_id="s1", start_rel=0, end_rel=2, original="a", text="б"),
             SceneLine(line_id="l2", speaker_id="s2", start_rel=3, end_rel=4, original="c", text="г")]
    return Scene(id="sc001", title="Кухня", start=0, end=10, dubbed_speakers=["s1", "s2"], lines=lines,
                 mix=MixSettings(**mix_kw))


SPEAKERS = [Speaker(id="s1", label="Голос 1", name="Мила", color="#fff"),
            Speaker(id="s2", label="Голос 2", name="Барсик", color="#000")]


@pytest.mark.asyncio
async def test_command_relative_gain(fake):
    sc = scene(tracks={"s2": TrackSettings(gain_db=-2.0)})
    p = fake({"patches": [{"target": "s2", "field": "gain_db", "value": 3, "relative": True}]})
    out = await director_command("кот громче", sc, p, speakers=SPEAKERS)
    assert out == [MixPatch(target="s2", field="gain_db", value=1.0, note="Барсик: громкость +1 дБ")]       # −2 + 3
    assert "Барсик" in p.calls[0]["user"] + p.calls[0]["system"]
    assert "cave" in p.calls[0]["system"] and "robot" in p.calls[0]["system"]
    assert p.calls[0]["schema"]["required"] == ["patches"]


@pytest.mark.asyncio
async def test_command_by_name_and_clamp_and_relative_chain(fake):
    p = fake({"patches": [
        {"target": "Барсик", "field": "gain_db", "value": 6, "relative": True},
        {"target": "барсик", "field": "gain_db", "value": 10, "relative": True},     # накапливается, но зажато
        {"target": "background", "field": "gain_db", "value": -8, "relative": False},
    ]})
    out = await director_command("громче кота", scene(tracks={"s2": TrackSettings(gain_db=0)}), p, speakers=SPEAKERS)
    by = {(x.target, x.field): x.value for x in out}
    assert by[("s2", "gain_db")] == 12 and by[("background", "gain_db")] == -8
    assert len(out) == 2                                                    # повтор одной цели схлопнут


@pytest.mark.asyncio
async def test_command_room_for_everyone(fake):
    p = fake({"patches": [{"target": "s1", "field": "room", "value": "cave"}, {"target": "s2", "field": "room", "value": "Cave"}]})
    out = await director_command("как в пещере", scene(), p)
    assert [(x.target, x.value) for x in out] == [("s1", "cave"), ("s2", "cave")]


@pytest.mark.asyncio
async def test_command_drops_invented(fake):
    p = fake({"patches": [
        {"target": "s7", "field": "gain_db", "value": 3, "relative": True},        # нет такого персонажа
        {"target": "s1", "field": "room", "value": "moon_base"},
        {"target": "s1", "field": "volume", "value": 3},                            # нет такого поля
        {"target": "master", "field": "gain_db", "value": 3},
        {"target": "s1", "field": "gain_db", "value": "громко"},
        {"target": "s1", "field": "voice_fx", "value": "ghost"},                    # единственный годный
        "мусор",
        {"target": "background", "field": "mute", "value": "да"},
    ]})
    out = await director_command("сделай что-нибудь", scene(), p)
    assert {(x.target, x.field, x.value) for x in out} == {("s1", "voice_fx", "ghost"), ("background", "mute", True)}


@pytest.mark.asyncio
async def test_command_empty_and_unrelated(fake):
    p = fake({"patches": []})
    assert await director_command("спой песню", scene(), p) == []
    assert await director_command("  ", scene(), p) == [] and len(p.calls) == 1


@pytest.mark.asyncio
async def test_command_needs_provider_and_bad_format(fake):
    with pytest.raises(EngineError, match="подключи модель"):
        await director_command("громче", scene(), None)
    with pytest.raises(EngineError):
        await director_command("громче", scene(), fake("не json"))
    with pytest.raises(EngineError, match="формате"):
        await director_command("громче", scene(), fake({"patches": "много"}))


@pytest.mark.asyncio
async def test_command_then_apply_end_to_end(fake):
    sc = scene()
    p = fake({"patches": [{"target": "background", "field": "gain_db", "value": -3, "relative": True},
                          {"target": "master", "field": "target_lufs", "value": -12}]})
    patches = await director_command("музыку потише и в целом погромче", sc, p)
    mix = apply_patches(sc.mix, patches)
    assert mix.background.gain_db == -3 and mix.target_lufs == -12
