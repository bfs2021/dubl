"""Распознавание речи: язык по речи, выбор нейросети (Whisper / GigaAM), нарезка для GigaAM, добор пропусков Whisper'ом.

Быстрые тесты — без нейросетей (поддельные результаты); медленные (@slow) — настоящие GigaAM и Whisper.
"""
from __future__ import annotations

import itertools
import shutil
import subprocess
import sys

import numpy as np
import pytest

from dubl_engine.analysis import gigaam, speech
from dubl_engine.analysis import transcribe as T
from dubl_engine.models import Line, Word


def _w(text, s, e, p=None):
    return Word(start=s, end=e, text=" " + text, prob=p)


def _line(text, s, e, p):
    toks = text.split()
    d = (e - s) / len(toks)
    return Line(id="x", start=s, end=e, text=text,
                words=[_w(t, s + i * d, s + (i + 1) * d, p) for i, t in enumerate(toks)])


def _seg(words, lp=-0.3):
    return T.Seg(words[0].start, words[-1].end, "".join(w.text for w in words), words, avg_logprob=lp,
                 no_speech_prob=0.0, compression_ratio=1.1)


# ───────────── язык по речи ─────────────
def test_choose_language_trusts_speech_over_track_label():
    # «Мегамозг»: дорожка помечена eng, а в окнах — русский с уверенностью 0,83–0,998
    probs = [{"ru": 0.998, "en": 0.001}, {"ru": 0.97, "en": 0.02}, {"ru": 0.83, "en": 0.15}]
    assert T.choose_language(probs, "en") == ("ru", "speech")


def test_choose_language_sums_windows():
    # одно окно (песня) «английское», два — русские: по сумме русский
    probs = [{"en": 0.9, "ru": 0.1}, {"ru": 0.8, "en": 0.2}, {"ru": 0.7, "en": 0.3}]
    assert T.choose_language(probs, None) == ("ru", "speech")


def test_choose_language_low_confidence_uses_track_hint():
    probs = [{"en": 0.35, "de": 0.3, "nl": 0.25}]
    assert T.choose_language(probs, "de") == ("de", "track")          # неуверенно и метка правдоподобна — метка
    assert T.choose_language(probs, "ja") == ("en", "speech")         # метка неправдоподобна — речь
    assert T.choose_language(probs, None) == ("en", "speech")


def test_choose_language_without_windows():
    assert T.choose_language([], "en") == ("en", "track")
    assert T.choose_language([], None) == (None, "speech")


def test_speech_windows_pick_loud_places():
    db = np.full(int(300 / T.ENV_HOP), -70.0)                        # 5 мин тишины…
    db[int(100 / T.ENV_HOP):int(140 / T.ENV_HOP)] = -20.0           # …речь на 100–140 с
    db[int(200 / T.ENV_HOP):int(220 / T.ENV_HOP)] = -20.0           # …и на 200–220 с
    wins = T.speech_windows(T.Envelope(db), 300.0)
    assert 1 <= len(wins) <= 3
    assert all(b - a == pytest.approx(30.0) for a, b in wins)
    assert any(a >= 95 and b <= 145 for a, b in wins)                 # окно внутри речи 100–140
    assert all(not (b <= 95 or a >= 225) for a, b in wins)            # чистую тишину не слушаем
    for (a1, b1), (a2, _b2) in itertools.pairwise(wins):
        assert a2 >= b1                                               # окна не пересекаются


def test_speech_windows_short_file():
    assert T.speech_windows(None, 12.0) == [(0.0, 12.0)]
    assert T.speech_windows(T.Envelope(np.full(100, -20.0)), 5.0) == [(0.0, 5.0)]


def test_language_note_only_when_speech_overrides_label():
    note = speech.language_note("ru", "en", "speech")
    assert note and "английский" in note and "слышно русский" in note and "разбери заново" in note
    assert speech.language_note("ru", "ru", "speech") is None
    assert speech.language_note("ru", "en", "user") is None          # выбор человека — без спора
    assert speech.language_note("ru", None, "speech") is None        # метки нет — не с чем спорить


# ───────────── выбор нейросети ─────────────
@pytest.mark.parametrize(("setting", "lang", "ok", "want", "note"), [
    ("auto", "ru", True, "gigaam", False),
    ("auto", "en", True, "whisper", False),
    ("auto", "ru", False, "whisper", False),           # GigaAM нет — тихо Whisper (по умолчанию так и было)
    ("whisper", "ru", True, "whisper", False),
    ("gigaam", "ru", True, "gigaam", False),
    ("gigaam", "en", True, "whisper", True),           # «GigaAM знает только русский — взял Whisper»
    ("gigaam", "ru", False, "whisper", True),
    ("gigaam", None, True, "whisper", True),
    ("nonsense", "ru", True, "gigaam", False),         # неизвестное значение — как auto
])
def test_pick_engine(setting, lang, ok, want, note):
    eng, why = speech.pick_engine(setting, lang, gigaam_ok=ok)
    assert eng == want and bool(why) == note
    if setting == "gigaam" and lang == "en":
        assert "только русский" in why


# ───────────── нарезка для GigaAM ─────────────
def test_energy_chunks_cover_everything_and_are_short():
    rng = np.random.default_rng(0)
    dur = 125.0
    db = -20 + 3 * rng.standard_normal(int(dur / T.ENV_HOP))
    quiet = [37.0, 51.5, 70.2]
    for q in quiet:
        db[int(q / T.ENV_HOP):int((q + 0.5) / T.ENV_HOP)] = -80.0
    ch = gigaam.energy_chunks(db, dur, hop=T.ENV_HOP)
    assert ch[0][0] == 0.0 and ch[-1][1] == dur
    assert all(b - a <= gigaam.CHUNK_MAX_S + 1e-6 for a, b in ch)      # модель берёт ≤ 25 с, режем ≤ 20
    assert all(x[1] == y[0] for x, y in itertools.pairwise(ch))               # без дыр и нахлёстов: ничего не теряем
    cuts = [b for _, b in ch[:-1]]
    for q in quiet:                                                    # тихие места в зоне поиска — там и режем
        near = [c for c in cuts if abs(c - (q + 0.25)) < 0.3]
        assert near or all(not (a + gigaam.CHUNK_MAX_S - gigaam.CHUNK_SEARCH_S <= q <= a + gigaam.CHUNK_MAX_S)
                           for a, _ in ch)


def test_energy_chunks_short_audio_is_one_piece():
    assert gigaam.energy_chunks(np.full(100, -20.0), 5.0, hop=0.05) == [(0.0, 5.0)]


def test_tokens_to_words_space_tokens_and_probs():
    toks = ["▁При", "вет", " ", "!", " ", "▁как", "▁дела"]
    ts = [0.0, 0.08, 0.16, 0.2, 0.3, 0.4, 0.6]
    lps = [np.log(0.9), np.log(0.7), 0.0, np.log(0.5), 0.0, np.log(1.0), np.log(0.8)]
    ws = gigaam.tokens_to_words(toks, ts, lps, offset=10.0)
    assert [w.text.strip() for w in ws] == ["Привет", "!", "как", "дела"]
    assert ws[0].start == 10.0 and ws[0].end == pytest.approx(10.12)
    assert ws[0].prob == pytest.approx(np.exp((np.log(0.9) + np.log(0.7)) / 2), abs=1e-3)


def test_group_words_to_lines():
    ws = [_w("Привет!", 0.0, 0.4), _w("Как", 0.6, 0.7), _w("дела?", 0.75, 1.0),      # конец предложения + пауза 0,2
          _w("—", 1.05, 1.06), _w("Нормально.", 1.1, 1.6),                          # тире — смена говорящего
          _w("А", 2.5, 2.6), _w("ты", 2.65, 2.8)]                                     # пауза 0,9
    groups = gigaam.group_words(ws)
    texts = [" ".join(w.text.strip() for w in g) for g in groups]
    assert texts == ["Привет!", "Как дела?", "Нормально.", "А ты"]


# ───────────── добор пропусков (поддельные результаты) ─────────────
def test_merge_fill_replaces_doubtful_line_with_confident_whisper():
    lines = [_line("И ещё мегамост!", 10.0, 11.0, 0.95),
             _line("Дай моу, дай боу!", 14.0, 16.2, 0.52),
             _line("Игра окончена!", 17.0, 18.0, 0.97)]
    heard = [((4.0, 26.0), [_seg([_w("Тайм-аут!", 13.7, 14.5, 0.95), _w("Тайм-аут!", 14.8, 15.3, 0.97),
                                  _w("Тайм-аут!", 15.5, 16.0, 0.99)]),
                            _seg([_w("Игра", 16.7, 17.1, 1.0), _w("окончена!", 17.2, 17.8, 1.0)])])]
    out, rep, add = speech.merge_fill(lines, heard)
    assert rep == 1 and add == 0
    assert [ln.text for ln in out] == ["И ещё мегамост!", "Тайм-аут! Тайм-аут! Тайм-аут!", "Игра окончена!"]
    assert out[1].start == 14.0 and out[1].end == 16.2                  # границы GigaAM остаются
    assert all(14.0 <= w.start <= w.end <= 16.2 for w in out[1].words)
    assert [ln.id for ln in out] == ["l0001", "l0002", "l0003"]


def test_merge_fill_keeps_line_when_whisper_not_better_or_not_russian():
    lines = [_line("Уботый!", 20.0, 20.8, 0.66)]
    weak = [((10.0, 30.0), [_seg([_w("Уботы!", 19.8, 20.3, 0.5)])])]
    assert speech.merge_fill(lines, weak)[0][0].text == "Уботый!"            # Whisper не увереннее — не меняем
    english = [((10.0, 30.0), [_seg([_w("Welcome", 19.8, 20.2, 0.95), _w("jungle", 20.3, 20.7, 0.95)])])]
    assert speech.merge_fill(lines, english)[0][0].text == "Уботый!"         # песня по-английски — не реплика
    caption = [((10.0, 30.0), [_seg([_w("АПЛОДИСМЕНТЫ", 19.9, 20.6, 0.95)])])]
    assert speech.merge_fill(lines, caption)[0][0].text == "Уботый!"         # пометка из субтитров


def test_merge_fill_ignores_window_edge_and_neighbour_words():
    lines = [_line("Блокирую!", 10.0, 10.7, 0.78), _line("Ука!", 10.9, 11.4, 0.7), _line("Блокирую!", 11.6, 12.4, 0.98)]
    # реплика у самого края окна — Whisper слышит её плохо, не берём
    edge = [((10.5, 30.0), [_seg([_w("Укол!", 10.7, 11.2, 0.98)])])]
    assert speech.merge_fill(lines, edge)[1] == 0
    # слова соседних реплик не уходят в чужую; «Ука!» → «Укол!»
    ok = [((0.0, 25.0), [_seg([_w("Блокирую!", 9.8, 10.5, 0.95), _w("Укол!", 10.7, 11.2, 0.98),
                               _w("Блокирую!", 11.4, 12.2, 1.0)])])]
    out, rep, _ = speech.merge_fill(lines, ok)
    assert [ln.text for ln in out] == ["Блокирую!", "Укол!", "Блокирую!"] and rep == 1


def test_merge_fill_restores_lost_start_of_shout():
    # GigaAM слышит «гард!» (конец «Ангард!»): Whisper даёт слово целиком — начало реплики сдвигается к нему
    lines = [_line("гард!", 478.36, 478.68, 0.75)]
    heard = [((465.0, 495.0), [_seg([_w("Ангард!", 477.42, 478.54, 0.84)])])]
    out, rep, _ = speech.merge_fill(lines, heard)
    assert rep == 1 and out[0].text == "Ангард!"
    assert 477.5 < out[0].start < 478.36 and out[0].end == 478.68


def test_merge_fill_gaps_adds_only_confident_russian():
    lines = [_line("Привет.", 1.0, 1.5, 0.95)]
    heard = [((0.0, 20.0), [_seg([_w("Тайм-аут!", 8.0, 8.6, 0.93)]),
                            _seg([_w("Субтитры", 12.0, 12.5, 0.9), _w("сделал", 12.5, 12.9, 0.9),
                                  _w("DimaTorzok", 12.9, 13.5, 0.9)]),
                            _seg([_w("I", 15.0, 15.1, 0.9), _w("wanna", 15.1, 15.4, 0.9)])])]
    gaps = [(7.9, 8.8), (11.9, 13.6), (14.9, 15.5)]
    # выдумки убирает фильтр приложения до слияния (как в fill_with_whisper)
    heard = [(heard[0][0], T.drop_hallucinations(heard[0][1], T.Envelope(np.full(400, -20.0)), loops=False))]
    out, _rep, add = speech.merge_fill(lines, heard, gaps=gaps)
    assert add == 1 and [ln.text for ln in out] == ["Привет.", "Тайм-аут!"]


def test_find_gaps_and_windows():
    db = np.full(int(60 / T.ENV_HOP), -70.0)
    db[int(10 / T.ENV_HOP):int(11 / T.ENV_HOP)] = -20.0          # громко 1 с, слов нет → пропуск
    db[int(20 / T.ENV_HOP):int(22 / T.ENV_HOP)] = -20.0          # громко, но тут реплика GigaAM
    db[int(30 / T.ENV_HOP):int(30.2 / T.ENV_HOP)] = -20.0        # 0,2 с — щелчок, не речь
    db[int(40 / T.ENV_HOP):int(52 / T.ENV_HOP)] = -20.0          # 12 с — музыка, не выкрик
    env = T.Envelope(db)
    gaps = speech.find_gaps(env, [_line("Да.", 20.2, 21.8, 0.9)], 60.0)
    assert len(gaps) == 1 and gaps[0][0] == pytest.approx(10.0, abs=0.1)
    wins = speech.fill_windows([(10.0, 11.0), (12.0, 12.5), (50.0, 51.0)], 60.0, ctx=8.0)
    assert wins == [(2.0, 20.5), (42.0, 59.0)]                  # соседние места — в одном окне
    assert all(b - a <= speech.FILL_WIN_MAX_S for a, b in wins)


# ───────────── запасной путь ─────────────
def _wav(tmp_path, sec=3.0):
    import soundfile as sf
    p = tmp_path / "a.wav"
    t = np.arange(int(sec * 16000)) / 16000
    sf.write(str(p), (0.2 * np.sin(2 * np.pi * 300 * t)).astype("float32"), 16000)
    return p


def _fake_whisper(monkeypatch, lang_probs, calls):
    monkeypatch.setattr(speech, "detect_language_probs", lambda *a, **k: lang_probs)

    def ws(path, *, language, **k):
        calls.append(("whisper", language))
        return [_seg([_w("hello", 0.5, 0.9, 0.9)])], language or "en"
    monkeypatch.setattr(speech, "whisper_segments", ws)


def test_auto_russian_without_gigaam_falls_back_to_whisper(monkeypatch, tmp_path):
    calls: list = []
    _fake_whisper(monkeypatch, [{"ru": 0.99}], calls)
    monkeypatch.setattr(gigaam, "installed", lambda: False)
    info: dict = {}
    _lines, lang = T.transcribe(_wav(tmp_path), language=None, engine="auto", track_language="en", info=info)
    assert lang == "ru" and calls == [("whisper", "ru")]                # язык речи, а не метка «en»
    assert info["asr_engine"] == "whisper" and info["language_source"] == "speech" and info["notes"] == []


def test_gigaam_crash_falls_back_to_whisper_with_note(monkeypatch, tmp_path):
    calls: list = []
    notes: list = []
    _fake_whisper(monkeypatch, [{"ru": 0.99}], calls)
    monkeypatch.setattr(gigaam, "installed", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("модель не скачалась")
    monkeypatch.setattr(speech, "gigaam_lines", boom)
    info: dict = {}
    _lines, _lang = T.transcribe(_wav(tmp_path), engine="gigaam", on_note=notes.append, info=info)
    assert calls == [("whisper", "ru")] and info["asr_engine"] == "whisper"
    assert notes == ["GigaAM не сработал — распознал Whisper."] and info["notes"] == notes


def test_gigaam_setting_for_english_uses_whisper(monkeypatch, tmp_path):
    calls: list = []
    notes: list = []
    _fake_whisper(monkeypatch, [{"en": 0.97}], calls)
    monkeypatch.setattr(gigaam, "installed", lambda: True)
    monkeypatch.setattr(speech, "gigaam_lines", lambda *a, **k: pytest.fail("GigaAM не для английского"))
    _lines, lang = T.transcribe(_wav(tmp_path), engine="gigaam", on_note=notes.append)
    assert lang == "en" and calls == [("whisper", "en")] and "только русский" in notes[0]


def test_gigaam_path_then_fill(monkeypatch, tmp_path):
    monkeypatch.setattr(speech, "detect_language_probs", lambda *a, **k: [{"ru": 0.99}])
    monkeypatch.setattr(gigaam, "installed", lambda: True)
    monkeypatch.setattr(speech, "gigaam_lines", lambda *a, **k: [_line("Дай боу!", 1.0, 1.6, 0.5)])
    monkeypatch.setattr(speech, "whisper_window",
                        lambda x, *, offset, **k: [_seg([_w("Тайм-аут!", 0.9, 1.5, 0.95)])])
    monkeypatch.setattr(speech, "whisper_segments", lambda *a, **k: pytest.fail("полный Whisper не нужен"))
    info: dict = {}
    lines, lang = T.transcribe(_wav(tmp_path), engine="auto", info=info)
    assert lang == "ru" and info["asr_engine"] == "gigaam" and info["filled"] == 1
    assert [ln.text for ln in lines] == ["Тайм-аут!"] and lines[0].start == 1.0


def test_chosen_language_is_law(monkeypatch, tmp_path):
    monkeypatch.setattr(speech, "detect_language_probs", lambda *a, **k: pytest.fail("язык выбран человеком"))
    monkeypatch.setattr(gigaam, "installed", lambda: False)
    calls: list = []
    monkeypatch.setattr(speech, "whisper_segments",
                        lambda path, *, language, **k: (calls.append(language) or ([], language)))
    info: dict = {}
    _lines, lang = T.transcribe(_wav(tmp_path), language="en", engine="auto", track_language="ru", info=info)
    assert lang == "en" and calls == ["en"] and info["language_source"] == "user"


def test_whisper_engine_with_chosen_language_skips_lid(monkeypatch, tmp_path):
    """Старый путь (язык выбран, Whisper) не трогает новый код: дубли и прочее работают как раньше."""
    import dubl_engine.analysis.speech as sp
    monkeypatch.setattr(sp, "recognize", lambda *a, **k: pytest.fail("не нужно"))
    monkeypatch.setattr(T, "_transcribe_faster", lambda path, **k: ([_seg([_w("hi", 0.1, 0.4, 0.9)])], "en"))
    monkeypatch.setattr(T, "pick_backend", lambda *a, **k: "faster")
    lines, lang = T.transcribe(_wav(tmp_path), language="en")
    assert lang == "en" and [ln.text for ln in lines] == ["hi"]


# ───────────── настоящие нейросети ─────────────
def _say(tmp_path, voice, text):
    aiff, wav = tmp_path / "x.aiff", tmp_path / "x.wav"
    subprocess.run(["say", "-v", voice, text, "-o", str(aiff)], check=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1",
                    str(wav)], check=True)
    return wav


@pytest.fixture
def shared_models(monkeypatch):
    import os

    from platformdirs import user_data_dir
    monkeypatch.setenv("DUBL_DATA_DIR", os.environ.get("DUBL_TEST_DATA_DIR") or user_data_dir("Dubl", "Dubl"))


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("say"), reason="нужна macOS `say`")
def test_gigaam_real_russian(tmp_path, shared_models):
    pytest.importorskip("onnx_asr")
    wav = _say(tmp_path, "Milena", "Привет, это проверка. Ты меня слышишь?")
    info: dict = {}
    lines, lang = T.transcribe(wav, language="ru", engine="gigaam", model="tiny", info=info)
    assert lang == "ru" and info["asr_engine"] == "gigaam"
    text = " ".join(ln.text for ln in lines).lower()
    assert "привет" in text and "провер" in text
    assert all(ln.words and ln.start <= ln.end for ln in lines)
    assert all(w.prob is not None for ln in lines for w in ln.words)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("say"), reason="нужна macOS `say`")
def test_language_by_speech_real(tmp_path, shared_models):
    wav = _say(tmp_path, "Milena", "Это русская речь, хотя у дорожки написано, что она английская.")
    info: dict = {}
    _lines, lang = T.transcribe(wav, language=None, engine="whisper", model="tiny", track_language="en", info=info)
    assert lang == "ru" and info["language_source"] == "speech"
