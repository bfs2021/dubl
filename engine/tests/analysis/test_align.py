from __future__ import annotations

import pytest

from dubl_engine.analysis.align import align_embedded_to_speech
from dubl_engine.models import Line, Word


def emb(i, s, e, text, sid=None):
    return Line(id=f"l{i:04d}", start=s, end=e, text=text, speaker_id=sid, source="embedded")


def sp(s, e, text, words=None):
    return Line(id="w", start=s, end=e, text=text, source="whisper", words=words or [])


def test_empty_inputs():
    assert align_embedded_to_speech([], [sp(0, 1, "a")]) == []
    lines = [emb(1, 1, 2, "Hello")]
    assert align_embedded_to_speech(lines, []) == lines


def test_same_text_shifted_follows_words():
    lines = [emb(1, 10.0, 12.0, "Who touched my sausage?", "s1")]
    speech = [sp(10.4, 12.3, "Who touched my sausage",
                 [Word(start=10.5, end=10.8, text="Who"), Word(start=12.0, end=12.2, text="sausage")])]
    out = align_embedded_to_speech(lines, speech)
    assert out[0].start == pytest.approx(10.5) and out[0].end == pytest.approx(12.2)
    assert out[0].text == lines[0].text and out[0].speaker_id == "s1" and out[0].id == "l0001"
    assert out[0].source == "embedded"
    assert lines[0].start == 10.0   # исходные не тронуты


def test_text_match_uses_neighbours_not_just_overlap():
    # встроенная строка сдвинута на 2.5 с (больше 1 с) — но текст нашёлся рядом
    lines = [emb(1, 5.0, 7.0, "I was asleep, honestly.")]
    speech = [sp(0, 1, "Something else entirely"), sp(7.5, 9.0, "I was asleep honestly")]
    out = align_embedded_to_speech(lines, speech)
    assert (out[0].start, out[0].end) == (7.5, 9.0)


def test_text_match_merges_split_speech():
    lines = [emb(1, 1.0, 5.0, "I will not do that, you know it well.")]
    speech = [sp(1.2, 2.8, "I will not do that"), sp(3.0, 4.6, "you know it well"), sp(9, 10, "unrelated")]
    out = align_embedded_to_speech(lines, speech)
    assert (out[0].start, out[0].end) == (1.2, 4.6)


def test_different_language_time_only_clamped():
    lines = [emb(1, 10.0, 12.0, "Кто трогал мою колбасу?")]
    speech = [sp(11.8, 14.5, "Who touched my sausage?")]
    out = align_embedded_to_speech(lines, speech)
    assert out[0].start == pytest.approx(10.79, abs=1e-6)       # сдвиг ограничен (< 0,8 с)
    assert out[0].end <= 12.8
    assert out[0].start > 10.0 and out[0].end > 12.0
    assert out[0].text == "Кто трогал мою колбасу?"


def test_different_language_small_shift():
    lines = [emb(1, 10.0, 12.0, "Кто трогал мою колбасу?")]
    speech = [sp(10.3, 11.7, "Who touched my sausage?")]
    out = align_embedded_to_speech(lines, speech)
    assert (out[0].start, out[0].end) == (10.3, 11.7)


def test_same_language_dissimilar_text_time_only():
    lines = [emb(1, 10.0, 12.0, "Completely different subtitle text here")]
    speech = [sp(10.6, 12.9, "banana pyjamas orbit")]
    out = align_embedded_to_speech(lines, speech)
    assert 10.0 <= out[0].start <= 11.0 and out[0].end <= 13.0


def test_no_overlap_unchanged():
    lines = [emb(1, 20.0, 22.0, "Nothing here")]
    out = align_embedded_to_speech(lines, [sp(0, 3, "far away"), sp(40, 42, "also far")])
    assert (out[0].start, out[0].end) == (20.0, 22.0)


def test_never_shifts_more_than_1s_without_text():
    lines = [emb(1, 10.0, 12.0, "Привет")]
    out = align_embedded_to_speech(lines, [sp(11.4, 20.0, "Hello there my friend")])
    assert out[0].start <= 11.0 and out[0].end <= 13.0


def test_preserves_order_and_count():
    lines = [emb(1, 2, 4, "One"), emb(2, 6, 8, "Two"), emb(3, 30, 31, "Lost")]
    speech = [sp(2.1, 4.1, "One"), sp(6.2, 8.2, "Two")]
    out = align_embedded_to_speech(lines, speech)
    assert [o.id for o in out] == ["l0001", "l0002", "l0003"]
    assert (out[2].start, out[2].end) == (30, 31)
    assert out[0].start == pytest.approx(2.1) and out[1].end == pytest.approx(8.2)


def test_minimum_duration_and_non_negative():
    lines = [emb(1, 0.0, 0.5, "Hi")]
    out = align_embedded_to_speech(lines, [sp(0.0, 0.5, "Hi", [Word(start=0.2, end=0.2, text="Hi")])])
    assert out[0].start >= 0 and out[0].end - out[0].start >= 0.1 - 1e-9


def _words(text, start, per=0.3):
    return [Word(start=start + i * per, end=start + (i + 1) * per - 0.02, text=" " + w) for i, w in enumerate(text.split())]


def test_neighbours_in_one_whisper_segment_are_not_collapsed():
    """Баг 1: три коротких реплики попали в один сегмент whisper — раньше все получали один интервал на весь сегмент."""
    lines = [emb(1, 100.0, 102.0, "We have to follow our passions."), emb(2, 103.0, 105.0, "You have your robotics..."),
             emb(3, 106.0, 109.0, "...and I just want to be awesome in space.")]
    ws = (_words("we have to follow our passions", 100.4) + _words("you have your robotics", 103.4)
          + _words("and I just want to be awesome in space", 106.4))
    speech = [sp(100.4, 109.0, "we have to follow our passions you have your robotics and I just want to be awesome in space", ws)]
    out = align_embedded_to_speech(lines, speech)
    iv = [(o.start, o.end) for o in out]
    assert len(set(iv)) == 3
    assert all(b[0] >= a[1] - 1e-6 for a, b in zip(iv, iv[1:]))
    assert abs(out[0].start - 100.4) < 0.1 and abs(out[1].start - 103.4) < 0.1 and abs(out[2].start - 106.4) < 0.1
    assert out[0].end < 103.0 and out[1].end < 106.0


def test_no_new_overlaps_and_monotonic_on_messy_input():
    import random
    rnd = random.Random(7)
    texts = ["I know", "You forgot me on earth", "I know", "I should just crush you", "Wait", "Go go go", "No way", "Okay then"]
    t, lines, speech = 10.0, [], []
    for i, tx in enumerate(texts):
        lines.append(emb(i + 1, t, t + 1.0, tx))
        st = t + rnd.uniform(-0.9, 0.9)
        speech.append(sp(st, st + 1.2, tx.lower(), _words(tx.lower(), st, 0.2)))
        t += 1.3
    out = align_embedded_to_speech(lines, speech)
    assert [o.id for o in out] == [ln.id for ln in lines]
    assert all(b.start >= a.start - 1e-9 for a, b in zip(out, out[1:]))
    assert all(b.start >= a.end - 1e-6 or (lines[i + 1].start < lines[i].end) for i, (a, b) in enumerate(zip(out, out[1:])))
    assert all(abs(o.start - ln.start) <= 3.0 and o.end - o.start >= 0.1 for o, ln in zip(out, lines))


def test_weak_evidence_keeps_srt_timing():
    lines = [emb(1, 20.0, 22.0, "A completely different sentence about birds")]
    speech = [sp(21.9, 23.0, "unrelated words", _words("unrelated words", 21.9))]
    out = align_embedded_to_speech(lines, speech)
    assert out[0].start >= 20.0 and out[0].start <= 21.0 and out[0].end - out[0].start < 4


def test_original_overlap_in_srt_is_preserved_not_made_worse():
    lines = [emb(1, 10.0, 12.0, "First line here"), emb(2, 11.5, 13.0, "Second line there")]
    speech = [sp(10.0, 12.0, "first line here", _words("first line here", 10.0, 0.6)),
              sp(11.6, 13.0, "second line there", _words("second line there", 11.6, 0.45))]
    out = align_embedded_to_speech(lines, speech)
    assert out[1].start >= out[0].start
    assert out[0].end - out[1].start <= 0.5 + 0.2          # перекрытие не больше исходного
