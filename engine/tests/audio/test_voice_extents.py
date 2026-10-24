"""voice_extents: реплики расширяются до реальных границ речи — хвост фразы за таймингом субтитров тоже заменяется."""
import numpy as np

from dubl_engine.audio.separate import voice_extents

SR = 48000


def _speech(spans, dur=10.0, level=0.3):
    x = np.zeros(int(dur * SR), dtype=np.float32)
    rng = np.random.default_rng(0)
    for a, b in spans:
        i0, i1 = int(a * SR), int(b * SR)
        x[i0:i1] = level * rng.standard_normal(i1 - i0).astype(np.float32)
    x += 1e-5 * rng.standard_normal(len(x)).astype(np.float32)
    return x


def test_tail_after_subtitle_is_included():
    x = _speech([(1.0, 3.6)])                     # фраза кончается на 3.6, а субтитр — на 3.0
    (a, b), = voice_extents(x, SR, [(1.0, 3.0)])
    assert a == 1.0 and 3.5 <= b <= 3.7


def test_head_before_subtitle_is_included():
    x = _speech([(0.6, 3.0)])
    (a, b), = voice_extents(x, SR, [(1.0, 3.0)])
    assert 0.5 <= a <= 0.7 and b == 3.0


def test_does_not_grow_into_neighbour_or_across_long_pause():
    x = _speech([(1.0, 3.0), (3.2, 5.0), (7.0, 8.0)])
    ext = voice_extents(x, SR, [(1.0, 2.8), (3.3, 5.0)])
    assert ext[0][1] <= 3.3                        # не залезает на соседнюю реплику
    assert ext[1][1] < 6.0                         # через паузу 2 с к чужой речи не тянется


def test_silence_keeps_spans():
    x = np.zeros(SR * 4, dtype=np.float32)
    assert voice_extents(x, SR, [(1.0, 2.0)]) == [(1.0, 2.0)]
    assert voice_extents(x, SR, []) == []
