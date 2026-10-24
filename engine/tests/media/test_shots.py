from __future__ import annotations

import pytest

from dubl_engine.media.shots import detect_shot_cuts, parse_showinfo_times
from dubl_engine.util import EngineError


def test_detect_cut_at_5s(rich_mkv):
    cuts = detect_shot_cuts(rich_mkv)
    assert len(cuts) == 1
    assert cuts[0] == pytest.approx(5.0, abs=0.1)


def test_no_cuts_in_simple(rich_mkv):
    # диапазон целиком внутри синего плана
    assert detect_shot_cuts(rich_mkv, start=6.0, end=9.0) == []


def test_range_keeps_absolute_time(rich_mkv):
    cuts = detect_shot_cuts(rich_mkv, start=3.0, end=8.0)
    assert len(cuts) == 1 and cuts[0] == pytest.approx(5.0, abs=0.1)


def test_range_excluding_cut(rich_mkv):
    assert detect_shot_cuts(rich_mkv, end=4.0) == []


def test_sorted(rich_mkv):
    cuts = detect_shot_cuts(rich_mkv, threshold=0.1)
    assert cuts == sorted(cuts)


def test_bad_args(rich_mkv):
    with pytest.raises(EngineError):
        detect_shot_cuts(rich_mkv, threshold=1.5)
    with pytest.raises(EngineError):
        detect_shot_cuts(rich_mkv, start=5, end=1)


def test_missing_file(tmp_path):
    with pytest.raises(EngineError):
        detect_shot_cuts(tmp_path / "nope.mp4")


def test_parse_showinfo():
    err = ("[Parsed_showinfo_1 @ 0x1] n:   0 pts: 125 pts_time:5     pos: 1 fmt:yuv420p\n"
           "[Parsed_showinfo_1 @ 0x1] n:   1 pts: 300 pts_time:12.52 pos: 1 fmt:yuv420p\n"
           "frame=  1 fps=0.0 pts_time:99\n")
    assert parse_showinfo_times(err) == [5.0, 12.52]
