from __future__ import annotations

import pytest

from dubl_engine.analysis.snap import dialogue_around, shot_around, snap_selection
from dubl_engine.models import Line, Word


def L(s, e, words=None):
    return Line(id="l", start=s, end=e, text="t", words=words or [])


LINES = [L(2, 4), L(4.5, 6), L(10, 12)]


def test_empty():
    assert snap_selection(1, 2, [], []) == (1, 2)
    assert dialogue_around(5, []) == (5, 5)
    assert shot_around(5, [], 20) == (0.0, 20)


def test_swapped():
    assert snap_selection(9, 8, [], []) == (8, 9)


def test_between_lines_untouched():
    assert snap_selection(7, 9, LINES, []) == (7, 9)


def test_inside_line_expands_with_pad():
    a, b = snap_selection(10.5, 11.5, LINES, [])
    assert a == pytest.approx(9.85) and b == pytest.approx(12.15)


def test_inside_line_does_not_run_into_neighbour():
    # между 4 и 4.5 меньше двух запасов: край не должен уходить за середину паузы
    a, b = snap_selection(3.0, 5.0, LINES, [])
    assert a == pytest.approx(1.85)
    assert b == pytest.approx(6.15)
    a2, _ = snap_selection(4.7, 5.5, LINES, [])
    assert a2 == pytest.approx(4.35) or a2 >= 4.25     # не раньше середины паузы 4.0–4.5


def test_edge_on_line_boundary_is_not_inside():
    a, b = snap_selection(12.0, 15.0, LINES, [])
    assert b == 15 and a == 12.0


def test_start_cannot_go_negative():
    a, _ = snap_selection(0.1, 0.5, [L(0, 1)], [])
    assert a == 0.0


def test_words_stick_out_of_line():
    ln = L(2, 4, [Word(start=1.9, end=2.3, text="a"), Word(start=3.5, end=4.3, text="b")])
    a, b = snap_selection(3, 3.5, [ln], [])
    assert a == pytest.approx(1.75) and b == pytest.approx(4.45)


def test_cut_snaps_edge_within_tolerance():
    a, b = snap_selection(7.0, 8.3, [], [7.3, 8.0])
    assert a == 7.3
    assert b == 8.0


def test_cut_outside_tolerance_ignored():
    assert snap_selection(7.0, 9.0, [], [7.7, 9.8]) == (7.0, 9.0)


def test_cut_picks_nearest():
    a, _ = snap_selection(7.0, 9.0, [], [6.6, 7.1])
    assert a == 7.1


def test_cut_near_expanded_edge():
    # левый край внутри реплики → 9.85; склейка на 9.5 в пределах 0.6 → прилипаем
    a, b = snap_selection(10.5, 11.5, LINES, [9.5, 12.5])
    assert a == 9.5 and b == 12.5


def test_cut_inside_line_is_ignored():
    # склейка 3.0 — посреди реплики 2–4: не прилипаем, иначе режем слово
    a, _ = snap_selection(3.2, 5.0, LINES, [3.0])
    assert a == pytest.approx(1.85)


def test_cuts_collapsing_selection_are_ignored():
    a, b = snap_selection(7.0, 7.2, [], [7.1])
    assert (a, b) == (7.0, 7.2)


def test_result_ordered():
    a, b = snap_selection(7.0, 7.0, LINES, [7.2])
    assert a <= b


# ---- dialogue_around ----
def test_dialogue_chain():
    lines = [L(1, 2), L(3, 4), L(5, 6), L(20, 21), L(22, 23)]
    assert dialogue_around(3.5, lines) == (1, 6)
    assert dialogue_around(21.5, lines) == (20, 23)


def test_dialogue_gap_boundary():
    lines = [L(0, 1), L(3.5, 4)]
    assert dialogue_around(0.5, lines, max_gap=2.5) == (0, 4)
    assert dialogue_around(0.5, lines, max_gap=2.4) == (0, 1)


def test_dialogue_t_in_pause_picks_nearest():
    lines = [L(1, 2), L(10, 11)]
    assert dialogue_around(2.5, lines) == (1, 2)
    assert dialogue_around(9.0, lines) == (10, 11)


def test_dialogue_t_far_from_everything():
    assert dialogue_around(6.0, [L(1, 2), L(10, 11)]) == (6.0, 6.0)


def test_dialogue_unsorted_input():
    lines = [L(5, 6), L(1, 2), L(3, 4)]
    assert dialogue_around(1.5, lines) == (1, 6)


def test_dialogue_overlapping_lines():
    lines = [L(0, 5), L(1, 2), L(6, 7)]
    assert dialogue_around(1.5, lines) == (0, 7)


# ---- shot_around ----
def test_shot_between_cuts():
    cuts = [5.0, 12.0, 30.0]
    assert shot_around(8, cuts, 40) == (5.0, 12.0)
    assert shot_around(2, cuts, 40) == (0.0, 5.0)
    assert shot_around(35, cuts, 40) == (30.0, 40)


def test_shot_on_cut_belongs_to_next():
    assert shot_around(5.0, [5.0, 12.0], 20) == (5.0, 12.0)


def test_shot_unsorted_and_out_of_range_cuts():
    assert shot_around(8, [12.0, 5.0, -1.0, 99.0, 0.0], 20) == (5.0, 12.0)


def test_shot_clamps_t():
    assert shot_around(-3, [5.0], 10) == (0.0, 5.0)
    assert shot_around(50, [5.0], 10) == (5.0, 10)


def test_shot_zero_duration():
    assert shot_around(1, [], 0) == (0.0, 0.0)
