"""Диаризация без токена на настоящих фильмах: Tears of Steel (7 персонажей), Sintel (2), Cosmos Laundromat (2).

Эталон — `tests/e2e/data/*_speakers.json`: кто говорит в каждой реплике SRT (по сценарию, смыслу реплик и кадрам;
`unsure` из метрик исключены). Метрики и подробности — `diar_eval.py`, итоговая таблица «было/стало» — docs/QUALITY.md.
Пороги — с запасом от замеров: параметры алгоритма подбирались по сумме трёх фильмов, а не по одному.
Реплики — как в продукте: субтитры, выровненные по whisper turbo.
"""
from __future__ import annotations

import pytest

import diar_eval as DE

pytestmark = pytest.mark.slow

NEED = {k: not DE.available(f) for k, f in DE.FILMS.items()}


def _skip_if_missing(key: str) -> None:
    if NEED[key]:
        pytest.skip(f"нет файлов фильма {key}: python3 samples/fetch.py")


_cache: dict = {}


def _diarize(key: str, src: str = "mix", **kw):
    """Один и тот же разбор переиспользуется между тестами модуля (диаризация быстрая, но whisper — нет)."""
    k = (key, src, tuple(sorted(kw.items())))
    if k not in _cache:
        from dubl_engine.analysis.diarize import diarize
        film = DE.FILMS[key]
        lines = DE.aligned_lines(film)
        _cache[k] = (diarize(DE.audio_wav(film, src), lines, hf_token=None, **kw), lines)
    return _cache[k]


def _score(key: str, src: str = "mix"):
    (new, speakers, sugg), _ = _diarize(key, src)
    gt, _conf = DE.load_gt(DE.FILMS[key])
    return DE.score(new, gt), new, speakers, sugg


def test_labeling_covers_every_line_and_is_consistent():
    """Разметка: для каждой реплики SRT есть запись; unsure не больше 10 %; число персонажей — как в фильме."""
    import json
    for key, film in DE.FILMS.items():
        d = json.loads((DE.DATA / film.gt_file).read_text("utf-8"))
        n = len(d["speakers"])
        assert sorted(map(int, d["speakers"])) == list(range(1, n + 1))
        assert set(d["confidence"]) == set(d["speakers"])
        unsure = [k for k, v in d["speakers"].items() if v == "unsure"]
        assert len(unsure) <= 0.1 * n and set(unsure) == set(d["unsure_guess"])
        if not NEED[key]:
            assert n == len(DE.srt_lines(film))


@pytest.mark.parametrize("src", ["mix", "center"])
def test_tos_clusters(src):
    """ToS, 7 персонажей (Том, старый Том, Селия, Капитан, Барли, техник, помощница). Замер (микс / центр 5.1): 8 / 7 голосов,
    чистота 0,85 / 0,82, охват 0,74 / 0,77; Селия 6/6 в одном кластере, Том 15/18 / 17/18, Капитан в 2 / 3 кластерах."""
    _skip_if_missing("tos")
    s, new, speakers, _ = _score("tos", src)
    assert 5 <= s["n_found"] <= 9                               # реально 7; не упёрлись ни в потолок, ни в одного
    assert s["purity"] >= 0.78 and s["coverage"] >= 0.70
    assert s["coverage_by_char"]["Celia"] >= 0.85
    assert s["coverage_by_char"]["Thom"] >= 0.80
    assert s["clusters_per_char"]["Captain"] <= 3
    assert s["mixed"] <= 2                                      # было 3 (Селия+Барли+помощница в одном «Голосе»)
    assert sum(sp.line_count for sp in speakers) == len(new)


def test_tos_main_characters_do_not_share_a_cluster():
    """Главные (Том, Селия, Капитан) никогда не лежат вместе в одном кластере с долей ≥ 20 % каждый — то, что нашёл тестировщик по кадрам."""
    _skip_if_missing("tos")
    for src in ("mix", "center"):
        s, *_ = _score("tos", src)
        for m in s["mixed_detail"]:
            big = [g for g in ("Thom", "Celia", "Captain") if m.get(g, 0) / sum(v for kk, v in m.items() if kk != "cluster") >= 0.2]
            assert len(big) <= 1, (src, m)


@pytest.mark.parametrize("key", ["sintel", "cosmos"])
def test_two_voice_films(key):
    """Sintel (Синтел и шаман) и Cosmos Laundromat (овца и продавец): ровно 2–3 голоса, ни одного смешанного кластера.
    Замер: 2 голоса, чистота 0,96, охват 0,96 (Sintel: шаман 10/11; Cosmos: Victor 38/40)."""
    _skip_if_missing(key)
    s, *_ = _score(key)
    assert 2 <= s["n_found"] <= 3
    assert s["mixed"] == 0 and s["purity"] >= 0.9 and s["coverage"] >= 0.88


@pytest.mark.parametrize("key", ["tos", "sintel", "cosmos"])
def test_diarize_is_deterministic(key):
    """Два разбора одного файла (и разное число потоков torch, и другой порядок реплик на входе) дают одни и те же голоса."""
    _skip_if_missing(key)
    import random

    import torch

    from dubl_engine.analysis.diarize import diarize
    film = DE.FILMS[key]
    wav = DE.audio_wav(film, "mix")
    lines = DE.aligned_lines(film)
    runs = []
    old = torch.get_num_threads()
    try:
        for threads, order in ((1, lines), (4, list(reversed(lines))), (old, random.Random(7).sample(lines, len(lines)))):
            torch.set_num_threads(threads)
            new, speakers, sugg = diarize(wav, order, hf_token=None)
            runs.append(({ln.id: ln.speaker_id for ln in new}, [(s.id, s.line_count, s.total_speech) for s in speakers], sugg))
    finally:
        torch.set_num_threads(old)
    assert runs[0] == runs[1] == runs[2]


def test_merge_suggestions_are_mostly_right():
    """Предлагаем слить пары, которые на разметке чаще всего правда один человек. Принудительно дробим фильмы мельче
    (порог 0,9 и 0,95), берём лучшее предложение: замер 15 из 16 верны (ToS, Sintel, Cosmos; микс и центр)."""
    for key in DE.FILMS:
        _skip_if_missing(key)
    n = right = 0
    for key in DE.FILMS:
        gt, _ = DE.load_gt(DE.FILMS[key])
        for thr in (0.9, 0.95):
            (new, _spk, sugg), _ = _diarize(key, "mix", cluster_threshold=thr)
            m = DE.merge_precision(sugg, new, gt, top=1)
            n += m["n"]
            right += m["right"]
    assert n >= 4 and right / n >= 0.75, (right, n)


def test_default_run_gives_no_false_merge_suggestions_on_known_films():
    """В обычном режиме на этих фильмах предлагать нечего или предложения верны: на ToS раньше предлагали слить Селию с помощницей."""
    for key in DE.FILMS:
        _skip_if_missing(key)
    for key in DE.FILMS:
        gt, _ = DE.load_gt(DE.FILMS[key])
        (new, _spk, sugg), _ = _diarize(key, "mix")
        m = DE.merge_precision(sugg, new, gt, top=3)
        assert m["right"] >= m["n"] - 1                     # допускается одна ошибка из показанных
        assert all(0.0 <= v <= 1.0 for _, _, v in sugg)
