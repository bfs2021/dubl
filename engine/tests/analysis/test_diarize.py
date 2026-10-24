from __future__ import annotations

import shutil
import subprocess
import sys

import numpy as np
import pytest

from dubl_engine.analysis import diarize as D
from dubl_engine.models import Line


def _synthetic(centers: list[np.ndarray], per: list[int], noise: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    emb, truth = [], []
    for k, (c, n) in enumerate(zip(centers, per)):
        for _ in range(n):
            emb.append(c + rng.normal(0, noise, c.shape))
            truth.append(k)
    return np.array(emb), np.array(truth)


def _basis(dim: int, idx: int) -> np.ndarray:
    v = np.zeros(dim)
    v[idx] = 1.0
    return v


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    return len({(int(x), int(y)) for x, y in zip(a, b)}) == len(set(a.tolist())) == len(set(b.tolist()))


def test_cluster_finds_number_of_speakers_automatically():
    for k in (1, 2, 3, 4):
        emb, truth = _synthetic([_basis(16, i) for i in range(k)], [6] * k)
        lab = D.cluster_embeddings(emb)
        assert len(set(lab.tolist())) == k
        assert _same_partition(lab, truth)


def test_cluster_with_shared_background_component():
    """ECAPA на речи поверх музыки: у всех реплик общая составляющая (сходство 0,8+). Без вычитания среднего порог ничего не режет."""
    rng = np.random.default_rng(1)
    common = rng.normal(0, 1, 64) * 3.0
    centers = [rng.normal(0, 1, 64) for _ in range(5)]
    emb = np.array([common + centers[k] + rng.normal(0, 0.35, 64) for k in range(5) for _ in range(8)])
    truth = np.repeat(np.arange(5), 8)
    lab = D.cluster_embeddings(emb)
    assert len(set(lab.tolist())) == 5
    assert _same_partition(lab, truth)


def test_number_of_speakers_not_stuck_at_old_cap():
    emb, truth = _synthetic([_basis(64, i) for i in range(12)], [10] * 12, noise=0.05, seed=2)
    lab = D.cluster_embeddings(emb)
    assert len(set(lab.tolist())) == 12 and _same_partition(lab, truth)      # раньше всегда ровно 10


def test_small_clusters_absorbed():
    emb, truth = _synthetic([_basis(16, i) for i in range(3)], [12, 12, 12], noise=0.05)
    odd = np.array([emb[0] + 0.0])
    emb2 = np.vstack([emb, odd * 0.2 + _basis(16, 7) * 0.8])             # одинокая реплика ни на кого не похожа
    lab = D.cluster_embeddings(emb2, center=False, threshold=0.6, min_size=2)
    assert len(set(lab.tolist())) == 3 and len(lab) == 37
    assert D.absorb_small_clusters(emb2, np.arange(37) % 1, 3).tolist() == [0] * 37


def test_min_cluster_size_grows_with_film():
    assert D.min_cluster_size(5) == 1 and D.min_cluster_size(20) == 2 and D.min_cluster_size(200) == 3


def test_centered_merge_suggestions_are_in_0_1_range():
    ls = [Line(id=f"l{i}", start=i, end=i + 1, text="") for i in range(4)]
    c = {0: np.array([1.0, 0.0]), 1: np.array([0.9, 0.44]), 2: np.array([-1.0, 0.0])}
    new, spk, sugg = D._build_result(ls, [0, 1, 2, 0], c, centered=True)
    assert sugg and all(0.0 <= v <= 1.0 for _, _, v in sugg)
    assert all(v >= D.MERGE_SIMILARITY_CENTERED for _, _, v in sugg)


def test_cluster_edge_cases():
    assert D.cluster_embeddings(np.zeros((0, 4))).tolist() == []
    assert D.cluster_embeddings(np.ones((1, 4))).tolist() == [0]


def test_cluster_max_clusters():
    emb, _ = _synthetic([_basis(16, i) for i in range(6)], [3] * 6)
    assert len(set(D.cluster_embeddings(emb, max_clusters=3).tolist())) == 3


def test_assign_short_to_nearest():
    emb, truth = _synthetic([_basis(8, 0), _basis(8, 1)], [5, 5])
    cents = D.centroids(emb, truth)
    q = np.array([_basis(8, 1) * 2 + 0.05, _basis(8, 0) * 0.3 + 0.02])
    assert D.assign_to_nearest(q, cents).tolist() == [1, 0]


def test_overlap_detection():
    ls = [Line(id="l1", start=0, end=2, text=""), Line(id="l2", start=1.5, end=3, text=""),
          Line(id="l3", start=3, end=4, text=""), Line(id="l4", start=3.5, end=4.5, text="")]
    # l1/l2 — разные говорящие и пересекаются; l3/l4 — один говорящий, не считается
    assert D.find_overlaps(ls, ["a", "b", "c", "c"]) == [True, True, False, False]
    assert D.find_overlaps(ls, [None, "b", "c", "c"]) == [False, False, False, False]


def test_merge_suggestions():
    c = np.array([[1.0, 0.0], [0.95, 0.31], [0.0, 1.0]])
    got = D.merge_suggestions_from(c, ["s1", "s2", "s3"])
    assert [(a, b) for a, b, _ in got] == [("s1", "s2")]
    assert 0.8 < got[0][2] <= 1.0


def test_build_result_sorted_by_speech_and_labels():
    ls = [Line(id="l1", start=0, end=1, text=""), Line(id="l2", start=1, end=5, text=""),
          Line(id="l3", start=5, end=6, text=""), Line(id="l4", start=6, end=8, text="")]
    new, spk, sugg = D._build_result(ls, [0, 1, 0, 1], None)
    assert [s.id for s in spk] == ["s1", "s2"]
    assert spk[0].total_speech == 6.0 and spk[0].line_count == 2  # метка 1 говорит больше → s1
    assert [l.speaker_id for l in new] == ["s2", "s1", "s2", "s1"]
    assert [s.label for s in spk] == ["Голос 1", "Голос 2"]
    assert spk[0].color != spk[1].color and spk[0].color.startswith("#")
    assert sugg == []
    assert not any(l.overlap for l in new)


def test_diarize_empty():
    assert D.diarize("nope.wav", [], hf_token=None) == ([], [], [])


@pytest.fixture
def shared_models(monkeypatch):
    import os

    from platformdirs import user_data_dir
    monkeypatch.setenv("DUBL_DATA_DIR", os.environ.get("DUBL_TEST_DATA_DIR") or user_data_dir("Dubl", "Dubl"))


def _voices_available() -> list[str]:
    if sys.platform != "darwin" or not shutil.which("say"):
        return []
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=False).stdout
    names = [l.split("  ")[0].strip() for l in out.splitlines()]
    return [n for n in ("Milena", "Samantha", "Daniel", "Yuri", "Alex") if n in names]


@pytest.mark.slow
@pytest.mark.skipif(len(_voices_available()) < 2, reason="нужны две голосовые модели macOS `say`")
def test_diarize_two_real_voices(tmp_path, shared_models):
    import soundfile as sf
    voices = _voices_available()[:2]
    phrases = ["Привет, как твои дела сегодня?", "Мы пойдём гулять в парк после обеда.",
               "Я думаю, что это отличная идея для всех."]
    parts, lines, truth, t = [], [], [], 0.0
    sr = 16000
    gap = np.zeros(int(0.5 * sr), dtype="float32")
    for k in range(6):
        v = voices[k % 2]
        txt = phrases[(k // 2) % 3] if v == "Milena" else ["Hello there, how are you doing today?",
                                                          "We will go for a walk in the park later.",
                                                          "I think this is a great idea for everybody."][(k // 2) % 3]
        aiff, wav = tmp_path / f"{k}.aiff", tmp_path / f"{k}.wav"
        subprocess.run(["say", "-v", v, txt, "-o", str(aiff)], check=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1",
                        str(wav)], check=True)
        x, _ = sf.read(str(wav), dtype="float32")
        lines.append(Line(id=f"l{k + 1:04d}", start=t, end=t + len(x) / sr, text=txt))
        parts += [x, gap]
        truth.append(v)
        t += len(x) / sr + 0.5
    mix = tmp_path / "mix.wav"
    sf.write(str(mix), np.concatenate(parts), sr)
    progress: list[float] = []
    new, spk, _sugg = D.diarize(mix, lines, hf_token=None, on_progress=progress.append)
    assert len(spk) == 2
    assert all(l.speaker_id for l in new)
    ids = [l.speaker_id for l in new]
    for v in voices:  # одинаковый голос → одинаковый speaker_id
        assert len({i for i, tv in zip(ids, truth) if tv == v}) == 1
    assert len(set(ids)) == 2
    assert sum(s.line_count for s in spk) == 6
    assert progress and progress[-1] == 1.0


# ───────────────────────── число голосов по силуэту, перекрытия, детерминизм ─────────────────────────
def _scene(k: int, per: int, noise: float = 0.35, dim: int = 64, seed: int = 3):
    """k «людей» с общей фоновой составляющей (как ECAPA на речи поверх музыки), реплики идут по очереди."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 1, dim) * 3.0
    centers = [rng.normal(0, 1, dim) for _ in range(k)]
    emb, truth = [], []
    for j in range(per):
        for c in range(k):
            emb.append(common + centers[c] + rng.normal(0, noise, dim))
            truth.append(c)
    return np.array(emb), np.array(truth)


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_auto_number_of_speakers_without_threshold(k):
    emb, truth = _scene(k, 10)
    lab = D.cluster_embeddings(emb)             # порога нет: число голосов выбирает силуэт
    assert len(set(lab.tolist())) == k and _same_partition(lab, truth)


def test_single_speaker_is_not_split():
    rng = np.random.default_rng(5)
    emb = rng.normal(0, 1, 192) * 3 + rng.normal(0, 1.0, (40, 192))     # размерность ECAPA
    assert len(set(D.cluster_embeddings(emb).tolist())) == 1


def test_overlapping_lines_are_not_merged():
    """Две реплики звучат одновременно → это разные люди, даже если эмбеддинги почти совпали."""
    emb, truth = _scene(2, 12, noise=0.2)
    emb[1] = emb[0] + 1e-3                                 # реплики 0 и 1 — «разные люди» по разметке, но эмбеддинги неотличимы
    spans = np.array([[i * 3.0, i * 3.0 + 2.0] for i in range(len(emb))])
    spans[1] = spans[0]                                    # звучат в одно время
    lab = D.cluster_embeddings(emb, spans=spans, threshold=1.0, min_size=1)
    assert lab[0] != lab[1]
    lab_no = D.cluster_embeddings(emb, threshold=1.0, min_size=1)
    assert lab_no[0] == lab_no[1]


def test_result_does_not_depend_on_row_order():
    emb, _ = _scene(4, 10)
    perm = np.random.default_rng(0).permutation(len(emb))
    a = D.cluster_embeddings(emb)
    b = D.cluster_embeddings(emb[perm])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    assert _same_partition(a, b[inv])


def test_cluster_is_deterministic_and_float_noise_safe():
    emb, _ = _scene(3, 15)
    a = D.cluster_embeddings(emb)
    for _ in range(3):
        assert D.cluster_embeddings(emb).tolist() == a.tolist()
    noisy = emb + np.random.default_rng(1).normal(0, 1e-9, emb.shape)     # шум float32 от числа потоков
    assert D.cluster_embeddings(noisy).tolist() == a.tolist()


def test_max_clusters_for_scales_with_lines():
    assert D.max_clusters_for(10) == 6 and D.max_clusters_for(90) == 10 and D.max_clusters_for(2000) == D.MAX_SPEAKERS


def test_merge_suggestions_only_mutual_nearest():
    c = np.array([[1.0, 0.0], [0.9, 0.44], [0.6, 0.8]])      # 0↔1 взаимные соседи; 2 ближе к 1, но 1 ближе к 0
    got = D.merge_suggestions_from(c, ["a", "b", "c"], min_sim=0.0, mutual=True)
    assert [(x, y) for x, y, _ in got] == [("a", "b")]
    got_all = D.merge_suggestions_from(c, ["a", "b", "c"], min_sim=0.0)
    assert len(got_all) == 3 and [(x, y) for x, y, _ in got_all][0] == ("a", "b")


def test_diarize_sorts_lines_with_id_tiebreak(monkeypatch):
    """Одинаковые start/end не должны зависеть от порядка во входе."""
    seen: list[list[str]] = []

    def fake(audio, lines, on_progress, thr=None):
        seen.append([l.id for l in lines])
        return lines, [], []
    monkeypatch.setattr(D, "_diarize_embeddings", fake)
    ls = [Line(id=i, start=1.0, end=2.0, text="") for i in ("b", "a", "c")]
    D.diarize("x.wav", ls, hf_token=None)
    D.diarize("x.wav", list(reversed(ls)), hf_token=None)
    assert seen[0] == seen[1] == ["a", "b", "c"]
