"""Оценка диаризации по разметке «кто говорит» на нескольких фильмах (ToS, Sintel, Cosmos Laundromat).

Разметка — `tests/e2e/data/*_speakers.json` (номер реплики в SRT → персонаж; `unsure` исключается из метрик).
Метрики на уровне реплик (взвешенных по числу реплик):
  * purity (чистота кластеров) — доля реплик, у которых персонаж совпал с самым частым персонажем их кластера;
  * coverage (охват персонажа) — доля реплик персонажа в его главном кластере; по каждому персонажу отдельно;
  * mixed — сколько кластеров смешивают ≥ 2 персонажей, у каждого из которых ≥ 20 % реплик кластера (по размеченным);
  * n_speakers — число найденных голосов против реального числа персонажей.
Запуск без pytest: `cd engine && .venv/bin/python tests/e2e/diar_eval.py [tos sintel cosmos]`.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers as H  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
DIAR_WORK = H.WORK / "diar"
COSMOS_MP4 = H.RAW / "cosmos_laundromat_1080p.mp4"
COSMOS_EN = H.RAW / "cosmos_en.srt"


@dataclass
class Film:
    key: str
    video: Path
    srt: Path | None            # None → субтитры из контейнера (дорожка eng)
    gt_file: str
    whisper_cache: Path | None = None   # готовый json whisper (если уже есть)


FILMS: dict[str, Film] = {
    "tos": Film("tos", H.TOS_MOV, H.TOS_EN, "tos_speakers.json", H.WORK / "analysis" / "whisper_large-v3-turbo.json"),
    "sintel": Film("sintel", H.SINTEL_MKV, None, "sintel_speakers.json"),
    "cosmos": Film("cosmos", COSMOS_MP4, COSMOS_EN, "cosmos_speakers.json"),
}


def load_gt(film: Film) -> tuple[dict[int, str], dict[int, str]]:
    """(номер реплики → персонаж без unsure, номер → уверенность)."""
    d = json.loads((DATA / film.gt_file).read_text("utf-8"))
    sp = {int(k): v for k, v in d["speakers"].items() if v != "unsure"}
    return sp, {int(k): v for k, v in d["confidence"].items()}


def available(film: Film) -> bool:
    return film.video.exists() and (film.srt is None or film.srt.exists())


def film_dir(film: Film) -> Path:
    d = DIAR_WORK / film.key
    d.mkdir(parents=True, exist_ok=True)
    return d


def audio_wav(film: Film, kind: str = "mix") -> Path:
    """mono 16 кГц: микс или центр 5.1 (если дорожка 5.1)."""
    from dubl_engine.media.extract import extract_audio, extract_center_channel
    from dubl_engine.media.probe import pick_original_audio, probe
    d = film_dir(film)
    out = d / f"{kind}_16k.wav"
    if out.exists():
        return out
    src = H.TOS_RELEASE if film.key == "tos" and H.TOS_RELEASE.exists() else film.video
    st = pick_original_audio(probe(src))
    if kind == "center" and st.channels > 2:
        extract_center_channel(src, st.index, out, sample_rate=16000, sample_fmt="s16")
    else:
        extract_audio(src, st.index, out, sample_rate=16000, mono=True, sample_fmt="s16")
    return out


def srt_lines(film: Film):
    from dubl_engine.analysis.subtitles import parse_subtitle_file
    from dubl_engine.media.extract import extract_subtitles
    from dubl_engine.media.probe import probe
    if film.srt is not None:
        return parse_subtitle_file(film.srt)
    st = next(s for s in probe(film.video).subtitle_streams if s.language in ("eng", "en"))
    return parse_subtitle_file(extract_subtitles(film.video, st.index, film_dir(film) / "subs_eng"))


def whisper_lines(film: Film):
    """whisper turbo (mlx) по миксу, с кэшем на диске — как в analyze_project."""
    from dubl_engine.models import Line
    cache = film.whisper_cache or film_dir(film) / "whisper_turbo.json"
    if cache.exists():
        return [Line(**d) for d in json.loads(cache.read_text("utf-8"))]
    from dubl_engine.analysis.transcribe import release_memory, transcribe
    lines, _ = transcribe(audio_wav(film, "mix"), language="en", model="large-v3-turbo")
    release_memory()
    cache.write_text(json.dumps([ln.model_dump() for ln in lines], ensure_ascii=False), "utf-8")
    return lines


def aligned_lines(film: Film):
    """Реплики как в продукте: SRT, выровненный по whisper."""
    from dubl_engine.analysis.align import align_embedded_to_speech
    return align_embedded_to_speech(srt_lines(film), whisper_lines(film))


# ───────────────────────── метрики ─────────────────────────
def score(out_lines, gt: dict[int, str], *, mixed_share: float = 0.2) -> dict:
    """out_lines — реплики после diarize (по порядку SRT, индекс i → номер i+1)."""
    pairs = [(gt[i], out_lines[i - 1].speaker_id) for i in sorted(gt) if i - 1 < len(out_lines)]
    by_cl: dict[str, list[str]] = {}
    by_gt: dict[str, list[str]] = {}
    for g, c in pairs:
        by_cl.setdefault(c, []).append(g)
        by_gt.setdefault(g, []).append(c)
    n = max(1, len(pairs))
    purity = sum(Counter(v).most_common(1)[0][1] for v in by_cl.values()) / n
    coverage = {g: Counter(v).most_common(1)[0][1] / len(v) for g, v in by_gt.items()}
    cov_w = sum(Counter(v).most_common(1)[0][1] for v in by_gt.values()) / n
    mixed = []
    for c, gs in by_cl.items():
        cnt = Counter(gs)
        big = [g for g, k in cnt.items() if k / len(gs) >= mixed_share]
        if len(big) >= 2:
            mixed.append((c, dict(cnt)))
    clusters_per_char = {g: len(set(v)) for g, v in by_gt.items()}
    all_cl = {ln.speaker_id for ln in out_lines}
    return {"n_found": len(all_cl), "n_real": len(by_gt), "lines": len(pairs),
            "purity": round(purity, 3), "coverage": round(cov_w, 3),
            "coverage_by_char": {g: round(v, 2) for g, v in sorted(coverage.items())},
            "chars": {g: dict(Counter(v)) for g, v in sorted(by_gt.items())},
            "clusters_per_char": clusters_per_char,
            "mixed": len(mixed), "mixed_detail": [{"cluster": c, **d} for c, d in mixed]}


def merge_precision(sugg, out_lines, gt: dict[int, str], top: int = 3) -> dict:
    """Доля верных пар среди топ-предложений: верно, если главные персонажи двух кластеров совпадают."""
    main: dict[str, str] = {}
    cl: dict[str, list[str]] = {}
    for i, g in gt.items():
        if i - 1 < len(out_lines):
            cl.setdefault(out_lines[i - 1].speaker_id, []).append(g)
    for c, gs in cl.items():
        main[c] = Counter(gs).most_common(1)[0][0]
    res = []
    for a, b, v in sugg[:top]:
        if a in main and b in main:
            res.append({"pair": f"{a}+{b}", "sim": v, "same": main[a] == main[b], "chars": f"{main[a]}/{main[b]}"})
    return {"n": len(res), "right": sum(r["same"] for r in res), "detail": res}


def run(keys: list[str], *, source: str = "mix", save: str | None = None, **kw) -> dict:
    from dubl_engine.analysis.diarize import diarize
    out: dict = {}
    for k in keys:
        film = FILMS[k]
        if not available(film):
            print(f"{k}: нет файлов — пропуск")
            continue
        gt, _ = load_gt(film)
        lines = aligned_lines(film)
        new, spk, sugg = diarize(audio_wav(film, source), lines, hf_token=None, **kw)
        s = score(new, gt)
        s["merge"] = merge_precision(sugg, new, gt)
        s["suggestions"] = sugg[:6]
        out[k] = s
        print(f"{k}: голосов {s['n_found']} (реально {s['n_real']}), purity {s['purity']}, coverage {s['coverage']}, "
              f"смешанных {s['mixed']}, по персонажам {s['coverage_by_char']}, кластеров на персонажа {s['clusters_per_char']}")
        for m in s["mixed_detail"]:
            print("    смесь:", m)
    if save:
        p = H.WORK / "bench" / save
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    src = "mix"
    if "--center" in args:
        args.remove("--center")
        src = "center"
    run(args or list(FILMS), source=src, save=f"diar_eval_{src}.json")
