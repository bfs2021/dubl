"""Метрики бенчмарка ASR по эталону reference.json.

  .venv/bin/python score.py [--ref reference.json] [--hyp-dir hyp] [--audio-dir DIR] [--out scores.md]
  (то же через переменные окружения DUBL_BENCH_REF / DUBL_BENCH_HYP / DUBL_BENCH_AUDIO)

reference.json:
{"sets": {"<set>": {"lang": "ru", "tags": ["mm_full", "mm_voices"],      # tag = имя входного wav без .wav
                     "full_file": {"tag": "mm_full", "voice_env": "audio/mm_voices.wav"},   # для выдумок по всему файлу
                     "pieces": [{"id": "p1", "start": 0, "end": 60, "title": "...",
                                 "nonverbal": [[a, b], ...],     # смех/крики без слов — гипотезу там игнорируем
                                 "lines": [{"start": 0.0, "end": 2.8, "text": "...", "conf": "high|mid|low", "note": ""}]}]}}}
Правка эталона: меняй text/start/end в reference.json и перезапусти score.py — всё пересчитается.
Гипотезы: hyp/hyp_<model>_<tag>.json (см. bench_common.py).

Метрики на кусках (нормализация: регистр, ё→е, пунктуация, дефис склеивает слова):
  WER, CER — по всему тексту куска (слова гипотезы берутся по середине слова внутри [start, end] куска);
  lost — реплика эталона, у которой в окне [start-0.7, end+0.7] найдено < 50 % её слов;
  fab_w — слова гипотезы внутри куска, не попавшие ни в одно окно реплики эталона ±1 с (текст там, где речи нет);
  first_short — реплики, начинающиеся со служебного слова (SHORT_FIRST: в, а, и, о, но, ну, ах…), где его нет у гипотезы в [start-0.7, start+0.8];
  start/end err — для найденной реплики: расстояние до ближайшего начала (конца) реплики гипотезы в пределах 2 с,
                  mean / p90 / средний сдвиг со знаком (− раньше эталона) в мс — только по эталонным границам с
                  bound=energy|manual (не зависят от моделей); «> 300 мс» — сколько границ ошиблись больше чем на 300 мс;
                  «нет границы» — если ближайшей нет в 2 с (обычно реплика слита с соседней);
  merged — реплики гипотезы, перекрывающие ≥ 2 реплик эталона (каждую ≥ 30 % её длины);
  split — реплики эталона, разрезанные на ≥ 2 реплики гипотезы;
  punct — доля реплик гипотезы с . , ! ? …
"""
from __future__ import annotations

import argparse
import os
import json
import re
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HYP = Path(os.environ.get("DUBL_BENCH_HYP", ROOT / "hyp"))   # папка гипотез; можно задать --hyp-dir
AUDIO = Path(os.environ.get("DUBL_BENCH_AUDIO", ROOT))           # от неё считаются пути «audio/…» в reference.json
# короткие служебные слова в начале реплики (предлоги, союзы, частицы, междометия) — их Whisper любит терять
SHORT_FIRST = {"в", "во", "а", "и", "о", "но", "да", "ну", "ах", "ох", "эй", "к", "ко", "с", "со", "у", "не", "ни", "же",
               "вот", "ой", "из", "на", "по", "за", "от", "до",
               "a", "an", "and", "but", "oh", "so", "the", "to", "in", "on", "i", "we", "it", "of"}


def norm_words(t: str) -> list[str]:
    t = t.lower().replace("ё", "е")
    t = re.sub(r"(?<=\w)-(?=\w)", "", t)            # супер-злодеев → суперзлодеев, тайм-аут → таймаут
    t = re.sub(r"\b(супер|тайм) (злоде\w*|аут)\b", r"\1\2", t)   # «супер злодеев», «тайм аут» — то же слово
    t = re.sub(r"[^\w\s']", " ", t)
    return t.split()


def edit_distance(a, b) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m]


def hyp_words(hyp: dict) -> list[dict]:
    """Слова гипотезы с таймингами; если у сегмента нет слов — раскладываем текст равномерно по сегменту."""
    out = []
    for s in hyp["segments"]:
        ws = []
        for w in s.get("words") or []:                 # «Тайм» + «-аут!» (так режет слова Whisper) → одно слово
            if ws and w["text"].strip().startswith("-") and len(w["text"].strip()) > 1:
                ws[-1] = {**ws[-1], "text": ws[-1]["text"].strip() + w["text"].strip(), "end": w["end"]}
            else:
                ws.append(w)
        if ws:
            for w in ws:
                for k, nw in enumerate(norm_words(w["text"])):
                    out.append({"start": w["start"], "end": w["end"], "w": nw, "raw": w["text"]})
        else:
            nws = norm_words(s["text"])
            d = (s["end"] - s["start"]) / max(1, len(nws))
            for k, nw in enumerate(nws):
                out.append({"start": s["start"] + k * d, "end": s["start"] + (k + 1) * d, "w": nw, "raw": nw})
    return out


def mid(w):
    return (w["start"] + w["end"]) / 2


def in_win(w, a, b):
    return a <= mid(w) <= b


def score_piece(hyp: dict, piece: dict) -> dict:
    a, b = piece["start"], piece["end"]
    lines = piece["lines"]
    nv = piece.get("nonverbal", [])          # смех/крики/вздохи: слова гипотезы там не считаются ни ошибкой, ни выдумкой
    hw = [w for w in hyp_words(hyp) if in_win(w, a, b) and not any(x <= mid(w) <= y for x, y in nv)]
    hsegs = [s for s in hyp["segments"] if s["end"] > a and s["start"] < b and norm_words(s["text"])]
    ref_w = [w for ln in lines for w in norm_words(ln["text"])]
    hyp_w = norm_words(" ".join(w["w"] for w in sorted(hw, key=lambda w: w["start"])))   # склейка «супер злодеев»
    wer_e = edit_distance(ref_w, hyp_w)
    rc, hc = " ".join(ref_w), " ".join(hyp_w)
    cer_e = edit_distance(rc, hc)
    lost, first_short_tot, first_short_lost = [], 0, []
    st_err, en_err, no_start, no_end = [], [], 0, 0
    st_sgn, en_sgn = [], []
    split = 0
    for ln in lines:
        rw = norm_words(ln["text"])
        win = [w for w in hw if in_win(w, ln["start"] - 0.7, ln["end"] + 0.7)]
        bag = [w["w"] for w in win]
        hit = 0
        tmp = list(bag)
        for x in rw:
            if x in tmp:
                hit += 1
                tmp.remove(x)
        found = hit >= 0.5 * len(rw)
        if not found:
            lost.append(ln["text"])
            continue
        if rw[0] in SHORT_FIRST:
            first_short_tot += 1
            early = [w["w"] for w in hw if ln["start"] - 0.7 <= mid(w) <= ln["start"] + 0.8]
            if rw[0] not in early:
                first_short_lost.append(ln["text"])
        starts = [s["start"] for s in hsegs]
        ends = [s["end"] for s in hsegs]
        sds = min((s - ln["start"] for s in starts), key=abs, default=99)   # со знаком: < 0 — гипотеза раньше
        sde = min((e - ln["end"] for e in ends), key=abs, default=99)
        ds, de = abs(sds), abs(sde)
        # границы считаем только там, где эталонная граница независима от моделей (энергия стема / ручная);
        # если у реплики нет поля bound — считаем всегда
        b = ln.get("bound", {})
        ok_s = not b or b.get("start") in ("energy", "manual")
        ok_e = not b or b.get("end") in ("energy", "manual")
        if ds <= 2.0:
            if ok_s:
                st_err.append(ds)
                st_sgn.append(sds)
        else:
            no_start += 1
        if de <= 2.0:
            if ok_e:
                en_err.append(de)
                en_sgn.append(sde)
        else:
            no_end += 1
        ov = [s for s in hsegs if min(s["end"], ln["end"]) - max(s["start"], ln["start"]) > 0.2]
        if len(ov) >= 2:
            split += 1
    merged = 0
    for s in hsegs:
        n = 0
        for ln in lines:
            o = min(s["end"], ln["end"]) - max(s["start"], ln["start"])
            if o > 0.3 * max(0.2, ln["end"] - ln["start"]):
                n += 1
        if n >= 2:
            merged += 1
    fab = [w for w in hw if not any(in_win(w, ln["start"] - 1.0, ln["end"] + 1.0) for ln in lines)]
    punct = np.mean([bool(re.search(r"[.,!?…]", s["text"])) for s in hsegs]) if hsegs else 0.0
    ms = lambda v, q: round(1000 * float(np.percentile(v, q) if q else np.mean(v)), 0) if v else None
    return {"n_ref_words": len(ref_w), "wer_err": wer_e, "cer_err": cer_e, "n_ref_chars": len(rc),
            "n_lines": len(lines), "lost": lost, "fab_words": [w["raw"] for w in fab],
            "first_short_tot": first_short_tot, "first_short_lost": first_short_lost,
            "start_err": st_err, "end_err": en_err, "start_sgn": st_sgn, "end_sgn": en_sgn, "no_start": no_start, "no_end": no_end,
            "merged": merged, "split": split, "punct": float(punct), "n_hyp_lines": len(hsegs),
            "start_mean_ms": ms(st_err, 0), "start_p90_ms": ms(st_err, 90),
            "end_mean_ms": ms(en_err, 0), "end_p90_ms": ms(en_err, 90)}


def agg(rows: list[dict]) -> dict:
    s = lambda k: sum(r[k] for r in rows)
    se = [x for r in rows for x in r["start_err"]]
    ee = [x for r in rows for x in r["end_err"]]
    ms = lambda v, q: int(round(1000 * float(np.percentile(v, q) if q else np.mean(v)))) if v else None
    se_s = [x for r in rows for x in r["start_sgn"]]
    ee_s = [x for r in rows for x in r["end_sgn"]]
    return {"WER": round(s("wer_err") / max(1, s("n_ref_words")), 3), "CER": round(s("cer_err") / max(1, s("n_ref_chars")), 3),
            "lines": s("n_lines"), "lost": sum(len(r["lost"]) for r in rows),
            "fab_words": sum(len(r["fab_words"]) for r in rows),
            "first_short": f'{sum(len(r["first_short_lost"]) for r in rows)}/{s("first_short_tot")}',
            "start_ms": ms(se, 0), "start_p90": ms(se, 90), "end_ms": ms(ee, 0), "end_p90": ms(ee, 90), "n_bound": f"{len(se)}+{len(ee)}",
            "start_bias": ms(se_s, 0), "end_bias": ms(ee_s, 0),
            "late300": f'{sum(abs(x) > 0.3 for x in se)}/{len(se)}+{sum(abs(x) > 0.3 for x in ee)}/{len(ee)}',
            "no_bound": s("no_start") + s("no_end"), "merged": s("merged"), "split": s("split"),
            "punct": round(float(np.mean([r["punct"] for r in rows])), 2), "hyp_lines": s("n_hyp_lines")}


CAPTION = ("музык", "music", "смех", "аплодис", "applause", "♪", "субтитр", "subtitle", "продолжение следует",
           "редактор", "динамичн", "напряженн", "напряжённ", "веселая", "весёлая", "спокойная", "тревожн")


def full_file_fabrications(hyp: dict, voice_env: np.ndarray | None, others: list[dict], hop=0.05) -> dict:
    """Выдумки по всему файлу: пометки-надписи, зацикливания, текст там, где в голосовом стеме тишина,
    реплики, которых нет ни у одной другой модели."""
    segs = [s for s in hyp["segments"] if norm_words(s["text"])]
    cap = [s for s in segs if any(c in s["text"].lower() for c in CAPTION) or
           (sum(ch.isupper() for ch in s["text"]) >= 6 and s["text"].upper() == s["text"])]
    seen, loops = {}, []
    for s in segs:
        k = " ".join(norm_words(s["text"]))
        if len(k.split()) >= 2:
            seen.setdefault(k, []).append(s)
    for k, v in seen.items():
        if len(v) >= 3:
            loops += v
    for s in segs:
        b = s["text"].encode()
        if len(b) > 40 and len(b) / len(zlib.compress(b)) > 2.4:
            loops.append(s)
    novoice = []
    if voice_env is not None:
        ref = float(np.percentile(voice_env, 90))
        for s in segs:
            i, j = int(s["start"] / hop), max(int(s["start"] / hop) + 1, int(s["end"] / hop))
            seg = voice_env[i:j]
            if len(seg) and float(np.percentile(seg, 90)) < ref - 30:
                novoice.append(s)
    lonely = []
    ow = [hyp_words(o) for o in others]
    for s in segs:
        sup = 0
        for ws in ow:
            if any(s["start"] - 0.5 <= mid(w) <= s["end"] + 0.5 for w in ws):
                sup += 1
        if ow and sup == 0:
            lonely.append(s)
    fmt = lambda v: [f'{s["start"]:.1f}–{s["end"]:.1f} «{s["text"]}»' for s in v]
    return {"segments": len(segs), "caption": fmt(cap), "loops": fmt(loops), "no_voice": fmt(novoice), "lonely": fmt(lonely)}


def env_of(path) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    h = int(sr * 0.05)
    n = len(x) // h
    return 10 * np.log10((x[: n * h].reshape(n, h) ** 2).mean(axis=1) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.environ.get("DUBL_BENCH_REF", str(ROOT / "reference.json")), help="эталон (или переменная DUBL_BENCH_REF)")
    ap.add_argument("--hyp-dir", default=None, help="папка с hyp_*.json (или DUBL_BENCH_HYP)")
    ap.add_argument("--audio-dir", default=None, help="от неё считается voice_env из эталона (или DUBL_BENCH_AUDIO)")
    ap.add_argument("--out", default=str(ROOT / "scores.md"))
    args = ap.parse_args()
    global HYP, AUDIO
    if args.hyp_dir:
        HYP = Path(args.hyp_dir)
    if args.audio_dir:
        AUDIO = Path(args.audio_dir)
    ref = json.loads(Path(args.ref).read_text())
    md, allres = [], {}
    for set_name, st in ref["sets"].items():
        for tag in st["tags"]:
            files = sorted(HYP.glob(f"hyp_*_{tag}.json"))
            if not files:
                continue
            md.append(f"\n### {set_name} · вход `{tag}`\n")
            md.append("| модель | WER | CER | потеряно реплик | выдумано слов | короткое 1-е слово потеряно | начало, мс (ср/p90/сдвиг) | конец, мс (ср/p90/сдвиг) | границ > 300 мс (нач+кон) | нет границы | слито | разрезано | пунктуация | реплик гип. |")
            md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for f in files:
                hyp = json.loads(f.read_text())
                model = f.name[len("hyp_"):-len(f"_{tag}.json")]
                rows = [score_piece(hyp, p) for p in st["pieces"]]
                a = agg(rows)
                allres[f"{set_name}/{tag}/{model}"] = {"total": a, "pieces": {p["id"]: r for p, r in zip(st["pieces"], rows)}}
                md.append(f'| {model} | {a["WER"]:.3f} | {a["CER"]:.3f} | {a["lost"]}/{a["lines"]} | {a["fab_words"]} | {a["first_short"]} | '
                          f'{a["start_ms"]}/{a["start_p90"]}/{a["start_bias"]} | {a["end_ms"]}/{a["end_p90"]}/{a["end_bias"]} | {a["late300"]} | {a["no_bound"]} | {a["merged"]} | {a["split"]} | '
                          f'{a["punct"]:.2f} | {a["hyp_lines"]} |')
            md.append("\nПо кускам (WER):\n")
            md.append("| модель | " + " | ".join(p["id"] + " " + p.get("title", "") for p in st["pieces"]) + " |")
            md.append("|---|" + "---|" * len(st["pieces"]))
            for k, v in allres.items():
                if k.startswith(f"{set_name}/{tag}/"):
                    md.append(f'| {k.split("/")[-1]} | ' + " | ".join(
                        f'{v["pieces"][p["id"]]["wer_err"] / max(1, v["pieces"][p["id"]]["n_ref_words"]):.3f}' for p in st["pieces"]) + " |")
        ff = st.get("full_file")
        if ff:
            files = sorted(HYP.glob(f'hyp_*_{ff["tag"]}.json'))
            env = env_of(AUDIO / ff["voice_env"]) if ff.get("voice_env") and (AUDIO / ff["voice_env"]).exists() else None
            hyps = {f.name[len("hyp_"):-len(f'_{ff["tag"]}.json')]: json.loads(f.read_text()) for f in files}
            base = {k: v for k, v in hyps.items() if k in ff.get("support_models", hyps.keys())}
            md.append(f'\n### {set_name} · выдумки по всему файлу `{ff["tag"]}`\n')
            md.append("| модель | реплик | надписи-пометки | повторы/зацикливание | в голосовом стеме тишина | нет ни у одной другой модели |")
            md.append("|---|---|---|---|---|---|")
            for m, h in hyps.items():
                others = [v for k, v in base.items() if k != m and k.split("_")[0] != m.split("_")[0]]
                r = full_file_fabrications(h, env, others)
                allres[f"{set_name}/full/{m}"] = r
                md.append(f'| {m} | {r["segments"]} | {len(r["caption"])} | {len(r["loops"])} | {len(r["no_voice"])} | {len(r["lonely"])} |')
    Path(args.out).write_text("\n".join(md) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(allres, ensure_ascii=False, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
