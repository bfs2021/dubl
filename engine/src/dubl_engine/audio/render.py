"""Рендер видео: кроп под аспект, масштаб, звук из сведения, субтитры (srt/ass)."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from ..models import RenderAspect, RenderFit, RenderSettings, SceneLine, SubtitleText
from ..util import _NO_WINDOW, EngineError, ffmpeg_failure, ffprobe_json, find_binary

_ASPECT = {RenderAspect.H16_9: (16, 9), RenderAspect.V9_16: (9, 16), RenderAspect.SQ1_1: (1, 1)}


def _even(v: float) -> int:
    return max(2, round(v / 2) * 2)


def _floor_even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


PORTRAIT_FRAME_MIN = 1280      # для 9:16 height ≥ этого — уже высота кадра (1280 → 720×1280, 1920 → 1080×1920)
DEFAULT_SIDE = 1080            # height=None для 9:16 и 1:1: 1080×1920 и 1080×1080


def _target_size(settings: RenderSettings) -> tuple[int, int] | None:
    """Итоговый размер для 9:16 и 1:1 (чётный). None — для original и 16:9 (там размер считает compute_geometry).

    9:16: height — «качество» как у обычного видео, то есть короткая сторона: 720 → 720×1280, 1080 → 1080×1920,
    None → 1080×1920. Значение ≥ 1280 считаем высотой кадра (1280 → 720×1280, 1920 → 1080×1920).
    1:1: height — сторона квадрата, None → 1080×1080.
    """
    h = settings.height
    if settings.aspect == RenderAspect.SQ1_1:
        side = _even(h or DEFAULT_SIDE)
        return side, side
    if settings.aspect == RenderAspect.V9_16:
        if h and h >= PORTRAIT_FRAME_MIN:
            oh = _even(h)
            ow = _even(oh * 9 / 16)
        else:
            ow = _even(h or DEFAULT_SIDE)
            oh = _even(ow * 16 / 9)
        return ow, oh
    return None


def blur_layout(src_w: int, src_h: int, settings: RenderSettings) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Раскладка «размытые поля»: ((ширина, высота) кадра внутри, (ширина, высота) итога) или None, если не нужна.

    Нужна для 9:16 и 1:1 при fit='blur', когда пропорции источника заметно отличаются от итоговых.
    Кадр целиком вписывается в итог (по ширине для широкого источника), остальное закрывает его размытая копия."""
    tgt = _target_size(settings)
    if tgt is None or settings.fit != RenderFit.BLUR:
        return None
    ow, oh = tgt
    if abs(src_w / src_h - ow / oh) <= 0.01:
        return None
    k = min(ow / src_w, oh / src_h)
    fw, fh = min(_floor_even(src_w * k), ow), min(_floor_even(src_h * k), oh)
    return (fw, fh), (ow, oh)


def compute_geometry(src_w: int, src_h: int, settings: RenderSettings
                     ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int]]:
    """(кроп (w, h, x, y) или None, итоговый размер). Все числа чётные (требование yuv420p/libx264).
    original / 16:9: height — высота кадра (None — как у кропа). 9:16 и 1:1 — см. _target_size.
    При fit='blur' (по умолчанию) для 9:16 и 1:1 кропа нет — см. blur_layout.
    src_w/src_h — размер картинки с квадратными пикселями (SAR уже учтён, см. render_video)."""
    ar = _ASPECT.get(settings.aspect)
    tgt = _target_size(settings)
    if tgt is not None and settings.fit == RenderFit.BLUR:
        return None, tgt
    crop = None
    cw, ch = _floor_even(src_w), _floor_even(src_h)
    target = ar[0] / ar[1] if ar else None
    if target is not None and abs(src_w / src_h - target) > 0.01:
        if src_w / src_h > target:
            cw = min(_floor_even(ch * target), cw)
        else:
            ch = min(_floor_even(cw / target), ch)
    if (cw, ch) != (src_w, src_h):
        crop = (cw, ch, (src_w - cw) // 2 // 2 * 2, (src_h - ch) // 2 // 2 * 2)
    if tgt is not None:
        return crop, tgt
    out_h = _even(settings.height) if settings.height else ch
    out_w = _even(out_h * target) if target is not None else _even(cw * out_h / ch)
    return crop, (out_w, out_h)


def _clip_rate(stream: dict) -> str | None:
    """Частота кадров клипа для фильтра fps (ровная сетка). None — не похоже на постоянную частоту, фильтр не ставим."""
    raw = str(stream.get("r_frame_rate") or "")
    try:
        num, _, den = raw.partition("/")
        n, d = int(num), int(den or 1)
    except ValueError:
        return None
    return f"{n}/{d}" if n > 0 and d > 0 and 1.0 <= n / d <= 120.0 else None


def _sar(stream: dict) -> float:
    """Отношение сторон пикселя (анаморфное видео: 1.333, 0.909…). Нет данных — 1."""
    raw = str(stream.get("sample_aspect_ratio") or "1:1")
    try:
        n, _, d = raw.partition(":")
        v = float(n) / float(d or 1)
        return v if 0.2 < v < 5 else 1.0
    except (ValueError, ZeroDivisionError):
        return 1.0


def _blur_graph(pre: list[str], layout: tuple[tuple[int, int], tuple[int, int]], *, with_subs: bool) -> str:
    """filter_complex для «размытых полей»: кадр целиком по центру, фон — его же увеличенная размытая копия."""
    (fw, fh), (ow, oh) = layout
    radius = max(8, min(ow, oh) // 24)
    head = ("[0:v:0]" + ",".join(pre) + ",") if pre else "[0:v:0]"
    g = (f"{head}split=2[bgsrc][fgsrc];"
         f"[bgsrc]scale={ow}:{oh}:force_original_aspect_ratio=increase:flags=bilinear,crop={ow}:{oh},"
         f"boxblur=luma_radius={radius}:luma_power=2:chroma_radius={radius}:chroma_power=2[bg];"
         f"[fgsrc]scale={fw}:{fh}:flags=lanczos[fg];"
         f"[bg][fg]overlay=x=(W-w)/2:y=(H-h)/2,setsar=1")
    return g + (",subtitles=filename=subs.ass[v]" if with_subs else "[v]")


def render_video(clip_mp4, mixed_wav, out_mp4, settings: RenderSettings, *, subtitles_ass: Path | None = None,
                 on_progress: Callable[[float], None] | None = None) -> Path:
    """Видео из clip_mp4 + звук из mixed_wav. libx264 crf 20 veryfast, yuv420p, aac 192k, faststart.

    Анаморфный SAR сначала «выпрямляется» (картинка растягивается до квадратных пикселей), потом кроп/масштаб.
    Путь к субтитрам в фильтр не подставляем (кавычки, двоеточия, Windows-диски): ASS копируется во временную папку
    под простым именем, и ffmpeg запускается в этой папке.
    """
    clip_mp4 = Path(clip_mp4).absolute()
    info = ffprobe_json(clip_mp4, "-show_streams", "-show_format")
    vs = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if vs is None:
        raise EngineError("В видео нет картинки.")
    w, h = int(vs["width"]), int(vs["height"])
    sar = _sar(vs)
    sq_w = _even(w * sar) if abs(sar - 1.0) > 0.005 else w
    duration = float(info.get("format", {}).get("duration") or vs.get("duration") or 0) or None
    crop, (ow, oh) = compute_geometry(sq_w, h, settings)
    layout = blur_layout(sq_w, h, settings)
    # Звук и картинка кончаются одновременно: длину задаёт видео клипа (-t), а не «кто короче» (-shortest обрезал
    # кадры в хвосте и сдвигал начало). Звук короче — добиваем тишиной; длиннее — режем по -t.
    vend = video_end(clip_mp4)
    vf = []
    rate = _clip_rate(vs)
    if rate:      # клип, начавшийся не с нуля (старые сцены), выравниваем: первый кадр держим от нуля, синхрон не двигаем
        vf.append(f"fps=fps={rate}:start_time=0")
    if sq_w != w:
        vf.append(f"scale={sq_w}:{h}:flags=lanczos")
    if layout is None:
        if crop:
            vf.append(f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}")
        if (ow, oh) != ((crop[0], crop[1]) if crop else (sq_w, h)):
            vf.append(f"scale={ow}:{oh}:flags=lanczos")
        vf.append("setsar=1")
    out = Path(out_mp4).absolute()
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="dubl-render-"))
    try:
        if subtitles_ass is not None:
            shutil.copyfile(subtitles_ass, work / "subs.ass")
        if layout is None:
            if subtitles_ass is not None:
                vf.append("subtitles=filename=subs.ass")
            vmap = ["-map", "0:v:0", "-vf", ",".join(vf)]
        else:
            vmap = ["-filter_complex", _blur_graph(vf, layout, with_subs=subtitles_ass is not None), "-map", "[v]"]
        cmd = [find_binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-progress", "pipe:1",
               "-i", str(clip_mp4), "-i", str(Path(mixed_wav).absolute()), *vmap, "-map", "1:a:0",
               "-map_chapters", "-1", "-map_metadata", "-1",       # главы и теги исходника в ролик не тянем
               "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p",
               "-af", "apad", "-c:a", "aac", "-b:a", "192k", "-t", f"{vend:.4f}",
               "-movflags", "+faststart", str(out)]
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace", cwd=work, creationflags=_NO_WINDOW)
        err_buf: list[str] = []
        t = threading.Thread(target=lambda: err_buf.append(p.stderr.read() if p.stderr else ""), daemon=True)
        t.start()
        try:
            assert p.stdout is not None
            for line in p.stdout:
                m = re.match(r"out_time_(?:us|ms)=(\d+)", line.strip())
                if m and duration and on_progress:
                    on_progress(min(1.0, int(m.group(1)) / 1e6 / duration))   # out_time_ms в ffmpeg фактически мкс
            rc = p.wait()
        except BaseException:           # отмена задачи (исключение из on_progress) и любые сбои: ffmpeg не оставляем
            p.kill()
            p.wait()
            out.unlink(missing_ok=True)
            raise
        t.join(timeout=5)
        if rc != 0:
            out.unlink(missing_ok=True)
            raise ffmpeg_failure(err_buf[0] if err_buf else "", rc)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if on_progress:
        on_progress(1.0)
    return out


# ─────────────────────────── мастеринг с проверкой после AAC ───────────────────────────
AAC_TP_TOLERANCE_DB = 0.2          # после AAC пик может быть выше цели не более чем на столько


def aac_true_peak(wav, *, bitrate: str = "192k") -> float:
    """True peak (dBTP) звука после кодирования AAC (те же параметры, что в render_video): кодируем, декодируем, меряем."""
    from ..util import run_ffmpeg
    from .io import read_wav
    from .master import true_peak_db

    with tempfile.TemporaryDirectory(prefix="dubl-aac-") as tmp:
        m4a, back = Path(tmp) / "a.m4a", Path(tmp) / "back.wav"
        run_ffmpeg(["-i", Path(wav), "-c:a", "aac", "-b:a", bitrate, m4a])
        run_ffmpeg(["-i", m4a, "-c:a", "pcm_f32le", back])
        y, _sr = read_wav(back)
    return true_peak_db(y)


# Первые ~20 мс звука кодер AAC (ffmpeg aac, twoloop) передаёт плохо: в первом кадре после «разгона» кодера пик
# выскакивает на 1–6 дБ выше, чем в WAV (ролик слепого теста 7: мастер −1.15 dBTP → в mp4 −0.4 dBTP на 12-й мс, весь
# остальной ролик −2.3). Сцена вырезана посреди звука и начинается сразу громко — это как раз такой случай.
# Короткое нарастание (≈ 43 мс, косинус) убирает выброс: на 96 кусках реальных сцен выброс в начале ≤ +0.16 дБ
# (без него — до +2.4 дБ, в 18 из 96). На слух нарастание короче кадра видео не заметно, заодно нет щелчка в начале.
AAC_FADE_IN_SAMPLES = 2048


def _fade_in(wav) -> None:
    import numpy as np

    from .io import read_wav, write_wav

    x, sr = read_wav(wav)
    k = min(AAC_FADE_IN_SAMPLES, len(x))
    if k < 2:
        return
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, k, dtype=np.float32))).astype(np.float32)
    x = np.array(x, dtype=np.float32, copy=True)
    x[:k] *= ramp if x.ndim == 1 else ramp[:, None]
    write_wav(wav, x, sr)


def _scale(wav, gain_db: float) -> None:
    import numpy as np

    from .io import read_wav, write_wav

    x, sr = read_wav(wav)
    write_wav(wav, (np.asarray(x, dtype=np.float32) * np.float32(10 ** (gain_db / 20))).astype(np.float32), sr)


def master_for_aac(in_wav, out_wav, *, target_lufs: float = -14.0, true_peak_db: float = -1.0,
                   attempts: int = 3):
    """master() + проверка после AAC: если пик в кодированном звуке выше цели + 0.2 дБ, пересводим с меньшей целью.
    Возвращает Loudness мастер-файла (WAV).

    Проверяется КАЖДЫЙ вариант, включая последний (раньше последний пересвод уходил в ролик без проверки — так в
    «Для YouTube» вышло −0.4 dBTP при цели −1.0). Если пересвод с меньшим потолком не помог (выброс делает сам кодер, а
    не пики сведения), весь мастер тише на недостающее — громкость чуть ниже цели лучше, чем пик выше."""
    from .io import read_wav
    from .master import master, measure

    limit = true_peak_db + AAC_TP_TOLERANCE_DB
    ceiling = true_peak_db
    master(in_wav, out_wav, target_lufs=target_lufs, true_peak_db=ceiling)
    _fade_in(out_wav)
    tp = aac_true_peak(out_wav)
    for _ in range(max(attempts - 1, 0)):
        if tp <= limit:
            break
        ceiling -= (tp - true_peak_db) + 0.05
        master(in_wav, out_wav, target_lufs=target_lufs, true_peak_db=ceiling)
        _fade_in(out_wav)
        tp = aac_true_peak(out_wav)
    for _ in range(4):                       # запасной путь: постоянный гейн (AAC почти линеен по уровню)
        if tp <= limit:
            break
        _scale(out_wav, -((tp - true_peak_db) + 0.05))
        tp = aac_true_peak(out_wav)
    y, sr = read_wav(out_wav)
    return measure(y, sr)


# ─────────────────────────── склейка нескольких сцен ───────────────────────────
SEAM_FADE_MS = 80.0


def video_duration(path) -> float:
    """Длина видеодорожки в секундах (если нет — длина контейнера)."""
    info = ffprobe_json(path, "-show_streams", "-show_format")
    vs = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    for src in (vs or {}, info.get("format", {})):
        try:
            v = float(src.get("duration") or 0)
        except (TypeError, ValueError):
            v = 0.0
        if v > 0:
            return v
    raise EngineError("Не удалось узнать длину видео.")


def video_end(path) -> float:
    """Где кончается картинка: начало + длина видеодорожки (мп4 может начинаться не с нуля). Нет — длина контейнера."""
    info = ffprobe_json(path, "-show_streams", "-show_format")
    vs = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if vs is not None:
        try:
            d = float(vs.get("duration") or 0)
            if d > 0:
                return max(float(vs.get("start_time") or 0), 0.0) + d
        except (TypeError, ValueError):
            pass
    return video_duration(path)


def join_audio(parts, sr: int, *, seam_ms: float = SEAM_FADE_MS):
    """Склеить куски звука (stereo float32) встык; на каждом стыке — переход длиной seam_ms: вторая половина уходит
    в тишину у конца предыдущего куска, первая половина нарастает в начале следующего (косинус, без щелчков).
    Общая длина не меняется — звук остаётся синхронным с жёстко склеенным видео."""
    import numpy as np

    if not parts:
        raise EngineError("Нечего склеивать.")
    half = max(int(seam_ms / 2000 * sr), 1)
    out = []
    for i, p in enumerate(parts):
        p = np.array(p, dtype=np.float32, copy=True)
        k = min(half, len(p) // 2)
        if k > 0:
            ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, k, dtype=np.float32)))
            shape = (-1,) + (1,) * (p.ndim - 1)
            if i > 0:
                p[:k] *= ramp.reshape(shape)
            if i < len(parts) - 1:
                p[-k:] *= ramp[::-1].reshape(shape)
        out.append(p)
    return np.concatenate(out, axis=0)


def fit_length(x, n: int):
    """Обрезать или добить тишиной до ровно n отсчётов."""
    import numpy as np

    if len(x) >= n:
        return x[:n]
    pad = np.zeros((n - len(x),) + x.shape[1:], dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


def _concat_quote(path) -> str:
    """Путь для списка concat-демуксера ffmpeg: одинарные кавычки внутри экранируются как '\\''."""
    return Path(path).absolute().as_posix().replace("'", "'\\''")


def concat_videos(videos: list[Path], out_mp4, *, size: tuple[int, int] | None = None) -> Path:
    """Жёсткая склейка видео (без звука). Сначала без перекодирования (все сцены закодированы одинаково);
    если контейнер «поплыл» по длине — перекодируем через фильтр concat."""
    from ..util import run_ffmpeg

    out = Path(out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(videos) == 1:
        shutil.copyfile(videos[0], out)
        return out
    want = sum(video_duration(v) for v in videos)
    with tempfile.TemporaryDirectory(prefix="dubl-concat-") as tmp:
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{_concat_quote(v)}'\n" for v in videos), encoding="utf-8")
        try:
            run_ffmpeg(["-f", "concat", "-safe", "0", "-i", lst, "-an", "-c:v", "copy", "-map_chapters", "-1",
                        "-map_metadata", "-1", "-movflags", "+faststart", out])
            if abs(video_duration(out) - want) <= 0.15:
                return out
        except EngineError:
            pass
    ins: list[str | Path] = []
    for v in videos:
        ins += ["-i", Path(v)]
    graph = "".join(f"[{i}:v:0]setsar=1[v{i}];" for i in range(len(videos)))
    graph += "".join(f"[v{i}]" for i in range(len(videos))) + f"concat=n={len(videos)}:v=1:a=0[v]"
    run_ffmpeg([*ins, "-filter_complex", graph, "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "20", "-preset",
                "veryfast", "-pix_fmt", "yuv420p", "-map_chapters", "-1", "-movflags", "+faststart", out])
    return out


def mux_video_audio(video_mp4, audio_wav, out_mp4) -> Path:
    """Видео как есть + звук из WAV в AAC 192k, faststart, без глав и тегов исходника."""
    from ..util import run_ffmpeg

    out = Path(out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    # длину задаёт видео (-t по концу картинки): -shortest при копировании видео с B-кадрами отрезал хвост кадров
    vend = video_end(video_mp4)
    run_ffmpeg(["-i", Path(video_mp4), "-i", Path(audio_wav), "-map", "0:v:0", "-map", "1:a:0",
                "-map_chapters", "-1", "-map_metadata", "-1", "-c:v", "copy", "-af", "apad", "-c:a", "aac",
                "-b:a", "192k", "-t", f"{vend:.4f}", "-movflags", "+faststart", out])
    return out


# ─────────────────────────── субтитры ───────────────────────────
def _one_block(text: str) -> str:
    """Текст реплики для SRT: пустая строка внутри закончила бы блок."""
    return "\n".join(row.strip() for row in text.strip().splitlines() if row.strip())


def _ts_srt(t: float) -> str:
    ms = round(max(t, 0) * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _ts_ass(t: float) -> str:
    cs = round(max(t, 0) * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _norm(s: str) -> str:
    return " ".join(s.split()).casefold()


def subtitle_items(lines: list[SceneLine], text: SubtitleText | str = SubtitleText.DUB) -> list[tuple[SceneLine, str]]:
    """Реплики для субтитров → (реплика, текст), по времени.

    dub: текст дубля (`SceneLine.text`: перевод, русский текст из субтитров файла или свой). Нет текста дубля или он
    совпадает с оригиналом — берём оригинал: реплика в субтитрах не пропадает.
    original: текст оригинала (`SceneLine.original`). Реплики совсем без текста пропускаются."""
    original = SubtitleText(text) == SubtitleText.ORIGINAL
    out = []
    for l in sorted(lines, key=lambda l: l.start_rel):
        t = l.original if original else (l.text if l.text.strip() else l.original)
        if not t.strip():
            continue
        out.append((l, t))
    return out


def write_subtitles(lines: list[SceneLine], out_path, *, fmt: Literal["srt", "ass"] = "srt",
                    play_res: tuple[int, int] = (1920, 1080), text: SubtitleText | str = SubtitleText.DUB) -> Path:
    """Субтитры из реплик сцены; какой текст — см. subtitle_items (по умолчанию текст дубля, нет перевода — оригинал).
    ASS: Arial, ~5% высоты, белый с чёрной обводкой."""
    items = subtitle_items(lines, text)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "srt":
        body = "\n".join(f"{i}\n{_ts_srt(l.start_rel)} --> {_ts_srt(l.end_rel)}\n{_one_block(t)}\n"
                         for i, (l, t) in enumerate(items, 1))
    elif fmt == "ass":
        rw, rh = play_res
        size = round(rh * 0.05)
        head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {rw}\nPlayResY: {rh}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
                "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
                "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n"
                f"Style: Default,Arial,{size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,"
                f"{max(2, size // 14)},1,2,{rw // 20},{rw // 20},{int(rh * 0.05)},1\n\n"
                "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        ev = []
        for l, t in items:
            txt = t.strip().replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", "\\N")
            ev.append(f"Dialogue: 0,{_ts_ass(l.start_rel)},{_ts_ass(l.end_rel)},Default,,0,0,0,,{txt}")
        body = head + "\n".join(ev) + "\n"
    else:
        raise EngineError(f"Неизвестный формат субтитров: {fmt}")
    out.write_text(body, encoding="utf-8")
    return out
