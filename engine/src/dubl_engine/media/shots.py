"""Монтажные склейки: ffmpeg scene detection."""
from __future__ import annotations

import re
from pathlib import Path

from ..util import EngineError, ffmpeg_failure, find_binary, run_process
from .probe import main_video_index

_PTS = re.compile(r"pts_time:\s*(-?\d+(?:\.\d+)?)")
ANALYSIS_WIDTH = 384      # детектор склеек работает по уменьшенной картинке: в разы быстрее на длинных фильмах


def parse_showinfo_times(stderr: str) -> list[float]:
    """Вытащить pts_time из строк showinfo."""
    return [float(m.group(1)) for line in stderr.splitlines() if "showinfo" in line
            for m in [_PTS.search(line)] if m]


def detect_shot_cuts(src: str | Path, *, threshold: float = 0.3, start: float | None = None,
                     end: float | None = None) -> list[float]:
    """Времена склеек (абсолютные секунды исходника), по возрастанию."""
    if not 0.0 < threshold < 1.0:
        raise EngineError("Порог поиска склеек должен быть между 0 и 1.")
    if start is not None and end is not None and end <= start:
        raise EngineError("Конец фрагмента должен быть позже начала.")
    offset = max(start or 0.0, 0.0)
    cmd = [find_binary("ffmpeg"), "-hide_banner", "-loglevel", "info", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{offset:.3f}"]
    if end is not None:
        cmd += ["-t", f"{end - offset:.3f}"]
    # scale перед select: оценка «насколько кадр изменился» почти не зависит от размера, а читать 4K незачем
    cmd += ["-i", str(Path(src).absolute()), "-map", f"0:{main_video_index(src)}", "-an", "-sn", "-dn",
            "-vf", f"scale='min({ANALYSIS_WIDTH},iw)':-2:flags=fast_bilinear,select='gt(scene,{threshold})',showinfo",
            "-f", "null", "-"]
    p = run_process(cmd)
    if p.returncode != 0:
        raise ffmpeg_failure(p.stderr, p.returncode)
    times = [round(t + offset, 3) for t in parse_showinfo_times(p.stderr)]
    # самый первый кадр ffmpeg иногда считает «склейкой» — это не она
    times = [t for t in times if t - offset > 0.05]
    return sorted(set(times))
