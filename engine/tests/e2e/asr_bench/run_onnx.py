"""onnx-asr (onnxruntime CPU, без torch) — кросс-платформенный путь для Windows/Linux:
GigaAM v3 e2e-rnnt и Parakeet TDT 0.6B v3 (+int8). Нарезка — та же energy_chunks (≤ 20 с без потерь), что у run_gigaam.
Тайминги — по токенам (onnx-asr отдаёт время начала каждого токена), слово = токены до следующего пробела.
  run_onnx.py <wav> <gigaam-v3-e2e-rnnt|nemo-parakeet-tdt-0.6b-v3> [int8]
"""
import sys

import numpy as np

from bench_common import Timer, energy_chunks, load16k, save, words_to_lines

inp, name = sys.argv[1], sys.argv[2]
quant = sys.argv[3] if len(sys.argv) > 3 else None
import onnx_asr  # noqa: E402

x = load16k(inp)
with Timer() as tl:
    m = onnx_asr.load_model(name, quantization=quant, providers=["CPUExecutionProvider"]).with_timestamps()
chunks = energy_chunks(x)
segs = []
with Timer() as t:
    for a, b in chunks:
        r = m.recognize(x[int(a * 16000):int(b * 16000)], sample_rate=16000)
        toks, ts = r.tokens or [], r.timestamps or []
        words = []
        new = True
        for k, (tk, st) in enumerate(zip(toks, ts)):
            nxt = ts[k + 1] if k + 1 < len(ts) else st + 0.3
            # как в пакете gigaam (frames_to_words): конец слова = последний кадр токена + 1 кадр энкодера
            en = min(nxt, st + (0.04 if "gigaam" in name else 0.08))
            if not tk.strip(" ▁"):                      # отдельный токен-пробел (GigaAM)
                new = True
                continue
            if tk.startswith((" ", "▁")) or new or not words:
                words.append({"start": round(st + a, 3), "end": round(en + a, 3), "text": tk.strip(" ▁")})
            else:
                words[-1]["text"] += tk
                words[-1]["end"] = round(en + a, 3)
            new = False
        if words:
            segs += words_to_lines(words, gap=0.5 if "gigaam" in name else 99.0)
short = {"gigaam-v3-e2e-rnnt": "gigaam_e2e_rnnt", "nemo-parakeet-tdt-0.6b-v3": "parakeet"}.get(name, name)
save(f"onnx_{short}" + (f"_{quant}" if quant else ""), inp,
     {"lang": None, "device": "CPU (onnxruntime)", "load_s": tl.s, "asr_s": t.s, "audio_s": round(len(x) / 16000, 2),
      "chunking": f"energy: {len(chunks)} кусков ≤ 20 с", "segments": segs})
