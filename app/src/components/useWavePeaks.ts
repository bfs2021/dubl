import { useEffect, useState } from "react";
import { decodeWav } from "../audio/wav";
import { envelope } from "../audio/analyze";
import { rng } from "../lib/rng";

const cache = new Map<string, number[]>();

/** Скачивает WAV дубля и считает огибающую. Не вышло — рисует правдоподобную заглушку по seed. */
export function useWavePeaks(url: string | null, seed: string, bins = 90): number[] {
  const [peaks, setPeaks] = useState<number[]>(() => (url && cache.get(url)) || []);
  useEffect(() => {
    let dead = false;
    const fake = () => {
      const r = rng([...seed].reduce((a, c) => a + c.charCodeAt(0), 7));
      return Array.from({ length: bins }, (_, i) => (0.15 + r() * 0.5) * (0.4 + 0.6 * Math.sin((i / bins) * Math.PI)));
    };
    if (!url) { setPeaks(fake()); return; }
    const hit = cache.get(url);
    if (hit) { setPeaks(hit); return; }
    fetch(url)
      .then((r) => r.arrayBuffer())
      .then((b) => {
        const p = envelope(decodeWav(b).samples, bins);
        const mx = Math.max(0.001, ...p);
        const norm = p.map((v) => v / mx);
        cache.set(url, norm);
        if (!dead) setPeaks(norm);
      })
      .catch(() => { if (!dead) setPeaks(fake()); });
    return () => { dead = true; };
  }, [url, seed, bins]);
  return peaks;
}
