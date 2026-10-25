import { useEffect, useReducer } from "react";

/**
 * Часы воспроизведения. Если к плееру привязан настоящий <video> — время берём из него,
 * иначе идёт виртуальный таймер (демо-режим: картинки нет, но всё остальное работает).
 * `t` — секунды исходного видео (offset — какое время исходника соответствует нулю файла).
 */
export class Player {
  t = 0;
  playing = false;
  offset = 0;
  /** играть только до этой секунды (null — до конца) */
  stopAt: number | null = null;
  rate = 1;
  private video: HTMLVideoElement | null = null;
  private raf = 0;
  private last = 0;
  private subs = new Set<() => void>();
  private ends = new Set<() => void>();

  constructor(public duration: number, t0 = 0) {
    this.t = t0;
  }

  /** привязанный <video> (null — виртуальный таймер) */
  get media(): HTMLVideoElement | null { return this.video; }

  attach(video: HTMLVideoElement | null) {
    this.video = video;
    if (video) {
      try { video.currentTime = Math.max(0, this.t - this.offset); } catch { /* ещё не загружено */ }
      if (this.playing) void video.play().catch(() => {});
    }
  }

  subscribe(fn: () => void): () => void {
    this.subs.add(fn);
    return () => { this.subs.delete(fn); };
  }
  onEnd(fn: () => void): () => void {
    this.ends.add(fn);
    return () => { this.ends.delete(fn); };
  }
  private notify() { this.subs.forEach((f) => f()); }

  seek(t: number) {
    this.t = Math.min(Math.max(0, t), this.duration);
    if (this.video) {
      try { this.video.currentTime = Math.max(0, this.t - this.offset); } catch { /* пока не готово */ }
    }
    this.notify();
  }

  play(stopAt: number | null = this.stopAt) {
    this.stopAt = stopAt;
    if (this.playing) return;
    if (this.t >= (stopAt ?? this.duration) - 0.05) this.seek(stopAt === null ? 0 : Math.max(0, this.t));
    this.playing = true;
    this.last = performance.now();
    if (this.video) void this.video.play().catch(() => {});
    this.loop();
    this.notify();
  }

  pause() {
    if (!this.playing) return;
    this.playing = false;
    cancelAnimationFrame(this.raf);
    this.video?.pause();
    this.notify();
  }

  toggle() { this.playing ? this.pause() : this.play(); }

  private loop = () => {
    if (!this.playing) return;
    const now = performance.now();
    if (this.video && this.video.readyState >= 2 && !this.video.paused) this.t = this.video.currentTime + this.offset;
    else this.t += ((now - this.last) / 1000) * this.rate;
    this.last = now;
    const end = this.stopAt ?? this.duration;
    if (this.t >= end) {
      this.t = end;
      this.playing = false;
      this.video?.pause();
      this.notify();
      this.ends.forEach((f) => f());
      return;
    }
    this.notify();
    this.raf = requestAnimationFrame(this.loop);
  };

  dispose() {
    cancelAnimationFrame(this.raf);
    this.playing = false;
    this.subs.clear();
    this.ends.clear();
  }
}

/** Перерисовывать компонент при изменении плеера (не чаще hz раз в секунду во время игры). */
export function usePlayerTick(p: Player, hz = 20) {
  const [, bump] = useReducer((x: number) => x + 1, 0);
  useEffect(() => {
    let lastDraw = 0;
    let wasPlaying = p.playing;
    return p.subscribe(() => {
      const now = performance.now();
      if (p.playing !== wasPlaying || !p.playing || now - lastDraw > 1000 / hz) {
        wasPlaying = p.playing;
        lastDraw = now;
        bump();
      }
    });
  }, [p, hz]);
  return p;
}
