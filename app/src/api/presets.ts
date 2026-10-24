/** Пресеты, которые знает движок (audio/rooms.py, audio/fx.py), с русскими названиями. */
export const ROOMS: { id: string; title: string }[] = [
  { id: "auto", title: "Подобрать сам" },
  { id: "dry", title: "Сухо" },
  { id: "room", title: "Комната" },
  { id: "kitchen", title: "Кухня" },
  { id: "hall", title: "Большой зал" },
  { id: "street", title: "Улица" },
  { id: "cave", title: "Пещера" },
  { id: "car", title: "Машина" },
  { id: "bathroom", title: "Ванная" },
  { id: "phone", title: "Телефон" },
  { id: "behind_door", title: "Из-за двери" },
];

export const VOICE_FX: { id: string; title: string }[] = [
  { id: "none", title: "Без эффекта" },
  { id: "robot", title: "Робот" },
  { id: "chipmunk", title: "Бурундук" },
  { id: "giant", title: "Гигант" },
  { id: "villain", title: "Злодей" },
  { id: "radio", title: "Рация" },
  { id: "phone", title: "Телефон" },
  { id: "ghost", title: "Призрак" },
];

export const roomTitle = (id: string) => ROOMS.find((r) => r.id === id)?.title ?? id;
export const fxTitle = (id: string) => VOICE_FX.find((r) => r.id === id)?.title ?? id;

export const LLM_TASKS: { id: import("./types").LLMTask; title: string; hint: string }[] = [
  { id: "translate", title: "Перевод", hint: "перевести реплики на русский" },
  { id: "fit_length", title: "Подогнать под длину", hint: "предложить текст, который успеешь произнести за время реплики" },
  { id: "name_speakers", title: "Имена персонажей", hint: "угадать, кто есть кто, по диалогам" },
  { id: "director", title: "Помощник-режиссёр", hint: "замечания по дублям и просьбы обычными словами" },
];

export const PLATFORMS: { id: string; title: string; lufs: number }[] = [
  { id: "yt", title: "YouTube / VK Видео", lufs: -14 },
  { id: "shorts", title: "Reels / Shorts / TikTok", lufs: -14 },
  { id: "tg", title: "Telegram", lufs: -16 },
  { id: "tv", title: "ТВ и кино", lufs: -23 },
];

/** Какую площадку предлагать, когда меняют формат ролика: вертикальное — для коротких роликов, остальное — как раньше. */
export const platformForAspect = (aspect: string, current: string): string =>
  aspect === "9:16" ? "shorts" : current === "shorts" ? "yt" : current;

export const defaultTrack = () => ({ gain_db: 0, mute: false, solo: false, room: "auto", voice_fx: "none", boost_capped: false });

export const defaultMix = () => ({
  mode: "dub" as const,
  autopilot: true,
  duck_background: true,
  background: defaultTrack(),
  original_voices_db: -14,
  tracks: {},
  target_lufs: -14,
  true_peak_db: -1,
  keep_breaths: true,
  loudness_target: "online" as "online" | "film",
  denoise: true,
});
