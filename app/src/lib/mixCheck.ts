/**
 * Честное предупреждение по звуку сцены: голос тонет в фоне или ролик приходится сильно поднимать.
 * Главный источник — Scene.mix_check (движок мерит после каждой сборки превью/ролика). Пока сборки после правок не было,
 * прикидываем сами по замерам дублей, чтобы сказать заранее, ещё до рендера.
 */
import type { Scene } from "../api/types";
import { QUIET_LUFS, takeIsUsable, type TakeX } from "./takeReview";

/** Как в движке (audio.mix): голос над фоном меньше — тонет; мастер поднимает больше — вылезут шумы фона. */
export const VOICE_BURIED_DB = 4;
export const MASTER_BOOST_WARN_DB = 10;

export type MixWarnKind = "autopilot" | "record_louder" | "buried";
export interface MixWarning {
  kind: MixWarnKind;
  text: string;
}

export const MIX_WARN_TEXT = {
  /** голос тише фона, автопилот выключен */
  autopilot: "Голос тише музыки — включи автопилот, он сам выровняет громкость.",
  /** голос нормальный, но весь ролик пришлось сильно поднимать, автопилот выключен */
  autopilot_boost: "Ролик вышел тихим, и его пришлось сильно поднять — вместе с голосом вылезут шумы. Включи автопилот, он сам выровняет громкость.",
  record_louder: "Голос записан тихо — перезапиши громче или поднеси микрофон ближе.",
  buried: "Голос тонет в музыке — убавь «Музыку и шумы» ползунком в «Дорожках».",
} as const;

/** Что сказать про звук сцены. null — всё в порядке (или проверить нечем): ничего не показываем. */
export function mixWarning(scene: Pick<Scene, "mix" | "takes"> & { mix_check?: Scene["mix_check"] }): MixWarning | null {
  const chk = scene.mix_check;
  if (chk) {
    if (!chk.warning) return null;
    if (chk.warning === "autopilot") {
      const buried = chk.voice_over_bg_db != null && chk.voice_over_bg_db < VOICE_BURIED_DB;
      return { kind: "autopilot", text: buried ? MIX_WARN_TEXT.autopilot : MIX_WARN_TEXT.autopilot_boost };
    }
    return { kind: chk.warning, text: MIX_WARN_TEXT[chk.warning] };
  }
  // сборки после правок ещё не было — прикидка по замерам дублей
  const mix = scene.mix;
  if (mix.autopilot) {
    if (Object.values(mix.tracks).some((t) => t.boost_capped)) return { kind: "record_louder", text: MIX_WARN_TEXT.record_louder };
    return null;
  }
  // автопилот выключен: дубль, который и после усиления дорожки тише QUIET_LUFS, в ролике утонет
  const quiet = scene.takes.some((t) => {
    const x = t as TakeX;
    if (!x.selected || !takeIsUsable(x) || x.loudness_lufs == null) return false;
    const gain = (x.speaker_id && mix.tracks[x.speaker_id]?.gain_db) || 0;
    return x.loudness_lufs + gain < QUIET_LUFS;
  });
  return quiet ? { kind: "autopilot", text: MIX_WARN_TEXT.autopilot } : null;
}
