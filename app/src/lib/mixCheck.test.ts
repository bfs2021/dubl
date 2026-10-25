import { describe, expect, it } from "vitest";
import { MIX_WARN_TEXT, mixWarning } from "./mixCheck";
import { applyPatches, takeRemark, takeReport } from "./director";
import { isQuiet, visibleRemark } from "./takeReview";
import { quietAdvice, verdictOf } from "./micProfile";
import { defaultMix, defaultTrack } from "../api/presets";
import type { MixSettings, Take } from "../api/types";

const take = (x: Partial<Take>) => ({ id: "t1", line_id: "l1", speaker_id: "s1", selected: true, peak_db: -8, loudness_lufs: -21, ...x } as Take);
const mix = (x: Partial<MixSettings> = {}) => ({ ...(defaultMix() as MixSettings), ...x });

describe("mixWarning: голос тонет / ролик сильно поднимают", () => {
  it("проверка движка после сборки — главная", () => {
    const off = mix({ autopilot: false });
    expect(mixWarning({ mix: off, takes: [], mix_check: { voice_over_bg_db: -1.7, master_gain_db: 19.3, warning: "autopilot" } })!.text).toBe(MIX_WARN_TEXT.autopilot);
    expect(mixWarning({ mix: off, takes: [], mix_check: { voice_over_bg_db: 8, master_gain_db: 14, warning: "autopilot" } })!.text).toBe(MIX_WARN_TEXT.autopilot_boost);
    expect(mixWarning({ mix: mix(), takes: [], mix_check: { voice_over_bg_db: 2, master_gain_db: 5, warning: "buried" } })!.kind).toBe("buried");
    // движок сказал «всё хорошо» — молчим, даже если дубль тихий
    expect(mixWarning({ mix: off, takes: [take({ loudness_lufs: -40 })], mix_check: { voice_over_bg_db: 12, master_gain_db: 5, warning: null } })).toBeNull();
  });
  it("до сборки — прикидка по дублям: тихий дубль без автопилота → «включи автопилот»", () => {
    const tracks = { s1: { ...defaultTrack(), gain_db: -2 } };
    // случай пользователя: −35,5 LUFS и дорожка −2 дБ
    expect(mixWarning({ mix: mix({ autopilot: false, tracks }), takes: [take({ loudness_lufs: -35.5, peak_db: -15.8 })] })!.kind).toBe("autopilot");
    // дорожку подняли руками до нормы — молчим
    expect(mixWarning({ mix: mix({ autopilot: false, tracks: { s1: { ...defaultTrack(), gain_db: 12 } } }), takes: [take({ loudness_lufs: -35.5 })] })).toBeNull();
    // автопилот включён — он сам поднимет; молчим, пока не упёрся в потолок
    expect(mixWarning({ mix: mix(), takes: [take({ loudness_lufs: -35.5 })] })).toBeNull();
    expect(mixWarning({ mix: mix({ tracks: { s1: { ...defaultTrack(), boost_capped: true } } }), takes: [take({ loudness_lufs: -50 })] })!.text).toBe(MIX_WARN_TEXT.record_louder);
    expect(mixWarning({ mix: mix({ autopilot: false }), takes: [take({})] })).toBeNull();
  });
});

describe("тихий дубль — по громкости речи, а не только по пику", () => {
  const base = { offset_ms: 0, stretch: 1, duration: 2, fit_confidence: 0.9 };
  it("пик нормальный, речь −35,5 LUFS → «Тихо записано…»", () => {
    const r = takeRemark({ ...base, peak_db: -15.8, loudness_lufs: -35.5 }, 2);
    expect(r).toEqual({ severity: "warn", text: "Тихо записано — поднеси микрофон ближе или прибавь громкость микрофона в системе." });
    expect(takeReport({ id: "t", ...base, peak_db: -15.8, loudness_lufs: -35.5 }, "l1", 2).loudness).toBe("too_quiet");
    expect(takeRemark({ ...base, peak_db: -8, loudness_lufs: -21 }, 2).severity).toBe("ok");
    expect(isQuiet({ peak_db: -15.8, loudness_lufs: -35.5 })).toBe(true);
    expect(isQuiet({ peak_db: -8, loudness_lufs: -21 })).toBe(false);
    expect(isQuiet({ peak_db: -60, loudness_lufs: -70 })).toBe(false); // это «почти пусто», не «тихо»
  });
  it("строка «тихо» уже есть — тот же совет помощника второй раз не показываем, другой — оставляем", () => {
    expect(visibleRemark({ text: "Тебя еле слышно — подойди ближе.", severity: "warn" as const }, { lips: true, file: false, quietShown: true })).toBeUndefined();
    expect(visibleRemark({ text: "Тихо записано — поднеси микрофон ближе.", more: "Начни чуть раньше.", severity: "warn" as const }, { lips: true, file: false, quietShown: true })!.text).toBe("Начни чуть раньше.");
    expect(visibleRemark({ text: "Вступил поздно.", severity: "warn" as const }, { lips: true, file: false, quietShown: true })!.text).toBe("Вступил поздно.");
  });
});

describe("автопилот выключают только правки громкости", () => {
  it("комната, эффект, «выкл.» — автопилот остаётся", () => {
    const m = applyPatches(mix(), [{ target: "s1", field: "room", value: "street", note: "" }, { target: "s1", field: "voice_fx", value: "robot", note: "" }, { target: "s2", field: "mute", value: true, note: "" }]);
    expect(m.autopilot).toBe(true);
    expect(m.tracks.s1.room).toBe("street");
  });
  it("громкость, цель, приглушение фона, режим — выключают", () => {
    for (const p of [{ target: "s1", field: "gain_db", value: 3, note: "" }, { target: "master", field: "target_lufs", value: -16, note: "" }, { target: "master", field: "duck_background", value: false, note: "" }, { target: "master", field: "mode", value: "voiceover", note: "" }] as const) {
      expect(applyPatches(mix(), [p]).autopilot).toBe(false);
    }
  });
});

describe("проверка микрофона: тихо по средней громкости", () => {
  it("профиль с ПК (речь −35,4, пик −14,6, шум −67) — «тихо», совет про настройки системы", () => {
    expect(verdictOf(-67, -14.6, -35.4)).toBe("quiet");
    expect(verdictOf(-67, -10, -24)).toBe("good");
    expect(quietAdvice("Windows")).toBe("Микрофон тихий: прибавь ему громкость в настройках звука Windows или подойди ближе к микрофону — на ширину ладони.");
    expect(quietAdvice("macOS")).toContain("настройках звука macOS");
    expect(quietAdvice("")).toBe("Микрофон тихий: прибавь ему громкость в настройках звука или подойди ближе к микрофону — на ширину ладони.");
  });
});
