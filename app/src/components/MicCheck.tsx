import { useCallback, useEffect, useRef, useState } from "react";
import { Check, Headphones, Mic, RotateCw, Volume2, X } from "lucide-react";
import { MicRecorder, listMicrophones, measureLatency, measureLatencyEngine, micErrorText, micPermission, MIC_ASKING_TEXT, MIC_DENIED_HELP, levelStep, type LevelReading } from "../audio/recorder";
import { buildProfile, noiseFloor, speechStats, VERDICT_TEXT, ZONE_TEXT, zoneOf, type MicProfile, type MicVerdict } from "../lib/micProfile";
import { wavBlob } from "../audio/wav";
import { fmtDb } from "../lib/format";
import { useStore } from "../state/store";
import { ZoneMeter } from "./ZoneMeter";

type Step = "intro" | "asking" | "silence" | "speech" | "beeps" | "result" | "error";

/** Что умеет запись микрофона в мастере (MicRecorder; в тестах — заглушка). */
export type CheckRecorder = Pick<MicRecorder, "open" | "close" | "getLevel" | "startCapture" | "stopCapture" | "ctx" | "isOpen">;

interface Props {
  open: boolean;
  onClose: () => void;
  /** сколько длится тишина и фраза, мс (по умолчанию 2 с и 5 с — вместе с бипами ≈ 10 секунд) */
  timing?: { silenceMs: number; speechMs: number };
  /** подмена записи в тестах */
  makeRecorder?: () => CheckRecorder;
  /** подмена замера задержки в тестах */
  latencyFn?: (rec: CheckRecorder) => Promise<number>;
}

const PHRASE = "Раз, два, три — проверка микрофона. Говорю, как будто озвучиваю кадр.";
const DEFAULT_TIMING = { silenceMs: 2000, speechMs: 5000 };

/**
 * Мастер «Проверка микрофона» на ~10 секунд:
 *  1) 2 секунды тишины — шум комнаты;
 *  2) фраза вслух — живая шкала с цветной зоной «сюда попадай» и совет;
 *  3) бипы через динамики — задержка записи (только без наушников; с наушниками шаг пропускается).
 * Итог — профиль «Мой микрофон»: хранится в настройках устройства, задержка уходит в настройки движка.
 */
export function MicCheck({ open, onClose, timing = DEFAULT_TIMING, makeRecorder, latencyFn }: Props) {
  const ui = useStore((s) => s.ui);
  const setUi = useStore((s) => s.setUi);
  const settings = useStore((s) => s.settings);
  const engine = useStore((s) => s.engine);
  const saveSettings = useStore((s) => s.saveSettings);
  const toast = useStore((s) => s.toast);

  const [step, setStep] = useState<Step>("intro");
  const [err, setErr] = useState<string | null>(null);
  const [level, setLevel] = useState(-120);
  const [hold, setHold] = useState(-120);
  const [t, setT] = useState(0);
  const [phones, setPhones] = useState(ui.headphones);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string | null>(settings?.input_device ?? null);
  const [profile, setProfile] = useState<MicProfile | null>(null);
  const [beepNote, setBeepNote] = useState<string | null>(null);
  const [engineAdvice, setEngineAdvice] = useState<string | null>(null);
  const [needSystem, setNeedSystem] = useState(false);
  const rec = useRef<CheckRecorder | null>(null);
  const run = useRef(0);
  const meter = useRef<LevelReading>({ db: -120, peak: -120, heard: false });

  const stop = useCallback(async () => {
    run.current++;
    const r = rec.current;
    rec.current = null;
    try { await r?.close(); } catch { /* уже закрыт */ }
  }, []);

  // закрыли окно или ушли с экрана — микрофон отпускаем
  useEffect(() => { if (!open) { void stop(); } return () => { void stop(); }; }, [open, stop]);
  useEffect(() => {
    if (!open) return;
    setStep("intro"); setErr(null); setProfile(null); setBeepNote(null); setEngineAdvice(null); setLevel(-120); setHold(-120); setNeedSystem(false);
    setPhones(useStore.getState().ui.headphones);
    setDeviceId(useStore.getState().settings?.input_device ?? null);
    void listMicrophones().then(setDevices).catch(() => {});
  }, [open]);

  // живая шкала: читаем уровень ~30 раз в секунду, пока открыт микрофон
  useEffect(() => {
    if (step !== "silence" && step !== "speech") return;
    meter.current = { db: -120, peak: -120, heard: false };
    let last = performance.now();
    const iv = setInterval(() => {
      const now = performance.now();
      meter.current = levelStep(meter.current, rec.current?.getLevel() ?? -120, now - last);
      last = now;
      setLevel(meter.current.db);
      setHold((h) => Math.max(h - 0.3, meter.current.db));
    }, 33);
    return () => clearInterval(iv);
  }, [step]);

  const wait = (ms: number, token: number) => new Promise<boolean>((res) => {
    const t0 = performance.now();
    const iv = setInterval(() => {
      const p = Math.min(1, (performance.now() - t0) / ms);
      setT(p);
      if (token !== run.current) { clearInterval(iv); res(false); }
      else if (p >= 1) { clearInterval(iv); res(true); }
    }, 50);
  });

  const start = async () => {
    const token = ++run.current;
    setErr(null); setBeepNote(null); setHold(-120); setNeedSystem(false);
    setStep("asking");
    // если система ещё не спрашивала, подскажем, что делать с окном macOS
    void micPermission().then((p) => { if (token === run.current && p === "prompt") setNeedSystem(true); });
    const r = (makeRecorder ?? (() => new MicRecorder()))();
    rec.current = r;
    try {
      await r.open(deviceId);
    } catch (e) {
      if (token !== run.current) return;
      rec.current = null;
      setErr(micErrorText(e));
      setStep("error");
      return;
    }
    if (token !== run.current) return;
    setNeedSystem(false);
    void listMicrophones().then(setDevices).catch(() => {});
    setUi({ headphones: phones });

    // 1) тишина
    setStep("silence"); setT(0);
    r.startCapture();
    if (!(await wait(timing.silenceMs, token))) return;
    const quiet = r.stopCapture();
    const noiseDb = noiseFloor(quiet.samples);

    // 2) фраза
    setStep("speech"); setT(0); setHold(-120);
    r.startCapture();
    if (!(await wait(timing.speechMs, token))) return;
    const said = r.stopCapture();
    const sp = speechStats(said.samples, said.sampleRate, noiseDb);

    // разбор делает движок (он же кладёт результат в профиль «Мой микрофон»); если он не ответил — считаем сами
    let verdictFromEngine: MicVerdict | null = null;
    let adviceFromEngine: string | null = null;
    if (engine) {
      try {
        const all = new Float32Array(quiet.samples.length + said.samples.length);
        all.set(quiet.samples, 0);
        all.set(said.samples, quiet.samples.length);
        const r = await engine.micCheck(wavBlob(all, said.sampleRate), quiet.samples.length / quiet.sampleRate);
        if (r.speech_found) { verdictFromEngine = r.verdict as MicVerdict; adviceFromEngine = r.advice; }
      } catch { /* старый движок или демо — выводы сделаем по своим замерам */ }
      if (token !== run.current) return;
    }

    // 3) бипы через динамики
    let latency: number | null = null;
    if (!phones) {
      setStep("beeps"); setT(0);
      try {
        const viaEngine = async (x: CheckRecorder) => {
          if (!engine) throw new Error("no engine");
          const sig = await engine.micSignal();
          return measureLatencyEngine(x as MicRecorder, sig, (w, off) => engine.micLatency(w, off));
        };
        latency = await (latencyFn ?? (async (x) => { try { return await viaEngine(x); } catch (e) { if (/Нет эталонного|no engine/.test((e as Error).message) || !(e instanceof Error)) return measureLatency(x as MicRecorder); throw e; } }))(r);
      } catch (e) {
        setBeepNote((e as Error).message);
      }
      if (token !== run.current) return;
    }
    await stop();
    const dev = devices.find((d) => d.deviceId === deviceId);
    const p = buildProfile({ noiseDb, speechPeak: sp.peak, speechRms: sp.rms, latencyMs: latency, headphones: phones, deviceId, deviceLabel: dev?.label ?? null });
    if (verdictFromEngine) p.verdict = verdictFromEngine;
    setEngineAdvice(adviceFromEngine);
    setProfile(p);
    setStep("result");
    // профиль — на устройстве; задержка — в настройки движка (её использует подгонка дублей)
    setUi({ mic_profile: p });
    if (settings) {
      // mic_profile движок из PUT /settings не берёт (его меняют только роуты /mic/*) — шлём задержку и выбранный микрофон
      const { mic_profile: _drop, ...rest } = settings;
      const next = { ...rest, ...(latency != null ? { latency_ms: latency } : {}), input_device: deviceId } as typeof settings;
      void saveSettings(next).catch(() => toast("Профиль микрофона сохранён только на этом устройстве.", "info"));
    }
  };

  const close = () => { void stop(); onClose(); };

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;
  const zone = zoneOf(level);
  const running = step === "silence" || step === "speech" || step === "beeps" || step === "asking";
  const stepNo = step === "silence" ? 1 : step === "speech" ? 2 : step === "beeps" ? 3 : 0;
  const total = phones ? 2 : 3;

  return (
    <div className="modal-back" onMouseDown={(e) => { if (e.target === e.currentTarget) close(); }}>
      <div className="modal micwiz" role="dialog" aria-modal="true" aria-labelledby="micwiz-t" data-testid="mic-check">
        <div className="between">
          <h3 id="micwiz-t"><Mic size={15} style={{ verticalAlign: "-2px" }} /> Проверка микрофона <span className="mute" style={{ fontWeight: 400 }}>· 10 секунд</span></h3>
          <button className="btn ghost sm iconbtn" aria-label="Закрыть" onClick={close}><X size={14} /></button>
        </div>

        {step === "intro" && (
          <>
            <p className="note">Проверим три вещи: тихо ли в комнате, хватает ли голоса и не опаздывает ли запись. Результат сохраним как «Мой микрофон» — потом это можно не повторять.</p>
            <label className="field"><span>Какой микрофон</span>
              <select className="inp" aria-label="Микрофон" value={deviceId ?? ""} onChange={(e) => setDeviceId(e.target.value || null)}>
                <option value="">По умолчанию</option>
                {devices.map((d, i) => <option key={d.deviceId || i} value={d.deviceId}>{d.label || `Микрофон ${i + 1}`}</option>)}
              </select></label>
            <div className="opt"><span className="l"><Headphones size={13} style={{ verticalAlign: "-2px" }} /> Я в наушниках<br /><span className="note">С наушниками бипы из динамиков не нужны — этот шаг пропустим.</span></span>
              <label className="sw"><input type="checkbox" role="switch" aria-label="Я в наушниках" checked={phones} onChange={(e) => setPhones(e.target.checked)} /><i /></label></div>
            <div className="modal-actions"><button className="btn" onClick={close}>Не сейчас</button><button className="btn primary" onClick={() => void start()} data-primary="micstart"><Mic size={13} />Начать проверку</button></div>
          </>
        )}

        {step === "asking" && (
          <div className="micstep" role="status" data-testid="mic-asking">
            <p><b>Включаю микрофон…</b></p>
            {needSystem && <p className="note warnnote">{MIC_ASKING_TEXT}</p>}
          </div>
        )}

        {running && step !== "asking" && (
          <div className="micstep">
            <div className="steps-dots" aria-label={`Шаг ${stepNo} из ${total}`}>
              {Array.from({ length: total }, (_, i) => <i key={i} className={i + 1 < stepNo ? "done" : i + 1 === stepNo ? "cur" : ""} />)}
              <span className="note">Шаг {stepNo} из {total}</span>
            </div>
            {step === "silence" && (<><p><b>Тишина.</b> Помолчи пару секунд — запомню шум комнаты.</p><ZoneMeter db={level} hold={undefined} /></>)}
            {step === "speech" && (
              <>
                <p><b>Скажи вслух:</b> <span className="phrase">«{PHRASE}»</span></p>
                <ZoneMeter db={level} hold={hold} />
                <div className={"zoneadvice " + zone} role="status" aria-live="polite" data-testid="zone-advice">{ZONE_TEXT[zone]} <span className="mono mute">{Math.max(-60, Math.round(level))} дБ</span></div>
              </>
            )}
            {step === "beeps" && (
              <p><Volume2 size={14} style={{ verticalAlign: "-2px" }} /> <b>Слушаю бипы из динамиков.</b> Включи звук погромче — замеряю, на сколько запись опаздывает.</p>
            )}
            <div className="calbar"><i style={{ width: Math.round(t * 100) + "%" }} /></div>
          </div>
        )}

        {step === "error" && (
          <div className="micstep" data-testid="mic-error">
            <p className="calres bad" role="alert">{err}</p>
            {err === MIC_DENIED_HELP && <p className="note">После этого вернись сюда и нажми «Попробовать ещё раз».</p>}
            <div className="modal-actions"><button className="btn" onClick={close}>Закрыть</button><button className="btn primary" onClick={() => void start()}><RotateCw size={13} />Попробовать ещё раз</button></div>
          </div>
        )}

        {step === "result" && profile && (
          <div className="micstep" data-testid="mic-result">
            <div className={"verdict " + profile.verdict}>
              {profile.verdict === "good" ? <Check size={16} /> : <Mic size={16} />}
              <div><b>{VERDICT_TEXT[profile.verdict].title}</b><div className="note">{engineAdvice ?? VERDICT_TEXT[profile.verdict].advice}</div></div>
            </div>
            <dl className="micnums">
              <div><dt>Шум комнаты</dt><dd className="mono">{fmtDb(profile.noise_floor_db, 0)}</dd></div>
              <div><dt>Голос (пик)</dt><dd className="mono">{fmtDb(profile.speech_peak_db, 0)}</dd></div>
              <div><dt>Голос выше шума</dt><dd className="mono">{Math.round(profile.snr_db)} дБ</dd></div>
              <div><dt>Задержка записи</dt><dd className="mono">{profile.latency_ms != null ? Math.round(profile.latency_ms) + " мс" : "не мерили"}</dd></div>
            </dl>
            {beepNote && <p className="note warnnote" role="status">{beepNote}</p>}
            {profile.latency_ms == null && profile.headphones && <p className="note">Ты в наушниках, поэтому задержку не мерили. Её можно замерить позже без наушников — «Настройки → Микрофон».</p>}
            <p className="note"><Check size={11} style={{ verticalAlign: "-1px" }} /> Сохранено как «Мой микрофон».</p>
            <div className="modal-actions"><button className="btn" onClick={() => void start()}><RotateCw size={13} />Проверить ещё раз</button><button className="btn primary" onClick={close} data-primary="micdone">Готово</button></div>
          </div>
        )}
      </div>
    </div>
  );
}
