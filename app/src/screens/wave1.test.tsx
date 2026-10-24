import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { freshStore, openFridge, renderRoute } from "../test/helpers";
import { useStore } from "../state/store";
import { RecordScreen } from "./Record";
import { MixScreen } from "./Mix";
import { RenderScreen } from "./Render";
import { ProjectsScreen } from "./Projects";
import { MicCheck, type CheckRecorder } from "../components/MicCheck";
import { MIC_ASKING_TEXT, MIC_DENIED_HELP, micUi } from "../audio/recorder";
import type { Take } from "../api/types";

beforeEach(async () => { await freshStore(); });

/** Запись-заглушка: тишина −62 дБ, потом «речь» (по умолчанию −14 дБ). */
function fakeRecorder(opts: { fail?: DOMException; speechDb?: number } = {}) {
  const calls: string[] = [];
  let level = -62;
  let capture = 0;
  const rec: CheckRecorder = {
    ctx: null, isOpen: true,
    open: async () => { calls.push("open"); if (opts.fail) throw opts.fail; },
    close: async () => { calls.push("close"); },
    getLevel: () => level,
    startCapture: () => { capture++; level = capture % 2 === 1 ? -62 : (opts.speechDb ?? -14); },
    stopCapture: () => {
      const amp = Math.pow(10, level / 20);
      const samples = Float32Array.from({ length: 48000 }, (_, i) => amp * Math.sin(i / 7));
      return { samples, sampleRate: 48000, duration: 1, peakDb: level, rmsDb: level - 3 };
    },
  };
  return { rec, calls };
}

describe("мастер проверки микрофона", () => {
  const timing = { silenceMs: 80, speechMs: 80 };
  it("закрыт — ничего не рисует", () => {
    const { container } = render(<MicCheck open={false} onClose={() => {}} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("тишина → фраза → результат; с наушниками бипы пропускаются; профиль сохраняется", async () => {
    const { rec, calls } = fakeRecorder();
    const onClose = vi.fn();
    const latency = vi.fn();
    render(<MicCheck open onClose={onClose} timing={timing} makeRecorder={() => rec} latencyFn={latency} />);
    await userEvent.click(screen.getByRole("switch", { name: "Я в наушниках" }));
    await userEvent.click(screen.getByRole("button", { name: /Начать проверку/ }));
    const res = await screen.findByTestId("mic-result", {}, { timeout: 3000 });
    expect(res).toHaveTextContent("Микрофон в порядке");
    expect(res).toHaveTextContent(/Ты в наушниках, поэтому задержку не мерили/);
    expect(latency).not.toHaveBeenCalled();
    expect(calls).toEqual(["open", "close"]);
    const p = useStore.getState().ui.mic_profile!;
    expect(p.verdict).toBe("good");
    expect(p.latency_ms).toBeNull();
    expect(p.headphones).toBe(true);
    await userEvent.click(screen.getByRole("button", { name: "Готово" }));
    expect(onClose).toHaveBeenCalled();
  });

  it("без наушников меряет задержку бипами и записывает её в настройки", async () => {
    const { rec } = fakeRecorder();
    const saved = vi.spyOn(useStore.getState(), "saveSettings");
    render(<MicCheck open onClose={() => {}} timing={timing} makeRecorder={() => rec} latencyFn={async () => 87} />);
    await userEvent.click(screen.getByRole("button", { name: /Начать проверку/ }));
    const res = await screen.findByTestId("mic-result", {}, { timeout: 3000 });
    expect(res).toHaveTextContent("87 мс");
    expect(useStore.getState().ui.mic_profile?.latency_ms).toBe(87);
    await waitFor(() => expect(saved).toHaveBeenCalled());
    const sent = saved.mock.calls[0][0] as unknown as Record<string, unknown>;
    expect(sent.latency_ms).toBe(87);
    expect("mic_profile" in sent).toBe(false); // профиль в PUT /settings не шлём: его меняют только роуты /mic/*
  });

  it("тихий голос → совет «придвинься ближе»", async () => {
    const { rec } = fakeRecorder({ speechDb: -38 });
    render(<MicCheck open onClose={() => {}} timing={timing} makeRecorder={() => rec} latencyFn={async () => 50} />);
    await userEvent.click(screen.getByRole("button", { name: /Начать проверку/ }));
    const res = await screen.findByTestId("mic-result", {}, { timeout: 3000 });
    expect(res).toHaveTextContent("Тихо");
    expect(res).toHaveTextContent(/ближе к микрофону/i); // совет приходит от движка (/mic/check)
  });

  it("разбор отдаёт движок: его совет и вердикт попадают в профиль; когда он молчит, считаем сами", async () => {
    const { rec } = fakeRecorder();
    const engine = useStore.getState().engine!;
    const spy = vi.spyOn(engine, "micCheck").mockResolvedValue({ noise_floor_db: -58, speech_db: -20, peak_db: -9, snr_db: 38, clipping: false, speech_found: true, verdict: "noisy", advice: "В комнате шумно — закрой окно.", saved: true });
    render(<MicCheck open onClose={() => {}} timing={timing} makeRecorder={() => rec} latencyFn={async () => 70} />);
    await userEvent.click(screen.getByRole("button", { name: /Начать проверку/ }));
    const res = await screen.findByTestId("mic-result", {}, { timeout: 3000 });
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0][1]).toBeCloseTo(1, 1); // сколько секунд тишины в начале записи
    expect(res).toHaveTextContent("В комнате шумно — закрой окно.");
    expect(useStore.getState().ui.mic_profile?.verdict).toBe("noisy");
    spy.mockRejectedValue(new Error("404"));
    await userEvent.click(screen.getByRole("button", { name: /Проверить ещё раз/ }));
    await waitFor(() => expect(screen.queryByTestId("mic-result")).toBeNull());
    const again = await screen.findByTestId("mic-result", {}, { timeout: 3000 });
    expect(again).toHaveTextContent("Микрофон в порядке");
  });

  it("доступ запрещён → объяснение, где включить, и «Попробовать ещё раз»", async () => {
    const { rec } = fakeRecorder({ fail: new DOMException("no", "NotAllowedError") });
    render(<MicCheck open onClose={() => {}} timing={timing} makeRecorder={() => rec} />);
    await userEvent.click(screen.getByRole("button", { name: /Начать проверку/ }));
    const err = await screen.findByTestId("mic-error");
    expect(err).toHaveTextContent("Системные настройки");
    expect(within(err).getByRole("button", { name: /Попробовать ещё раз/ })).toBeInTheDocument();
  });

  it("Esc закрывает окно и отпускает микрофон", async () => {
    const { rec, calls } = fakeRecorder();
    const onClose = vi.fn();
    render(<MicCheck open onClose={onClose} timing={{ silenceMs: 3000, speechMs: 3000 }} makeRecorder={() => rec} latencyFn={async () => 1} />);
    await userEvent.click(screen.getByRole("button", { name: /Начать проверку/ }));
    await screen.findByText(/Помолчи пару секунд/);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
    await waitFor(() => expect(calls).toContain("close"));
  });
});

describe("микрофон: «Разреши доступ» вместо «выключен»", () => {
  it("состояния панели", () => {
    expect(micUi({ open: true, opening: false, permission: "granted", failed: false })).toBe("on");
    expect(micUi({ open: false, opening: true, permission: "prompt", failed: false })).toBe("asking");
    expect(micUi({ open: false, opening: true, permission: "unknown", failed: false })).toBe("asking");
    expect(micUi({ open: false, opening: false, permission: "denied", failed: false })).toBe("denied");
    expect(micUi({ open: false, opening: false, permission: "unknown", failed: true })).toBe("denied");
    expect(micUi({ open: false, opening: false, permission: "prompt", failed: false })).toBe("off");
    expect(MIC_ASKING_TEXT).toMatch(/окне macOS/);
    expect(MIC_DENIED_HELP).toMatch(/Системные настройки/);
  });
  it("на экране записи нет слов «Микрофон выключен»", async () => {
    await openFridge();
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    expect(screen.queryByText(/Микрофон выключен/)).toBeNull();
    expect(screen.getByTestId("mic-state")).toBeInTheDocument();
  });
});

describe("Запись: проверка микрофона из панели", () => {
  it("нет профиля — подсказка «Проверить микрофон» открывает мастер; есть — «Мой микрофон»", async () => {
    await openFridge();
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    const prompt = screen.getByTestId("mic-prompt");
    await userEvent.click(within(prompt).getByRole("button", { name: /Проверить микрофон/ }));
    expect(await screen.findByTestId("mic-check")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Закрыть" }));
    expect(screen.queryByTestId("mic-check")).toBeNull();
    act(() => useStore.getState().setUi({ mic_profile: { checked_at: new Date().toISOString(), device_id: null, device_label: null, noise_floor_db: -58, speech_peak_db: -10, speech_rms_db: -22, snr_db: 36, latency_ms: 99, headphones: false, verdict: "good" } }));
    expect(await screen.findByTestId("mic-profile")).toHaveTextContent(/Мой микрофон: Микрофон в порядке · задержка 99 мс/);
    expect(screen.queryByTestId("mic-prompt")).toBeNull();
  });
  it("на главной шаг «Перед первой записью» запускает мастер", async () => {
    renderRoute(<ProjectsScreen />, "/", "/");
    const step = await screen.findByTestId("mic-step");
    await userEvent.click(within(step).getByRole("button", { name: /Проверить/ }));
    expect(await screen.findByTestId("mic-check")).toBeInTheDocument();
  });
});

describe("Репетиция", () => {
  it("кнопка есть; сначала оригинал; «Стоп» прерывает; запись не создаётся", async () => {
    await openFridge();
    const up = vi.spyOn(useStore.getState().engine!, "uploadTake");
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    const btn = screen.getByTestId("rehearse");
    expect(btn).toHaveTextContent("Репетиция");
    await userEvent.click(btn);
    const badge = await screen.findByTestId("rehearsal-badge");
    expect(badge).toHaveTextContent(/слушай оригинал/);
    expect(screen.getByRole("button", { name: /Стоп репетиции/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Записать реплику/ })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: /Стоп репетиции/ }));
    await waitFor(() => expect(screen.queryByTestId("rehearsal-badge")).toBeNull());
    expect(screen.getByTestId("rehearse")).toBeEnabled();
    expect(up).not.toHaveBeenCalled();
  });
});

describe("Счётчик слогов в редакторе реплики", () => {
  it("«Свой текст»: меняется живьём — влезет / не влезет; кнопка ИИ со скобкой", async () => {
    await openFridge();
    await useStore.getState().setTextMode("custom");
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    const item = document.querySelector("[data-line]") as HTMLElement;
    const box = within(item).getByRole("textbox");
    fireEvent.change(box, { target: { value: "Да" } });
    expect(within(item).getByTestId("budget-label")).toHaveTextContent("влезет");
    fireEvent.change(box, { target: { value: "Ну что же ты наделал, несчастный ты наш Барсик, посмотри на этот холодильник и скажи что-нибудь" } });
    expect(within(item).getByTestId("budget-label")).toHaveTextContent("не влезет");
    expect(within(item).getByTestId("budget-trim")).toHaveTextContent(/убрать примерно \d+ сл/);
    expect(within(item).getByTestId("syllable-budget")).toHaveTextContent(/оригинал/);
    expect(within(item).getByTestId("syllable-budget")).toHaveTextContent(/сл\/с/);
    expect(within(item).getByRole("button", { name: /Подогнать под длину \(ИИ\)/ })).toBeInTheDocument();
  });
});

const TAKE: Take = {
  id: "tx", line_id: "", speaker_id: null, file: "takes/tx.wav", duration: 2, preroll: 1, start_rel: 0, offset_ms: 700, stretch: 1, fit_confidence: 0.64,
  peak_db: -8, loudness_lufs: -20, recognized_text: null, selected: false, source: "mic", empty: false, chosen_by_user: false, tail_trimmed: false,
};

async function withTakes(make: () => Partial<Take>[]) {
  await openFridge();
  const st = useStore.getState();
  const sc = st.project!.scenes[0];
  const line = sc.lines[0];
  const takes = make().map((p, i) => ({ ...TAKE, id: `x${i}`, line_id: line.line_id, speaker_id: line.speaker_id, ...p }));
  useStore.setState({ project: { ...st.project!, scenes: [{ ...sc, takes }, ...st.project!.scenes.slice(1)] } });
  return takes;
}

describe("Разбор дубля по тексту", () => {
  it("«Всё сказал ✓» и ★ с причиной у самого полного", async () => {
    await withTakes(() => [{ text_match: 0.5, complete: false }, { text_match: 1, complete: true, selected: true }]);
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    const reviews = await screen.findAllByTestId("take-review");
    expect(reviews.map((r) => r.getAttribute("data-kind")).sort()).toEqual(["full", "partial"]);
    expect(screen.getByText("Всё сказал ✓")).toBeInTheDocument();
    expect(screen.getByTestId("star-why")).toHaveTextContent("Самый полный: сказано 100% текста");
  });

  it("повтор в конце → «отрезать?» и кнопка работает через движок", async () => {
    await withTakes(() => [{ complete: true, text_match: 1, tail_cut_s: 1.4, tail_text: "давай ещё раз", selected: true }]);
    const engine = useStore.getState().engine!;
    const spy = vi.spyOn(engine, "trimTakeTail").mockResolvedValue({ ...TAKE });
    renderRoute(<RecordScreen />);
    const note = await screen.findByTestId("take-review");
    expect(note).toHaveTextContent("В конце повтор («давай ещё раз») — отрезать?");
    await userEvent.click(within(note).getByRole("button", { name: /Отрезать/ }));
    await waitFor(() => expect(spy).toHaveBeenCalledWith("p-fridge", "sc001", "x0"));
  });

  it("пустой дубль помечен и не выбирается; «Запись почти пустая» уходит вместе с ним", async () => {
    await withTakes(() => [{ empty: true, peak_db: -60 }]);
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    expect(await screen.findByTestId("take-empty")).toHaveTextContent(/Запись почти пустая/);
    expect(screen.getByRole("button", { name: /Взять этот дубль в ролик/ })).toBeDisabled();
    expect(screen.queryByTestId("star-why")).toBeNull();
    const st = useStore.getState();
    act(() => useStore.setState({ project: { ...st.project!, scenes: [{ ...st.project!.scenes[0], takes: [] }, ...st.project!.scenes.slice(1)] } }));
    await waitFor(() => expect(screen.queryByTestId("take-empty")).toBeNull());
    expect(screen.queryByTestId("empty-take")).toBeNull();
  });

  it("перегруз: честная подпись", async () => {
    await withTakes(() => [{ clip_pct: 0.6, selected: true }]);
    renderRoute(<RecordScreen />);
    expect(await screen.findByTestId("take-clipped")).toHaveTextContent("Записано с перегрузом, подальше от микрофона — перегруз починили, но лучше переписать.");
  });

  it("дубль из файла: нет «Начал на … раньше», нет совета про губы, пометка «из файла»", async () => {
    const t = await withTakes(() => [{ source: "file", offset_ms: 700, fit_confidence: 0.64, selected: true }]);
    useStore.setState({ remarks: { [`p-fridge/sc001/${t[0].id}`]: { text: "Ты сильно спешишь вперёд губ — выжди паузу.", severity: "warn" } } });
    renderRoute(<RecordScreen />);
    await screen.findByText(/Записать реплику/);
    await screen.findByTestId("take-from-file");
    expect(document.body.textContent).not.toMatch(/Начал на/);
    expect(document.body.textContent).not.toMatch(/Совпадение/);
    expect(screen.queryByText(/спешишь вперёд губ/)).toBeNull();
    expect(screen.getByText("из файла")).toBeInTheDocument();
  });

  it("дубль с микрофона: «Начал на … раньше» остаётся", async () => {
    await withTakes(() => [{ source: "mic", offset_ms: 700, selected: true }]);
    renderRoute(<RecordScreen />);
    expect(await screen.findByText(/Начал на 0,7 с раньше оригинала/)).toBeInTheDocument();
  });
});

describe("Сведение и «Готовый ролик» без дублей", () => {
  it("после удаления дубля «Дальше» не активна и объясняет почему; озвучки нет — ползунок серый", async () => {
    await openFridge();
    const st = useStore.getState();
    useStore.setState({ project: { ...st.project!, scenes: st.project!.scenes.map((x) => ({ ...x, takes: [] })) } });
    renderRoute(<MixScreen />);
    await screen.findByRole("region", { name: "Помощник-режиссёр" });
    expect(screen.getByRole("button", { name: /Дальше: готовый ролик/ })).toBeDisabled();
    expect(screen.getByTestId("mix-next-why")).toHaveTextContent(/Нет дублей/);
    expect(screen.getAllByText(/дублей пока нет — звучит оригинальный голос/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/— оригинал$/).length).toBeGreaterThan(0);
  });
  it("пустой дубль не считается озвучкой", async () => {
    await withTakes(() => [{ empty: true, selected: true }]);
    renderRoute(<MixScreen />);
    await screen.findByRole("region", { name: "Помощник-режиссёр" });
    expect(screen.getByRole("button", { name: /Дальше: готовый ролик/ })).toBeDisabled();
  });
  it("на экране ролика без дублей — объяснение вместо молча серой кнопки", async () => {
    await openFridge();
    const st = useStore.getState();
    useStore.setState({ project: { ...st.project!, scenes: st.project!.scenes.map((x) => ({ ...x, takes: [] })) } });
    renderRoute(<RenderScreen />);
    expect(await screen.findByTestId("no-voice")).toHaveTextContent(/нет дублей/);
    expect(screen.getByRole("button", { name: /Собрать ролик/ })).toBeDisabled();
  });
});

describe("Готовый ролик: имя файла и громкость", () => {
  it("переключатель «Как в фильме» / «Для YouTube и VK» запоминается в сцене, пояснение — одной строкой", async () => {
    await openFridge();
    renderRoute(<RenderScreen />);
    await screen.findByText(/Что собираем/);
    expect(screen.getByRole("radio", { name: "Для YouTube и VK" })).toBeChecked();
    await userEvent.click(screen.getByRole("radio", { name: "Как в фильме" }));
    expect(screen.getByTestId("loud-note")).toHaveTextContent(/таким же по громкости, как в оригинале/);
    await waitFor(() => expect(useStore.getState().project!.scenes[0].mix.loudness_target).toBe("film"));
    const spy = vi.spyOn(useStore.getState().engine!, "render");
    await userEvent.click(screen.getByRole("button", { name: /Собрать ролик/ }));
    await waitFor(() => expect(spy).toHaveBeenCalled());
    expect(screen.queryByText(/LUFS/)).toBeNull(); // жаргон спрятан в подсказку «?»
  });

  it("если движок сохранил под другим именем — показываем итоговое", async () => {
    await openFridge();
    renderRoute(<RenderScreen />);
    await screen.findByText(/Что собираем/);
    const path = screen.getByLabelText("Куда сохранить") as HTMLTextAreaElement;
    await waitFor(() => expect(path.value).toMatch(/\.mp4$/));
    // первый раз — имя как просили, плашки нет
    await userEvent.click(screen.getByRole("button", { name: /Собрать ролик/ }));
    const first = await screen.findByTestId("done", {}, { timeout: 8000 });
    expect(within(first).queryByTestId("renamed")).toBeNull();
    // второй раз с тем же именем — движок добавил « (2)», и мы это говорим
    await userEvent.click(within(first).getByRole("button", { name: "Ещё раз" }));
    await userEvent.click(await screen.findByRole("button", { name: /Собрать ролик/ }));
    const second = await screen.findByTestId("done", {}, { timeout: 8000 });
    expect(within(second).getByTestId("saved-path")).toHaveTextContent("(2).mp4");
    expect(within(second).getByTestId("renamed")).toHaveTextContent(/уже был, поэтому ролик сохранён как/);
  });
});

describe("Проекты: слова", () => {
  it("кнопка называется «Открыть» (видео не копируем)", async () => {
    renderRoute(<ProjectsScreen />, "/", "/");
    await screen.findByText(/Перетащи сюда/);
    const form = screen.getByLabelText("Ссылка или путь").closest("form") as HTMLElement;
    expect(within(form).getByRole("button", { name: "Открыть" })).toBeInTheDocument();
    expect(within(form).queryByRole("button", { name: /Загрузить/ })).toBeNull();
  });

  it("название проекта из имени файла чистится: без расширения, точек и «720p / x264»", async () => {
    const engine = useStore.getState().engine!;
    const spy = vi.spyOn(engine, "createProject");
    renderRoute(<ProjectsScreen />, "/", "/");
    await screen.findByText(/Перетащи сюда/);
    await userEvent.type(screen.getByLabelText("Ссылка или путь"), "/Users/me/Movies/Sintel.2010.720p.x264.mkv");
    await userEvent.click(within(screen.getByLabelText("Ссылка или путь").closest("form") as HTMLElement).getByRole("button", { name: "Открыть" }));
    await userEvent.click(await screen.findByRole("button", { name: /Начать разбор/ }));
    await waitFor(() => expect(spy).toHaveBeenCalled());
    expect(spy.mock.calls[0][0].title).toBe("Sintel 2010");
  });
});
