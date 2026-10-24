import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { freshStore, renderRoute, wait } from "../test/helpers";
import { useStore } from "../state/store";
import { MockEngine, mockProbe } from "../api/mock";
import { ConfirmHost } from "../components/ConfirmDialog";
import { FoundInFile } from "../components/FoundInFile";
import { AnalyzeScreen, asrSummary } from "./Analyze";
import { SettingsScreen } from "./Settings";
import type { Analysis } from "../api/types";

let engine: MockEngine;
beforeEach(async () => { engine = await freshStore(80); });

async function analyzed(path: string, language: string | null = null) {
  const { project } = await engine.createProject({ path, language });
  await waitFor(async () => expect((await engine.getProject(project.id)).analysis).not.toBeNull(), { timeout: 4000 });
  await useStore.getState().openProject(project.id);
  return project.id;
}

describe("распознавание речи: язык по речи и GigaAM", () => {
  it("Настройки → Обработка: выбор распознавателя сохраняется, пояснение без жаргона, про скачивание GigaAM", async () => {
    useStore.setState({ demo: false, health: await engine.health() });
    renderRoute(<SettingsScreen />, "/settings/:page?", "/settings/processing");
    const sel = await screen.findByRole("combobox", { name: "Распознавание речи" });
    expect(within(sel).getAllByRole("option").map((o) => o.textContent)).toEqual(["Авто (рекомендуется)", "Whisper", "GigaAM — только русский"]);
    expect((sel as HTMLSelectElement).value).toBe("auto");
    expect(screen.getByTestId("asr-explain")).toHaveTextContent(/Если не уверен, оставь так/);
    expect(screen.getByTestId("asr-explain")).toHaveTextContent(/скачается при первом разборе русской речи, ~230 МБ/);
    expect(screen.getByTestId("asr-explain")).toHaveTextContent(/Разобрать снова/);
    await userEvent.selectOptions(sel, "gigaam");
    await waitFor(async () => expect((await engine.getSettings()).asr_engine).toBe("gigaam"));
    expect(screen.getByTestId("asr-explain")).toHaveTextContent(/всё равно возьмёт Whisper/);
  });

  it("дорожка помечена «eng», а речь русская: в разборе фраза про расхождение и «Разобрать заново» с языком", async () => {
    const pid = await analyzed("/tmp/Мегамозг дубляж.mkv");
    const an = (await engine.getProject(pid)).analysis!;
    expect(an.language).toBe("ru");
    expect(an.track_language).toBe("en");
    expect(an.asr_engine).toBe("gigaam");
    renderRoute(<><AnalyzeScreen /><ConfirmHost /></>);
    const note = await screen.findByTestId("language-note");
    expect(note).toHaveTextContent("В файле написано «английский», а слышно русский — взял русский. Не так? Выбери язык и разбери заново.");
    const spy = vi.spyOn(engine, "reanalyze");
    await userEvent.click(within(note).getByRole("button", { name: "Выбрать язык" }));
    await userEvent.selectOptions(within(note).getByRole("combobox", { name: "Язык оригинала" }), "en");
    await userEvent.click(within(note).getByRole("button", { name: /Разобрать заново/ }));
    await userEvent.click(await screen.findByRole("button", { name: "Разобрать" }));
    await waitFor(() => expect(spy).toHaveBeenCalledWith(pid, expect.objectContaining({ language: "en" })));
    await wait(1500);
    const after = (await engine.getProject(pid)).analysis!;
    expect(after.language).toBe("en");
    expect(after.language_source).toBe("user");          // выбор человека — закон, без спора
    expect(after.language_note).toBeNull();
    expect(after.asr_engine).toBe("whisper");
  });

  it("язык совпал с меткой — фразы нет; во вкладке «Текст» серым — чем распознано", async () => {
    await analyzed("/tmp/Серия.mkv");
    renderRoute(<AnalyzeScreen />);
    await screen.findByRole("tab", { name: "Текст" });
    expect(screen.queryByTestId("language-note")).toBeNull();
    await userEvent.click(screen.getByRole("tab", { name: "Текст" }));
    expect(screen.getByTestId("asr-summary")).toHaveTextContent("Текст распознал Whisper · язык: английский (по речи)");
  });

  it("asrSummary: GigaAM, выбор вручную, пояснения", () => {
    const base = { audio_stream_index: 1, lines: [], speakers: [], shot_cuts: [], chapters: [], merge_suggestions: [], merged_lines: {}, asr_notes: [] } as Analysis;
    expect(asrSummary({ ...base, asr_engine: "gigaam", language: "ru", language_source: "speech" })).toBe("Текст распознал GigaAM · язык: русский (по речи)");
    expect(asrSummary({ ...base, asr_engine: "whisper", language: "en", language_source: "user", asr_notes: ["GigaAM знает только русский — взял Whisper."] }))
      .toBe("Текст распознал Whisper · язык: английский (выбран вручную). GigaAM знает только русский — взял Whisper.");
    expect(asrSummary(base)).toBeNull();                  // старый разбор без полей — ничего не пишем
  });

  it("«Что нашли в файле»: язык дорожки — подпись файла, а не факт", () => {
    render(<FoundInFile media={mockProbe("Серия.mkv")} audioIdx={1} />);
    expect(screen.getByTestId("lang-hint")).toHaveTextContent("Язык у дорожки — только подпись в файле, она бывает неверной. Если язык оригинала не выбран, программа определит его по речи.");
    expect(screen.getAllByText("в файле: английский").length).toBeGreaterThan(0);
  });
});
