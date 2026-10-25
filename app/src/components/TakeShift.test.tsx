import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { freshStore, openFridge } from "../test/helpers";
import { useStore } from "../state/store";
import { Player } from "../lib/player";
import type { MockEngine } from "../api/mock";
import type { Take } from "../api/types";

const listen = vi.hoisted(() => vi.fn());
vi.mock("../lib/takePlayback", async (orig) => ({ ...(await orig<typeof import("../lib/takePlayback")>()), listenTake: listen }));

import { TakeShift } from "./TakeShift";

let engine: MockEngine;
beforeEach(async () => {
  engine = await freshStore();
  await openFridge();
  engine.takeUrl = () => "blob:take";
  listen.mockReset();
  listen.mockImplementation(async (a: { onEnd: () => void }) => ({ stop: () => a.onEnd(), from: 0, to: 1, placement: {}, bg: true }));
});

const scene = () => useStore.getState().project!.scenes[0];
const take = (i = 0) => scene().takes[i];
const player = new Player(22, 0);

function Bound({ id, settleMs = 30 }: { id: string; settleMs?: number }) {
  const t = useStore((s) => s.project!.scenes[0].takes.find((x) => x.id === id))!;
  return <TakeShift take={t} player={player} settleMs={settleMs} />;
}

describe("строка «Сдвиг» в карточке дубля", () => {
  it("есть у дубля с микрофона в дубляже: подпись, кнопки, значение со знаком", () => {
    render(<Bound id={take().id} />);
    expect(screen.getByTestId("take-shift")).toHaveTextContent("Сдвиг");
    expect(screen.getByRole("button", { name: "← раньше" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "позже →" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "▶ с видео" })).toBeInTheDocument();
    expect(screen.getByTestId("take-shift-value")).toHaveTextContent("−0,08 с");   // offset_ms −80
  });

  it("клики меняют offset_ms на ±50 мс: серия — один запрос и одно проигрывание", async () => {
    const spy = vi.spyOn(engine, "patchTake");
    const id = take().id;
    render(<Bound id={id} />);
    const later = screen.getByRole("button", { name: "позже →" });
    fireEvent.click(later);
    fireEvent.click(later);
    fireEvent.click(later);
    expect(screen.getByTestId("take-shift-value")).toHaveTextContent("+0,07 с");   // −80 + 150
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    expect(spy.mock.calls[0][3]).toEqual({ offset_ms: 70 });
    await waitFor(() => expect(take().offset_ms).toBe(70));
    expect(listen).toHaveBeenCalledTimes(1);
    expect((listen.mock.calls[0][0] as { take: Take }).take.offset_ms).toBe(70);

    fireEvent.click(screen.getByRole("button", { name: "← раньше" }));
    expect(screen.getByTestId("take-shift-value")).toHaveTextContent("+0,02 с");
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
    expect(spy.mock.calls[1][3]).toEqual({ offset_ms: 20 });
  });

  it("не выходит за пределы ползунка Сведения (±3 с)", async () => {
    await act(() => useStore.getState().patchTake(take().id, { offset_ms: 2980 }));
    render(<Bound id={take().id} />);
    const later = screen.getByRole("button", { name: "позже →" });
    fireEvent.click(later);
    expect(screen.getByTestId("take-shift-value")).toHaveTextContent("+3,00 с");
    expect(later).toBeDisabled();
  });

  it("«▶ с видео» играет, пока играет — «■ стоп»; стоп глушит", async () => {
    let stop = vi.fn();
    listen.mockImplementation(async (a: { onEnd: () => void }) => { stop = vi.fn(() => a.onEnd()); return { stop, from: 0, to: 1, placement: {}, bg: true }; });
    render(<Bound id={take().id} />);
    fireEvent.click(screen.getByRole("button", { name: "▶ с видео" }));
    const btn = await screen.findByRole("button", { name: "■ стоп" });
    expect(listen).toHaveBeenCalledTimes(1);
    fireEvent.click(btn);
    expect(stop).toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "▶ с видео" })).toBeInTheDocument();
  });

  it("уход с карточки останавливает звук", async () => {
    const stop = vi.fn();
    listen.mockImplementation(async () => ({ stop, from: 0, to: 1, placement: {}, bg: true }));
    const { unmount } = render(<Bound id={take().id} />);
    fireEvent.click(screen.getByRole("button", { name: "▶ с видео" }));
    await screen.findByRole("button", { name: "■ стоп" });
    unmount();
    expect(stop).toHaveBeenCalled();
  });

  it("нет строки у дубля из файла, у пустого дубля и в закадровом режиме", async () => {
    const base = take();
    const { rerender, container } = render(<TakeShift take={{ ...base, source: "file" }} player={player} />);
    expect(container).toBeEmptyDOMElement();
    rerender(<TakeShift take={{ ...base, empty: true }} player={player} />);
    expect(container).toBeEmptyDOMElement();
    rerender(<TakeShift take={base} player={player} />);
    expect(screen.getByTestId("take-shift")).toBeInTheDocument();
    await act(() => useStore.getState().patchScene({ mix: { ...scene().mix, mode: "voiceover" } }));
    expect(container).toBeEmptyDOMElement();
  });
});
