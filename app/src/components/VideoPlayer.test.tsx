import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { VideoPlayer } from "./VideoPlayer";
import { Player } from "../lib/player";
import type { ProxySource } from "../api/types";

const player = () => new Player(60, 0);

describe("VideoPlayer: исходник → копия → заглушка", () => {
  it("сначала играет исходное видео", () => {
    const { container } = render(<VideoPlayer player={player()} src="http://e/files/p/source" proxy={{ url: "http://e/files/p/proxy", status: async () => "ready" }} />);
    expect(container.querySelector("video")?.getAttribute("src")).toBe("http://e/files/p/source");
  });

  it("если исходник не играется, а копия готова — переключается на копию", async () => {
    const proxy: ProxySource = { url: "http://e/files/p/proxy", status: vi.fn(async () => "ready" as const) };
    const { container } = render(<VideoPlayer player={player()} src="http://e/files/p/source" proxy={proxy} />);
    fireEvent.error(container.querySelector("video")!);
    await waitFor(() => expect(container.querySelector("video")?.getAttribute("src")).toBe("http://e/files/p/proxy"));
    expect(screen.queryByText(/Готовим видео/)).toBeNull();
  });

  it("пока копия готовится — подпись «Готовим видео для просмотра…» и канвас вместо видео", async () => {
    let state: "preparing" | "ready" = "preparing";
    vi.useFakeTimers();
    try {
      const proxy: ProxySource = { url: "http://e/files/p/proxy", status: async () => state };
      const { container } = render(<VideoPlayer player={player()} src="http://e/files/p/source" proxy={proxy} />);
      fireEvent.error(container.querySelector("video")!);
      await act(async () => { await vi.advanceTimersByTimeAsync(10); });
      expect(screen.getByText("Готовим видео для просмотра…")).toBeInTheDocument();
      expect(container.querySelector("video")).toBeNull();
      expect(container.querySelector("canvas")).not.toBeNull();
      state = "ready";
      await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
      expect(container.querySelector("video")?.getAttribute("src")).toBe("http://e/files/p/proxy");
      expect(screen.queryByText(/Готовим видео/)).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("копии нет и исходник не играется — остаётся заглушка без подписи", async () => {
    const proxy: ProxySource = { url: "http://e/files/p/proxy", status: async () => "missing" };
    const { container } = render(<VideoPlayer player={player()} src="http://e/files/p/source" proxy={proxy} />);
    fireEvent.error(container.querySelector("video")!);
    await waitFor(() => expect(container.querySelector("canvas")).not.toBeNull());
    expect(screen.queryByText(/Готовим видео/)).toBeNull();
  });

  it("без proxy ошибка видео сразу даёт заглушку (клип сцены)", () => {
    const { container } = render(<VideoPlayer player={player()} src="http://e/files/p/scenes/s/clip.mp4" />);
    fireEvent.error(container.querySelector("video")!);
    expect(container.querySelector("canvas")).not.toBeNull();
  });
});
