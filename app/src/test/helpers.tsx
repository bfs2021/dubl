import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import type { ReactElement } from "react";
import { MockEngine } from "../api/mock";
import { useStore, defaultUi } from "../state/store";

/** Свежий мок-движок в хранилище; таймеры задач ускорены. */
export async function freshStore(speed = 60) {
  localStorage.clear();
  const engine = new MockEngine({ speed });
  useStore.setState({ ui: defaultUi(), toasts: [], remarks: {}, asIs: {}, project: null, sceneId: null });
  await useStore.getState().attachEngine(engine, true, "");
  return engine;
}

export async function openFridge(sceneId: string | null = "sc001") {
  await useStore.getState().openProject("p-fridge");
  useStore.getState().setSceneId(sceneId);
}

export function renderRoute(ui: ReactElement, path = "/p/:pid/x", url = "/p/p-fridge/x") {
  return render(
    <MemoryRouter initialEntries={[url]}>
      <Routes><Route path={path} element={ui} /><Route path="*" element={<div>other</div>} /></Routes>
    </MemoryRouter>,
  );
}

export const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
