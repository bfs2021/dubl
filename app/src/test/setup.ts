import "@testing-library/jest-dom/vitest";

// jsdom не умеет canvas, ResizeObserver и матчи — подставляем безвредные заглушки
if (!("ResizeObserver" in globalThis)) {
  (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}
HTMLCanvasElement.prototype.getContext = (() => null) as unknown as HTMLCanvasElement["getContext"];
Element.prototype.scrollIntoView = () => {};
if (!window.matchMedia) {
  window.matchMedia = ((q: string) => ({
    matches: false, media: q, onchange: null,
    addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}, dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
}

// Тесты не должны ходить в настоящую сеть (на машине может работать живой движок на :8765)
globalThis.fetch = (async () => { throw new TypeError("offline in tests"); }) as typeof fetch;
