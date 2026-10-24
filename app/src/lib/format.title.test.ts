import { describe, expect, it } from "vitest";
import { cleanTitle } from "./format";

describe("название проекта из имени файла", () => {
  it.each([
    ["elephantsdream-720-h264-st-aac.mov", "elephantsdream"],
    ["Sintel.2010.720p.mkv", "Sintel 2010"],
    ["Tears_of_Steel_1080p_x264.mp4", "Tears of Steel"],
    ["Барсик.и.Мила.S01E07.WEB-DL.1080p.x265.mkv", "Барсик и Мила S01E07"],
    ["/Users/me/Movies/Cosmos Laundromat [1080p] [RuTracker].mp4", "Cosmos Laundromat"],
    ["Spider-Man.2002.BluRay.mkv", "Spider-Man 2002"],
    ["Фильм (2019) (1080p BluRay).mkv", "Фильм (2019)"],
    ["clip.mp4", "clip"],
  ])("%s → %s", (src, want) => {
    expect(cleanTitle(src)).toBe(want);
  });
  it("если вычистили всё, оставляем имя без расширения", () => {
    expect(cleanTitle("1080p.mkv")).toBe("1080p");
  });
});
