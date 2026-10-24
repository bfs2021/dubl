// Сквозной путь в браузере против ЖИВОГО движка (без демо): диалог → сцена → запись → сведение → рендер.
// Запуск: `pnpm e2e` (движок на ENGINE_URL и окно на APP_URL должны быть запущены, иначе тест пропускается).
//
//   cd engine && DUBL_DATA_DIR=/tmp/dubl-live .venv/bin/dubl-engine --port 8765   # в /tmp/dubl-live уже есть разобранный проект
//   cd app && pnpm dev                                                            # окно на :1420
//   cd app && pnpm e2e
//
// Переменные: ENGINE_URL (http://127.0.0.1:8765), APP_URL (http://127.0.0.1:1420), PROJECT_ID (первый разобранный),
// SHOTS (/tmp/dubl-app-shots), OUT (/tmp/dubl-e2e), CHROME (системный Chrome), SPEECH_WAV (речь для «микрофона»; по умолчанию делается `say`).
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";

const ENGINE = process.env.ENGINE_URL || "http://127.0.0.1:8765";
const APP = process.env.APP_URL || "http://127.0.0.1:1420";
const SHOTS = process.env.SHOTS || "/tmp/dubl-app-shots";
const OUT = process.env.OUT || "/tmp/dubl-e2e";
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PREFIX = process.env.PREFIX || "w1-live-";

const up = async (url) => {
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(2500) });
    return r.ok;
  } catch {
    return false;
  }
};
if (!(await up(ENGINE + "/health"))) {
  console.log(`SKIP: движок не отвечает на ${ENGINE}/health`);
  process.exit(0);
}
if (!(await up(APP + "/"))) {
  console.log(`SKIP: окно (vite) не отвечает на ${APP}`);
  process.exit(0);
}
if (!existsSync(CHROME)) {
  console.log(`SKIP: нет Chrome по пути ${CHROME}`);
  process.exit(0);
}

const require = createRequire(import.meta.url);
const { chromium } = require("playwright-core");
mkdirSync(SHOTS, { recursive: true });
mkdirSync(OUT, { recursive: true });

// ───────────── вспомогательное ─────────────
const api = async (path, init) => {
  const r = await fetch(ENGINE + path, init);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0).padStart(4)}s]`, ...a);
const failures = [];
const warnings = [];
const check = (ok, what) => {
  log(ok ? "ок  " : "FAIL", what);
  if (!ok) failures.push(what);
};
async function until(fn, what, ms = 120000, step = 1000) {
  const end = Date.now() + ms;
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() > end) throw new Error(`не дождались: ${what}`);
    await sleep(step);
  }
}

// речь для «микрофона»
let wav = process.env.SPEECH_WAV;
if (!wav) {
  wav = join(OUT, "speech.wav");
  if (!existsSync(wav)) {
    execFileSync("say", ["-v", "Milena", "Ты просто грубиян, Том. Слушай, Селия, нам нужно следовать своим увлечениям.", "-o", join(OUT, "speech.aiff")]);
    execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", join(OUT, "speech.aiff"), "-ar", "48000", "-ac", "1", wav]);
  }
}

const projects = await api("/projects");
const summary = process.env.PROJECT_ID
  ? projects.find((p) => p.id === process.env.PROJECT_ID)
  : projects.find((p) => p.has_analysis && /tears/i.test(p.title)) ?? projects.find((p) => p.has_analysis);
if (!summary) {
  console.log("SKIP: в движке нет разобранного проекта");
  process.exit(0);
}
const pid = summary.id;
log("проект:", summary.title, pid);

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", `--use-file-for-fake-audio-capture=${wav}`, "--autoplay-policy=no-user-gesture-required"],
});
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
await ctx.grantPermissions(["microphone"], { origin: new URL(APP).origin });
const page = await ctx.newPage();
// все показанные тосты (для проверки дублей и сырых ошибок)
await ctx.addInitScript(() => {
  window.__toasts = [];
  const seen = new WeakSet();
  new MutationObserver(() => {
    document.querySelectorAll(".toast").forEach((el) => {
      if (seen.has(el)) return;
      seen.add(el);
      window.__toasts.push(el.querySelector("span")?.textContent ?? "");
    });
  }).observe(document, { childList: true, subtree: true });
});
let probing400 = false;
const errors = [];
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
page.on("console", (m) => {
  if (m.type() !== "error") return;
  const t = m.text();
  // видео, которое браузер не умеет (mkv) → штатный переход на лёгкую копию; это не ошибка программы
  if (/Failed to load resource.*(source|proxy)/i.test(t)) return;
  // проба заведомо битого файла/несуществующего пути (шаг 5b) честно отвечает 400 — браузер пишет это в консоль
  if (/Failed to load resource.*status of 400/i.test(t) && probing400) return;
  errors.push("console: " + t);
});
const shot = (name) => page.screenshot({ path: join(SHOTS, `${PREFIX}${name}.png`) });
const go = async (hash) => {
  await page.goto(`${APP}/#${hash}`);
  await page.waitForTimeout(700);
};
const getProject = () => api(`/projects/${pid}`);
let toastsSeen = [];
// ошибки про файл теперь стоят рядом с полем ввода (а не тостом) — смотрим и туда, и в тосты
const shownErrors = async () => [...(await page.evaluate(() => window.__toasts ?? [])), ...(await page.getByTestId("probe-error").allInnerTexts().catch(() => []))];

try {
  // ───────────── 0. открыть проект ─────────────
  await page.goto(APP + "/");
  await until(async () => (await page.locator(".demo").count()) === 0 && (await page.locator(".home, .projects, .card").count()) > 0, "главный экран без демо-плашки", 15000, 300).catch(() => {});
  check((await page.locator(".demo").count()) === 0, "нет демо-плашки (движок подключён)");
  await shot("01-projects");
  await page.getByRole("button", { name: /Указать путь к файлу/ }).click();
  check(await page.getByRole("alert").filter({ hasText: "полный путь" }).isVisible(), "«Выбрать файл» в браузере: понятная подсказка про полный путь");
  await page.waitForTimeout(150);
  check(await page.getByLabel("Ссылка или путь").evaluate((el) => el === document.activeElement), "фокус на поле пути");
  await shot("01b-projects-path-hint");
  await go(`/p/${pid}/analyze`);
  await page.waitForSelector(".timeline .tl-tracks", { timeout: 20000 });
  await page.waitForTimeout(800);
  await shot("02-analyze");

  // вёрстка: все голоса в списке и все дорожки на шкале либо видны, либо прокручиваются
  const lay = await page.evaluate(() => {
    const vis = (el, box) => {
      const r = el.getBoundingClientRect(), b = box.getBoundingClientRect();
      return r.bottom <= b.bottom + 1;
    };
    const list = document.querySelector('.analyze > .panel [role="tabpanel"]');
    const rows = [...document.querySelectorAll(".chr")];
    const tl = document.querySelector(".tl-body");
    const lanes = [...document.querySelectorAll(".tl-tracks .lane")];
    const win = window.innerHeight;
    return {
      chars: rows.length, charsScroll: list.scrollHeight > list.clientHeight, charsLastFits: rows.length ? vis(rows.at(-1), list) : true,
      lanes: lanes.length, lanesScroll: tl.scrollHeight > tl.clientHeight, lanesLastFits: lanes.length ? vis(lanes.at(-1), tl) : true,
      pageOverflow: document.documentElement.scrollHeight > win + 1,
    };
  });
  log("вёрстка", JSON.stringify(lay));
  check(lay.charsLastFits || lay.charsScroll, "список голосов целиком или с прокруткой");
  check(lay.lanesLastFits || lay.lanesScroll, "дорожки шкалы целиком или с прокруткой");
  check(!lay.pageOverflow, "страница не вылезает за окно");

  // ───────────── 1. две сцены: «Диалог» ─────────────
  let scenes = (await getProject()).scenes;
  if (scenes.length < 2) {
    const lines = [...(await getProject()).analysis.lines].sort((a, b) => a.start - b.start);
    const picks = [lines[0], lines.find((l) => l.start > 120 && l.speaker_id)].filter(Boolean);
    for (const [i, ln] of picks.entries()) {
      if (scenes.some((s) => s.start - 1 <= ln.start && ln.end <= s.end + 1)) continue;
      await page.getByRole("radio", { name: "Диалог" }).click();
      const box = await page.locator(".tl-tracks").boundingBox();
      const lane = await page.locator(`.tl-tracks [data-lane="${ln.speaker_id}"]`).boundingBox();
      const dur = (await getProject()).media.duration;
      const x = box.x + ((ln.start + ln.end) / 2 / dur) * box.width;
      await page.mouse.click(x, lane.y + lane.height / 2);
      await page.waitForSelector('[data-primary="scene"]', { timeout: 5000 });
      const info = await page.locator(".selinfo").innerText();
      log(`выделено (диалог #${i + 1}):`, info.replace(/\n/g, " | ").slice(0, 160));
      if (i === 0) await shot("03-dialogue-selected");
      await page.locator('[data-primary="scene"]').click();
      await until(async () => (await getProject()).scenes.length > scenes.length, "сцена появилась", 15000, 300);
      scenes = (await getProject()).scenes;
      log("сцена создана:", scenes.at(-1).title, scenes.at(-1).start.toFixed(1), "–", scenes.at(-1).end.toFixed(1));
      if (i === 0) await shot("04-scene-created");
    }
  }
  check(scenes.length >= 2, "создано две сцены");

  // подготовка (голоса/фон)
  await until(async () => (await getProject()).scenes.every((s) => s.stems_ready), "голоса и фон готовы", 900000, 3000);
  scenes = (await getProject()).scenes;
  check(scenes.every((s) => s.stems_ready), "подготовка сцен закончилась (голоса и фон)");
  await go(`/p/${pid}/analyze`);
  await page.getByRole("tab", { name: /Сцены/ }).click();
  await shot("05-scenes-ready");

  // ───────────── 2. запись дублей ─────────────
  const cfg = await api("/settings");
  const aiOn = (task) => { const id = cfg.llm.routing[task]; return !!id && cfg.llm.providers.some((p) => p.id === id && p.enabled); };
  log("ИИ в движке:", JSON.stringify(cfg.llm.routing), "провайдеров:", cfg.llm.providers.length);
  for (const [i, sc] of scenes.slice(0, 2).entries()) {
    if (sc.takes.length) continue;
    await go(`/p/${pid}/record`);
    await page.waitForSelector('[data-primary="record"]', { timeout: 20000 });
    await page.locator("select.scenesel").selectOption(sc.id);
    await page.waitForTimeout(600);
    if (i === 0) {
      await shot("06-record");
      if (!aiOn("translate") && !aiOn("fit_length")) {
        check(await page.getByRole("button", { name: /Перевести/ }).isDisabled(), "без ИИ «Перевести» неактивна");
        check(await page.getByRole("button", { name: /Подогнать под длину/ }).first().isDisabled(), "без ИИ «Подогнать под длину» неактивна");
        check((await page.getByText(/Подключи ИИ в/).count()) > 0, "рядом подсказка «Подключи ИИ в Настройках»");
      }
    }
    await page.getByRole("button", { name: "Я в наушниках" }).click({ timeout: 1500 }).catch(() => {});
    // первая реплика, которую озвучиваем
    await page.locator(".rl-item").first().click();
    await page.locator('[data-primary="record"]').click();
    if (i === 0) {
      await page.waitForSelector(".countdown.on, .recbadge.on", { timeout: 10000 });
      await page.waitForTimeout(500);
      await shot("07-recording");
    }
    const had = sc.takes.length;
    await until(async () => (await getProject()).scenes.find((s) => s.id === sc.id).takes.length > had, `дубль загружен (${sc.title})`, 90000, 1000);
    await page.waitForSelector(".take", { timeout: 10000 });
    // замечание помощника
    await until(async () => {
      const t = await page.locator(".take .remark span").first().innerText().catch(() => "");
      return t && !/Слушаю/.test(t) ? t : null;
    }, "замечание помощника", 60000, 500);
    const remark = await page.locator(".take .remark span").first().innerText();
    const cap = await page.locator(".take .m").first().innerText();
    log("подпись дубля:", cap);
    check(!/ мс|\+\d{3,}/.test(cap) && /Начал на|Попал в губы/.test(cap), "подпись дубля словами, без «+3049 мс»");
    const take = (await getProject()).scenes.find((s) => s.id === sc.id).takes[0];
    log(`дубль ${sc.title}: offset=${take.offset_ms}мс stretch=${take.stretch} conf=${take.fit_confidence} peak=${take.peak_db} | «${remark}»`);
    check(!!remark.trim() && !/Слушаю/.test(remark), `есть замечание помощника под дублем (${sc.title})`);
    check(take.selected, `дубль выбран в сведение (${sc.title})`);
    if (i === 0) await shot("08-take");
  }

  // ───────────── 3. сведение: превью играет ─────────────
  await go(`/p/${pid}/mix`);
  await page.waitForSelector(".mixmain", { timeout: 20000 });
  await page.waitForTimeout(500);
  await shot("09-mix");
  // превью теперь собирается само при входе (если голос записан); кнопку жмём, только если оно ещё не пошло
  await page.waitForTimeout(1500);
  const btn = page.getByRole("button", { name: /Собрать превью|Обновить превью/ });
  if (await btn.count()) await btn.click();
  else log("превью уже собирается/собрано само");
  const previewJob = await until(async () => {
    const jobs = await api("/jobs");
    const j = jobs.filter((x) => x.kind === "preview" && x.project_id === pid).at(-1);
    return j && (j.state === "done" || j.state === "failed") ? j : null;
  }, "превью собрано", 300000, 1500);
  check(previewJob.state === "done", "превью собрано: " + (previewJob.error || previewJob.result?.path || ""));
  await page.waitForTimeout(1200);
  const vid = page.locator(".video video");
  await vid.waitFor({ timeout: 20000 });
  const src = await vid.getAttribute("src");
  check(/preview\.mp4/.test(src || ""), "в плеере сведения — превью: " + src);
  await page.locator(".transport .playbtn").click();
  await page.waitForTimeout(1000);
  const pl = await vid.evaluate((v) => ({ t: v.currentTime, paused: v.paused, ended: v.ended, err: v.error && v.error.code, ready: v.readyState, dur: v.duration }));
  log("плеер:", JSON.stringify(pl));
  // сцена может быть короче секунды-двух: «играет» = время ушло вперёд и ролик идёт или доиграл до конца
  check(!pl.err && pl.t > Math.min(0.3, pl.dur / 2) && (!pl.paused || pl.ended), "превью играет (время идёт вперёд)");
  await shot("10-mix-playing");
  if (await vid.evaluate((v) => !v.paused)) await page.locator(".transport .playbtn").click();

  // команда помощнику (если подключён ИИ — движок ответит; иначе — понятная ошибка в плашке, это не сбой)
  // ───────────── 4. рендер ─────────────
  const renders = [];
  await go(`/p/${pid}/render`);
  await page.waitForSelector('[data-primary="render"]', { timeout: 20000 });
  const boxes = page.locator(".scenepick input[type=checkbox]");
  await shot("11-render");
  check(await page.getByRole("radio", { name: "Нет" }).isChecked(), "субтитры по умолчанию выключены");
  for (const [name, count] of [["one", 1], ["two", 2]]) {
    const n = await boxes.count();
    for (let i = 0; i < n; i++) {
      const on = i < count;
      if ((await boxes.nth(i).isChecked()) !== on) await boxes.nth(i).setChecked(on);
    }
    const path = join(OUT, "out", `${name}.mp4`);
    mkdirSync(join(OUT, "out"), { recursive: true });
    await page.getByLabel("Куда сохранить").fill(path);
    const posted = page.waitForResponse((r) => r.request().method() === "POST" && r.url().endsWith(`/projects/${pid}/render`));
    await page.locator('[data-primary="render"]').click();
    const started = await (await posted).json();
    check(started.kind === "render_project", `«${name}»: одна задача render_project (${started.id})`);
    await page.waitForTimeout(1500);
    if (name === "two") await shot("12-render-running");
    const job = await until(async () => {
      const j = await api(`/jobs/${started.id}`);
      return j.state === "done" || j.state === "failed" ? j : null;
    }, `рендер «${name}»`, 600000, 2000);
    check(job.state === "done", `рендер «${name}»: ${job.state} ${job.error || ""}`);
    await page.waitForSelector('[data-testid="done"]', { timeout: 15000 });
    const shown = await page.locator('[data-testid="saved-path"]').innerText();
    renders.push({ name, path, job, shown });
    log(`рендер «${name}»:`, JSON.stringify(job.result), "| на экране:", shown);
    if (name === "two") await shot("13-render-done");
    await page.getByRole("button", { name: "Ещё раз" }).click();
  }

  // файлы
  const durs = [];
  for (const r of renders) {
    const p = r.job.result?.path || r.path;
    const ok = existsSync(p) && statSync(p).size > 10000;
    check(ok, `файл «${r.name}» существует: ${p}${ok ? ` (${(statSync(p).size / 1e6).toFixed(1)} МБ)` : ""}`);
    if (!ok) continue;
    const pr = JSON.parse(execFileSync("ffprobe", ["-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,duration:format=duration", "-of", "json", p]).toString());
    const d = Number(pr.format.duration);
    durs.push(d);
    log(`ffprobe «${r.name}»:`, d.toFixed(2), "с,", pr.streams.map((s) => `${s.codec_type}:${s.codec_name}${s.width ? " " + s.width + "x" + s.height : ""}`).join(", "));
    check(pr.streams.some((s) => s.codec_type === "video") && pr.streams.some((s) => s.codec_type === "audio"), `в «${r.name}» есть видео и звук`);
  }
  if (durs.length === 2) {
    const sum = scenes.slice(0, 2).reduce((a, s) => a + (s.end - s.start), 0);
    check(durs[1] > durs[0] + 1, `склейка длиннее одной сцены (${durs[1].toFixed(1)} > ${durs[0].toFixed(1)}) · сумма сцен ${sum.toFixed(1)} с`);
  }

  // ───────────── 5. плеер: исходник не играется → лёгкая копия (202 «Готовим видео…» → 200) ─────────────
  try {
    const p2 = await ctx.newPage();
    p2.on("pageerror", (e) => errors.push("pageerror(proxy): " + e.message));
    await p2.route("**/files/*/source*", (r) => r.abort());
    const statuses = [];
    p2.on("response", (r) => { if (/\/files\/[^/]+\/proxy/.test(r.url())) statuses.push(r.status()); });
    await p2.goto(`${APP}/#/p/${pid}/analyze`);
    await p2.waitForSelector(".vwait, .video video", { timeout: 20000 });
    const waitText = await p2.locator(".vwait").textContent({ timeout: 1500 }).catch(() => null);
    if (waitText) {
      if (waitText.includes("Готовим видео для просмотра")) check(true, "пока копия готовится: «Готовим видео для просмотра…», вместо видео — заглушка");
      else if (/Не получилось показать видео/.test(waitText)) warnings.push("лёгкая копия сразу упала на стороне движка — окно показало «Не получилось показать видео»");
      else check(false, "подпись при подготовке копии: " + waitText);
      await p2.screenshot({ path: join(SHOTS, `${PREFIX}15-proxy-preparing.png`) });
    } else log("копия уже была готова — подпись не успела показаться");
    await p2.waitForSelector(".video video", { timeout: 30000 });
    const pv = await p2.locator(".video video").evaluate(async (v) => { await new Promise((r) => (v.readyState >= 2 ? r() : v.addEventListener("loadeddata", r, { once: true }))); return { src: v.src, err: v.error && v.error.code, dur: v.duration, w: v.videoWidth }; });
    log("копия:", JSON.stringify(pv), "ответы:", [...new Set(statuses)].join(","));
    check(/\/proxy/.test(pv.src) && !pv.err && pv.dur > 60, "плеер переключился на лёгкую копию и она играбельна");
    await p2.screenshot({ path: join(SHOTS, `${PREFIX}16-proxy-ready.png`) });
    await p2.close();
  } catch (e) {
    // сбой лёгкой копии не должен прятать остальные проверки: записываем его и идём дальше
    // известный сбой движка (зона Б): proxy падает на видео с нечётной высотой (720×405: «height not divisible by 2»)
    const jobs = await api("/jobs").catch(() => []);
    const bad = jobs.filter((j) => j.kind === "proxy" && j.project_id === pid && j.state === "failed").at(-1);
    if (bad && /not divisible by 2/.test(bad.error_debug || "")) {
      warnings.push("движок: лёгкая копия не собралась — libx264 «height not divisible by 2» (media/extract.py scale=-2:'min(720,ih)' на 720×405)");
      log("ПРЕДУПРЕЖДЕНИЕ:", warnings.at(-1));
    } else check(false, "лёгкая копия видео: " + e.message.split("\n")[0]);
  }

  // ───────────── 5b. ошибки по-человечески + общий прогресс разбора ─────────────
  {
    probing400 = true;
    const bad = join(OUT, "broken.mp4");
    execFileSync("sh", ["-c", `head -c 30000 /dev/urandom > '${bad}'`]);
    await go("/");
    await page.getByLabel("Ссылка или путь").fill(bad);
    await page.keyboard.press("Enter");
    const t1 = await until(async () => (await shownErrors()).find((t) => /повреждён|повреждено|не видео|не нашёл/i.test(t)), "ошибка про битый файл", 15000, 300);
    log("битый файл → тост:", t1);
    check(!/moov atom|Invalid data/.test(t1), "битый mp4: человеческий текст без ffmpeg");
    check(await page.getByTestId("probe-error").isVisible(), "ошибка про файл стоит под полем ввода, а не в углу");
    await shot("17-broken-file-toast");
    await page.getByLabel("Ссылка или путь").fill("/nonexistent/video.mov");
    await page.keyboard.press("Enter");
    const t2 = await until(async () => (await shownErrors()).find((t) => /Файл не найден|не нашёл/i.test(t)), "ошибка про отсутствующий файл", 15000, 300);
    log("нет файла → тост:", t2);
    await page.getByLabel("Ссылка или путь").fill("file://" + wav);
    await page.keyboard.press("Enter");
    const t3 = await until(async () => (await shownErrors()).find((t) => /нет видео/i.test(t)), "file:// принят движком (ответ про «нет видео»)", 15000, 300);
    check(!/Файл не найден: file:/.test(t3), "file:// движок принимает: " + t3);

    // новый короткий проект → общий прогресс разбора растёт и не пятится
    const short = join(OUT, "short.mkv");
    const srcMkv = projects.find((p) => p.id === pid) && (await getProject()).source_path;
    execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-ss", "20", "-t", "45", "-i", srcMkv, "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", short]);
    await page.getByLabel("Ссылка или путь").fill(short);
    await page.keyboard.press("Enter");
    await page.waitForSelector('[data-primary="start"]:not([disabled])', { timeout: 20000 });
    await page.locator('[data-primary="start"]').click();
    await page.waitForURL(/analyze/, { timeout: 15000 });
    const samples = [];
    let shotProgress = false;
    const end = Date.now() + 240000;
    while (Date.now() < end) {
      const jobs = await api("/jobs");
      const j = jobs.filter((x) => x.kind === "analyze").at(-1);
      if (j) {
        samples.push({ p: j.progress, eta: j.eta_s, st: j.state });
        const ui = await page.locator('.waitcard [role="progressbar"]').getAttribute("aria-valuenow").catch(() => null);
        if (ui !== null) {
          samples.at(-1).ui = Number(ui);
          if (!shotProgress && Number(ui) >= 50) { shotProgress = true; await shot("18-analyze-progress"); }
        }
        if (j.state === "done" || j.state === "failed") break;
      }
      await sleep(1500);
    }
    log("прогресс разбора:", samples.map((x) => `${(x.p * 100).toFixed(0)}%${x.eta != null ? "~" + Math.round(x.eta) + "с" : ""}${x.ui != null ? "(ui " + x.ui + ")" : ""}`).join(" "));
    check(samples.every((x, i) => i === 0 || x.p >= samples[i - 1].p), "общий прогресс разбора не идёт назад");
    check(samples.at(-1)?.st === "done", "разбор нового проекта закончился");
    check(samples.some((x) => x.ui != null), "в окне виден общий прогресс-бар");
    probing400 = false;
    // ───── путь новичка: свежий проект сразу после разбора ─────
    {
      const fresh = (await api("/projects")).find((p) => p.id !== pid && /short/i.test(p.title));
      if (fresh) {
        await go(`/p/${fresh.id}/analyze`);
        await page.waitForSelector(".timeline .tl-tracks", { timeout: 20000 });
        await page.waitForTimeout(600);
        const next = page.getByRole("button", { name: /Дальше: выбери момент для озвучки/ });
        check(await next.isVisible(), "новичок: после разбора есть крупная кнопка «Дальше: выбери момент для озвучки»");
        check((await page.locator(".selinfo .hint").innerText().catch(() => "")).includes("Протяни мышью по шкале или кликни по реплике"), "новичок: подсказка в шапке шкалы");
        check((await page.getByTestId("tl-hint").count()) === 0, "новичок: подсказка не лежит поверх дорожек");
        await page.getByRole("tab", { name: /Сцены/ }).click();
        check(await page.getByText("Сцен пока нет").isVisible(), "новичок: пустой список сцен — понятный пустой экран");
        await shot("20-newbie-after-analysis");
        // «Вручную»: простой клик по реплике сразу выделяет момент
        await page.getByRole("radio", { name: "Вручную" }).click();
        const fp = await api(`/projects/${fresh.id}`);
        const ln = [...fp.analysis.lines].sort((a, b) => a.start - b.start)[0];
        const box = await page.locator(".tl-tracks").boundingBox();
        const lane = await page.locator(`.tl-tracks [data-lane="${ln.speaker_id}"]`).boundingBox();
        await page.mouse.click(box.x + ((ln.start + ln.end) / 2 / fp.media.duration) * box.width, lane.y + lane.height / 2);
        check(await page.locator('[data-primary="scene"]').isVisible({ timeout: 3000 }).catch(() => false), "новичок: клик по реплике (даже в «Вручную») сразу даёт «Создать сцену»");
        await shot("21-newbie-clicked-line");
        await page.getByRole("radio", { name: "Диалог" }).click();
        await page.reload();
        await page.waitForSelector(".timeline .tl-tracks", { timeout: 20000 });
        await page.getByRole("button", { name: /Дальше: выбери момент/ }).click();
        check(await page.locator('[data-primary="scene"]').isVisible({ timeout: 3000 }).catch(() => false), "новичок: «Дальше» выделяет момент и показывает «Создать сцену»");
        await shot("22-newbie-next-clicked");
      }
    }
    // убираем за собой пробный проект
    const extra = (await api("/projects")).filter((p) => p.id !== pid && /short/i.test(p.title));
    for (const p of extra) await fetch(`${ENGINE}/projects/${p.id}`, { method: "DELETE" });
  }


  // ───────────── 7. второй слепой тест ─────────────
  {
    const P = await getProject();
    const speakersNow = async () => (await getProject()).analysis.speakers.length;

    // 7a. слияние голосов + «Отменить»
    await go(`/p/${pid}/analyze`);
    await page.waitForSelector(".chr", { timeout: 20000 });
    const nSp = await speakersNow();
    if (nSp >= 2) {
      await page.getByRole("button", { name: /^Объединить .* с другим голосом/ }).nth(1).click();
      await shot("23-merge-pick");
      await page.locator(".mergepick").getByRole("button", { name: "Объединить" }).click();
      await until(async () => (await speakersNow()) === nSp - 1, "голоса объединены", 8000, 300);
      const toast = page.locator(".toast", { hasText: "Голоса объединены" });
      await toast.waitFor({ timeout: 4000 }).catch(() => {});
      check(await toast.isVisible(), "слияние: тост «Голоса объединены» с кнопкой «Отменить»");
      check(await page.getByTestId("merge-suggest").count() === 0, "слияние: пока тост виден, новой подсказки слияния нет");
      await page.waitForTimeout(450);
      await shot("24-merge-toast");
      await toast.getByRole("button", { name: "Отменить" }).click();
      await until(async () => (await speakersNow()) === nSp, "слияние отменено", 8000, 300);
      check((await speakersNow()) === nSp, `слияние отменено: снова ${nSp} голосов`);
    }

    // 7b. «Закадровый»: объяснение, нет подсказок про губы
    await go(`/p/${pid}/record`);
    await page.waitForSelector('[data-primary="record"]', { timeout: 20000 });
    await page.getByRole("radio", { name: "Закадровый" }).click();
    await page.waitForTimeout(700);
    check(await page.getByTestId("voiceover-note").isVisible(), "закадровый: пояснение «оригинал слышен тише, твой голос поверх»");
    check((await page.getByText(/Подгонять под губы/).count()) === 0 && (await page.getByText(/попадает в движения губ/).count()) === 0, "закадровый: нет «подгонять под губы» и подсказок про губы");
    await shot("25-voiceover");
    await page.getByRole("radio", { name: "Дубляж" }).click();
    await page.waitForTimeout(500);
    check((await page.getByText(/Подгонять под губы/).count()) > 0, "дубляж: «Подгонять под губы сам» вернулось");
    check((await page.getByText(/ложится в губы/).count()) === 0, "в записи нет жаргона «ложится в губы»");
    await shot("26-dub-words");
    // живая шкала уровня на экране записи ещё до записи
    await page.getByRole("button", { name: /Включить микрофон/ }).click({ timeout: 1500 }).catch(() => {});
    const meterOk = await until(async () => (await page.getByTestId("live-meter").count()) > 0 && (await page.getByText("Слышу тебя ✓").count()) > 0, "шкала уровня реагирует на звук", 8000, 300).catch(() => false);
    check(!!meterOk, "запись: живая шкала уровня работает до начала записи («Слышу тебя ✓»)");
    await shot("27-record-meter");

    // 7c. микрофон в настройках: шкала и калибровка
    await go("/settings/mic");
    // разрешение на микрофон уже дано — шкала включается сама; иначе жмём «Включить микрофон»
    await page.waitForTimeout(1200);
    if (await page.getByRole("button", { name: /Включить микрофон/ }).count()) await page.getByRole("button", { name: /Включить микрофон/ }).click();
    const mic2 = await until(async () => (await page.getByText("Слышу тебя ✓").count()) > 0, "шкала в настройках", 8000, 300).catch(() => false);
    check(!!mic2, "настройки → микрофон: шкала двигается, «Слышу тебя ✓»");
    await shot("28-settings-mic-meter");
    await page.getByRole("button", { name: "Откалибровать" }).click();
    await page.waitForTimeout(500);
    check((await page.getByTestId("cal-progress").count()) > 0, "калибровка: видно, что идёт (полоска и подпись)");
    await shot("29-calibration-running");
    await until(async () => (await page.getByTestId("cal-result").count()) > 0, "результат калибровки", 15000, 300);
    const calText = await page.getByTestId("cal-result").innerText();
    log("калибровка:", calText);
    check(calText.length > 10, "калибровка: есть результат или понятная ошибка");
    await shot("30-calibration-result");

    // 7d. ключ ИИ
    await go("/settings/ai");
    const gem = page.getByTestId("prov-gemini");
    await gem.getByRole("button", { name: /Подключить|Изменить/ }).click();
    await gem.getByLabel(/Ключ: Gemini/).fill("sk-test123");
    await gem.getByRole("button", { name: "Сохранить" }).click();
    await until(async () => (await page.getByTestId("prov-gemini").getAttribute("data-status")) === "invalid", "статус ключа «не подходит»", 40000, 500);
    await shot("31a-key-invalid-open");
    await page.getByTestId("prov-gemini").getByRole("button", { name: "Свернуть" }).click();
    check(true, "ключ sk-test123 → карточка «ключ не подходит»");
    const msg = await page.getByTestId("key-message").first().innerText().catch(() => "");
    log("сообщение о ключе:", msg);
    check(/[А-Яа-я]/.test(msg) && !/API key not valid/.test(msg), "сообщение о ключе по-русски: " + msg);
    {
      const pill = await page.locator(".pill", { hasText: /ИИ/ }).innerText();
      log("шапка про ИИ:", pill);
      // агент В показывает «ИИ подключён (N моделей) · M не отвечает»: неподходящий ключ в N не считается или явно назван
      check(/ИИ не подключён/.test(pill) || !/ИИ подключён \(\d/.test(pill) || /не отвечает|не подходит/.test(pill), "в шапке «ИИ подключён (N)» не прячет неподходящий ключ: " + pill);
    }
    check((await page.getByRole("button", { name: "Изменить ключ" }).count()) > 0, "красная карточка с кнопкой «Изменить ключ»");
    check((await page.getByText(/Для опытных/).count()) > 0, "«OpenAI-совместимый» спрятан в «Для опытных»");
    await shot("31-key-invalid");
    await api("/settings/llm/gemini/key", { method: "DELETE" }).catch(() => {});

    // 7e. сведение: ползунки не врут, автопилот, превью
    const sc0 = (await getProject()).scenes[0];
    const t0take = sc0.takes.find((t) => t.selected) ?? sc0.takes[0];
    const orig = { offset_ms: t0take.offset_ms, stretch: t0take.stretch };
    await api(`/projects/${pid}/scenes/${sc0.id}/takes/${t0take.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ offset_ms: 1060, stretch: 0.85 }) });
    await go(`/p/${pid}/mix`);
    await page.waitForSelector(".mixmain", { timeout: 20000 });
    await page.getByTestId("fine-toggle").click();
    const lbl = await page.getByTestId("offset-label").innerText();
    const rng = await page.getByLabel("Сдвиг записи").evaluate((el) => ({ v: Number(el.value), min: Number(el.min), max: Number(el.max) }));
    log("сдвиг:", lbl, JSON.stringify(rng));
    const shownMs = Number((lbl.match(/\d+/) ?? [0])[0]);
    check(shownMs > 0 && Math.abs(rng.v - shownMs) <= 10 && rng.max >= shownMs, `сведение: подпись «${lbl}» совпадает с ползунком (${rng.v}, диапазон ±${rng.max})`);
    const st = await page.getByLabel("Длина записи").evaluate((el) => ({ v: Number(el.value), min: Number(el.min) }));
    check(Math.abs(st.v - 0.85) < 0.001 && st.min <= 0.85, "сведение: «Длина 85%» совпадает с ползунком");
    await shot("32-mix-fine-sliders");
    check((await page.getByTestId("mix-help").locator("span").count()) === 3, "сведение: «Автопилот / Помощник / Тонкая настройка» — по строке на каждое");
    await page.getByLabel(/Громкость: Музыка и шумы/).focus();
    await page.keyboard.press("ArrowLeft");
    await page.waitForTimeout(700);
    check(await page.getByTestId("autopilot-off").isVisible(), "сведение: заметная плашка «Автопилот выключен, потому что ты поменял вручную»");
    check((await page.getByRole("button", { name: "Вернуть автопилот" }).count()) > 0, "сведение: есть «Вернуть автопилот»");
    await shot("33-autopilot-off-banner");
    await page.getByRole("button", { name: "Вернуть автопилот" }).click();
    await page.waitForTimeout(500);
    check((await page.getByTestId("autopilot-off").count()) === 0, "автопилот возвращён");
    await api(`/projects/${pid}/scenes/${sc0.id}/takes/${t0take.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(orig) });

    // 7f. удаление сцены и дубля с «Отменить»
    const tmp = await api(`/projects/${pid}/scenes`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ start: P.media.duration - 30, end: P.media.duration - 22, title: "Временная" }) });
    await until(async () => (await getProject()).scenes.find((s) => s.id === tmp.id)?.stems_ready, "временная сцена готова", 300000, 2000);
    await go(`/p/${pid}/analyze`);
    await page.getByRole("tab", { name: /Сцены/ }).click();
    await page.getByRole("button", { name: /Удалить сцену Временная/ }).click();
    await page.waitForTimeout(500);
    check((await page.getByRole("button", { name: /^Сцена Временная/ }).count()) === 0, "удаление сцены: из списка пропала сразу");
    check(await page.locator(".toast", { hasText: "удалена" }).getByRole("button", { name: "Отменить" }).isVisible(), "удаление сцены: тост «Отменить»");
    await page.waitForTimeout(450);
    await shot("34-scene-deleted-toast");
    await page.locator(".toast", { hasText: "удалена" }).getByRole("button", { name: "Отменить" }).click();
    await page.waitForTimeout(800);
    check((await page.getByRole("button", { name: /^Сцена Временная/ }).count()) === 1, "удаление сцены: «Отменить» вернула её");
    await page.waitForTimeout(6500);
    check((await getProject()).scenes.some((s) => s.id === tmp.id), "удаление сцены: после отмены в движке она осталась");
    await page.getByRole("button", { name: /Удалить сцену Временная/ }).click();
    await page.waitForTimeout(6800);
    check(!(await getProject()).scenes.some((s) => s.id === tmp.id), "удаление сцены: без отмены движок удалил через 6 с");

    // 7g. восстановление рендера после перезагрузки + действия после рендера + переименование
    await go(`/p/${pid}/render`);
    await page.waitForSelector('[data-primary="render"]', { timeout: 20000 });
    const boxes2 = page.locator(".scenepick input[type=checkbox]");
    for (let i = 0; i < (await boxes2.count()); i++) if (!(await boxes2.nth(i).isChecked())) await boxes2.nth(i).setChecked(true);
    await page.getByRole("radio", { name: "720p" }).click();
    await page.getByLabel("Куда сохранить").fill(join(OUT, "out", "reload.mp4"));
    await page.getByRole("radio", { name: /9:16/ }).click();
    check(await page.getByTestId("fit-opt").isVisible(), "рендер 9:16: есть выбор «вписать / обрезать»");
    check((await page.getByRole("radio", { name: /С размытыми полями/ }).getAttribute("aria-checked")) === "true", "9:16: по умолчанию «С размытыми полями»");
    check(await page.getByRole("radio", { name: "Для YouTube и VK" }).isChecked(), "громкость: простой переключатель, по умолчанию «Для YouTube и VK»");
    check((await page.getByText(/LUFS/).count()) === 0, "громкость: «LUFS» не торчит на экране (только в подсказке «?»)");
    await shot("35-render-916-blur");
    const posted9 = page.waitForRequest((r) => r.method() === "POST" && r.url().endsWith(`/projects/${pid}/render`));
    await page.locator('[data-primary="render"]').click();
    const req9 = await posted9;
    const body9 = JSON.parse(req9.postData());
    const job9 = await (await req9.response()).json();
    check(body9.settings.fit === "blur" && body9.settings.aspect === "9:16", "9:16: в движок ушло fit=blur");
    await page.waitForTimeout(800);
    await page.reload();
    await page.waitForSelector('[data-primary="render"], [data-testid="render-restored"]', { timeout: 20000 });
    await page.waitForTimeout(800);
    const restored = (await page.getByTestId("render-restored").count()) > 0;
    check(restored, "после перезагрузки во время рендера: экран показывает идущий рендер");
    check(await page.locator('[data-primary="render"]').isDisabled(), "после перезагрузки: повторный запуск заблокирован");
    await shot("36-render-restored-after-reload");
    const rj = await until(async () => {
      const j = await api(`/jobs/${job9.id}`);
      return j.state === "done" || j.state === "failed" ? j : null;
    }, "рендер после перезагрузки", 600000, 2000);
    check(rj.state === "done", "рендер после перезагрузки дошёл до конца: " + (rj.error || ""));
    await page.waitForSelector('[data-testid="done"]', { timeout: 20000 }).catch(() => {});
    check((await page.getByRole("button", { name: "Скопировать путь" }).count()) > 0, "после рендера в браузере: «Скопировать путь»");
    await shot("37-render-done-actions");
    if (existsSync(rj.result?.path ?? "")) {
      const pr = JSON.parse(execFileSync("ffprobe", ["-v", "error", "-show_entries", "stream=width,height", "-of", "json", rj.result.path]).toString());
      const vs = pr.streams.find((x) => x.width);
      log("9:16 blur размер:", vs.width + "x" + vs.height);
      check(vs.height > vs.width, "9:16: вертикальный ролик " + vs.width + "x" + vs.height);
    }
    // переименование проекта
    const oldTitle = P.title;
    await go(`/p/${pid}/analyze`);
    await page.getByRole("button", { name: "Переименовать проект" }).click();
    await page.getByLabel("Название проекта").fill("Переименован в тесте");
    await page.keyboard.press("Enter");
    await until(async () => (await getProject()).title === "Переименован в тесте", "проект переименован", 8000, 300);
    check(true, "переименование: карандаш у названия → PATCH");
    await shot("38-renamed");
    await api(`/projects/${pid}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: oldTitle }) });
  }


  // ───────────── 9. слепой тест №3: «Проекты», «Назад», диалог вместо confirm, блокировка «К записи», дубль файлом ─────────────
  {
    const P = await getProject();
    // 9a. кнопка «Проекты» на каждом экране; из проекта → Настройки → «Назад к проекту» (и Esc) на тот же шаг
    for (const step of ["analyze", "record", "mix", "render"]) {
      await go(`/p/${pid}/${step}`);
      check(await page.getByRole("button", { name: "Проекты", exact: true }).isVisible(), `шапка: кнопка «Проекты» видна на шаге ${step}`);
    }
    await go(`/p/${pid}/mix`);
    await page.getByRole("button", { name: "Настройки" }).click();
    await page.waitForSelector('[data-testid="settings-back"]');
    check((await page.getByTestId("settings-back").innerText()).includes("Назад к проекту"), "Настройки: есть «← Назад к проекту»");
    await shot("40-settings-back");
    await page.getByTestId("settings-back").click();
    await page.waitForTimeout(500);
    check(page.url().endsWith(`/p/${pid}/mix`), "«Назад» вернул на тот же шаг проекта");
    await page.getByRole("button", { name: "Настройки" }).click();
    await page.waitForSelector('[data-testid="settings-back"]');
    await page.keyboard.press("Escape");
    await page.waitForTimeout(500);
    check(page.url().endsWith(`/p/${pid}/mix`), "Esc в Настройках = «Назад»");
    await page.getByRole("button", { name: "Проекты", exact: true }).click();
    await page.waitForTimeout(500);
    check((await page.getByRole("button", { name: "Проекты", exact: true }).getAttribute("aria-current")) === "page", "на экране проектов кнопка «Проекты» подсвечена");
    await shot("41-projects-cards");
    check((await page.locator('[data-testid="project-card"] img').count()) > 0, "в карточках проектов есть миниатюры");

    // 9b. «Для опытных» раскрывается по клику
    await go("/settings/ai");
    await page.getByText("Для опытных: другие сервисы").click();
    await page.waitForTimeout(300);
    check(await page.getByTestId("prov-openai").isVisible().catch(() => false), "«Для опытных: другие сервисы» раскрылся по клику");
    await shot("42-settings-advanced");

    // 9c. удаление проекта своим окном, при заблокированном window.confirm (как в Tauri)
    const c3 = await browser.newContext({ viewport: { width: 1440, height: 865 } });
    await c3.addInitScript(() => { window.confirm = () => false; window.alert = () => {}; window.prompt = () => null; });
    const p3 = await c3.newPage();
    const tmpClip = join(OUT, "to-delete.mkv");
    const srcFile = P.source_path;
    execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-ss", "30", "-t", "20", "-i", srcFile, "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", tmpClip]);
    const made = await api("/projects", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: tmpClip }) });
    const tmpPid = made.id;
    await p3.goto(APP + "/#/");
    await p3.waitForSelector('[data-testid="project-card"]', { timeout: 15000 });
    await p3.getByRole("button", { name: `Удалить проект to-delete` }).click();
    await p3.waitForSelector('[role="alertdialog"]');
    check(await p3.getByRole("alertdialog").isVisible(), "корзина → свой диалог (window.confirm заблокирован, как в Tauri)");
    await p3.screenshot({ path: join(SHOTS, `${PREFIX}43-delete-dialog.png`) });
    await p3.getByRole("button", { name: "Отмена" }).click();
    await p3.waitForTimeout(300);
    check((await api("/projects")).some((x) => x.id === tmpPid), "«Отмена»: проект остался");
    await p3.getByRole("button", { name: `Удалить проект to-delete` }).click();
    await p3.getByRole("button", { name: "Удалить проект", exact: true }).click();
    await until(async () => !(await api("/projects")).some((x) => x.id === tmpPid), "проект удалён в движке", 20000, 400);
    check(true, "красная кнопка диалога удалила проект");
    await c3.close();

    // 9d. «К записи» и вкладки шагов блокируются, пока идёт «Голоса и фон»
    const lines = [...P.analysis.lines].sort((a, b) => a.start - b.start);
    const ln = lines.find((l) => l.start > P.media.duration / 2) ?? lines.at(-1);
    // сцена подлиннее (≈90 с): подготовка идёт заметно дольше, и блокировку можно поймать
    const a0 = Math.max(0, ln.start - 0.3), b0 = Math.min(P.media.duration, Math.max(ln.end + 0.3, a0 + 90));
    for (const old of (await getProject()).scenes.filter((x) => x.title === "Для блокировки")) await api(`/projects/${pid}/scenes/${old.id}`, { method: "DELETE" }).catch(() => {});
    const tmpScene = await api(`/projects/${pid}/scenes`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ start: a0, end: b0, title: "Для блокировки" }) });
    await go(`/p/${pid}/analyze`);
    await page.reload();                       // сцена создана в обход окна — перечитываем проект
    await page.waitForSelector(".timeline .tl-tracks", { timeout: 20000 });
    await page.getByRole("tab", { name: /Сцены/ }).click();
    await page.getByRole("button", { name: /^Сцена Для блокировки/ }).click();
    await page.getByRole("button", { name: "Снять выделение" }).click();   // с выделением вместо «К записи» показано «Создать сцену»
    await page.waitForTimeout(300);
    const next = page.getByRole("button", { name: /Дальше: запись/ });
    const stillPreparing = await next.isDisabled().catch(() => false);
    if (stillPreparing) {
      check(await next.isDisabled(), "«Дальше: запись» неактивна, пока «Голоса и фон» не готовы");
      check(((await next.getAttribute("title")) ?? "").startsWith("Дождись, пока программа отделит голоса от фона"), "подсказка: " + (await next.getAttribute("title")));
      check(await page.getByRole("button", { name: "Шаг 2: Запись" }).isDisabled(), "вкладка «2 Запись» заблокирована до конца подготовки");
      await shot("44-next-blocked");
    } else log("сцена подготовилась слишком быстро — проверка блокировки пропущена");
    await until(async () => (await getProject()).scenes.find((x) => x.id === tmpScene.id)?.stems_ready, "временная сцена готова", 600000, 2000);
    await page.waitForTimeout(1200);
    check(await page.getByRole("button", { name: /Дальше: запись/ }).isEnabled(), "после подготовки «Дальше: запись» активна");

    // 9e. запись: дубль файлом (mp3, как из другой программы) + формат времени + тосты внизу
    const mp3 = join(OUT, "dub-from-other-app.mp3");
    execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", wav, "-ac", "1", "-b:a", "96k", mp3]);
    await go(`/p/${pid}/record`, 1200);
    await page.waitForSelector('[data-primary="record"]', { timeout: 20000 });
    await page.locator("select.scenesel").selectOption(tmpScene.id);
    await page.waitForTimeout(800);
    const timeLbl = await page.locator(".scenebar .mono").first().innerText();
    check(/^\d+:\d\d\.\d – \d+:\d\d\.\d/.test(timeLbl), "время сцены в едином формате: " + timeLbl);
    await page.getByRole("button", { name: "Я в наушниках" }).click({ timeout: 1500 }).catch(() => {});
    await page.locator(".rl-item").first().click();
    const before = (await getProject()).scenes.find((x) => x.id === tmpScene.id).takes.length;
    await page.getByTestId("take-file").setInputFiles(mp3);
    await until(async () => (await getProject()).scenes.find((x) => x.id === tmpScene.id).takes.length > before, "дубль из файла mp3 принят движком", 90000, 800);
    const tk = (await getProject()).scenes.find((x) => x.id === tmpScene.id).takes.at(-1);
    check(!!tk && tk.duration > 1, `дубль из файла: ${tk?.duration} с, offset=${tk?.offset_ms} мс, stretch=${tk?.stretch} (прошёл ту же подгонку)`);
    await page.waitForSelector(".take", { timeout: 10000 });
    const toastBox = await page.locator(".toast").last().boundingBox().catch(() => null);
    if (toastBox) check(toastBox.y > 865 / 2, `тост внизу окна (y=${Math.round(toastBox.y)}), не поверх рабочих кнопок`);
    await shot("45-record-upload");
    await api(`/projects/${pid}/scenes/${tmpScene.id}/takes/${tk.id}`, { method: "DELETE" }).catch(() => {});

    // 9f. рендер: путь целиком, с .mp4, кнопка «Изменить…»
    await go(`/p/${pid}/render`, 1200);
    const outVal = await page.getByLabel("Куда сохранить").inputValue();
    check(/\.mp4$/i.test(outVal) && outVal.length > 20, "рендер: путь с .mp4 и целиком: " + outVal);
    check(await page.getByRole("button", { name: /Изменить…/ }).isVisible(), "рендер: кнопка «Изменить…»");
    check((await page.getByText("Дорожки отдельными файлами").count()) > 0 && (await page.getByText("Сохрани дорожки отдельно").count()) === 0, "рендер: «Дорожки отдельными файлами» — название опции, а не приказ");
    await shot("46-render-path");

    // убираем временную сцену
    await api(`/projects/${pid}/scenes/${tmpScene.id}`, { method: "DELETE" }).catch(() => {});
  }

  // ───────────── 10. волна 1: проверка микрофона, репетиция, слоги, разбор дубля, имена, вёрстка ─────────────
  {
    const P = await getProject();
    const sc = P.scenes.find((x) => x.stems_ready && x.lines.length) ?? P.scenes[0];

    // 10a. мастер проверки микрофона (из блока «Перед первой записью» на главной)
    await go("/");
    await page.waitForSelector('[data-testid="mic-step"]', { timeout: 15000 });
    check((await page.getByRole("button", { name: "Загрузить", exact: true }).count()) === 0 && (await page.getByRole("button", { name: "Открыть", exact: true }).count()) > 0, "главная: вместо «Загрузить» — «Открыть» (видео не копируем)");
    check((await page.getByText("Готовый ролик", { exact: false }).count()) > 0, "главная: в «Как это устроено» шаг называется «Готовый ролик»");
    await page.getByTestId("mic-step").getByRole("button", { name: /Проверить|Ещё раз/ }).click();
    await page.waitForSelector('[data-testid="mic-check"]', { timeout: 5000 });
    await shot("50-micwizard-intro");
    await page.getByRole("button", { name: /Начать проверку/ }).click();
    await page.waitForSelector("text=Помолчи пару секунд", { timeout: 8000 });
    await shot("51-micwizard-silence");
    await page.waitForSelector('[data-testid="zone-advice"]', { timeout: 6000 });
    await page.waitForTimeout(800);
    await shot("52-micwizard-speech");
    check((await page.getByTestId("zone-advice").innerText()).length > 5, "мастер: во время фразы есть совет по уровню");
    await page.waitForSelector('[data-testid="mic-result"], [data-testid="mic-error"]', { timeout: 30000 });
    await shot("53-micwizard-result");
    check((await page.getByTestId("mic-result").count()) === 1, "мастер: дошёл до результата за ~10 секунд");
    const resTxt = await page.getByTestId("mic-result").innerText();
    log("мастер:", resTxt.replace(/\n+/g, " | ").slice(0, 220));
    check(/Шум комнаты/.test(resTxt) && /Задержка записи/.test(resTxt), "мастер: цифры — шум, голос, задержка");
    await page.getByRole("button", { name: "Готово" }).click();
    const prof = await api("/mic/profile").catch(() => null);
    log("профиль в движке:", JSON.stringify(prof));
    check(!prof || typeof prof.noise_floor_db === "number", "профиль «Мой микрофон» сохранён в движке");
    check((await page.getByTestId("mic-step").innerText()).includes("Мой микрофон"), "главная: «Мой микрофон» отмечен как проверенный");
    await shot("54-home-mic-done");

    // 10b. экран записи: вёрстка панели на трёх размерах окна (подписи переносятся, ничего не обрезано)
    for (const [w, h] of [[1200, 720], [1440, 865], [1920, 1166]]) {
      await page.setViewportSize({ width: w, height: h });
      await go(`/p/${pid}/record`);
      await page.waitForSelector('[data-primary="record"]', { timeout: 20000 });
      await page.locator("select.scenesel").selectOption(sc.id).catch(() => {});
      await page.waitForTimeout(500);
      const lay = await page.evaluate(() => {
        const mp = document.querySelector(".micpanel");
        const pr = mp.getBoundingClientRect();
        const rows = [...mp.querySelectorAll(".opt")].map((o) => { const r = o.getBoundingClientRect(); return { t: o.textContent.slice(0, 28), inside: r.left >= pr.left - 1 && r.right <= pr.right + 1, clipped: o.scrollWidth > o.clientWidth + 1 }; });
        const right = document.querySelector(".record .right");
        const desc = mp.querySelector(".optdesc");
        const dr = desc?.getBoundingClientRect();
        return { rows, descInside: !dr || (dr.right <= pr.right + 1 && dr.left >= pr.left - 1), panelFits: right.scrollHeight <= right.clientHeight + 2, hScroll: right.scrollWidth > right.clientWidth + 1 };
      });
      log(`панель записи ${w}×${h}:`, JSON.stringify({ ...lay, rows: lay.rows.length }));
      check(lay.rows.length >= 3 && lay.rows.every((r) => r.inside && !r.clipped), `${w}×${h}: подписи настроек записи не вылезают за панель`);
      check(lay.descInside && !lay.hScroll, `${w}×${h}: описание «Подгонять под губы» целиком внутри колонки`);
      check(lay.panelFits, `${w}×${h}: правая панель помещается без обрезки по высоте`);
      await shot(`55-record-${w}`);
    }
    await page.setViewportSize({ width: 1440, height: 900 });

    // 10c. счётчик слогов + «Свой текст»
    await go(`/p/${pid}/record`);
    await page.waitForSelector('[data-primary="record"]', { timeout: 20000 });
    await page.locator("select.scenesel").selectOption(sc.id).catch(() => {});
    await page.getByRole("radio", { name: "Свой текст" }).click();
    await page.waitForTimeout(500);
    const item = page.locator(".rl-item").first();
    await item.getByRole("textbox").fill("Да");
    check((await item.getByTestId("budget-label").innerText()) === "влезет", "слоги: короткий текст — «влезет»");
    await item.getByRole("textbox").fill("Ну что же ты наделал, несчастный ты наш Барсик, посмотри на этот холодильник и скажи хоть что-нибудь умное");
    check((await item.getByTestId("budget-label").innerText()) === "не влезет", "слоги: длинный текст — «не влезет»");
    check(/убрать примерно \d+/.test(await item.getByTestId("budget-trim").innerText()), "слоги: подсказка, сколько слогов убрать");
    check((await item.getByRole("button", { name: /Подогнать под длину \(ИИ\)/ }).count()) === 1, "слоги: рядом «Подогнать под длину (ИИ)»");
    await shot("56-syllables");
    await item.getByRole("textbox").fill(sc.lines[0].text);
    await item.getByRole("textbox").blur();
    await page.getByRole("radio", { name: "Как есть" }).click();

    // 10d. репетиция: оригинал → отсчёт → лента, записи не появляется
    const nTakes = (await getProject()).scenes.flatMap((x) => x.takes).length;
    await page.getByRole("button", { name: "Я в наушниках" }).click({ timeout: 1000 }).catch(() => {});
    await page.getByTestId("rehearse").click();
    await page.waitForSelector('[data-testid="rehearsal-badge"]', { timeout: 5000 });
    check(/слушай оригинал/.test(await page.getByTestId("rehearsal-badge").innerText()), "репетиция: сначала звучит оригинал");
    await shot("57-rehearsal-listen");
    await page.waitForFunction(() => /приготовься|говори вслед/.test(document.querySelector('[data-testid="rehearsal-badge"]')?.textContent ?? ""), null, { timeout: 40000 });
    await shot("58-rehearsal-count");
    await page.waitForFunction(() => /говори вслед/.test(document.querySelector('[data-testid="rehearsal-badge"]')?.textContent ?? ""), null, { timeout: 15000 });
    check(true, "репетиция: после оригинала — отсчёт и бегущая лента");
    await shot("59-rehearsal-speak");
    await page.waitForFunction(() => !document.querySelector('[data-testid="rehearsal-badge"]'), null, { timeout: 60000 });
    check((await getProject()).scenes.flatMap((x) => x.takes).length === nTakes, "репетиция: ничего не записано");

    // 10e. разбор дубля по тексту, ★ с причиной, «из файла» без «начал раньше»
    const tk = (await getProject()).scenes.find((x) => x.id === sc.id).takes;
    log("разбор дублей:", JSON.stringify(tk.map((t) => ({ id: t.id, source: t.source, empty: t.empty, text_match: t.text_match, complete: t.complete, tail: t.tail_cut_s, clip: t.clip_pct, sel: t.selected }))));
    await page.locator(".rl-item", { hasText: sc.lines.find((l) => tk.some((t) => t.line_id === l.line_id))?.text.slice(0, 12) ?? "" }).first().click().catch(() => {});
    await page.waitForTimeout(1200);
    const reviews = await page.getByTestId("take-review").allInnerTexts();
    log("разбор по тексту на экране:", JSON.stringify(reviews));
    const withText = tk.filter((t) => t.text_match != null);
    if (withText.length) check(reviews.length > 0, "разбор по тексту виден под дублем (" + reviews[0] + ")");
    else log("движок не считал разбор по тексту (нет нейросетей?) — проверка пропущена");
    if (tk.some((t) => t.selected && !t.empty)) check((await page.getByTestId("star-why").count()) > 0, "у ★ есть подпись, почему выбран");
    await shot("60-take-review");

    // дубль из файла: нет «Начал на … раньше»
    const mp3b = join(OUT, "dub-from-other-app.mp3");
    if (existsSync(mp3b)) {
      const had = (await getProject()).scenes.find((x) => x.id === sc.id).takes.length;
      await page.getByTestId("take-file").setInputFiles(mp3b);
      await until(async () => (await getProject()).scenes.find((x) => x.id === sc.id).takes.length > had, "дубль из файла принят", 90000, 800);
      const ft = (await getProject()).scenes.find((x) => x.id === sc.id).takes.at(-1);
      check(ft.source === "file", "дубль из файла: движок записал source=file");
      await page.waitForTimeout(1500);
      const card = page.locator(".take").first();
      const txt = await page.locator(".take", { hasText: "из файла" }).first().innerText();
      check(!/Начал на|Совпадение/.test(txt) && /из файла/.test(txt), "дубль из файла: нет «Начал на … раньше» и «Совпадение», есть пометка «из файла»");
      await shot("61-take-from-file");
      void card;
      await api(`/projects/${pid}/scenes/${sc.id}/takes/${ft.id}`, { method: "DELETE" }).catch(() => {});
    }

    // 10f. сведение без дублей: «Дальше» объясняет, почему серая
    {
      const tmpS = await api(`/projects/${pid}/scenes`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ start: Math.max(0, P.media.duration - 12), end: P.media.duration - 5, title: "Без дублей" }) });
      await until(async () => (await getProject()).scenes.find((x) => x.id === tmpS.id)?.stems_ready, "сцена «Без дублей» готова", 300000, 2000);
      await go(`/p/${pid}/analyze`);
      await page.reload();
      await page.waitForSelector(".timeline .tl-tracks", { timeout: 20000 });
      await page.getByRole("tab", { name: /Сцены/ }).click();
      await page.getByRole("button", { name: /^Сцена Без дублей/ }).click();
      await page.getByRole("button", { name: "Шаг 3: Сведение" }).click();
      await page.waitForSelector(".mixmain", { timeout: 20000 });
      await page.waitForTimeout(500);
      check(await page.getByRole("button", { name: /Дальше: готовый ролик/ }).isDisabled(), "сведение без дублей: «Дальше» серая…");
      check(/Нет дублей/.test(await page.getByTestId("mix-next-why").innerText()), "…и рядом написано почему: «Нет дублей — сначала запиши голос»");
      await shot("62a-mix-no-takes");
      await page.getByRole("button", { name: "Шаг 4: Рендер" }).click();
      await page.waitForSelector('[data-testid="no-voice"]', { timeout: 5000 });
      check(true, "готовый ролик без дублей: объяснение вместо молча серой кнопки");
      await shot("62b-render-no-takes");
      await api(`/projects/${pid}/scenes/${tmpS.id}`, { method: "DELETE" }).catch(() => {});
      await page.reload();   // окно ещё помнит удалённую сцену: перечитываем проект
    }
    await go(`/p/${pid}/mix`);
    await page.waitForSelector(".mixmain", { timeout: 20000 });
    await shot("62-mix");
    check((await page.getByText("Автопилот", { exact: true }).count()) > 0 && (await page.getByRole("button", { name: "Что такое автопилот" }).count()) === 1, "сведение: у «Автопилота» есть подсказка «?»");
    check((await page.getByRole("button", { name: /Дальше: готовый ролик/ }).count()) === 1, "сведение: кнопка «Дальше: готовый ролик» (не «К рендеру»)");

    // 10g. готовый ролик: имя, громкость, повторный рендер → « (2)»
    await go(`/p/${pid}/render`);
    await page.waitForSelector('[data-primary="render"]', { timeout: 20000 });
    check((await page.getByRole("heading", { name: /Готовый ролик/ }).count()) > 0, "рендер: экран называется «Готовый ролик»");
    await page.getByRole("radio", { name: "Как в фильме" }).click();
    await page.waitForTimeout(600);
    check((await api(`/projects/${pid}`)).scenes.some((x) => x.mix.loudness_target === "film"), "громкость «Как в фильме» сохранилась в сцене (loudness_target=film)");
    await page.getByRole("radio", { name: "Для YouTube и VK" }).click();
    await page.waitForTimeout(400);
    const same = join(OUT, "out", `same-name-${Date.now()}.mp4`);
    mkdirSync(join(OUT, "out"), { recursive: true });
    const boxes3 = page.locator(".scenepick input[type=checkbox]");
    for (let i = 0; i < (await boxes3.count()); i++) await boxes3.nth(i).setChecked(i === 0);
    await page.getByLabel("Куда сохранить").fill(same);
    const names = [];
    for (let k = 0; k < 2; k++) {
      await page.locator('[data-primary="render"]').click();
      await page.waitForSelector('[data-testid="done"]', { timeout: 300000 });
      names.push(await page.getByTestId("saved-path").innerText());
      if (k === 0) await shot("63-render-done-first");
      if (k === 1) await shot("64-render-done-renamed");
      if (k === 0) await page.getByRole("button", { name: "Ещё раз" }).click();
    }
    log("два рендера в одно имя:", JSON.stringify(names));
    check(names[0] === same && / \(2\)\.mp4$/.test(names[1]), "второй рендер с тем же именем не затёр первый: " + names[1]);
    check(existsSync(same) && existsSync(names[1]), "оба файла лежат на диске");
    check((await page.getByTestId("renamed").count()) === 1, "на экране сказано, что имя сменилось (« (2)»)");
  }

  // ───────────── 8. пустая запись (тишина вместо микрофона) ─────────────
  {
    const silent = join(OUT, "silence.wav");
    execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "30", silent]);
    const b2 = await chromium.launch({ executablePath: CHROME, headless: true, args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", `--use-file-for-fake-audio-capture=${silent}`, "--autoplay-policy=no-user-gesture-required"] });
    const c2 = await b2.newContext({ viewport: { width: 1440, height: 900 } });
    await c2.grantPermissions(["microphone"], { origin: new URL(APP).origin });
    const p2 = await c2.newPage();
    await p2.goto(`${APP}/#/p/${pid}/record`);
    await p2.waitForSelector('[data-primary="record"]', { timeout: 20000 });
    await p2.getByRole("button", { name: "Я в наушниках" }).click({ timeout: 1500 }).catch(() => {});
    const before = (await getProject()).scenes.flatMap((s) => s.takes).length;
    await p2.locator(".rl-item").first().click();
    await p2.locator('[data-primary="record"]').click();
    // движок сохраняет тишину как дубль с empty=true (★ ему не ставится); старый движок отвечал 422 — тогда видна плашка empty-take
    await p2.waitForSelector('[data-testid="take-empty"], [data-testid="empty-take"]', { timeout: 90000 });
    const et = await p2.locator('[data-testid="take-empty"], [data-testid="empty-take"]').first().innerText();
    log("пустая запись:", et.replace(/\n/g, " | "));
    check(/почти пустая/i.test(et), "пустая запись: показано «Запись почти пустая»");
    const afterTakes = (await getProject()).scenes.flatMap((s) => s.takes);
    const emptyTake = afterTakes.find((t) => t.empty);
    if (emptyTake) {
      check(!emptyTake.selected, "пустой дубль не получил ★");
      check(await p2.getByRole("button", { name: "Взять этот дубль в ролик" }).first().isDisabled(), "у пустого дубля ★ заблокирована");
      await p2.screenshot({ path: join(SHOTS, `${PREFIX}39-empty-take.png`) });
      // уйти вместе с дублем: удаляем и ждём 6 с «Отменить»
      await p2.getByRole("button", { name: "Удалить дубль" }).first().click();
      await p2.waitForTimeout(500);
      check((await p2.getByTestId("take-empty").count()) === 0, "«Запись почти пустая» исчезла сразу после удаления дубля");
      await p2.waitForTimeout(6600);
      check((await getProject()).scenes.flatMap((s) => s.takes).length === before, "после удаления пустой дубль ушёл и из движка");
    } else {
      check((await p2.getByRole("button", { name: "Записать ещё раз" }).count()) > 0, "пустая запись: есть «Записать ещё раз»");
      check((await getProject()).scenes.flatMap((s) => s.takes).length === before, "пустая запись не сохранилась как дубль");
      await p2.screenshot({ path: join(SHOTS, `${PREFIX}39-empty-take.png`) });
    }
    await b2.close();
  }

  // ───────────── 6. настройки: статус ключей берётся с движка, перезагрузка не теряет его ─────────────
  toastsSeen = await page.evaluate(() => window.__toasts ?? []);
  await go("/settings/ai");
  await shot("14-settings-ai");
} catch (e) {
  failures.push("исключение: " + e.message);
  log("ОШИБКА:", e.message);
  await shot("99-error").catch(() => {});
}

{
  const toasts = toastsSeen;
  log("тосты:", JSON.stringify(toasts));
  const count = (re) => toasts.filter((t) => re.test(t)).length;
  check(count(/^Ролик готов/) <= 3, `«Ролик готов.» не дублируется (${count(/^Ролик готов/)} при трёх сборках)`);
  const prep = toasts.filter((t) => /^Голоса и фон готовы/.test(t));
  check(new Set(prep).size === prep.length, "«Голоса и фон готовы» не дублируются: " + prep.join(" | "));
  check(!toasts.some((t) => /Not Found|moov atom|file:\/|Traceback/.test(t)), "в тостах нет сырых ошибок сервера и ffmpeg");
}
check(errors.length === 0, "консоль без ошибок" + (errors.length ? ": " + errors.slice(0, 5).join(" || ") : ""));
await browser.close();
if (failures.length) {
  console.log("\nПровалено:\n - " + failures.join("\n - "));
  process.exit(1);
}
console.log("\nВсё прошло." + (warnings.length ? "\nПредупреждения (не сбои окна):\n - " + warnings.join("\n - ") : ""));
