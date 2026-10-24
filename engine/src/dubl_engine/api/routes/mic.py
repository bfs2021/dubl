"""Проверка микрофона и профиль «Мой микрофон»."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from ... import settings as app_settings
from ...analysis import mic as micmod
from ...settings import MicProfile
from ...util import EngineError
from ..deps import SettingsDep, require_token
from ..schemas import ErrorResponse, LatencyResult, LatencySignal, MicCheckResult, Ok

router = APIRouter(prefix="/mic", tags=["mic"], dependencies=[Depends(require_token)],
                   responses={400: {"model": ErrorResponse}})

MAX_RECORD_BYTES = 50 * 1024 * 1024        # минутная запись 48 кГц float32 — 11 МБ; больше проверке микрофона не нужно


async def _read(file: UploadFile) -> bytes:
    data = await file.read(MAX_RECORD_BYTES + 1)
    if not data:
        raise EngineError("Пустая запись. Попробуй записать ещё раз.")
    if len(data) > MAX_RECORD_BYTES:
        raise EngineError("Запись слишком длинная: для проверки хватит пары десятков секунд.")
    return data


def _save_profile(request: Request, **fields) -> MicProfile:
    """Обновить профиль «Мой микрофон» (поля, которых не мерили, остаются прежними) и сохранить настройки."""
    cur: app_settings.AppSettings = request.app.state.settings
    prof = (cur.mic_profile or MicProfile()).model_copy(update={k: v for k, v in fields.items() if v is not None})
    prof.checked_at = datetime.now().astimezone()
    prof.device = cur.input_device
    new = cur.model_copy(update={"mic_profile": prof})
    if fields.get("latency_ms") is not None:
        new.latency_ms = float(min(max(fields["latency_ms"], 0.0), 2000.0))       # его же использует подгонка дублей
    request.app.state.settings = app_settings.save_settings(new)
    return prof


@router.get("/profile")
def get_profile(settings: SettingsDep) -> MicProfile | None:
    """Профиль «Мой микрофон»; null — микрофон ещё не проверяли."""
    return settings.mic_profile


@router.delete("/profile")
def delete_profile(request: Request) -> Ok:
    """Забыть профиль «Мой микрофон» (задержка записи AppSettings.latency_ms остаётся)."""
    cur: app_settings.AppSettings = request.app.state.settings
    request.app.state.settings = app_settings.save_settings(cur.model_copy(update={"mic_profile": None}))
    return Ok()


@router.post("/check")
async def check_mic(request: Request, file: Annotated[UploadFile, File(description="Запись микрофона: тишина, потом "
                    "фраза вслух (wav из браузера или любой звук, который понимает ffmpeg)")],
                    silence_s: Annotated[float | None, Form(ge=0.0, le=30.0, allow_inf_nan=False,
                    description="Сколько секунд от начала записи человек молчал (в мастере 2). Не указан — движок "
                    "ищет тишину сам")] = None,
                    save: Annotated[bool, Form(description="Записать результат в профиль «Мой микрофон»")] = True,
                    ) -> MicCheckResult:
    """Проверка микрофона: уровень шума в тишине, пик и средний уровень речи, перегруз и совет по-русски.
    Советы: «Всё хорошо…», «Тихо — подойди ближе…», «Громковато — отодвинься…» (и отдельно про перегруз),
    «В комнате шумно…», «Не слышу твой голос…». Профиль («Мой микрофон») обновляется только если речь нашлась."""
    data = await _read(file)

    def work() -> MicCheckResult:
        x = micmod.load_audio(data)
        m = micmod.analyze_mic(x, micmod.SR, silence_s=silence_s)
        return MicCheckResult(noise_floor_db=m.noise_floor_db, speech_db=m.speech_db, peak_db=m.peak_db, snr_db=m.snr_db,
                              clipping=m.clipping, speech_found=m.speech_found, verdict=m.verdict,  # type: ignore[arg-type]
                              advice=m.advice)
    res = await run_in_threadpool(work)
    if save and res.speech_found:
        _save_profile(request, noise_floor_db=res.noise_floor_db, speech_db=res.speech_db, peak_db=res.peak_db)
        res.saved = True
    return res


@router.get("/latency_signal")
def latency_signal_info() -> LatencySignal:
    """Описание сигнала для замера задержки петлёй. Сценарий: открыть микрофон и начать запись → через ≈0,3 с запустить
    проигрывание GET /mic/latency_signal.wav в динамик (запомнить, на какой секунде записи запустили) → писать не меньше
    min_record_s → отправить запись в POST /mic/latency с play_offset_s."""
    return LatencySignal(
        sample_rate=micmod.SIGNAL_SR, duration_s=micmod.SIGNAL_DURATION_S, beep_count=len(micmod.SIGNAL_BEEPS_S),
        beep_starts_s=list(micmod.SIGNAL_BEEPS_S), beep_s=micmod.SIGNAL_BEEP_S, freq_from_hz=micmod.SIGNAL_F0,
        freq_to_hz=micmod.SIGNAL_F1, min_record_s=micmod.MIN_RECORD_S, wav_url="/mic/latency_signal.wav",
        how_to="Сними наушники, включи колонки. Пока микрофон пишет, из динамика сыграют три коротких сигнала; по тому, "
               "насколько позже они попали в запись, считается задержка.")


@router.get("/latency_signal.wav", response_class=Response, responses={200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}}})
def latency_signal_wav() -> Response:
    """Сам сигнал: wav 48 кГц, mono, 16 бит, 2 с — полсекунды тишины и три чирика по 80 мс (тон растёт от 1,5 до 3,5 кГц)."""
    return Response(micmod.latency_signal_wav(), media_type="audio/wav",
                    headers={"Cache-Control": "public, max-age=86400"})


@router.post("/latency")
async def measure_latency(request: Request, file: Annotated[UploadFile, File(description="Запись микрофона, на которую "
                          "попал сигнал из динамика")],
                          play_offset_s: Annotated[float, Form(ge=0.0, le=30.0, allow_inf_nan=False, description="На "
                          "какой секунде записи запустили проигрывание сигнала")] = 0.0,
                          save: Annotated[bool, Form(description="Записать задержку в профиль и в настройки "
                          "записи (latency_ms)")] = True) -> LatencyResult:
    """Замер задержки петлёй: движок находит в записи эталонный сигнал (GET /mic/latency_signal) кросс-корреляцией.
    latency_ms — на сколько позже сигнал попал в запись, чем его запустили; confidence ≥ 0,5 — результату можно верить
    (ok=true). Не нашёл — ok=false и message с подсказкой (колонки, тишина). При saved=true задержка стала
    AppSettings.latency_ms, и автоподгонка дублей её учитывает."""
    data = await _read(file)

    def work():
        x = micmod.load_audio(data, micmod.SIGNAL_SR)
        return micmod.measure_latency(x, micmod.SIGNAL_SR, play_offset_s=play_offset_s)
    m = await run_in_threadpool(work)
    ok = m.latency_ms is not None
    res = LatencyResult(latency_ms=m.latency_ms, confidence=m.confidence, ok=ok, message=m.message)
    if ok and save:
        _save_profile(request, latency_ms=m.latency_ms)
        res.saved = True
    return res
