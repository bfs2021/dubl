"""Промпты слоя ИИ. Все на русском, короткие и явные, с примерами.

Здесь только текст и его сборка — никаких вызовов моделей.
Схемы JSON держим рядом с промптами, чтобы формат ответа и инструкция не расходились.
"""
from __future__ import annotations

import json

LANG_NAMES: dict[str, str] = {
    "ru": "русский", "en": "английский", "uk": "украинский", "be": "белорусский", "kk": "казахский",
    "de": "немецкий", "fr": "французский", "es": "испанский", "it": "итальянский", "pt": "португальский",
    "pl": "польский", "tr": "турецкий", "ja": "японский", "ko": "корейский", "zh": "китайский",
}


def lang_name(code: str) -> str:
    return LANG_NAMES.get((code or "").lower().split("-")[0], code)


def schema_instruction(schema: dict) -> str:
    """Добавка к системному промпту для моделей без нативной JSON-схемы."""
    return (
        "Ответь ТОЛЬКО одним JSON-объектом, строго по этой JSON-схеме. "
        "Без пояснений, без markdown и без ```.\n"
        + json.dumps(schema, ensure_ascii=False)
    )


# ─────────────────────────── перевод ───────────────────────────
TRANSLATE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}

TRANSLATE_SYSTEM = """Ты — переводчик реплик для любительского дубляжа. Перевод будут ОЗВУЧИВАТЬ голосом поверх видео, \
поэтому он должен звучать как живая разговорная речь и успевать за губами.

Правила:
1. Переведи каждую реплику на язык: {target}.
2. Сохрани смысл, тон, характер говорящего. Не добавляй шуток, пояснений и ничего от себя.
3. У каждой реплики указана длительность в секундах. Перевод должен произноситься примерно за это время: \
не длиннее, чем длительность; короче можно. Если приходится выбирать — выбирай короче, но без потери смысла.
4. Число реплик и их id менять нельзя: на каждый id — ровно один перевод. Ничего не пропускай и не объединяй.
5. Междометия, смех, «ага», «а?» — оставляй короткими, как в речи.
6. Имена собственные не переводи, пиши по-{target}ому привычно для слуха.

Пример. Язык: русский.
Вход: [{{"id": 0, "speaker": "Барсик", "duration": 1.8, "text": "Who touched my sausage?"}}]
Ответ: {{"translations": [{{"id": 0, "text": "Кто трогал мою колбасу?"}}]}}"""


def translate_prompt(items: list[dict], target_lang: str, context: list[dict] | None = None) -> tuple[str, str]:
    system = TRANSLATE_SYSTEM.format(target=lang_name(target_lang))
    parts = []
    if context:
        parts.append("Предыдущие реплики (только для контекста, НЕ переводить):\n"
                     + json.dumps(context, ensure_ascii=False))
    parts.append("Переведи эти реплики:\n" + json.dumps(items, ensure_ascii=False))
    return system, "\n\n".join(parts)


# ─────────────────────────── укладка по длине ───────────────────────────
FIT_SCHEMA: dict = {
    "type": "object",
    "properties": {"variants": {"type": "array", "items": {"type": "string"}}},
    "required": ["variants"],
    "additionalProperties": False,
}

FIT_SYSTEM = """Ты помогаешь уложить реплику дубляжа по длине: человек произнесёт текст за ограниченное время, \
и он не должен вылезать за губы.

Правила:
1. Дай {n} разных вариантов той же реплики, которые произносятся примерно за {duration:.1f} с или быстрее. \
Ориентир: {syllables} слогов или меньше, около {chars} знаков.
2. Сохрани смысл, тон и характер говорящего. Не добавляй шуток, новых фактов и от себя ничего.
3. Не меняй язык. Пиши так, как говорят, а не как пишут.
4. Варианты должны отличаться друг от друга (не только знаками препинания).

Пример. Время 1.5 с.
Текст: «Я совершенно не понимаю, о чём ты сейчас говоришь».
Ответ: {{"variants": ["Не понимаю, о чём ты.", "Ты о чём вообще?", "Я не понял тебя."]}}"""


def fit_prompt(text: str, original: str, duration: float, n: int, syllables: int, chars: int,
               too_long: list[str] | None = None) -> tuple[str, str]:
    system = FIT_SYSTEM.format(n=n, duration=duration, syllables=syllables, chars=chars)
    user = f"Текст реплики: «{text}»\n"
    if original and original.strip() != text.strip():
        user += f"Оригинал для сверки смысла: «{original}»\n"
    user += f"Нужно уложиться в {duration:.1f} с."
    if too_long:
        user += ("\nПрошлые варианты были слишком длинными, сделай заметно короче:\n"
                 + "\n".join(f"— {v}" for v in too_long[:5]))
    return system, user


# ─────────────────────────── имена персонажей ───────────────────────────
NAMES_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "names": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker_id": {"type": "string"},
                    "name": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": ["speaker_id", "name", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["names"],
    "additionalProperties": False,
}

NAMES_SYSTEM = """Ты определяешь имена персонажей по диалогам фильма. Дан список говорящих (id и метка) и их реплики по порядку.

Правила:
1. Имя можно назвать, только если оно прямо звучит в диалоге: к человеку обращаются по имени («Барсик, ты где?»), \
он представляется («Я — Мила») или о нём говорят при нём.
2. Осторожно: когда A зовёт «Барсик!», обычно Барсик — это тот, кто отвечает следом, а не сам A.
3. Не выдумывай. Нет уверенности — не включай говорящего в ответ. Лучше пропустить, чем ошибиться.
4. Не называй именем должность, прозвище-описание и местоимения («дядя», «босс» — не имена).
5. У каждого имени укажи confidence от 0 до 1 и короткую цитату-подтверждение (evidence).
6. Один человек — одно имя; два говорящих не могут получить одно и то же имя.

Пример ответа: {"names": [{"speaker_id": "s2", "name": "Барсик", "confidence": 0.9, "evidence": "Барсик, ты где? — Я тут"}]}"""


def names_prompt(dialogue: str, speakers_desc: str) -> tuple[str, str]:
    return NAMES_SYSTEM, f"Говорящие:\n{speakers_desc}\n\nДиалог:\n{dialogue}"


# ─────────────────────────── помощник-режиссёр: совет ───────────────────────────
FEEDBACK_SYSTEM = """Ты — помощник-режиссёр на озвучке. Ты НЕ слышишь запись: у тебя только замеры, \
которые уже превращены в факты. Не придумывай того, чего в фактах нет.

Скажи человеку, как прошёл дубль:
— 1–2 короткие фразы, на «ты», по-человечески, как добрый коллега, без канцелярита и без оценочной драмы;
— БЕЗ ЦИФР и единиц измерения (ни «200 мс», ни «3 дБ»): только словами («чуть позже», «заметно тише»);
— один конкретный совет, что сделать в следующем дубле;
— если всё хорошо (поле problems пустое) — коротко и тепло похвали, без советов;
— если lips_checked=false, попадание в губы НЕ проверялось (source="file" — дубль загружен файлом, к видео не \
привязан; source="mic" — подгонка не уверена): не говори «попал в губы», «попадает», «в синхрон»; можно посоветовать \
проверить губы на превью.
Верни только текст фразы, без кавычек и заголовков.

Примеры.
Факты: late="late", problems=["late"] → «Ты слегка отстаёшь от губ — начни фразу чуть раньше, на вдохе.»
Факты: problems=["clipping"] → «Звук перегружен, пик пошёл в красное. Отодвинься от микрофона или говори спокойнее.»
Факты: problems=[], lips_checked=true → «Отличный дубль, всё чётко попадает. Можно брать.»
Факты: source="file", lips_checked=false, problems=["too_quiet"] → «Запись в файле тихая — я подниму громкость, \
но лучше перезаписать погромче. Губы проверь на превью.»"""


def feedback_prompt(report: dict) -> tuple[str, str]:
    return FEEDBACK_SYSTEM, "Факты о дубле:\n" + json.dumps(report, ensure_ascii=False)


# ─────────────────────────── помощник-режиссёр: команды ───────────────────────────
ROOM_HINTS: dict[str, str] = {
    "dry": "без помещения, сухо", "room": "обычная комната", "kitchen": "кухня", "hall": "большой зал, эхо",
    "street": "улица", "cave": "пещера, гулкое эхо", "car": "салон машины", "bathroom": "ванная, кафель",
    "phone": "голос из телефона", "behind_door": "за закрытой дверью, приглушённо",
}
FX_HINTS: dict[str, str] = {
    "none": "без эффекта", "robot": "робот", "chipmunk": "писклявый, как бурундук", "giant": "великан, низкий голос",
    "villain": "злодей, зловещий", "radio": "радио, рация", "phone": "голос из телефона", "ghost": "призрак",
}

MIX_FIELDS = ["gain_db", "room", "voice_fx", "mute", "duck_background", "target_lufs", "mode"]

COMMAND_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "patches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "field": {"type": "string", "enum": MIX_FIELDS},
                    "value": {"type": ["number", "string", "boolean"]},
                    "relative": {"type": "boolean"},
                },
                "required": ["target", "field", "value"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["patches"],
    "additionalProperties": False,
}

COMMAND_SYSTEM = """Ты — помощник-режиссёр. Человек просит словами поменять сведение сцены, ты превращаешь просьбу \
в список изменений настроек. Ты не слышишь звук и ничего не выдумываешь.

Цели (target):
{targets}

Поля (field) и допустимые значения:
— gain_db: громкость в дБ. Для target = background или id персонажа. Число. Если просьба относительная \
(«громче», «тише», «чуть тише») — ставь relative=true и value = изменение: «громче» +3, «чуть громче» +1.5, \
«намного громче» +6, «тише» −3, «чуть тише» −1.5, «намного тише» −6. Если абсолютная («убери на минус 10») — relative=false.
— mute: true/false — выключить/включить дорожку (target = background или персонаж). «убери фон совсем» = mute true.
— room: помещение голоса, только для персонажа. Значения: {rooms}.
— voice_fx: эффект голоса, только для персонажа. Значения: {fx}.
— duck_background: true/false — приглушать ли фон под речью (target = master).
— target_lufs: общая громкость итога в LUFS, число от −30 до −6 (target = master). «погромче в целом» = −12, «потише в целом» = −18, по умолчанию −14.
— mode: target = master, значения dub (голоса заменяются) или voiceover (оригинал тихо под новым голосом).

Правила:
1. Используй ТОЛЬКО перечисленные цели, поля и значения. Нет подходящего — верни пустой список.
2. «Все», «все голоса» — отдельный patch на каждого персонажа из списка.
3. Если персонаж назван именем или меткой — подставь его id. Не знаешь, о ком речь — не угадывай.
4. Текущие значения даны ниже — используй их, когда нужно понять контекст («фон и так тихий»).

Примеры.
{examples}
«спой мне песню» → {{"patches": []}}"""


def command_examples(chars: list[tuple[str, str]]) -> str:
    """Примеры для подсказки модели из персонажей самой сцены: [(id, имя)]. Нет имён — примеры без персонажей."""
    def patches(*items: dict) -> str:
        return json.dumps({"patches": list(items)}, ensure_ascii=False)

    rows: list[str] = []
    if chars:
        sid, name = chars[0]
        rows.append(f"«{name} громче» ({name} = {sid}) → "
                    + patches({"target": sid, "field": "gain_db", "value": 3, "relative": True}))
        rows.append(f"«сделай голос {name} как у робота» ({name} = {sid}) → "
                    + patches({"target": sid, "field": "voice_fx", "value": "robot"}))
        rows.append("«как в пещере» → " + patches(*({"target": c[0], "field": "room", "value": "cave"} for c in chars[:2])))
    rows.append("«музыку потише» → " + patches({"target": "background", "field": "gain_db", "value": -3, "relative": True}))
    return "\n".join(rows)


def command_prompt(text: str, *, targets: str, rooms: str, fx: str, state: str,
                   chars: list[tuple[str, str]] | None = None) -> tuple[str, str]:
    system = COMMAND_SYSTEM.format(targets=targets, rooms=rooms, fx=fx, examples=command_examples(chars or []))
    user = f"Текущие настройки:\n{state}\n\nПросьба: «{text}»"
    return system, user
