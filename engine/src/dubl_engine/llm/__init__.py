"""Слой ИИ «Дубль»: провайдеры моделей, ключи, маршрутизация, задачи и помощник-режиссёр.

Подмодули импортируются по требованию (httpx/keyring — только при использовании).
"""
from __future__ import annotations

from .director import (
                       MixPatch,
                       apply_patches,
                       director_command,
                       director_feedback,
                       rule_based_feedback,
                       take_report,
)
from .providers import LLMProvider, list_models, make_provider
from .router import default_routing, provider_for
from .tasks import estimate_duration, fit_line, name_speakers, translate_lines

__all__ = [
    "LLMProvider", "MixPatch", "apply_patches", "default_routing", "director_command", "director_feedback",
    "estimate_duration", "fit_line", "list_models", "make_provider", "name_speakers", "provider_for",
    "rule_based_feedback", "take_report", "translate_lines",
]
