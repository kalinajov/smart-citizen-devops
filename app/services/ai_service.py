"""
AI service — HuggingFace zero-shot classification.

Uses facebook/bart-large-mnli to classify report text into
one of the candidate labels passed in (category names from DB).

The HuggingFace pipeline is lazy-loaded to avoid slow imports and
surprise model downloads during test runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import threading
from time import monotonic, perf_counter
from typing import Literal

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_classifier_lock = threading.Lock()
_classifier = None
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

_classification_cache_lock = threading.Lock()
_classification_cache: dict[tuple[str, tuple[str, ...], float | None], "_ClassificationCacheEntry"] = {}


@dataclass(frozen=True)
class _ClassificationCacheEntry:
    label: str
    expires_at_monotonic: float


def _normalize_text_cache_key(text: str) -> str:
    # Keep key reasonably stable across superficial user input differences.
    compacted = re.sub(r"\s+", " ", text).strip()
    return compacted.casefold()


def _classification_cache_get(
    *,
    key: tuple[str, tuple[str, ...], float | None],
    now_monotonic: float,
) -> str | None:
    with _classification_cache_lock:
        entry = _classification_cache.get(key)
        if entry is None:
            return None
        if entry.expires_at_monotonic <= now_monotonic:
            _classification_cache.pop(key, None)
            return None
        return entry.label


def _classification_cache_set(
    *,
    key: tuple[str, tuple[str, ...], float | None],
    label: str,
    ttl_seconds: int,
    now_monotonic: float,
) -> None:
    if ttl_seconds <= 0:
        return
    expires_at = now_monotonic + float(ttl_seconds)
    with _classification_cache_lock:
        _classification_cache[key] = _ClassificationCacheEntry(label=label, expires_at_monotonic=expires_at)


def _get_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier

    with _classifier_lock:
        if _classifier is None:
            # Imported here, not at module scope: transformers pulls in torch,
            # which adds several GB to the image. Deployments that run with
            # AI_ENABLED=false install the slim requirements.txt and never
            # reach this line, so the dependency stays optional.
            try:
                from transformers import pipeline
            except ImportError as exc:  # pragma: no cover - depends on install extras
                raise RuntimeError(
                    "AI_ENABLED is true but 'transformers' is not installed. "
                    "Install the optional extras with: pip install -r requirements-ai.txt"
                ) from exc

            settings = get_settings()
            # Downloads ~1.6 GB on first run, then cached locally by HuggingFace.
            logger.info(
                "Loading HuggingFace classifier model=%s revision=%s device=%s",
                settings.ai_hf_model,
                settings.ai_hf_revision,
                settings.ai_hf_device,
            )
            pipeline_kwargs: dict[str, object] = {
                "model": settings.ai_hf_model,
                "device": settings.ai_hf_device,
            }
            if settings.ai_hf_revision:
                pipeline_kwargs["revision"] = settings.ai_hf_revision

            _classifier = pipeline("zero-shot-classification", **pipeline_kwargs)
    return _classifier


def warmup_model() -> bool:
    """Eagerly load the model (optional). Returns True on success."""
    settings = get_settings()
    if not settings.ai_enabled:
        return False

    try:
        _get_classifier()
        return True
    except Exception:
        logger.exception("AI warmup failed.")
        return False


def _classify_text_openai(text: str, candidate_labels: list[str]) -> str | None:
    """
    OpenAI fallback classifier.

    Requires `OPENAI_API_KEY` to be available in the environment.
    """
    settings = get_settings()
    if not settings.ai_openai_fallback_enabled:
        return None
    if not settings.openai_api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.ai_openai_timeout_seconds)
        response = client.responses.create(
            model=settings.ai_openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You classify citizen reports into exactly one category label. "
                        "Return only a JSON object that matches the schema."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Report description:\n{text}\n\n"
                        "Choose the single best label from:\n"
                        + "\n".join(f"- {label}" for label in candidate_labels)
                    ),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "report_category",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "enum": candidate_labels},
                        },
                        "required": ["label"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        )

        parsed = json.loads(response.output_text)
        label = parsed.get("label")
        if isinstance(label, str) and label in candidate_labels:
            return label

        logger.warning("OpenAI fallback returned unexpected label=%r", label)
        return None
    except Exception:
        logger.exception("OpenAI fallback classification failed.")
        return None


def _should_try_openai_classification(
    *,
    text: str,
    top_label: str | None,
    top_score: float | None,
    min_confidence: float | None,
) -> bool:
    """
    Decide when the local zero-shot result is weak enough to ask OpenAI instead.

    The local BART zero-shot model is English-centric, so Macedonian/Cyrillic
    reports often need a stronger fallback path even when the call "succeeds".
    """
    settings = get_settings()
    if not settings.ai_openai_fallback_enabled or not settings.openai_api_key:
        return False

    if top_label is None:
        return True

    effective_threshold = min_confidence
    has_cyrillic = _CYRILLIC_RE.search(text) is not None
    if has_cyrillic and (effective_threshold is None or effective_threshold <= 0.0):
        effective_threshold = 0.55

    if effective_threshold is not None and top_score is not None and top_score < effective_threshold:
        return True

    if has_cyrillic and top_label.strip().casefold() == "other":
        return True

    return False


def classify_text(
    text: str,
    candidate_labels: list[str],
    *,
    min_confidence: float | None = None,
) -> str | None:
    """
    Classify free-form report text into one of the candidate category names.

    Args:
        text:             The report description submitted by the user.
        candidate_labels: Category names fetched live from the DB.
        min_confidence:   If set, return None when the top score is below this threshold.

    Returns:
        The top predicted category name, or None if classification fails.
    """
    start = perf_counter()
    cache_hit = False
    if not text or not candidate_labels:
        logger.warning("classify_text called with empty text or no candidate labels.")
        return None

    settings = get_settings()
    if not settings.ai_enabled:
        return None

    ttl_seconds = int(getattr(settings, "ai_cache_ttl_seconds", 0) or 0)
    cache_key = (
        _normalize_text_cache_key(text),
        tuple(sorted(candidate_labels, key=str.casefold)),
        float(min_confidence) if min_confidence is not None else None,
    )
    if ttl_seconds > 0:
        cached = _classification_cache_get(key=cache_key, now_monotonic=monotonic())
        if cached is not None:
            cache_hit = True
            elapsed_ms = (perf_counter() - start) * 1000
            logger.info("AI classification latency: %.0fms (cache hit)", elapsed_ms)
            return cached

    top_label = None
    top_score: float | None = None
    try:
        classifier = _get_classifier()
        result = classifier(text, candidate_labels)
        # result = {"labels": ["Roads", "Waste", ...], "scores": [0.91, 0.05, ...], ...}
        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
        logger.info("Classified text as '%s' (confidence %.2f)", top_label, top_score)
        if min_confidence is not None and top_score < min_confidence:
            logger.info(
                "Classification below min_confidence (%.2f < %.2f) - trying OpenAI fallback.",
                top_score,
                min_confidence,
            )
            top_label = None
    except Exception:
        logger.warning(
            "AI unavailable — skipping HuggingFace classification (trying OpenAI fallback).",
            exc_info=True,
        )
        top_label = None

    if _should_try_openai_classification(
        text=text,
        top_label=top_label,
        top_score=top_score,
        min_confidence=min_confidence,
    ):
        label = _classify_text_openai(text, candidate_labels)
        if isinstance(label, str) and label:
            if ttl_seconds > 0 and not cache_hit:
                _classification_cache_set(
                    key=cache_key,
                    label=label,
                    ttl_seconds=ttl_seconds,
                    now_monotonic=monotonic(),
                )
            elapsed_ms = (perf_counter() - start) * 1000
            logger.info("AI classification latency: %.0fms (OpenAI fallback)", elapsed_ms)
            return label
        top_label = None

    if top_label is not None:
        if ttl_seconds > 0 and not cache_hit:
            _classification_cache_set(
                key=cache_key,
                label=top_label,
                ttl_seconds=ttl_seconds,
                now_monotonic=monotonic(),
            )
        elapsed_ms = (perf_counter() - start) * 1000
        logger.info("AI classification latency: %.0fms", elapsed_ms)
        return top_label

    label = _classify_text_openai(text, candidate_labels)
    if isinstance(label, str) and label and ttl_seconds > 0 and not cache_hit:
        _classification_cache_set(
            key=cache_key,
            label=label,
            ttl_seconds=ttl_seconds,
            now_monotonic=monotonic(),
        )

    elapsed_ms = (perf_counter() - start) * 1000
    logger.info("AI classification latency: %.0fms", elapsed_ms)
    return label


PriorityLevel = Literal["Низок", "Среден", "Висок", "Итен"]

_HIGH_URGENCY_KEYWORDS = {
    "пожар",
    "оган",
    "гори",
    "се шири пожар",
    "критично",
    "ризик по живот",
    "повреда",
    "итна интервенција",
    "fire",
    "explosion",
    "urgent",
    "emergency",
    "critical",
    "life-threatening",
    "danger",
    "immediate risk",
    "поплава",
    "flood",
    "експлозија",
    "итно",
    "опасно",
    "загрозено",
}
_MEDIUM_URGENCY_KEYWORDS = {
    "опасност",
    "hazard",
    "струја",
    "ризик",
    "кабел",
    "оштетено",
    "може да падне",
    "небезбедно",
    "протекување",
    "лизгаво",
    "нестабилно",
    "risk",
    "unsafe",
    "damaged",
    "unstable",
    "leaking",
    "exposed wire",
    "dangerous",
}
_MILD_ISSUE_KEYWORDS = {
    "оштетен",
    "расипан",
    "не работи",
    "дефект",
    "дупка",
    "проблем",
    "валкано",
    "блокирано",
    "broken",
    "not working",
    "defect",
    "blocked",
    "dirty",
    "damaged",
    "оштетено",
    "незначителен проблем",
}
_LOW_ISSUE_KEYWORDS = {
    "мал проблем",
    "незначително",
    "козметички",
    "благо оштетување",
    "minor",
    "small issue",
    "cosmetic",
    "slight damage",
}
_PRIORITY_ORDER: tuple[PriorityLevel, ...] = ("Низок", "Среден", "Висок", "Итен")


def _normalize_priority_text(text: str) -> str:
    """Normalize free-form text for fast keyword and similarity checks."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def _contains_keyword(text: str, keywords: set[str]) -> bool:
    """Return True when any keyword appears as a substring in normalized text."""
    if not text:
        return False
    return any(keyword in text for keyword in keywords)


def _tokenize_priority_text(text: str) -> set[str]:
    """Tokenize normalized text into simple alphanumeric words."""
    return set(re.findall(r"\w+", text, flags=re.UNICODE))


def _base_priority_from_text(normalized_text: str) -> PriorityLevel:
    """Compute the initial priority from direct keyword evidence in text."""
    if _contains_keyword(normalized_text, _HIGH_URGENCY_KEYWORDS):
        return "Итен"
    if _contains_keyword(normalized_text, _MEDIUM_URGENCY_KEYWORDS):
        return "Висок"
    if _contains_keyword(normalized_text, _LOW_ISSUE_KEYWORDS):
        return "Низок"
    if _contains_keyword(normalized_text, _MILD_ISSUE_KEYWORDS):
        return "Среден"
    return "Низок"


def _history_should_boost(text: str, history: list[str]) -> bool:
    """Check whether similar historical reports suggest increasing one priority level."""
    normalized_text = _normalize_priority_text(text)
    if not normalized_text:
        return False

    base_tokens = _tokenize_priority_text(normalized_text)
    if not base_tokens:
        return False

    risk_keywords = _HIGH_URGENCY_KEYWORDS | _MEDIUM_URGENCY_KEYWORDS
    for entry in history:
        normalized_entry = _normalize_priority_text(entry)
        if not normalized_entry:
            continue
        if not _contains_keyword(normalized_entry, risk_keywords):
            continue

        if normalized_text in normalized_entry or normalized_entry in normalized_text:
            return True

        overlap = len(base_tokens & _tokenize_priority_text(normalized_entry))
        if overlap >= 2:
            return True

    return False


def _boost_priority_once(priority: PriorityLevel) -> PriorityLevel:
    """Increase priority by one level, capped at the highest level."""
    try:
        index = _PRIORITY_ORDER.index(priority)
    except ValueError:
        return priority
    if index >= len(_PRIORITY_ORDER) - 1:
        return priority
    return _PRIORITY_ORDER[index + 1]


def assign_priority(text: str, history: list[str]) -> PriorityLevel:
    """Assign report priority using keyword scoring and lightweight historical boost.

    Rules:
    - High-urgency keywords -> Итен
    - Medium-urgency keywords -> Висок
    - Mild issue words -> Среден
    - Otherwise -> Низок
    - If similar historical reports contain risk keywords, boost by one level
    """
    normalized_text = _normalize_priority_text(text)
    priority = _base_priority_from_text(normalized_text)
    if _history_should_boost(text, history):
        priority = _boost_priority_once(priority)
    return priority


def generate_confirmation_message(
    description: str,
    *,
    category_label: str | None = None,
    possible_duplicate_of: int | None = None,
) -> str | None:
    """
    Generate a short confirmation message for a newly created report.

    Uses OpenAI when configured; otherwise returns a deterministic template.
    """
    start = perf_counter()
    settings = get_settings()

    template_parts: list[str] = ["Thanks for your report — we’ve received it and will review it shortly."]
    if category_label:
        template_parts.append(f"Initial category: {category_label}.")
    if possible_duplicate_of is not None:
        template_parts.append(f"This may be a duplicate of report #{possible_duplicate_of}; our team will review.")
    template_message = " ".join(template_parts)

    if not settings.ai_enabled:
        elapsed_ms = (perf_counter() - start) * 1000
        logger.info("AI confirmation generation latency: %.0fms (AI disabled)", elapsed_ms)
        return template_message

    if not settings.openai_api_key or not settings.ai_openai_fallback_enabled:
        elapsed_ms = (perf_counter() - start) * 1000
        logger.info("AI confirmation generation latency: %.0fms (template)", elapsed_ms)
        return template_message

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.ai_openai_timeout_seconds)
        user_bits = [f"Description:\n{description.strip()}"]
        if category_label:
            user_bits.append(f"Category: {category_label}")
        if possible_duplicate_of is not None:
            user_bits.append(f"Possible duplicate of report id: {possible_duplicate_of}")

        response = client.responses.create(
            model=settings.ai_openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You write short confirmation messages for a city report app. "
                        "Be concise, friendly, and specific. Do not ask questions."
                    ),
                },
                {"role": "user", "content": "\n".join(user_bits)},
            ],
        )
        message = (response.output_text or "").strip()
        if not message:
            message = template_message

        elapsed_ms = (perf_counter() - start) * 1000
        logger.info("AI confirmation generation latency: %.0fms", elapsed_ms)
        return message
    except Exception:
        logger.warning("AI confirmation generation failed — using template.", exc_info=True)
        elapsed_ms = (perf_counter() - start) * 1000
        logger.info("AI confirmation generation latency: %.0fms (template fallback)", elapsed_ms)
        return template_message


# ---------------------------------------------------------------------------
# FR-03 / CR-02 — Macedonian confirmation via HuggingFace Inference API

import requests as _requests

# Maps English priority strings (free-text from DB) to Macedonian.
_PRIORITY_MK: dict[str, str] = {
    "low":      "низок",
    "medium":   "среден",
    "normal":   "среден",
    "high":     "висок",
    "urgent":   "итен",
    "critical": "критичен",
}


def _priority_to_mk(priority: str | None) -> str | None:
    """Translate a priority string to Macedonian. Returns None if unknown/missing."""
    if not priority:
        return None
    return _PRIORITY_MK.get(priority.strip().lower())


_MK_FALLBACK_TEMPLATE = (
    "Вашата пријава е успешно примена и ќе биде разгледана наскоро."
    "{category_part}"
    "{priority_part}"
    " Очекувајте одговор во рок од 3–5 работни дена."
    "{duplicate_part}"
    " Ви благодариме за придонесот кон подобрување на нашата заедница."
)


def _mk_fallback(
    category_label: str | None,
    priority: str | None,
    possible_duplicate_of: int | None,
) -> str:
    category_part = (
        f" Пријавата е класифицирана во категоријата: {category_label}."
        if category_label
        else ""
    )
    priority_mk = _priority_to_mk(priority)
    priority_part = (
        f" Приоритетот на пријавата е {priority_mk}."
        if priority_mk
        else (
            f" Приоритет: {priority}."
            if priority
            else ""
        )
    )
    duplicate_part = (
        f" Забележавме дека оваа пријава може да е слична на пријава #{possible_duplicate_of};"
        " нашиот тим ќе го разгледа тоа."
        if possible_duplicate_of is not None
        else ""
    )
    advice_part = (
        " Ве советуваме да ја документирате состојбата со фотографии и да останете достапни за контакт."
    )
    return (
        "Вашата пријава е успешно примена и ќе биде разгледана наскоро."
        + category_part
        + priority_part
        + " Очекувајте одговор во рок од 3–5 работни дена."
        + duplicate_part
        + advice_part
    )


def _build_mk_prompt(
    description: str,
    category_label: str | None,
    priority: str | None,
    possible_duplicate_of: int | None,
) -> str:
    """Build a Mistral-instruct-format prompt that requests a Macedonian reply."""
    priority_mk = _priority_to_mk(priority)
    priority_display = priority_mk or priority  # use MK if known, raw otherwise

    details: list[str] = [f"- Опис на пријавата: {description.strip()}"]
    if category_label:
        details.append(f"- Категорија: {category_label}")
    if priority_display:
        details.append(f"- Приоритет: {priority_display}")
    if possible_duplicate_of is not None:
        details.append(f"- Можна дупликат на пријава бр.: {possible_duplicate_of}")

    instruction = (
        "Ти си асистент на систем за управување со граѓански поплаки во македонски град. "
        "Генерирај кратка потврдна порака НА МАКЕДОНСКИ ЈАЗИК за граѓанин кој поднел нова пријава.\n\n"
        "Детали за пријавата:\n"
        + "\n".join(details)
        + "\n\n"
        "Пораката МОРА да содржи (во 3–4 реченици):\n"
        "1. Потврда дека пријавата е примена\n"
        "2. Класификацијата и приоритетот на проблемот (ако се дадени) — "
        "формулирај го вака: 'Вашата пријава е класифицирана како [категорија] со [приоритет] приоритет'\n"
        "3. Очекуваниот тек / следни чекори\n"
        "4. Краток совет за граѓанинот\n\n"
        "Одговори САМО со пораката на македонски. Без воведни фрази, без објаснувања."
    )
    return f"<s>[INST] {instruction} [/INST]"


def _generate_confirmation_mk_openai(
    description: str,
    *,
    category_label: str | None,
    priority: str | None,
    possible_duplicate_of: int | None,
) -> str | None:
    """Use OpenAI as a secondary fallback for Macedonian confirmation text."""
    settings = get_settings()
    if not settings.ai_openai_fallback_enabled or not settings.openai_api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.ai_openai_timeout_seconds)
        priority_mk = _priority_to_mk(priority)
        user_bits = [f"Опис на пријавата:\n{description.strip()}"]
        if category_label:
            user_bits.append(f"Категорија: {category_label}")
        if priority_mk or priority:
            user_bits.append(f"Приоритет: {priority_mk or priority}")
        if possible_duplicate_of is not None:
            user_bits.append(f"Можен дупликат на пријава бр.: {possible_duplicate_of}")

        response = client.responses.create(
            model=settings.ai_openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Напиши кратка потврдна порака на македонски јазик за градска пријава. "
                        "Биди јасен, конкретен и љубезен. Не поставувај прашања."
                    ),
                },
                {"role": "user", "content": "\n".join(user_bits)},
            ],
        )
        message = (response.output_text or "").strip()
        return message or None
    except Exception:
        logger.exception("FR-03: OpenAI fallback for Macedonian confirmation failed.")
        return None


def generate_confirmation_mk(
    description: str,
    *,
    category_label: str | None = None,
    priority: str | None = None,
    possible_duplicate_of: int | None = None,
) -> str:
    """
    FR-03 / CR-02: Generate a short AI confirmation message in Macedonian,
    including category, priority, expected process and citizen advice.

    Strategy:
      1. Call HuggingFace Inference API (free) with Mistral-7B-Instruct.
      2. On any failure → graceful fallback to deterministic MK template.
      Never raises.
    """
    start = perf_counter()
    settings = get_settings()

    if not settings.ai_enabled:
        logger.info("FR-03: AI disabled — using MK fallback template.")
        return _mk_fallback(category_label, priority, possible_duplicate_of)

    openai_message = _generate_confirmation_mk_openai(
        description,
        category_label=category_label,
        priority=priority,
        possible_duplicate_of=possible_duplicate_of,
    )
    if openai_message:
        logger.info("FR-03: MK confirmation generated via OpenAI fallback.")
        return openai_message

    prompt = _build_mk_prompt(description, category_label, priority, possible_duplicate_of)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.hf_api_token:
        headers["Authorization"] = f"Bearer {settings.hf_api_token}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.6,
            "top_p": 0.92,
            "do_sample": True,
            "return_full_text": False,
            "stop": ["</s>", "[INST]"],
        },
    }

    url = f"https://api-inference.huggingface.co/models/{settings.ai_hf_inference_model}"

    try:
        resp = _requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=settings.ai_hf_inference_timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"].strip()
        elif isinstance(data, dict) and "generated_text" in data:
            text = data["generated_text"].strip()
        else:
            logger.warning("FR-03: Unexpected HF response shape: %r — using fallback.", data)
            text = ""

        if not text:
            raise ValueError("Empty generated text from HF Inference API")

        elapsed_ms = (perf_counter() - start) * 1000
        logger.info("FR-03: MK confirmation generated via HF in %.0fms", elapsed_ms)
        return text

    except Exception:
        elapsed_ms = (perf_counter() - start) * 1000
        logger.warning(
            "FR-03: HF Inference API failed (%.0fms) — using MK fallback template.",
            elapsed_ms,
            exc_info=True,
        )
        return _mk_fallback(category_label, priority, possible_duplicate_of)
