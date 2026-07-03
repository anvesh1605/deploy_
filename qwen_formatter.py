from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

from lstm_lookup import normalize_whitespace


SYSTEM_PROMPT = (
    "You are only a response formatter for HPE Aruba AOS-CX release-note QA.\n\n"
    "You must not add new facts.\n"
    "You must not remove important facts.\n"
    "You must not change Bug IDs, Event IDs, switch models, AOS-CX versions, sub-versions, commands, category names, caveats, symptoms, scenarios, or workaround text.\n"
    "You must not invent causes, fixes, workarounds, commands, versions, links, or explanations.\n"
    "Use only the retrieved answer provided.\n"
    'If the retrieved answer says "No workaround is documented in the release notes", preserve that meaning exactly.\n'
    "Return a concise conversational answer for the user."
)

USER_PROMPT_TEMPLATE = """Question:
{question}

Predicted intent:
{predicted_intent}

Slots:
{slots_json}

Retrieved answer:
{answer}

Task:
Rewrite the retrieved answer into a short, clear, user-friendly response.

Rules:
1. Use only the retrieved answer.
2. Do not add any technical detail.
3. Do not change numbers, IDs, versions, or commands.
4. Do not change the meaning.
5. Keep the answer concise.
"""

DETERMINISTIC_MESSAGES = {
    "not_found": "No matching answer was found in the current release-note dataset.",
    "needs_disambiguation": "Multiple possible answers were found. Please provide more detail such as feature, bug ID, version, or sub-version.",
    "slot_missing": "I need more detail to answer this, such as the bug ID, feature, version, or sub-version.",
    "low_similarity": "No matching answer was found in the current release-note dataset.",
    "error": "No matching answer was found in the current release-note dataset.",
}

EXACT_ANSWER_INTENTS = {
    "bug_category",
    "version_date",
    "release_date",
    "event_id",
    "cli_syntax",
    "show_command_syntax",
}

FORMATTER_ELIGIBLE_INTENTS = {
    "bug_symptom",
    "bug_scenario",
    "bug_workaround",
    "release_caveat",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "please",
    "provide",
    "such",
    "that",
    "the",
    "to",
    "this",
    "was",
    "were",
    "with",
    "you",
}

COMMAND_PATTERNS = [
    r"\bshow\s+[a-z0-9_-]+(?:\s+[a-z0-9_-]+)*\b",
    r"\bconfigure\s+terminal\b",
    r"\bcopy\s+running-config(?:\s+[a-z0-9_-]+)*\b",
    r"\bwrite\s+memory\b",
    r"\breload\b",
    r"\bclear\s+[a-z0-9_-]+(?:\s+[a-z0-9_-]+)*\b",
    r"\bdelete\s+[a-z0-9_-]+(?:\s+[a-z0-9_-]+)*\b",
    r"\binterface\s+[a-z0-9/._:-]+(?:\s+[a-z0-9/._:-]+)*\b",
    r"\brouter\s+[a-z0-9_-]+(?:\s+[a-z0-9_-]+)*\b",
    r"\bvlan\s+[a-z0-9_-]+(?:\s+[a-z0-9_-]+)*\b",
    r"\bno\s+shutdown\b",
    r"\bshutdown\b",
]


def normalize_text(text: object) -> str:
    return normalize_whitespace(text)


def tokenize(text: object) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", normalize_text(text).lower())


def content_tokens(text: object) -> List[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def extract_bug_ids(text: object) -> List[str]:
    return re.findall(r"\b\d{4,7}\b", normalize_text(text))


def extract_version_strings(text: object) -> List[str]:
    versions = re.findall(r"\b\d+\.\d+(?:\.\d+)?\b", normalize_text(text))
    normalized = []
    for version in versions:
        if version not in normalized:
            normalized.append(version)
    return normalized


def extract_command_phrases(text: object) -> List[str]:
    source = normalize_text(text)
    phrases: List[str] = []
    for pattern in COMMAND_PATTERNS:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            phrase = normalize_whitespace(match.group(0))
            if phrase not in phrases:
                phrases.append(phrase)
    return phrases


def extract_switch_models(text: object) -> List[str]:
    source = normalize_text(text)
    models = re.findall(r"\b\d{4}[A-Za-z]?\b", source)
    unique: List[str] = []
    for model in models:
        if model not in unique:
            unique.append(model)
    return unique


def version_to_dotted(version: str, sub_version: str) -> str:
    if version and sub_version:
        return f"{version.replace('_', '.')}.{sub_version}"
    return version.replace("_", ".") if version else ""


def build_user_prompt(question: str, predicted_intent: str, slots: Dict[str, str], answer: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        question=normalize_text(question),
        predicted_intent=normalize_text(predicted_intent),
        slots_json=json.dumps(slots, ensure_ascii=False, sort_keys=True),
        answer=normalize_text(answer),
    )


def is_no_workaround_answer(answer: str) -> bool:
    text = normalize_text(answer).lower()
    return "no workaround is documented" in text or text == "no workaround is documented in the release notes."


def should_use_qwen_formatter(
    predicted_intent: str,
    lookup_answer: Optional[str],
    status: str,
    confidence: float,
    min_confidence: float = 0.75,
) -> Tuple[bool, str]:
    if status != "found":
        return False, f"status={status}"
    if not lookup_answer or not normalize_text(lookup_answer):
        return False, "missing lookup answer"
    if confidence < min_confidence:
        return False, "confidence below threshold"
    if predicted_intent in EXACT_ANSWER_INTENTS:
        return False, "exact-answer intent"
    if predicted_intent not in FORMATTER_ELIGIBLE_INTENTS:
        return False, "intent not eligible"
    if is_no_workaround_answer(lookup_answer):
        return False, "no-workaround answer"
    if len(normalize_text(lookup_answer).split()) < 8:
        return False, "short answer"
    return True, ""


def ollama_chat(model_name: str, system_prompt: str, user_prompt: str, timeout: int = 120) -> str:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "num_predict": 120,
            "repeat_penalty": 1.05,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "message" in payload and isinstance(payload["message"], dict):
        return normalize_text(payload["message"].get("content", ""))
    return normalize_text(payload.get("response", ""))


def call_formatter_backend(
    backend: str,
    model_name: str,
    question: str,
    predicted_intent: str,
    slots: Dict[str, str],
    answer: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if backend.lower() != "ollama":
        return None, None, "unsupported backend"

    user_prompt = build_user_prompt(question, predicted_intent, slots, answer)
    models_to_try: List[str] = []
    normalized_model = normalize_whitespace(model_name)

    def add_model(candidate: str) -> None:
        candidate = normalize_whitespace(candidate)
        if candidate and candidate not in models_to_try:
            models_to_try.append(candidate)

    def add_family_variants(prefix: str) -> None:
        if normalized_model.startswith(f"{prefix}-"):
            suffix = normalized_model[len(prefix) + 1 :]
            add_model(f"{prefix}:{suffix}")
        if normalized_model.startswith(f"{prefix}:"):
            suffix = normalized_model[len(prefix) + 1 :]
            add_model(f"{prefix}-{suffix}")

    add_model(normalized_model)
    add_family_variants("qwen2.5")
    if normalized_model.startswith("qwen2.5:3b") or normalized_model.startswith("qwen2.5-3b"):
        add_model("qwen2.5:3b-instruct")
        add_model("qwen2.5:3b")
    if normalized_model.startswith("qwen2.5:7b") or normalized_model.startswith("qwen2.5-7b"):
        add_model("qwen2.5:7b-instruct")
        add_model("qwen2.5:7b")

    last_error: Optional[str] = None
    for candidate_model in models_to_try:
        try:
            formatted = ollama_chat(candidate_model, SYSTEM_PROMPT, user_prompt)
            if formatted:
                return formatted, candidate_model, None
            last_error = "empty formatter response"
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else str(exc)
            last_error = error_text or str(exc)
            if "model not found" in error_text.lower() or "not found" in error_text.lower():
                continue
        except Exception as exc:  # pragma: no cover - backend failures are runtime/environment specific
            last_error = str(exc)
    return None, None, last_error


def original_has_no_workaround(answer: str) -> bool:
    text = normalize_text(answer).lower()
    return "no workaround is documented" in text or "no workaround" in text


def validate_formatter_output(
    question: str,
    predicted_intent: str,
    slots: Dict[str, str],
    original_answer: str,
    formatted_answer: str,
) -> Tuple[bool, str]:
    original = normalize_text(original_answer)
    formatted = normalize_text(formatted_answer)
    if not formatted:
        return False, "empty formatter output"
    if len(formatted.split()) > max(60, len(original.split()) * 2 + 12):
        return False, "answer too long"
    if len(formatted.split()) < max(1, int(len(original.split()) * 0.7)):
        return False, "too much shortening"

    original_tokens = content_tokens(original)
    formatted_tokens = content_tokens(formatted)
    if original_tokens:
        overlap = sum(1 for token in original_tokens if token in formatted_tokens)
        recall = overlap / max(1, len(original_tokens))
        similarity = SequenceMatcher(None, normalize_text(original).lower(), normalize_text(formatted).lower()).ratio()
        if recall < 0.70 and similarity < 0.45:
            return False, "meaning changed"

    allowed_bug_ids = set(extract_bug_ids(original))
    if slots.get("bug_id"):
        allowed_bug_ids.add(slots["bug_id"])
    output_bug_ids = set(extract_bug_ids(formatted))
    if output_bug_ids and not output_bug_ids.issubset(allowed_bug_ids):
        return False, "bug id changed"

    allowed_versions = set(extract_version_strings(original))
    slot_version = version_to_dotted(slots.get("version", ""), slots.get("sub_version", ""))
    if slot_version:
        allowed_versions.add(slot_version)
    output_versions = set(extract_version_strings(formatted))
    if output_versions and not output_versions.issubset(allowed_versions):
        return False, "version changed"

    allowed_models = set(extract_switch_models(original))
    if slots.get("switch"):
        allowed_models.add(slots["switch"])
    output_models = set(extract_switch_models(formatted))
    if output_models and not output_models.issubset(allowed_models):
        return False, "switch model changed"

    original_commands = set(extract_command_phrases(original))
    slot_command = normalize_text(slots.get("command", ""))
    if slot_command:
        original_commands.add(slot_command)
    output_commands = set(extract_command_phrases(formatted))
    if output_commands and not output_commands.issubset(original_commands):
        return False, "command changed"

    if original_has_no_workaround(original):
        if "no workaround" not in formatted.lower():
            return False, "invented workaround"
    elif "no workaround" in formatted.lower() or "not documented" in formatted.lower():
        return False, "invented no-workaround meaning"

    return True, None


def deterministic_message_for_status(status: str) -> str:
    return DETERMINISTIC_MESSAGES.get(status, DETERMINISTIC_MESSAGES["error"])


def format_lookup_answer(
    question: str,
    predicted_intent: str,
    slots: Dict[str, str],
    lookup_answer: Optional[str],
    status: str,
    confidence: float,
    backend: str,
    model_name: str,
    min_confidence: float = 0.75,
) -> Dict[str, object]:
    should_use_qwen, skip_reason = should_use_qwen_formatter(
        predicted_intent,
        lookup_answer,
        status,
        confidence,
        min_confidence=min_confidence,
    )
    if not should_use_qwen:
        if status != "found" or not lookup_answer or not normalize_text(lookup_answer):
            final_answer = deterministic_message_for_status(status)
        else:
            final_answer = normalize_text(lookup_answer)
        return {
            "qwen_used": False,
            "qwen_answer": None,
            "qwen_validation_passed": False,
            "final_answer": final_answer,
            "qwen_model_used": None,
            "reason": skip_reason,
        }

    formatted, model_used, error = call_formatter_backend(backend, model_name, question, predicted_intent, slots, lookup_answer)
    if not formatted:
        return {
            "qwen_used": True,
            "qwen_answer": None,
            "qwen_validation_passed": False,
            "final_answer": normalize_text(lookup_answer),
            "qwen_model_used": model_used,
            "reason": error or "formatter unavailable",
        }

    valid, reason = validate_formatter_output(question, predicted_intent, slots, lookup_answer, formatted)
    if not valid:
        return {
            "qwen_used": True,
            "qwen_answer": formatted,
            "qwen_validation_passed": False,
            "final_answer": normalize_text(lookup_answer),
            "qwen_model_used": model_used,
            "reason": reason,
        }

    return {
        "qwen_used": True,
        "qwen_answer": formatted,
        "qwen_validation_passed": True,
        "final_answer": formatted,
        "qwen_model_used": model_used,
        "reason": None,
    }
