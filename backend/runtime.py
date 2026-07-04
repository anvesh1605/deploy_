from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import (  # noqa: E402
    BACKEND_CACHE_DIR,
    DATA_ROOT,
    MODEL_ROOT,
    PRODUCT_LOOKUP_DATA_PATHS,
    PRODUCT_DOCS_DATA_DIR,
    PRODUCT_LSTM_MODEL_PATH,
    PRODUCT_LSTM_DATA_DIR,
    OLLAMA_BASE_URL,
    QWEN_MODEL_PATH,
    RELEASE_AVAILABILITY_PATH,
    RELEASE_BUG_METADATA_PATH,
    RELEASE_LSTM_DATA_DIR,
    RELEASE_LSTM_MODEL_PATH,
    RELEASE_NOTES_DATA_DIR,
    RELEASE_LOOKUP_DATA_PATH,
    RELEASE_LOOKUP_INDEX_PATH,
)
from live_release_chat import answer_question as answer_release_question  # noqa: E402
from lstm_lookup import (  # noqa: E402
    DATA_NOT_AVAILABLE_RESPONSE,
    build_availability_index,
    build_bug_metadata_index,
    build_lookup_entries,
    build_lookup_index,
    check_data_availability,
    extract_slots_from_question,
    load_or_build_availability_index,
    load_or_build_bug_metadata_index,
    normalize_whitespace,
    read_jsonl,
    write_jsonl,
)
from release_notes_qwen_pipeline import (  # noqa: E402
    _extract_cli_syntax,
    clean_cli_syntax,
    format_cli_syntax_answer,
    generate_qwen_answer,
    is_command_purpose_question,
    is_cli_syntax_answer,
    is_bad_syntax_artifact,
    load_lstm_support,
    load_lookup_resources,
    load_qwen_model,
    predict_intent,
    validate_qwen_answer,
)


RELEASE_LIKE_INTENTS = {
    "bug_category",
    "bug_scenario",
    "bug_symptom",
    "bug_workaround",
    "release_caveat",
}

PRODUCT_EXACT_INTENTS = {
    "cli_syntax",
    "show_command_syntax",
    "show_command_usage",
    "event_id_meaning",
    "event_id_action",
    "version_date",
    "release_date",
    "out_of_domain",
    "data_not_available",
}

PRODUCT_DATANOT_AVAILABLE_RESPONSE = (
    "This particular data is not available in the current Aruba product documentation dataset."
)

PRODUCT_NOT_FOUND_RESPONSE = "I could not find a matching answer in the current Aruba product documentation dataset."
PRODUCT_NEEDS_DISAMBIGUATION_RESPONSE = (
    "Multiple possible answers were found. Please provide more detail such as feature, command, version, or sub-version."
)
PRODUCT_SLOT_MISSING_RESPONSE = "I need more detail to answer this, such as the command, topic, version, or sub-version."

PRODUCT_SYNTAX_VALIDATION_FALLBACK = (
    "I found a related entry, but the retrieved text looks like a table-of-contents or index artifact, "
    "so I cannot safely return it as exact syntax."
)

PRODUCT_FOLLOWUP_CONTEXT_MISSING_RESPONSE = (
    "I need more context. Please specify the topic you want me to explain."
)

PRODUCT_FOLLOWUP_WORDS = [
    "above bug",
    "that bug",
    "this bug",
    "same bug",
    "above issue",
    "that issue",
    "this issue",
    "above limitation",
    "that limitation",
    "this limitation",
    "explain those",
    "explain that",
    "explain this",
    "explain more",
    "tell me more",
    "what about that",
    "what about this",
    "what does that mean",
    "those two types",
    "elaborate",
    "explain",
]

PRODUCT_FILLER_PREFIXES = (
    "the documented answer is:",
    "according to the documentation:",
    "according to the guide:",
    "the guide says:",
    "the guide states:",
)

PRODUCT_GENERATED_LABEL_PREFIXES = (
    "concept explanation:",
    "product documentation:",
    "answer:",
    "response:",
    "final answer:",
)

PRODUCT_QWEN_STOPWORDS = {
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


def _clean(value: object) -> str:
    return normalize_whitespace(value)


def _lower(value: object) -> str:
    return _clean(value).lower()


def _canonical_product_switch(value: object) -> str:
    text = _clean(value)
    upper = text.upper()
    if upper.startswith("CX") and len(text) > 2 and text[2:].isdigit():
        return text[2:]
    return text


def _unique(values: Sequence[str]) -> List[str]:
    seen: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen


def _dominant_answer(answers: Sequence[str]) -> Tuple[str, int, int]:
    counts: Dict[str, int] = {}
    order: List[str] = []
    for answer in answers:
        cleaned = _clean(answer)
        if not cleaned:
            continue
        if cleaned not in counts:
            order.append(cleaned)
        counts[cleaned] = counts.get(cleaned, 0) + 1
    if not counts:
        return "", 0, 0
    ordered = sorted(counts.items(), key=lambda item: (item[1], -order.index(item[0]) if item[0] in order else 0), reverse=True)
    top_answer, top_count = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0
    return top_answer, top_count, runner_up


def _domain_version(version: str, domain: str) -> str:
    version = _clean(version)
    if not version:
        return ""
    if domain == "product":
        return version.replace("_", ".")
    return version


def _product_version_aliases(version: str, sub_version: str = "") -> List[str]:
    version = _clean(version).replace("_", ".")
    sub_version = _clean(sub_version)
    aliases: List[str] = []
    if version and sub_version:
        combined = version if version.endswith(f".{sub_version}") else f"{version}.{sub_version}"
        aliases.append(combined)
        base_parts = version.split(".")
        if len(base_parts) > 2:
            aliases.append(".".join(base_parts[:2]))
        elif len(base_parts) == 2:
            aliases.append(version)
    elif version:
        aliases.append(version)
        base_parts = version.split(".")
        if len(base_parts) > 2:
            aliases.append(".".join(base_parts[:2]))
    return _unique(aliases)


def _product_primary_slot(slots: Dict[str, str]) -> str:
    for key in ("command", "topic", "feature", "section", "category", "event_id", "question_type"):
        value = _clean(slots.get(key, ""))
        if value:
            return value
    return ""


def _product_command_from_question(question: str) -> str:
    text = _clean(question)
    patterns = [
        r"\b(?:could\s+you\s+help\s+me\s+)?find\s+how\s+to\s+configure\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\b(?:could\s+you\s+help\s+me\s+)?find\s+how\s+to\s+use\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\b(?:could\s+you\s+help\s+me\s+)?find\s+how\s+to\s+set\s+up\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\b(?:could\s+you\s+help\s+me\s+)?find\s+(?:the\s+)?syntax\s+for\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+to\s+configure\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+to\s+use\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+to\s+set\s+up\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+do\s+i\s+configure\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+do\s+i\s+use\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+do\s+you\s+configure\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+do\s+you\s+use\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+can\s+i\s+configure\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bhow\s+can\s+i\s+use\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bwhat\s+is\s+the\s+syntax\s+of\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bwhat\s+is\s+the\s+syntax\s+for\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bsyntax of (?:the )?(?P<command>.+?) command\b",
        r"\bwhat does (?:the )?(?P<command>.+?) command do\b",
        r"\bwhat is the syntax of (?:the )?(?P<command>.+?) command\b",
        r"\bwhat is the purpose of (?:the )?(?P<command>.+?) command\b",
        r"\bwhat is (?:the )?(?P<command>.+?) command\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            command = _clean(match.group("command"))
            command = command.strip(" ?.")
            return command
    return ""


def _product_sentence_chunks(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'\[])", _clean(text))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _product_structured_segments(text: str) -> List[str]:
    normalized = _cleanup_product_markdown(_clean(text))
    if not normalized:
        return []
    if "\n" in normalized:
        return [line.strip(" -*") for line in normalized.splitlines() if line.strip()]
    if normalized.count(":") < 3 and len(normalized) < 140:
        return []

    segments = [normalized]
    markers = [
        r"(?=\bPOL\d+:\s*)",
        r"(?=\bShowing\b)",
        r"(?=\bSyntax\b)",
        r"(?=\bDescription\b)",
        r"(?=\bExamples?\b)",
        r"(?=\bAttached Access List\b)",
        r"(?=\bAttached Prefix List\b)",
        r"(?=\bPreference Range\b)",
        r"(?=\bApplied on VLAN\b)",
        r"(?=\bApplied on Port\b)",
    ]
    for pattern in markers:
        next_segments: List[str] = []
        for segment in segments:
            if len(segment) < 24:
                next_segments.append(segment)
                continue
            parts = [part.strip(" :") for part in re.split(pattern, segment) if part and part.strip(" :")]
            if len(parts) > 1:
                next_segments.extend(parts)
            else:
                next_segments.append(segment)
        segments = next_segments

    collapsed: List[str] = []
    for segment in segments:
        cleaned = re.sub(r"\s+", " ", segment).strip()
        if cleaned:
            collapsed.append(cleaned)
    return collapsed


def _strip_product_filler_prefix(answer: str) -> str:
    text = _cleanup_product_markdown(answer)
    lower = normalize_whitespace(text).lower()
    for prefix in PRODUCT_FILLER_PREFIXES:
        if lower.startswith(prefix):
            return _cleanup_product_markdown(text[len(prefix) :])
    return text


def _strip_product_generated_label(answer: str) -> str:
    text = _cleanup_product_markdown(answer)
    lower = normalize_whitespace(text).lower()
    for prefix in PRODUCT_GENERATED_LABEL_PREFIXES:
        if lower.startswith(prefix):
            return _cleanup_product_markdown(text[len(prefix) :])
    return text


def _cleanup_product_markdown(answer: str) -> str:
    text = "" if answer is None else str(answer)
    if not text:
        return ""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned_lines: List[str] = []
    in_code_block = False
    blank_run = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            cleaned_lines.append(stripped)
            blank_run = 0
            continue
        if in_code_block:
            cleaned_lines.append(line.rstrip())
            continue
        if not stripped:
            blank_run += 1
            if blank_run <= 2:
                cleaned_lines.append("")
            continue
        blank_run = 0
        stripped = re.sub(r"(?<!`) {2,}(?!`)", " ", stripped)
        stripped = re.sub(r"\.\.(?=\s|$)", ".", stripped)
        cleaned_lines.append(stripped)
    cleaned = "\n".join(cleaned_lines).strip()
    if cleaned.startswith("- "):
        candidate = cleaned[2:].strip()
        if "\n" not in candidate and not re.search(r"^\s*[-*]\s+", candidate, flags=re.MULTILINE):
            cleaned = candidate
    return cleaned


def _prompt_safe_text(text: object) -> str:
    value = "" if text is None else str(text)
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _product_looks_like_command_question(question: str) -> bool:
    text = _clean(question).lower()
    return any(
        phrase in text
        for phrase in [
            "how to configure",
            "how to use",
            "how to set up",
            "how do i configure",
            "how do i use",
            "how do you configure",
            "how do you use",
            "how can i configure",
            "how can i use",
            "help me find how to configure",
            "help me find how to use",
            "help me find the syntax",
            "what is the syntax",
            "what is the syntax for",
            "syntax of",
            "show syntax",
            "command syntax",
            "how is the command written",
        ]
    )


def _product_is_command_purpose_question(question: str) -> bool:
    return is_command_purpose_question(question)


def _product_answer_looks_like_cli_syntax(answer: str) -> bool:
    raw = _clean(answer)
    text = raw.lower()

    if not text:
        return False

    if re.search(r"\bsyntax\s*:", text):
        return True

    if re.search(r"\bcommand syntax\s*:", text):
        return True

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    if len(lines) <= 3:
        joined = " ".join(lines)

        has_cli_symbols = any(
            symbol in joined for symbol in ("<", ">", "[", "]", "{", "}", "|")
        )

        starts_like_command = re.match(
            r"^(?:no\s+)?(show|clear|ip|ipv6|interface|vlan|bfd|redundancy|apply|aaa|erps|mdns-sd)\b",
            joined.lower(),
        )

        if has_cli_symbols and starts_like_command and len(joined.split()) <= 40:
            return True

    return False


def _product_syntax_validation_result(question: str, lookup_answer: str, slots: Dict[str, str]) -> Tuple[bool, str, str]:
    command = _clean(slots.get("command", "")) or _product_command_from_question(question)
    raw_answer = _clean(lookup_answer)
    syntax_candidate = clean_cli_syntax(_extract_cli_syntax(raw_answer, command))
    if not syntax_candidate:
        return False, "no clean syntax candidate found", ""

    if is_bad_syntax_artifact(raw_answer, command):
        return False, "lookup answer contains syntax artifact markers", syntax_candidate

    candidate_lines = [line.strip() for line in raw_answer.splitlines() if line.strip()]
    command_like_lines = [
        line
        for line in candidate_lines
        if re.match(
            r"^(?:\d{1,4}\s+)?(?:no\s+)?(?:show|clear|ip|ipv6|interface|vlan|bfd|redundancy|apply|aaa|erps|mdns-sd)\b",
            line.lower(),
        )
    ]
    if len(command_like_lines) > 1:
        return False, "multiple command entries detected in lookup answer", syntax_candidate

    if any(re.search(r"\.{8,}", line) for line in candidate_lines):
        return False, "dotted leader artifact detected", syntax_candidate

    if any(re.search(r"^\s*\d{1,4}\s+[A-Za-z]", line) for line in candidate_lines):
        return False, "page-number prefix detected before syntax", syntax_candidate

    if len(candidate_lines) > 3 and syntax_candidate and len(syntax_candidate.split()) <= 40:
        return False, "multiple adjacent lines detected in syntax answer", syntax_candidate

    if command:
        syntax_lower = syntax_candidate.lower()
        if command.lower().startswith("show ") and "show " not in syntax_lower:
            return False, "expected show command syntax was not preserved", syntax_candidate
        if not command.lower().startswith("show ") and re.search(r"\bshow\s+", syntax_lower) and not re.search(r"\bshow\s+", command.lower()):
            return False, "unrelated show command detected in syntax answer", syntax_candidate

    return True, "", syntax_candidate


def _normalize_product_lookup_question(question: str, slots: Dict[str, str], predicted_intent: str) -> str:
    text = _clean(question)
    command = _clean(slots.get("command", ""))
    topic = _clean(slots.get("topic", ""))
    feature = _clean(slots.get("feature", ""))
    category = _clean(slots.get("category", ""))
    event_id = _clean(slots.get("event_id", ""))
    intent = _clean(predicted_intent)

    if command and _product_looks_like_command_question(text):
        return f"What is the syntax of {command} command?"
    if event_id:
        return f"What does event {event_id} mean?"
    if feature and category:
        return f"What is {category} {feature}?"
    if feature:
        return f"What is {feature}?"
    if topic:
        return f"What is {topic}?"

    lower = text.lower()
    if _product_looks_like_command_question(text):
        command_match = _product_command_from_question(text)
        if command_match:
            return f"What is the syntax of {command_match} command?"
    if intent in PRODUCT_EXACT_INTENTS:
        return text
    return text


def _product_intent_override(question: str, slots: Dict[str, str], predicted_intent: str) -> str:
    text = _clean(question).lower()
    if slots.get("command") and _product_looks_like_command_question(text):
        return "cli_syntax"
    if slots.get("command") and _product_is_command_purpose_question(text) and predicted_intent in {"cli_syntax", "show_command_syntax"}:
        return "concept_explanation"
    return predicted_intent


def _product_topic_from_question(question: str) -> str:
    text = _clean(question)
    patterns = [
        r"\bwhat\s+does\s+(?:the\s+)?(?:guide|documentation|docs|manual)\s+say\s+about\s+(?P<topic>.+?)(?:\?|$)",
        r"\bwhat\s+does\s+(?:the\s+)?(?:guide|documentation|docs|manual)\s+explain\s+about\s+(?P<topic>.+?)(?:\?|$)",
        r"\bwhat\s+can\s+you\s+tell\s+me\s+about\s+(?P<topic>.+?)(?:\?|$)",
        r"\btell\s+me\s+about\s+(?P<topic>.+?)(?:\?|$)",
        r"\bwhat\s+is\s+(?P<topic>.+?)(?:\?|$)",
        r"\bexplain\s+(?P<topic>.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        topic = _clean(match.group("topic")).strip(" ?.")
        if not topic or re.search(r"\bcommand\b", topic, flags=re.IGNORECASE):
            continue
        if re.search(r"\b(this|that|those|these|it)\b", topic, flags=re.IGNORECASE) and len(topic.split()) <= 6:
            continue
        return topic
    return ""


def _product_event_id_from_question(question: str) -> str:
    text = _clean(question)
    match = re.search(r"\b(?:event\s+id|event)\s*(?:is\s*)?(?P<event_id>\d{3,7})\b", text, flags=re.IGNORECASE)
    return match.group("event_id") if match else ""


def _product_slots_from_question(question: str) -> Dict[str, str]:
    slots = extract_slots_from_question(question)
    text = _clean(question)
    switch_match = re.search(
        r"\b(?:For\s+)?(?:an?\s+|the\s+)?(?P<switch>(?:CX\d{4}|\d{4,5}[A-Za-z]?))\s+(?:Switch\s+Series\s+)?(?:running\s+)?AOS-CX\s+(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<sub>\d+))?\b",
        text,
        flags=re.IGNORECASE,
    )
    if switch_match:
        slots["switch"] = _canonical_product_switch(switch_match.group("switch"))
        slots["version"] = f"{switch_match.group('major')}.{switch_match.group('minor')}"
        if switch_match.group("sub"):
            slots["sub_version"] = switch_match.group("sub")
    command = _product_command_from_question(text)
    if command:
        slots["command"] = command
    topic = _product_topic_from_question(text)
    if topic and not slots.get("topic"):
        slots["topic"] = topic
    event_id = _product_event_id_from_question(text)
    if event_id:
        slots["event_id"] = event_id
    return slots


def _merge_context_slots(
    slots: Dict[str, str],
    session_context: Dict[str, Optional[str]],
    selected_context: Dict[str, str],
    *,
    use_session_context: bool,
) -> Dict[str, str]:
    effective = {key: _clean(value) for key, value in slots.items() if _clean(value)}
    for key in ("switch", "version", "sub_version", "feature", "category", "bug_id", "command", "topic", "event_id"):
        value = _clean(selected_context.get(key, ""))
        if key == "switch":
            value = _canonical_product_switch(value)
        if value and not effective.get(key):
            effective[key] = value
    if not use_session_context:
        return effective
    for key in ("last_bug_id", "last_switch", "last_version", "last_sub_version", "last_feature", "last_category", "last_command", "last_topic", "last_event_id"):
        value = _clean(session_context.get(key))
        if not value:
            continue
        target = key.replace("last_", "")
        if target == "switch":
            value = _canonical_product_switch(value)
        if target == "version" and value:
            value = _domain_version(value, "product")
        if not effective.get(target):
            effective[target] = value
    return effective


def _build_product_lookup_index(entries) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = defaultdict(list)
    for entry in entries:
        slots = dict(entry.slots)
        slots.setdefault("switch", entry.switch)
        slots.setdefault("version", entry.version)
        slots.setdefault("sub_version", entry.sub_version)
        slots.setdefault("command", _clean(slots.get("command", "")))
        slots.setdefault("topic", _clean(slots.get("topic", "")))
        slots.setdefault("feature", _clean(slots.get("feature", "")))
        slots.setdefault("category", _clean(slots.get("category", "")))
        slots.setdefault("section", _clean(slots.get("section", "")))
        slots.setdefault("event_id", _clean(slots.get("event_id", "")))
        slots.setdefault("question_type", _clean(slots.get("question_type", "")))

        switch = _canonical_product_switch(slots.get("switch", ""))
        version = _clean(slots.get("version", "")).replace("_", ".")
        sub_version = _clean(slots.get("sub_version", ""))
        primary = _product_primary_slot(slots)

        candidates: List[str] = []
        version_aliases = _product_version_aliases(version, sub_version)
        for version_alias in version_aliases:
            if switch and version_alias and sub_version and primary:
                candidates.append("|".join([entry.intent, switch, version_alias, sub_version, primary]))
            if switch and version_alias and primary:
                candidates.append("|".join([entry.intent, switch, version_alias, primary]))
        if switch and primary:
            candidates.append("|".join([entry.intent, switch, primary]))
        if primary:
            candidates.append("|".join([entry.intent, primary]))
        for version_alias in version_aliases:
            if switch and version_alias and sub_version:
                candidates.append("|".join([entry.intent, switch, version_alias, sub_version]))
            if switch and version_alias:
                candidates.append("|".join([entry.intent, switch, version_alias]))
        if switch:
            candidates.append("|".join([entry.intent, switch]))
        candidates.append(entry.intent)

        for key in _unique(candidates):
            if entry.entry_id not in index[key]:
                index[key].append(entry.entry_id)
    return dict(index)


def _build_product_availability_index(entries) -> Dict[str, object]:
    product_docs: Dict[str, Dict[str, Dict[str, set[str]]]] = defaultdict(lambda: {"versions": defaultdict(set)})
    for entry in entries:
        switch = _canonical_product_switch(entry.switch)
        version = _clean(entry.version).replace("_", ".")
        sub_version = _clean(entry.sub_version)
        if switch and version:
            for version_alias in _product_version_aliases(version, sub_version):
                product_docs[switch]["versions"][version_alias].add(sub_version)
    normalized: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for switch, payload in product_docs.items():
        versions = payload.get("versions", {})
        normalized[switch] = {
            "versions": {version: sorted(value for value in values if value) for version, values in versions.items()}
        }
    return {"release_notes": {}, "product_docs": normalized}


def _build_product_bug_metadata_index(entries) -> Dict[str, List[Dict[str, str]]]:
    index: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for entry in entries:
        bug_id = _clean(entry.bug_id)
        if not bug_id:
            continue
        index[bug_id].append(
            {
                "switch": _clean(entry.switch),
                "version": _domain_version(entry.version, "product"),
                "sub_version": _clean(entry.sub_version),
                "intent": _clean(entry.intent),
                "feature": _clean(entry.feature),
                "category": _clean(entry.category),
                "question_type": _clean(entry.question_type),
            }
        )
    return dict(index)


def _select_primary_answer(answers: Sequence[str]) -> str:
    unique = _unique([_clean(answer) for answer in answers if _clean(answer)])
    return unique[0] if unique else ""


def _build_candidate_keys(domain: str, intent: str, slots: Dict[str, str]) -> List[str]:
    intent = _clean(intent)
    if not intent:
        return []
    switch = _clean(slots.get("switch", ""))
    version = _domain_version(slots.get("version", ""), domain)
    sub_version = _clean(slots.get("sub_version", ""))
    bug_id = _clean(slots.get("bug_id", ""))
    feature = _clean(slots.get("feature", ""))
    category = _clean(slots.get("category", ""))
    command = _clean(slots.get("command", ""))
    topic = _clean(slots.get("topic", ""))
    section = _clean(slots.get("section", ""))
    event_id = _clean(slots.get("event_id", ""))
    question_type = _clean(slots.get("question_type", ""))
    primary = _product_primary_slot(slots)

    candidates: List[str] = []
    if domain == "release":
        if intent == "release_caveat":
            if switch and version and sub_version and feature and question_type:
                candidates.append("|".join([intent, switch, version, sub_version, feature, question_type]))
            if switch and version and sub_version and feature:
                candidates.append("|".join([intent, switch, version, sub_version, feature]))
            if switch and version and feature and question_type:
                candidates.append("|".join([intent, switch, version, feature, question_type]))
            if switch and version and feature:
                candidates.append("|".join([intent, switch, version, feature]))
            if feature and question_type:
                candidates.append("|".join([intent, feature, question_type]))
            if feature:
                candidates.append("|".join([intent, feature]))
            return _unique(candidates)
        if intent.startswith("bug_"):
            if switch and version and sub_version and bug_id:
                candidates.append("|".join([intent, switch, version, sub_version, bug_id]))
            if bug_id:
                candidates.append("|".join([intent, bug_id]))
            if switch and version and sub_version and category and bug_id:
                candidates.append("|".join([intent, switch, version, sub_version, category, bug_id]))
            if category and bug_id:
                candidates.append("|".join([intent, category, bug_id]))
            return _unique(candidates)
        return _unique(candidates)

    # product lookups
    version_aliases = _product_version_aliases(version, sub_version)
    for version_alias in version_aliases:
        if switch and version_alias and sub_version and primary:
            candidates.append("|".join([intent, switch, version_alias, sub_version, primary]))
        if switch and version_alias and primary:
            candidates.append("|".join([intent, switch, version_alias, primary]))
    if switch and primary:
        candidates.append("|".join([intent, switch, primary]))
    if primary:
        candidates.append("|".join([intent, primary]))
    for version_alias in version_aliases:
        if switch and version_alias and sub_version:
            candidates.append("|".join([intent, switch, version_alias, sub_version]))
        if switch and version_alias:
            candidates.append("|".join([intent, switch, version_alias]))
    if switch:
        candidates.append("|".join([intent, switch]))
    if command:
        candidates.append("|".join([intent, command]))
    if topic:
        candidates.append("|".join([intent, topic]))
    if feature:
        candidates.append("|".join([intent, feature]))
    if section:
        candidates.append("|".join([intent, section]))
    if event_id:
        candidates.append("|".join([intent, event_id]))
    if category:
        candidates.append("|".join([intent, category]))
    if question_type:
        candidates.append("|".join([intent, question_type]))
    candidates.append(intent)
    return _unique(candidates)


def _tokenize(text: object) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", _clean(text).lower())


def _jaccard(left: object, right: object) -> float:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _score_entry(domain: str, question: str, slots: Dict[str, str], entry) -> float:
    q_norm = _clean(question).lower()
    e_norm = _clean(entry.input_text).lower()
    seq = SequenceMatcher(None, q_norm, e_norm).ratio()
    tok = _jaccard(q_norm, e_norm)
    score = 0.58 * seq + 0.25 * tok

    for key in ("bug_id", "switch", "version", "sub_version", "feature", "category", "command", "topic", "section", "event_id", "question_type"):
        slot_value = _clean(slots.get(key, ""))
        entry_value = _clean(getattr(entry, key, ""))
        if not slot_value or not entry_value:
            continue
        if key == "version":
            slot_value = _domain_version(slot_value, domain)
            entry_value = _domain_version(entry_value, domain)
        if slot_value.lower() == entry_value.lower():
            if key in {"bug_id", "switch", "command", "topic"}:
                score += 0.15
            elif key in {"version", "sub_version", "feature", "category", "section", "event_id"}:
                score += 0.08
            else:
                score += 0.05

    command_value = _clean(slots.get("command", "")).lower()
    if command_value:
        answer_text = _clean(getattr(entry, "answer", "")).lower()
        input_text = _clean(getattr(entry, "input_text", "")).lower()
        command_root = command_value.split()[0] if command_value else ""
        if command_value and command_value in answer_text:
            score += 0.3
        elif command_root and command_root in answer_text:
            score += 0.12
        elif command_value and command_value in input_text:
            score += 0.12
        elif command_root and command_root in input_text:
            score += 0.05

    return min(1.0, score)


def _rank_entries(domain: str, question: str, slots: Dict[str, str], candidates: Sequence[Any]) -> List[Tuple[Any, float]]:
    ranked = [(entry, _score_entry(domain, question, slots, entry)) for entry in candidates]
    ranked.sort(key=lambda item: (item[1], getattr(item[0], "entry_id", 0)), reverse=True)
    return ranked


def _resolve_generic_lookup(
    domain: str,
    question: str,
    intent: str,
    slots: Dict[str, str],
    entries,
    lookup_index: Dict[str, List[int]],
) -> Dict[str, object]:
    candidates = _build_candidate_keys(domain, intent, slots)
    exact_product_intent = domain == "product" and intent in PRODUCT_EXACT_INTENTS
    matching_entries: List[Any] = []
    seen_ids: set[int] = set()
    for key in candidates:
        for entry_id in lookup_index.get(key, []):
            if entry_id in seen_ids or not (0 <= entry_id < len(entries)):
                continue
            seen_ids.add(entry_id)
            matching_entries.append(entries[entry_id])

    if matching_entries:
        answers = [_clean(entry.answer) for entry in matching_entries if _clean(entry.answer)]
        unique_answers = _unique(answers)
        if len(unique_answers) == 1:
            return {
                "status": "found",
                "answer": unique_answers[0],
                "lookup_key_used": candidates[0] if candidates else None,
                "confidence": 0.98,
                "similarity": 0.98,
                "reason": None,
            }
        ranked = _rank_entries(domain, question, slots, matching_entries)
        if not ranked:
            return {
                "status": "not_found",
                "answer": None,
                "lookup_key_used": candidates[0] if candidates else None,
                "confidence": 0.0,
                "similarity": 0.0,
                "reason": "no ranked entries",
            }
        best_entry, best_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if exact_product_intent:
            if best_score < 0.5:
                return {
                    "status": "low_similarity",
                    "answer": None,
                    "lookup_key_used": candidates[0] if candidates else None,
                    "confidence": best_score,
                    "similarity": best_score,
                    "reason": "best similarity below exact-product threshold",
                }
            return {
                "status": "found",
                "answer": _clean(best_entry.answer),
                "lookup_key_used": candidates[0] if candidates else None,
                "confidence": best_score,
                "similarity": best_score,
                "reason": "exact product intent",
            }
        if best_score < 0.56:
            return {
                "status": "low_similarity",
                "answer": None,
                "lookup_key_used": candidates[0] if candidates else None,
                "confidence": best_score,
                "similarity": best_score,
                "reason": "best similarity below threshold",
            }
        if domain == "product" and intent == "concept_explanation":
            return {
                "status": "found",
                "answer": _clean(best_entry.answer),
                "lookup_key_used": candidates[0] if candidates else None,
                "confidence": best_score,
                "similarity": best_score,
                "reason": "best product explanation match",
            }
        if best_score - runner_up_score < 0.05:
            return {
                "status": "needs_disambiguation",
                "answer": None,
                "lookup_key_used": candidates[0] if candidates else None,
                "confidence": best_score,
                "similarity": best_score,
                "reason": "multiple close answers",
            }
        return {
            "status": "found",
            "answer": _clean(best_entry.answer),
            "lookup_key_used": candidates[0] if candidates else None,
            "confidence": best_score,
            "similarity": best_score,
            "reason": None,
        }

    if exact_product_intent:
        return {
            "status": "not_found",
            "answer": None,
            "lookup_key_used": candidates[0] if candidates else None,
            "confidence": 0.0,
            "similarity": 0.0,
            "reason": "no exact product syntax match",
        }

    intent_candidates = [entry for entry in entries if _clean(entry.intent) == _clean(intent)]
    ranked_pool = intent_candidates or list(entries)
    ranked = _rank_entries(domain, question, slots, ranked_pool)
    if not ranked:
        return {
            "status": "not_found",
            "answer": None,
            "lookup_key_used": candidates[0] if candidates else None,
            "confidence": 0.0,
            "similarity": 0.0,
            "reason": "no candidates",
        }
    best_entry, best_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score < 0.56:
        return {
            "status": "low_similarity",
            "answer": None,
            "lookup_key_used": candidates[0] if candidates else None,
            "confidence": best_score,
            "similarity": best_score,
            "reason": "best similarity below threshold",
        }
    if best_score - runner_up_score < 0.05 and len(ranked) > 1:
        return {
            "status": "needs_disambiguation",
            "answer": None,
            "lookup_key_used": candidates[0] if candidates else None,
            "confidence": best_score,
            "similarity": best_score,
            "reason": "similar candidates",
        }
    return {
        "status": "found",
        "answer": _clean(best_entry.answer),
        "lookup_key_used": candidates[0] if candidates else None,
        "confidence": best_score,
        "similarity": best_score,
        "reason": None,
    }


def _is_no_workaround(answer: str) -> bool:
    text = _clean(answer).lower()
    return "no workaround is documented" in text or text == "no workaround is documented in the release notes."


def _should_use_qwen(domain: str, intent: str, lookup_answer: str) -> bool:
    answer = _clean(lookup_answer)
    if not answer:
        return False
    if is_cli_syntax_answer(answer, intent):
        return False
    if _is_no_workaround(answer):
        return False
    if domain == "release" and intent in {"bug_category", "version_date", "release_date", "event_id", "cli_syntax", "show_command_syntax"}:
        return False
    if domain == "product":
        if intent in PRODUCT_EXACT_INTENTS:
            return False
        if intent == "concept_explanation":
            return True
        return len(answer.split()) > 8
    if len(answer.split()) < 8:
        return False
    return True


def _build_qwen_prompt(
    domain: str,
    question: str,
    predicted_intent: str,
    slots: Dict[str, str],
    lookup_answer: str,
    previous_context: Optional[Dict[str, str]] = None,
) -> str:
    title = "release-note" if domain == "release" else "product documentation"
    source_type = predicted_intent or "response_formatter"
    data_family = "release_notes" if domain == "release" else "product_documentation"
    if domain == "product":
        extra_guidance = (
            "For product documentation, keep the full grounded meaning and make it easier to read.\n"
            "Do not truncate the answer.\n"
            "If the grounded answer contains multiple facts, types, methods, goals, requirements, conditions, or steps, format them as bullet points or numbered steps.\n"
            "If the grounded answer contains commands, wrap the command text in backticks.\n"
            "If the user asks what a command does, explain the purpose only if the grounded answer already includes that purpose.\n"
            "If the grounded answer only contains syntax, say that only syntax was found.\n"
            "Avoid filler prefixes like 'The documented answer is' or 'According to the documentation'.\n"
            "Write concise bullet points that preserve every factual detail from the grounded answer.\n"
            "Do not add any facts that are not already grounded.\n"
        )
    else:
        extra_guidance = (
            "Keep the answer grounded and precise.\n"
            "Use short headings, bullets, or numbered steps only when they improve readability.\n"
            "Prefer bullets for multiple factual items.\n"
        )
    if domain == "product" and previous_context:
        previous_question = _clean(previous_context.get("last_question", ""))
        previous_lookup_answer = _clean(previous_context.get("last_lookup_answer", ""))
        previous_final_answer = _clean(previous_context.get("last_final_answer", ""))
        if previous_question and (previous_lookup_answer or previous_final_answer):
            return (
                "You are an HPE Aruba AOS-CX product documentation response formatter.\n\n"
                "Facts come only from the previous retrieved answer and previous final answer.\n"
                "You must not answer from your own knowledge.\n"
                "Your job is to answer the follow-up clearly and conversationally using only the previous context.\n"
                "Do not add new facts.\n"
                "Do not truncate the answer.\n"
                "If the previous answer explains multiple types or conditions, keep all of them and format them as bullets when helpful.\n"
                "If the previous context is missing, ask the user to specify the topic.\n\n"
                f"Previous question:\n{previous_question}\n\n"
                f"Previous retrieved answer:\n{previous_lookup_answer}\n\n"
                f"Previous final answer:\n{previous_final_answer}\n\n"
                f"Follow-up question:\n{_clean(question)}\n\n"
                "Task:\n"
                "Answer the follow-up using only the previous retrieved answer and previous final answer.\n"
                "Use bullets if it helps explain multiple documented items.\n"
                "Return only the final formatted answer."
            )
    return (
        f"You are an HPE Aruba AOS-CX {title} assistant.\n\n"
        "Use only the grounded answer provided below.\n"
        "Do not invent facts.\n"
        "Do not change Bug IDs, categories, versions, commands, workarounds, symptoms, scenarios, caveats, or feature names.\n"
        "If the grounded answer says no workaround is documented, preserve that meaning exactly.\n\n"
        f"{extra_guidance}\n"
        f"Question:\n{_clean(question)}\n\n"
        f"Predicted intent:\n{_clean(predicted_intent)}\n\n"
        f"Slots:\n{json.dumps(slots, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Metadata:\nSwitch: {_clean(slots.get('switch', ''))}\nVersion: {_clean(slots.get('version', ''))}\nSub-version: {_clean(slots.get('sub_version', ''))}\nSource type: {source_type}\nData family: {data_family}\n\n"
        f"Retrieved answer:\n{_prompt_safe_text(lookup_answer)}\n\n"
        "Task:\nAnswer the user naturally using only the grounded answer.\n"
        "Write a neat, complete response.\n"
        "If the grounded answer lists multiple documented items, format them as bullets.\n"
        "If the grounded answer is short, restate it clearly and do not add unsupported facts.\n"
        "Do not truncate the answer."
    )


def _session_template() -> Dict[str, Optional[str]]:
    return {
        "last_question": None,
        "last_final_answer": None,
        "last_lookup_answer": None,
        "last_source_type": None,
        "last_data_family": None,
        "last_bug_id": None,
        "last_valid_bug_id": None,
        "last_switch": None,
        "last_version": None,
        "last_sub_version": None,
        "last_feature": None,
        "last_category": None,
        "last_command": None,
        "last_topic": None,
        "last_event_id": None,
        "last_intent": None,
    }


def _update_session_context(
    session_context: Dict[str, Optional[str]],
    question: str,
    slots: Dict[str, str],
    predicted_intent: str,
    lookup_answer: str,
    final_answer: str,
    source_type: str,
    data_family: str,
) -> None:
    session_context["last_question"] = _clean(question)
    for key in ["bug_id", "switch", "version", "sub_version", "feature", "category", "command", "topic", "event_id"]:
        if slots.get(key):
            session_context[f"last_{key}"] = slots[key]
    if slots.get("bug_id"):
        session_context["last_bug_id"] = slots["bug_id"]
        session_context["last_valid_bug_id"] = slots["bug_id"]
    session_context["last_intent"] = predicted_intent
    session_context["last_lookup_answer"] = lookup_answer
    session_context["last_final_answer"] = final_answer
    session_context["last_source_type"] = source_type
    session_context["last_data_family"] = data_family


def _reuse_session_slots(slots: Dict[str, str], session_context: Dict[str, Optional[str]]) -> Dict[str, str]:
    effective = dict(slots)
    if not effective.get("bug_id") and _clean(session_context.get("last_bug_id")):
        effective["bug_id"] = _clean(session_context.get("last_bug_id"))
    for key in ["switch", "version", "sub_version", "feature", "category", "command", "topic", "event_id"]:
        if not effective.get(key) and _clean(session_context.get(f"last_{key}")):
            value = _clean(session_context.get(f"last_{key}"))
            if key == "version":
                value = _domain_version(value, "product")
            effective[key] = value
    return effective


def _format_deterministic(domain: str, status: str) -> str:
    if domain == "release":
        if status == "not_found":
            return "No matching answer was found in the current Aruba AOS-CX dataset."
        if status == "low_similarity":
            return "I found related documentation, but not a reliable exact match."
        if status == "needs_disambiguation":
            return "Multiple possible answers were found. Please provide more detail such as feature, bug ID, command, version, or sub-version."
        if status == "slot_missing":
            return "I need more detail to answer this, such as the bug ID, feature, command, version, or sub-version."
        return "Unable to answer from the current release-note dataset."
    if status == "not_found":
        return PRODUCT_NOT_FOUND_RESPONSE
    if status == "low_similarity":
        return "I found related documentation, but not a reliable exact match."
    if status == "needs_disambiguation":
        return PRODUCT_NEEDS_DISAMBIGUATION_RESPONSE
    if status == "slot_missing":
        return PRODUCT_SLOT_MISSING_RESPONSE
    return "Unable to answer from the current product documentation dataset."


def _polish_product_answer(answer: str, intent: str, slots: Optional[Dict[str, str]] = None) -> str:
    text = _cleanup_product_markdown(_strip_product_filler_prefix(answer))
    if not text:
        return text
    if intent in PRODUCT_EXACT_INTENTS or _is_no_workaround(text):
        return text
    if intent == "concept_explanation":
        return _format_product_concept_answer(text, slots or {})
    if text[-1] not in ".!?":
        return f"{text}."
    return text


def _looks_like_followup_question(question: str) -> bool:
    text = _clean(question).lower()
    if any(phrase in text for phrase in PRODUCT_FOLLOWUP_WORDS):
        return True
    if text in {"this", "that", "it", "explain", "elaborate"}:
        return True
    if len(text.split()) <= 10 and re.search(r"\b(this|that|those|these|it)\b", text):
        return True
    return False


def _is_product_followup(question: str) -> bool:
    return _looks_like_followup_question(question)


def _product_meaningful_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", _clean(text).lower())
    return [token for token in tokens if token not in PRODUCT_QWEN_STOPWORDS]


def _product_qwen_is_too_drifty(lookup_answer: str, qwen_answer: str) -> bool:
    candidate = _clean(qwen_answer)
    if not candidate:
        return True
    if re.search(r"\b[a-z0-9_-]+\(config\)#", candidate, flags=re.IGNORECASE):
        return True
    if "config)#" in candidate.lower():
        return True
    original_tokens = set(_product_meaningful_tokens(lookup_answer))
    candidate_tokens = set(_product_meaningful_tokens(candidate))
    if not original_tokens or not candidate_tokens:
        return True
    overlap = len(original_tokens & candidate_tokens) / max(1, len(original_tokens))
    return overlap < 0.6


def _format_product_concept_answer(answer: str, slots: Dict[str, str]) -> str:
    text = _cleanup_product_markdown(_strip_product_filler_prefix(answer))
    if not text:
        return ""
    if text.startswith(("**", "-", "1.", "*")) or "\n" in text:
        return text

    structured_segments = _product_structured_segments(text)
    if len(structured_segments) > 1:
        topic = _clean(slots.get("topic", "") or slots.get("feature", "") or slots.get("category", ""))
        lines: List[str] = []
        if topic:
            lines.append(f"**{topic}**")
            lines.append("")
        lines.extend(f"- {segment}" for segment in structured_segments)
        return _cleanup_product_markdown("\n".join(lines))

    sentences = _product_sentence_chunks(text)
    if len(sentences) <= 1:
        return text if text.endswith((".", "!", "?")) else f"{text}."

    topic = _clean(slots.get("topic", "") or slots.get("feature", "") or slots.get("category", ""))
    lead_sentence = sentences[0]
    list_like = bool(
        re.search(r"\b(two types|three types|four types|includes|consists|contains|requirements|conditions|methods|goals|steps|overview)\b", lead_sentence, flags=re.IGNORECASE)
        or ":" in lead_sentence
        or any(sentence[:1].isupper() and len(sentence.split()) <= 18 for sentence in sentences[1:])
    )

    if not list_like:
        return text if text.endswith((".", "!", "?")) else f"{text}."

    lines: List[str] = []
    if topic:
        lines.append(f"**{topic}**")
        lines.append("")
    if len(sentences) == 2 and re.search(r"\b(types?|methods?|steps?|requirements?|conditions?|goals?)\b", lead_sentence, flags=re.IGNORECASE):
        lines.append(lead_sentence)
        lines.append("")
        lines.extend(f"- {sentence}" for sentence in sentences[1:])
    else:
        lines.extend(f"- {sentence}" for sentence in sentences)
    return _cleanup_product_markdown("\n".join(line for line in lines if line is not None))


@dataclass
class QwenBundle:
    tokenizer: Any = None
    model: Any = None
    requested_path: str = ""
    resolved_path: str = ""
    model_kind: str = "disabled"
    resolution_reason: str = ""
    base_model_name: str = ""
    loaded: bool = False
    error: str = ""


@dataclass
class ReleaseRuntime:
    model_path: Path
    lookup_data_path: Path
    lookup_index_path: Path
    availability_path: Path
    bug_metadata_path: Path
    device: torch.device
    qwen: QwenBundle
    lstm_model: Any = field(init=False)
    lstm_tokenizer: Any = field(init=False)
    lstm_config: Dict[str, object] = field(init=False)
    lookup_entries: List[Any] = field(init=False)
    lookup_index: Dict[str, List[int]] = field(init=False)
    availability_index: Dict[str, object] = field(init=False)
    bug_metadata_index: Dict[str, List[Dict[str, str]]] = field(init=False)

    def __post_init__(self) -> None:
        self.lookup_entries, self.lookup_index = load_lookup_resources(self.lookup_index_path, self.lookup_data_path)
        self.availability_index = load_or_build_availability_index(self.availability_path, self.lookup_entries)
        self.bug_metadata_index = load_or_build_bug_metadata_index(self.bug_metadata_path, self.lookup_entries)
        self.lstm_model, self.lstm_tokenizer, self.lstm_config = load_lstm_support(self.model_path, self.device)

    def answer(
        self,
        question: str,
        session_context: Dict[str, Optional[str]],
        selected_context: Dict[str, str],
        show_debug: bool = False,
    ) -> Dict[str, object]:
        result = answer_release_question(
            question,
            self.lstm_model,
            self.lstm_tokenizer,
            self.lstm_config,
            self.lookup_entries,
            self.lookup_index,
            self.availability_index,
            self.bug_metadata_index,
            self.qwen.tokenizer,
            self.qwen.model,
            self.device,
            session_context,
        )
        return {
            "domain": "release",
            "question": result.get("question", _clean(question)),
            "predicted_intent": result.get("predicted_intent"),
            "raw_lstm_intent": result.get("raw_lstm_intent"),
            "slots": result.get("slots", {}),
            "lookup_status": result.get("lookup_status"),
            "lookup_key_used": result.get("lookup_key_used"),
            "lookup_answer": result.get("lookup_answer"),
            "qwen_used": result.get("qwen_used"),
            "qwen_answer": result.get("qwen_answer"),
            "qwen_validation_passed": result.get("qwen_validation_passed"),
            "final_answer": result.get("final_answer"),
            "answer_source": result.get("answer_source"),
            "source_type": result.get("source_type"),
            "data_family": result.get("data_family"),
            "confidence": result.get("availability_check", {}).get("available", True) if isinstance(result.get("availability_check"), dict) else None,
            "similarity": None,
            "debug": {
                "availability_check": result.get("availability_check"),
                "continuation_used": result.get("continuation_used"),
                "pending_intent_used": result.get("pending_intent_used"),
                "resolved_bug_id": result.get("resolved_bug_id"),
                "validation_reason": result.get("validation_reason"),
            },
        }


@dataclass
class ProductRuntime:
    model_path: Path
    data_paths: Sequence[Path]
    device: torch.device
    qwen: QwenBundle
    cache_dir: Path = BACKEND_CACHE_DIR
    lstm_model: Any = field(init=False)
    lstm_tokenizer: Any = field(init=False)
    lstm_config: Dict[str, object] = field(init=False)
    lookup_entries: List[Any] = field(init=False)
    lookup_index: Dict[str, List[int]] = field(init=False)
    availability_index: Dict[str, object] = field(init=False)
    bug_metadata_index: Dict[str, List[Dict[str, str]]] = field(init=False)

    def __post_init__(self) -> None:
        records: List[Dict[str, object]] = []
        for path in self.data_paths:
            if path.exists():
                records.extend(read_jsonl(path))
        self.lookup_entries = build_lookup_entries(records)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lookup_cache = self.cache_dir / "product_lookup_index.json"
        metadata_cache = self.cache_dir / "product_metadata_index.json"
        availability_cache = self.cache_dir / "product_availability_index.json"
        self.lookup_index = _build_product_lookup_index(self.lookup_entries)
        with lookup_cache.open("w", encoding="utf-8") as handle:
            json.dump(self.lookup_index, handle, indent=2, ensure_ascii=False)
        self.bug_metadata_index = _build_product_bug_metadata_index(self.lookup_entries)
        with metadata_cache.open("w", encoding="utf-8") as handle:
            json.dump(self.bug_metadata_index, handle, indent=2, ensure_ascii=False)
        self.availability_index = _build_product_availability_index(self.lookup_entries)
        with availability_cache.open("w", encoding="utf-8") as handle:
            json.dump(self.availability_index, handle, indent=2, ensure_ascii=False)
        self.lstm_model, self.lstm_tokenizer, self.lstm_config = load_lstm_support(self.model_path, self.device)

    def _availability_check(self, slots: Dict[str, str]) -> Dict[str, object]:
        release_notes = self.availability_index.get("product_docs", {})
        if not isinstance(release_notes, dict):
            release_notes = {}
        switch = _canonical_product_switch(slots.get("switch", ""))
        version = _clean(slots.get("version", "")).replace("_", ".")
        sub_version = _clean(slots.get("sub_version", ""))
        bug_id = _clean(slots.get("bug_id", ""))

        if switch and switch not in release_notes:
            return {"available": False, "status": "data_not_available", "reason": f"switch {switch} not in product availability"}
        if switch and version:
            payload = release_notes.get(switch, {})
            versions = payload.get("versions", {}) if isinstance(payload, dict) else {}
            version_aliases = _product_version_aliases(version, sub_version)
            matched_version = next((candidate for candidate in version_aliases if candidate in versions), "")
            if not matched_version:
                return {"available": False, "status": "data_not_available", "reason": f"version {version} not available for switch {switch}"}
            if sub_version:
                available_sub_versions = versions.get(matched_version, [])
                if available_sub_versions and sub_version not in available_sub_versions:
                    return {
                        "available": False,
                        "status": "data_not_available",
                        "reason": f"sub-version {sub_version} not available for switch {switch} version {version}",
                    }
        if bug_id and bug_id not in self.bug_metadata_index:
            return {"available": False, "status": "data_not_available", "reason": f"bug {bug_id} not found in product metadata"}
        return {"available": True, "status": "available", "reason": None}

    def answer(
        self,
        question: str,
        session_context: Dict[str, Optional[str]],
        selected_context: Dict[str, str],
        show_debug: bool = False,
    ) -> Dict[str, object]:
        cleaned_question = _clean(question)
        is_syntax_question = is_cli_syntax_answer("", "", cleaned_question)
        is_command_purpose = _product_is_command_purpose_question(cleaned_question)
        extracted_slots = _product_slots_from_question(cleaned_question)
        is_followup = _is_product_followup(cleaned_question)
        slots = _merge_context_slots(extracted_slots, session_context, selected_context, use_session_context=is_followup)
        slots["switch"] = _canonical_product_switch(slots.get("switch", ""))
        raw_lstm_intent = predict_intent(cleaned_question, self.lstm_model, self.lstm_tokenizer, self.lstm_config, self.device)
        predicted_intent = _product_intent_override(cleaned_question, slots, raw_lstm_intent)
        lookup_question = _normalize_product_lookup_question(cleaned_question, slots, predicted_intent)
        used_previous_context = False
        followup_context: Dict[str, str] = {}
        if is_followup:
            followup_context = {
                key: _clean(session_context.get(key, ""))
                for key in (
                    "last_question",
                    "last_final_answer",
                    "last_lookup_answer",
                    "last_source_type",
                    "last_data_family",
                    "last_switch",
                    "last_version",
                    "last_sub_version",
                    "last_topic",
                    "last_intent",
                )
            }
            has_previous_answer = bool(followup_context.get("last_lookup_answer") or followup_context.get("last_final_answer"))
            if not followup_context.get("last_question") or not has_previous_answer:
                return {
                    "domain": "product",
                    "question": cleaned_question,
                    "predicted_intent": predicted_intent,
                    "raw_lstm_intent": raw_lstm_intent,
                    "slots": slots,
                    "lookup_status": "slot_missing",
                    "lookup_key_used": None,
                    "lookup_answer": None,
                    "qwen_used": False,
                    "qwen_answer": None,
                    "qwen_validation_passed": False,
                    "final_answer": PRODUCT_FOLLOWUP_CONTEXT_MISSING_RESPONSE,
                    "answer_source": "followup_context_missing",
                    "source_type": predicted_intent,
                    "data_family": "product_documentation",
                    "confidence": 0.0,
                    "similarity": 0.0,
                    "debug": {"availability_check": {"available": True, "status": "available", "reason": None}},
                }
            if followup_context.get("last_intent") and not is_syntax_question:
                predicted_intent = _clean(followup_context.get("last_intent", predicted_intent)) or predicted_intent
                lookup_question = _normalize_product_lookup_question(cleaned_question, slots, predicted_intent)
            slots = _reuse_session_slots(slots, session_context)
            used_previous_context = True
        availability_check = self._availability_check(slots)

        common_slots = dict(slots)
        common_slots.pop("switch", None)
        common_slots.pop("version", None)
        common_slots.pop("sub_version", None)

        common_resolution = None
        common_lookup_answer = ""
        common_lookup_key_used = None
        common_confidence = 0.0
        common_similarity = 0.0
        if not availability_check.get("available", True):
            common_resolution = _resolve_generic_lookup(
                "product",
                lookup_question,
                predicted_intent,
                common_slots,
                self.lookup_entries,
                self.lookup_index,
            )
            if common_resolution.get("status") == "found" and common_resolution.get("answer"):
                common_lookup_answer = _clean(common_resolution.get("answer", ""))
                common_lookup_key_used = common_resolution.get("lookup_key_used")
                common_confidence = float(common_resolution.get("confidence", 0.0) or 0.0)
                common_similarity = float(common_resolution.get("similarity", 0.0) or 0.0)
                availability_check = {"available": True, "status": "available", "reason": "common product lookup fallback"}

        if not availability_check.get("available", True):
            return {
                "domain": "product",
                "question": cleaned_question,
                "predicted_intent": "data_not_available",
                "raw_lstm_intent": "data_not_available",
                "slots": slots,
                "lookup_status": "data_not_available",
                "lookup_key_used": None,
                "lookup_answer": None,
                "qwen_answer": None,
                "qwen_validation_passed": False,
                "final_answer": PRODUCT_DATANOT_AVAILABLE_RESPONSE,
                "answer_source": "deterministic_availability",
                "confidence": 0.0,
                "similarity": 0.0,
                "debug": {"availability_check": availability_check},
            }

        if _is_product_followup(cleaned_question) and not slots.get("command") and not slots.get("topic"):
            if _clean(session_context.get("last_command")):
                slots["command"] = _clean(session_context.get("last_command"))
            if _clean(session_context.get("last_topic")):
                slots["topic"] = _clean(session_context.get("last_topic"))
            if _clean(session_context.get("last_feature")) and not slots.get("feature"):
                slots["feature"] = _clean(session_context.get("last_feature"))
            if _clean(session_context.get("last_category")) and not slots.get("category"):
                slots["category"] = _clean(session_context.get("last_category"))

        resolution = _resolve_generic_lookup("product", lookup_question, predicted_intent, slots, self.lookup_entries, self.lookup_index)
        lookup_status = str(resolution.get("status", "error"))
        lookup_answer = _clean(resolution.get("answer", "")) if resolution.get("answer") else ""
        lookup_key_used = resolution.get("lookup_key_used")
        confidence = float(resolution.get("confidence", 0.0) or 0.0)
        similarity = float(resolution.get("similarity", 0.0) or 0.0)

        if common_resolution and common_lookup_answer and (
            resolution.get("status") != "found" or confidence < common_confidence
        ):
            resolution = common_resolution
            lookup_status = str(common_resolution.get("status", "error"))
            lookup_answer = common_lookup_answer
            lookup_key_used = common_lookup_key_used
            confidence = common_confidence
            similarity = common_similarity
        elif not lookup_answer and common_lookup_answer:
            lookup_status = str(common_resolution.get("status", "found")) if common_resolution else "found"
            lookup_answer = common_lookup_answer
            lookup_key_used = common_lookup_key_used
            confidence = common_confidence
            similarity = common_similarity

        if (
            predicted_intent == "concept_explanation"
            and lookup_status == "found"
            and (slots.get("switch") or slots.get("version") or slots.get("sub_version"))
        ):
            common_slots = dict(slots)
            common_slots.pop("switch", None)
            common_slots.pop("version", None)
            common_slots.pop("sub_version", None)
            common_resolution = _resolve_generic_lookup(
                "product",
                lookup_question,
                predicted_intent,
                common_slots,
                self.lookup_entries,
                self.lookup_index,
            )
            common_answer = _clean(common_resolution.get("answer", "")) if common_resolution.get("answer") else ""
            common_confidence = float(common_resolution.get("confidence", 0.0) or 0.0)
            current_is_weak = len(lookup_answer.split()) < 8 or "config)#" in lookup_answer.lower() or lookup_answer.endswith(":")
            if common_resolution.get("status") == "found" and common_answer and (
                common_confidence > confidence
                or (current_is_weak and common_confidence >= max(0.56, confidence - 0.1))
            ):
                resolution = common_resolution
                lookup_status = str(common_resolution.get("status", "error"))
                lookup_answer = common_answer
                lookup_key_used = common_resolution.get("lookup_key_used")
                confidence = common_confidence
                similarity = common_confidence

        previous_context_answer = ""
        if used_previous_context:
            previous_context_answer = _clean(session_context.get("last_lookup_answer")) or _clean(session_context.get("last_final_answer"))
            if previous_context_answer and lookup_status != "found":
                lookup_status = "found"
                lookup_answer = previous_context_answer
                lookup_key_used = "session_context"
                confidence = max(confidence, 0.9)
                similarity = max(similarity, 0.9)

        qwen_answer = None
        qwen_validation_passed = False
        qwen_used = False
        answer_source = "lookup_fallback"
        final_answer = _format_deterministic("product", lookup_status)
        source_type = predicted_intent
        data_family = "product_documentation"
        answer_type = predicted_intent
        validation_passed = True
        rejection_reason = ""
        bypass_qwen = False

        if lookup_status == "found" and lookup_answer:
            formatter_lookup_answer = lookup_answer
            if used_previous_context and previous_context_answer:
                formatter_lookup_answer = previous_context_answer

            is_strict_syntax_question = bool(
                is_syntax_question
                or is_command_purpose
                or _product_answer_looks_like_cli_syntax(formatter_lookup_answer)
            )
            if is_strict_syntax_question:
                answer_type = "cli_syntax"
                validation_passed, rejection_reason, syntax_candidate = _product_syntax_validation_result(
                    cleaned_question,
                    formatter_lookup_answer,
                    slots,
                )
                if validation_passed and syntax_candidate:
                    final_answer = format_cli_syntax_answer(
                        cleaned_question,
                        formatter_lookup_answer,
                        {"intent": predicted_intent, **slots},
                    )
                    answer_source = "deterministic_command_formatter" if is_command_purpose and not is_syntax_question else "deterministic_cli_syntax"
                    final_answer = _cleanup_product_markdown(_strip_product_generated_label(final_answer))
                else:
                    final_answer = PRODUCT_SYNTAX_VALIDATION_FALLBACK
                    answer_source = "deterministic_syntax_validation"
                    qwen_answer = None
                    qwen_validation_passed = False
                    bypass_qwen = True
            else:
                final_answer = _polish_product_answer(lookup_answer, predicted_intent, slots)
                answer_source = "deterministic_lookup"

            use_qwen_for_product = self.qwen.loaded and _should_use_qwen("product", predicted_intent, lookup_answer)
            if is_strict_syntax_question or not validation_passed:
                use_qwen_for_product = False
            if used_previous_context and len(formatter_lookup_answer.split()) < 15:
                use_qwen_for_product = False

            if use_qwen_for_product and validation_passed:
                prompt = _build_qwen_prompt(
                    "product",
                    cleaned_question,
                    predicted_intent,
                    slots,
                    formatter_lookup_answer,
                    followup_context if used_previous_context else None,
                )
                try:
                    qwen_used = True
                    qwen_answer = generate_qwen_answer(
                        self.qwen.tokenizer,
                        self.qwen.model,
                        prompt,
                        predicted_intent,
                        self.device,
                        data_family="product_documentation",
                    )
                    qwen_validation_passed, _reason = validate_qwen_answer(
                        predicted_intent,
                        slots,
                        formatter_lookup_answer,
                        qwen_answer,
                        data_family="product_documentation",
                    )
                    if qwen_validation_passed and not _product_qwen_is_too_drifty(formatter_lookup_answer, qwen_answer):
                        final_answer = qwen_answer
                        answer_source = "qwen_grounded"
                    else:
                        final_answer = _polish_product_answer(formatter_lookup_answer, predicted_intent, slots)
                        answer_source = "lookup_fallback"
                except Exception:
                    qwen_answer = None
                    qwen_validation_passed = False
            elif is_strict_syntax_question:
                bypass_qwen = True
        elif used_previous_context and previous_context_answer:
            final_answer = _polish_product_answer(previous_context_answer, predicted_intent, slots)
            answer_source = "session_context"
            lookup_status = "found"
            lookup_answer = previous_context_answer
            lookup_key_used = "session_context"

        final_answer = _cleanup_product_markdown(_strip_product_generated_label(final_answer))
        if (is_syntax_question or is_command_purpose or answer_type == "cli_syntax") and not validation_passed:
            final_answer = PRODUCT_SYNTAX_VALIDATION_FALLBACK

        if lookup_status == "found" and lookup_answer:
            formatter_lookup_answer = lookup_answer
            if used_previous_context and previous_context_answer:
                formatter_lookup_answer = previous_context_answer
            _update_session_context(
                session_context,
                cleaned_question,
                slots,
                predicted_intent,
                formatter_lookup_answer,
                final_answer,
                source_type,
                data_family,
            )

        if show_debug:
            print(f"[FORMATTER] question: {cleaned_question}")
            print(f"[FORMATTER] predicted_intent: {predicted_intent}")
            print(f"[FORMATTER] lookup_status: {lookup_status}")
            print(f"[FORMATTER] lookup_answer_length: {len(lookup_answer or '')}")
            print(f"[FORMATTER] final_answer_length: {len(final_answer or '')}")
            print(f"[FORMATTER] is_cli_syntax: {is_syntax_question}")
            print(f"[FORMATTER] is_command_purpose: {is_command_purpose}")
            print(f"[FORMATTER] answer_type: {answer_type}")
            print(f"[FORMATTER] validation_passed: {validation_passed}")
            print(f"[FORMATTER] rejection_reason: {rejection_reason}")
            print(f"[FORMATTER] bypass_qwen: {bypass_qwen}")
            print(f"[FORMATTER] qwen_used: {qwen_used}")
            print(f"[FORMATTER] validation_passed: {qwen_validation_passed}")
            print(f"[FOLLOWUP] is_followup: {is_followup}")
            print(f"[FOLLOWUP] used_previous_context: {used_previous_context}")

        return {
            "domain": "product",
            "question": cleaned_question,
            "predicted_intent": predicted_intent,
            "raw_lstm_intent": raw_lstm_intent,
            "slots": slots,
            "lookup_status": lookup_status,
            "lookup_key_used": lookup_key_used,
            "lookup_answer": lookup_answer or None,
            "qwen_used": qwen_used,
            "qwen_answer": qwen_answer,
            "qwen_validation_passed": qwen_validation_passed,
            "final_answer": final_answer,
            "answer_source": answer_source,
            "source_type": source_type,
            "data_family": data_family,
            "confidence": confidence,
            "similarity": similarity,
            "answer_type": answer_type,
            "validation_passed": validation_passed,
            "rejection_reason": rejection_reason or None,
            "debug": {
                "availability_check": availability_check,
                "is_followup": is_followup,
                "used_previous_context": used_previous_context,
                "answer_type": answer_type,
                "validation_passed": validation_passed,
                "rejection_reason": rejection_reason or None,
                "lookup_key_used": lookup_key_used,
                "bypass_qwen": bypass_qwen,
            },
        }


@dataclass
class AnswerService:
    device: torch.device
    release: ReleaseRuntime
    product: ProductRuntime
    qwen: QwenBundle
    sessions: Dict[str, Dict[str, Dict[str, Optional[str]]]] = field(default_factory=dict)

    @classmethod
    def create(cls, device: Optional[torch.device] = None) -> "AnswerService":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qwen_bundle = QwenBundle()
        try:
            tokenizer, model, meta = load_qwen_model(QWEN_MODEL_PATH, device)
            qwen_bundle = QwenBundle(
                tokenizer=tokenizer,
                model=model,
                requested_path=meta.get("requested_path", str(QWEN_MODEL_PATH)),
                resolved_path=meta.get("resolved_path", ""),
                model_kind=meta.get("model_kind", "adapter"),
                resolution_reason=meta.get("resolution_reason", ""),
                base_model_name=meta.get("base_model_name", ""),
                loaded=True,
            )
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            qwen_bundle = QwenBundle(
                requested_path=str(QWEN_MODEL_PATH),
                loaded=False,
                error=str(exc),
            )

        release = ReleaseRuntime(
            model_path=RELEASE_LSTM_MODEL_PATH,
            lookup_data_path=RELEASE_LOOKUP_DATA_PATH,
            lookup_index_path=RELEASE_LOOKUP_INDEX_PATH,
            availability_path=RELEASE_AVAILABILITY_PATH,
            bug_metadata_path=RELEASE_BUG_METADATA_PATH,
            device=device,
            qwen=qwen_bundle,
        )
        product = ProductRuntime(
            model_path=PRODUCT_LSTM_MODEL_PATH,
            data_paths=PRODUCT_LOOKUP_DATA_PATHS,
            device=device,
            qwen=qwen_bundle,
        )
        return cls(device=device, release=release, product=product, qwen=qwen_bundle)

    def new_session_id(self) -> str:
        return uuid4().hex

    def _session(self, session_id: str) -> Dict[str, Dict[str, Optional[str]]]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "release": _session_template(),
                "product": _session_template(),
            }
        return self.sessions[session_id]

    def resolve_domain(self, requested_domain: str, question: str, session: Optional[Dict[str, Dict[str, Optional[str]]]] = None) -> str:
        domain = _clean(requested_domain).lower()
        if domain in {"release", "product"}:
            return domain
        text = _clean(question).lower()
        if session and _looks_like_followup_question(question):
            product_session = session.get("product", {})
            release_session = session.get("release", {})
            if _clean(product_session.get("last_question")) and (
                _clean(product_session.get("last_final_answer")) or _clean(product_session.get("last_lookup_answer"))
            ):
                return "product"
            if _clean(release_session.get("last_question")) and (
                _clean(release_session.get("last_lookup_answer")) or _clean(release_session.get("last_final_answer"))
            ):
                return "release"
        if any(keyword in text for keyword in ["bug ", "bug id", "workaround", "scenario", "symptom", "release note", "caveat"]):
            return "release"
        if any(
            keyword in text
            for keyword in [
                "command",
                "syntax",
                "configuration",
                "rest api",
                "snmp",
                "event id",
                "show ",
                "how do i",
                "how do you",
                "how to",
                "what is the syntax",
                "what is ",
                "what does ",
                "overview",
                "feature",
                "guide",
                "purpose",
                "meaning",
            ]
        ):
            return "product"
        return "release"

    def chat(
        self,
        question: str,
        session_id: Optional[str] = None,
        domain: str = "auto",
        selected_switch: str = "",
        selected_version: str = "",
        selected_sub_version: str = "",
        show_debug: bool = False,
    ) -> Dict[str, object]:
        session_id = session_id or self.new_session_id()
        session = self._session(session_id)
        resolved_domain = self.resolve_domain(domain, question, session)
        selected_context = {
            "switch": selected_switch,
            "version": selected_version,
            "sub_version": selected_sub_version,
        }
        runtime = self.release if resolved_domain == "release" else self.product
        session_context = session[resolved_domain]
        result = runtime.answer(question, session_context, selected_context, show_debug=show_debug)

        return {
            "session_id": session_id,
            "domain": resolved_domain,
            "question": result.get("question"),
            "predicted_intent": result.get("predicted_intent"),
            "raw_lstm_intent": result.get("raw_lstm_intent"),
            "slots": result.get("slots", {}),
            "lookup_status": result.get("lookup_status"),
            "lookup_key_used": result.get("lookup_key_used"),
            "lookup_answer": result.get("lookup_answer"),
            "qwen_used": result.get("qwen_used"),
            "qwen_answer": result.get("qwen_answer"),
            "qwen_validation_passed": result.get("qwen_validation_passed"),
            "final_answer": result.get("final_answer"),
            "answer_source": result.get("answer_source"),
            "source_type": result.get("source_type"),
            "data_family": result.get("data_family"),
            "confidence": result.get("confidence"),
            "similarity": result.get("similarity"),
            "debug": result.get("debug", {}),
            "qwen_loaded": self.qwen.loaded,
            "qwen_model_path": self.qwen.resolved_path or self.qwen.requested_path,
            "qwen_model_kind": self.qwen.model_kind,
            "qwen_error": self.qwen.error or None,
        }

    def health(self) -> Dict[str, object]:
        return {
            "device": str(self.device),
            "qwen_loaded": self.qwen.loaded,
            "qwen_model_path": self.qwen.resolved_path or self.qwen.requested_path,
            "qwen_model_kind": self.qwen.model_kind,
            "qwen_error": self.qwen.error or None,
            "release_lstm_path": str(RELEASE_LSTM_MODEL_PATH),
            "product_lstm_path": str(PRODUCT_LSTM_MODEL_PATH),
            "model_root": str(MODEL_ROOT),
            "data_root": str(DATA_ROOT),
            "release_notes_data_dir": str(RELEASE_NOTES_DATA_DIR),
            "product_docs_data_dir": str(PRODUCT_DOCS_DATA_DIR),
            "release_lstm_data_dir": str(RELEASE_LSTM_DATA_DIR),
            "product_lstm_data_dir": str(PRODUCT_LSTM_DATA_DIR),
            "release_lookup_path": str(RELEASE_LOOKUP_INDEX_PATH),
            "release_bug_metadata_path": str(RELEASE_BUG_METADATA_PATH),
            "release_availability_path": str(RELEASE_AVAILABILITY_PATH),
            "product_lookup_data_paths": [str(path) for path in PRODUCT_LOOKUP_DATA_PATHS],
            "backend_cache_dir": str(BACKEND_CACHE_DIR),
            "ollama_base_url": OLLAMA_BASE_URL or None,
            "release_runtime": {
                "lookup_entries": len(self.release.lookup_entries),
                "lookup_keys": len(self.release.lookup_index),
                "availability_switches": len(self.release.availability_index.get("release_notes", {})),
            },
            "product_runtime": {
                "lookup_entries": len(self.product.lookup_entries),
                "lookup_keys": len(self.product.lookup_index),
                "availability_switches": len(self.product.availability_index.get("product_docs", {})),
            },
        }
