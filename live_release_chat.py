from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from lstm_lookup import (
    DATA_NOT_AVAILABLE_RESPONSE,
    check_data_availability,
    extract_slots_from_question,
    load_or_build_availability_index,
    load_or_build_bug_metadata_index,
    normalize_whitespace,
    resolve_lookup_answer,
)
from release_notes_qwen_pipeline import (
    DEFAULT_LOOKUP_DATA_PATH,
    DEFAULT_LOOKUP_INDEX_PATH,
    DEFAULT_LSTM_MODEL_PATH,
    DEFAULT_QWEN_MODEL_PATH,
    build_prompt,
    format_cli_syntax_answer,
    generate_qwen_answer,
    load_lstm_support,
    load_lookup_resources,
    load_qwen_model,
    is_command_purpose_question,
    predict_intent,
    is_cli_syntax_answer,
    is_no_workaround_answer,
    validate_qwen_answer,
)


SYSTEM_PROMPT = (
    "You are an HPE Aruba AOS-CX release-note assistant.\n\n"
    "Use only the grounded answer provided below.\n"
    "Do not invent facts.\n"
    "Do not change Bug IDs, categories, versions, sub-versions, commands, symptoms, scenarios, caveats, or workaround text.\n"
    "If the grounded answer says no workaround is documented, preserve that meaning exactly.\n"
    "If the user asks what a command does, only describe the purpose when the grounded answer actually includes that purpose.\n"
    "If the grounded answer only contains syntax, say that only syntax was found."
)

USER_PROMPT_TEMPLATE = """Question:
{user_question}

Predicted intent:
{predicted_intent}

Slots:
{slots_json}

Grounded answer:
{lookup_answer}

Task:
Answer the user naturally using only the grounded answer.
Do not truncate the answer.
Use bullets or headings when they improve readability.
Do not invent command purposes, bugs, versions, or workarounds.
"""

DEFAULT_OUTPUT_LOG = Path(r"C:\Hpe\Train\outputs_release_lstm\4100i\live_chat_log.jsonl")
DEFAULT_TEST_RESULTS_PATH = Path(r"C:\Hpe\Train\outputs_release_lstm\4100i\live_chat_test_results.jsonl")
DEFAULT_AVAILABILITY_INDEX_PATH = Path(r"C:\Hpe\Train\outputs_final\availability_index.json")
DEFAULT_BUG_METADATA_INDEX_PATH = Path(r"C:\Hpe\Train\outputs_release_lstm\all_switches\bug_metadata_index.json")
SIMILARITY_FALLBACK_THRESHOLD = 0.10
STRICT_BUG_INTENTS = {
    "bug_scenario",
    "bug_symptom",
    "bug_workaround",
    "bug_category",
}
RELEASE_QWEN_INTENTS = {
    "bug_scenario",
    "bug_symptom",
    "bug_workaround",
    "release_caveat",
}
RELEASE_EXACT_INTENTS = {
    "bug_category",
    "version_date",
    "release_date",
    "event_id",
    "cli_syntax",
    "show_command_syntax",
}
TYPO_FIXES = {
    "teh": "the",
    "ans": "answer",
    "work around": "workaround",
    "belogs": "belongs",
    "belongd": "belongs",
    "hte": "the",
}
FOLLOWUP_INTENT_MAP = {
    "scenario": "bug_scenario",
    "scenario?": "bug_scenario",
    "symptom": "bug_symptom",
    "symptom?": "bug_symptom",
    "workaround": "bug_workaround",
    "workaround?": "bug_workaround",
    "fix": "bug_workaround",
    "resolve": "bug_workaround",
    "version": "bug_metadata",
    "version?": "bug_metadata",
    "switch": "bug_metadata",
    "switch?": "bug_metadata",
    "category": "bug_metadata",
    "category?": "bug_metadata",
}
FOLLOWUP_WORDS = [
    "above bug",
    "that bug",
    "this bug",
    "same bug",
    "it",
    "above issue",
    "that issue",
    "this issue",
]
WORKAROUND_HINTS = [
    "how to resolve",
    "how to fix",
    "what is the fix",
    "workaround",
    "solution",
]
TEST_QUESTIONS = [
    "What about Bug 373540, when does that occur?",
    "how to resolve teh above bug?",
    "which switch does the bug belong to?",
    "what version and switch does the bug 373450 belong to?",
    "373450",
    "bug id is 373450",
    "what version and switch does bug 373540 belong to?",
    "Under what scenario does Bug 407418 occur?",
    "how to fix that bug?",
    "For 4100i AOS-CX 10.15.1020, what is the workaround for DHCP Bug 348727?",
    "What is the symptom of Bug 348886?",
    "My webserver is unresponsive but AFC is reachable. What should I do?",
    "I am getting Firmware image is invalid while uploading firmware. How do I fix it?",
    "For 4100i AOS-CX 10.18.0001, what limitation is mentioned for SNMP?",
    "what is the above limitation about?",
    "Which category does Bug 297755 belong to?",
    "what about bug 401936 workaround?",
    "what switch and version is this bug from?",
    "I have port flapping and STP instability. Which known issue is this?",
    "exit",
]


def build_qwen_prompt(question: str, predicted_intent: str, slots: Dict[str, str], lookup_answer: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        user_question=normalize_whitespace(question),
        predicted_intent=normalize_whitespace(predicted_intent),
        slots_json=json.dumps(slots, ensure_ascii=False, sort_keys=True),
        lookup_answer=str(lookup_answer or "").replace("\r\n", "\n").replace("\r", "\n").strip(),
    )


def normalize_question_for_routing(question: str) -> str:
    q = normalize_whitespace(question)
    for wrong, right in TYPO_FIXES.items():
        q = re.sub(rf"\b{re.escape(wrong)}\b", right, q, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", q).strip()


def override_intent_by_question(question: str, predicted_intent: str) -> str:
    text = normalize_question_for_routing(question).lower()

    scenario_patterns = [
        "under what scenario",
        "when does that occur",
        "when does it occur",
        "scenario",
    ]
    symptom_patterns = [
        "resolved issue",
        "what happens",
        "what problem",
        "symptom",
        "issue",
        "unresponsive",
        "reachable",
        "flapping",
        "instability",
        "invalid",
        "failure",
        "failed",
    ]
    workaround_patterns = [
        "how to resolve",
        "how to fix",
        "how to solve",
        "workaround",
        "resolution",
        "resolve",
        "fix",
        "solve",
    ]
    metadata_patterns = [
        r"\bwhich\s+switch\s+does\s+(?:the|this|that)?\s*bug\b",
        r"\bwhat\s+version\s+and\s+switch\s+does\s+(?:the|this|that)?\s*bug\b",
        r"\bwhat\s+switch\s+and\s+version\s+is\s+(?:the|this|that)?\s*bug\b",
        r"\bwhich\s+category\s+does\s+(?:the|this|that)?\s*bug\b",
        r"\bwhat\s+category\s+does\s+(?:the|this|that)?\s*bug\b",
        r"\bbug\s+id\s+is\s+\d+\b",
    ]

    if re.fullmatch(r"\d{4,7}", normalize_whitespace(text)):
        return "bug_metadata"
    if any(pattern in text for pattern in workaround_patterns):
        return "bug_workaround"
    if any(pattern in text for pattern in scenario_patterns):
        return "bug_scenario"
    if any(pattern in text for pattern in symptom_patterns):
        return "bug_symptom"
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in metadata_patterns):
        return "bug_metadata"
    return predicted_intent


def normalize_followup_key(question: str) -> str:
    return re.sub(r"[^\w]+", "", normalize_question_for_routing(question).lower())


def detect_followup_intent(question: str) -> Optional[str]:
    return FOLLOWUP_INTENT_MAP.get(normalize_followup_key(question))


def get_session_bug_id(session_context: Dict[str, Optional[str]]) -> str:
    return normalize_whitespace(session_context.get("last_valid_bug_id") or session_context.get("last_bug_id") or "")


def format_bug_metadata_answer(bug_id: str, entry, kind: str) -> str:
    def entry_value(name: str) -> str:
        if isinstance(entry, dict):
            return normalize_whitespace(entry.get(name, ""))
        return normalize_whitespace(getattr(entry, name, ""))

    switch = entry_value("switch")
    version = entry_value("version")
    sub_version = entry_value("sub_version")
    category = entry_value("category")
    version_text = format_version(version, sub_version)
    if kind == "category_only" and category:
        return f"Bug {bug_id} belongs to the {category} category."
    if switch and version_text and category:
        return f"Bug {bug_id} belongs to switch {switch}, AOS-CX {version_text}, category {category}."
    if switch and version_text:
        return f"Bug {bug_id} belongs to switch {switch}, AOS-CX {version_text}."
    if category:
        return f"Bug {bug_id} belongs to the {category} category."
    if switch:
        return f"Bug {bug_id} belongs to switch {switch}."
    return f"Bug {bug_id} metadata was found, but no additional details are available."


def choose_metadata_entry(question: str, slots: Dict[str, str], entries) -> Optional[object]:
    if not entries:
        return None
    normalized_question = normalize_question_for_routing(question).lower()
    preferred_switch = normalize_whitespace(slots.get("switch", "")).lower()
    preferred_version = normalize_whitespace(slots.get("version", "")).lower()
    preferred_sub_version = normalize_whitespace(slots.get("sub_version", "")).lower()
    preferred_category = normalize_whitespace(slots.get("category", "")).lower()

    scored: List[Tuple[float, object]] = []
    for entry in entries:
        if isinstance(entry, dict):
            entry_input_text = normalize_whitespace(entry.get("input_text", "")).lower()
            entry_switch = normalize_whitespace(entry.get("switch", "")).lower()
            entry_version = normalize_whitespace(entry.get("version", "")).lower()
            entry_sub_version = normalize_whitespace(entry.get("sub_version", "")).lower()
            entry_category = normalize_whitespace(entry.get("category", "")).lower()
        else:
            entry_input_text = normalize_whitespace(getattr(entry, "input_text", "")).lower()
            entry_switch = normalize_whitespace(getattr(entry, "switch", "")).lower()
            entry_version = normalize_whitespace(getattr(entry, "version", "")).lower()
            entry_sub_version = normalize_whitespace(getattr(entry, "sub_version", "")).lower()
            entry_category = normalize_whitespace(getattr(entry, "category", "")).lower()

        score = SequenceMatcher(None, normalized_question, entry_input_text).ratio()
        if preferred_switch and entry_switch == preferred_switch:
            score += 0.15
        if preferred_version and entry_version == preferred_version:
            score += 0.1
        if preferred_sub_version and entry_sub_version == preferred_sub_version:
            score += 0.05
        if preferred_category and entry_category == preferred_category:
            score += 0.08
        scored.append((score, entry))
    scored.sort(key=lambda item: (item[0], getattr(item[1], "entry_id", 0)), reverse=True)
    return scored[0][1]


def resolve_device(device_arg: str) -> torch.device:
    choice = normalize_whitespace(device_arg).lower()
    if choice in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def deterministic_message_for_status(lookup_status: str) -> str:
    if lookup_status == "not_found":
        return "No matching answer was found in the current Aruba AOS-CX dataset."
    if lookup_status == "low_similarity":
        return "I found related documentation, but not a reliable exact match."
    if lookup_status == "needs_disambiguation":
        return "Multiple possible answers were found. Please provide more detail such as feature, bug ID, command, version, or sub-version."
    if lookup_status == "slot_missing":
        return "I need more detail to answer this, such as the bug ID, feature, command, version, or sub-version."
    return "Unable to answer from the current release-note dataset."


def _should_use_release_qwen(
    predicted_intent: str,
    lookup_status: str,
    lookup_answer: str,
    confidence: float,
    question: str = "",
) -> bool:
    if lookup_status != "found":
        return False
    answer = normalize_whitespace(lookup_answer)
    if not answer or confidence < 0.6:
        return False
    if predicted_intent in RELEASE_EXACT_INTENTS:
        return False
    if is_cli_syntax_answer(answer, predicted_intent, question):
        return False
    if is_command_purpose_question(question):
        return False
    if is_no_workaround_answer(answer):
        return False
    if predicted_intent not in RELEASE_QWEN_INTENTS:
        return False
    return len(answer.split()) > 8


def is_followup_question(question: str) -> bool:
    text = normalize_whitespace(question).lower()
    for phrase in [item for item in FOLLOWUP_WORDS if item != "it"]:
        if phrase in text:
            return True
    if re.search(r"\bit\b", text) and len(text.split()) <= 6:
        return True
    return False


def is_workaround_request(question: str) -> bool:
    text = normalize_whitespace(question).lower()
    return any(hint in text for hint in WORKAROUND_HINTS)


def is_metadata_question(question: str) -> Optional[str]:
    text = normalize_whitespace(question).lower()
    patterns = {
        "category_only": [
            r"\bwhich\s+category\s+does\s+(?:the\s+|this\s+|that\s+)?bug(?:\s+\d+)?\s+belong\s+to\b",
            r"\bwhat\s+category\s+does\s+(?:the\s+|this\s+|that\s+)?bug(?:\s+\d+)?\s+belong\s+to\b",
            r"\bcategory\s+of\s+(?:the\s+|this\s+|that\s+)?bug(?:\s+\d+)?\b",
        ],
        "switch_only": [
            r"\bwhich\s+switch\s+does\s+the\s+bug\s+belong\s+to\b",
            r"\bwhich\s+switch\s+does\s+this\s+bug\s+belong\s+to\b",
            r"\bwhich\s+switch\s+does\s+(?:the\s+|this\s+|that\s+)?bug(?:\s+\d+)?\s+belong\s+to\b",
        ],
        "switch_version": [
            r"\bwhat\s+version\s+and\s+switch\s+does\s+(?:the\s+|this\s+|that\s+)?bug(?:\s+\d+)?\s+belong\s+to\b",
            r"\bwhat\s+switch\s+and\s+version\s+is\s+(?:the\s+|this\s+|that\s+)?bug(?:\s+\d+)?\s+from\b",
            r"\bwhat\s+switch\s+and\s+version\s+is\s+this\s+bug\s+from\b",
            r"\bwhat\s+version\s+does\s+(?:the\s+|this\s+|that\s+)?bug(?:\s+\d+)?\s+belong\s+to\b",
            r"\bwhat\s+version\s+and\s+switch\s+is\s+(?:the\s+|this\s+|that\s+)?bug(?:\s+\d+)?\s+from\b",
        ],
    }
    for kind, kind_patterns in patterns.items():
        for pattern in kind_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return kind
    return None


def looks_like_bare_bug_id(question: str) -> bool:
    return bool(re.fullmatch(r"\d{4,7}", normalize_whitespace(question)))


def format_version(version: str, sub_version: str) -> str:
    dotted = version.replace("_", ".")
    return f"{dotted}.{sub_version}" if sub_version else dotted


def build_metadata_answer(kind: str, entry) -> str:
    if kind == "switch_only":
        return normalize_whitespace(entry.switch)
    if kind == "switch_version":
        version_text = format_version(entry.version, entry.sub_version)
        if entry.switch and version_text:
            return f"{normalize_whitespace(entry.switch)} AOS-CX {version_text}"
        if version_text:
            return version_text
    return normalize_whitespace(entry.category or entry.switch or entry.answer)


def find_bug_entries(lookup_entries, bug_id: str):
    return [entry for entry in lookup_entries if normalize_whitespace(getattr(entry, "bug_id", "")) == normalize_whitespace(bug_id)]


def suggest_bug_id(bug_id: str, lookup_entries) -> Optional[str]:
    candidates: List[str] = []
    seen = set()
    for entry in lookup_entries:
        candidate = normalize_whitespace(getattr(entry, "bug_id", ""))
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    best_id = None
    best_score = 0.0
    for candidate in candidates:
        score = SequenceMatcher(None, bug_id, candidate).ratio()
        if score > best_score:
            best_score = score
            best_id = candidate
    if best_id and best_score >= 0.8:
        return best_id
    return None


def update_session_context(session_context: Dict[str, Optional[str]], slots: Dict[str, str], predicted_intent: str, lookup_answer: str) -> None:
    for key in ["bug_id", "switch", "version", "sub_version", "feature", "category"]:
        if slots.get(key):
            session_context[f"last_{key}"] = slots[key]
    if slots.get("bug_id"):
        session_context["last_bug_id"] = slots["bug_id"]
        session_context["last_valid_bug_id"] = slots["bug_id"]
    session_context["last_intent"] = predicted_intent
    session_context["last_lookup_answer"] = lookup_answer


def content_tokens(text: str) -> List[str]:
    stopwords = {
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
        "how",
        "do",
        "i",
        "my",
        "what",
        "should",
        "get",
        "getting",
        "while",
    }
    return [token for token in re.findall(r"[A-Za-z0-9_]+", normalize_whitespace(text).lower()) if token not in stopwords]


def ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]


def should_try_similarity_fallback(question: str, predicted_intent: str, slots: Dict[str, str]) -> bool:
    if slots.get("bug_id"):
        return False
    text = normalize_whitespace(question).lower()
    symptom_words = ["unresponsive", "invalid", "error", "crash", "crashes", "flapping", "instability", "reachable"]
    if predicted_intent in {"bug_symptom", "bug_workaround", "bug_scenario"}:
        return True
    if any(word in text for word in symptom_words):
        return True
    if "how do i fix" in text or "what should i do" in text:
        return True
    return False


def resolve_similarity_fallback(question: str, lookup_entries, predicted_intent: str) -> Tuple[Optional[object], float]:
    text = normalize_whitespace(question).lower()
    question_tokens = content_tokens(question)
    question_ngrams = set(ngrams(question_tokens, 3))
    phrase_bonuses = [
        "port flapping",
        "stp instability",
        "webserver becomes unresponsive",
        "afc is reachable",
        "firmware image is invalid",
        "authentication cycling",
    ]
    best_entry = None
    best_score = 0.0
    for entry in lookup_entries:
        if getattr(entry, "intent", "") != predicted_intent:
            continue
        answer_text = normalize_whitespace(getattr(entry, "answer", ""))
        input_text = normalize_whitespace(getattr(entry, "input_text", ""))
        answer_tokens = content_tokens(answer_text)
        input_tokens = content_tokens(input_text)
        raw_answer_score = SequenceMatcher(None, text, answer_text.lower()).ratio() if answer_text else 0.0
        raw_input_score = SequenceMatcher(None, text, input_text.lower()).ratio() if input_text else 0.0
        token_overlap = 0.0
        trigram_overlap = 0.0
        answer_token_set = set(answer_tokens)
        input_token_set = set(input_tokens)
        question_token_set = set(question_tokens)
        if question_token_set and answer_token_set:
            token_overlap = len(question_token_set & answer_token_set) / len(question_token_set | answer_token_set)
        answer_ngrams = set(ngrams(answer_tokens, 3))
        input_ngrams = set(ngrams(input_tokens, 3))
        if question_ngrams:
            trigram_overlap = max(
                len(question_ngrams & answer_ngrams) / len(question_ngrams),
                len(question_ngrams & input_ngrams) / len(question_ngrams),
            )
        score = max(raw_answer_score, raw_input_score) * 0.2 + token_overlap * 0.3 + trigram_overlap * 0.5
        for phrase in phrase_bonuses:
            if phrase in text and (phrase in answer_text.lower() or phrase in input_text.lower()):
                score += 0.08
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry, best_score


def log_row(log_path: Path, row: Dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer_question(
    question: str,
    lstm_model,
    lstm_tokenizer,
    lstm_config: Dict[str, object],
    lookup_entries,
    lookup_index: Dict[str, List[int]],
    availability_index: Dict[str, object],
    bug_metadata_index: Dict[str, List[object]],
    qwen_tokenizer,
    qwen_model,
    device: torch.device,
    session_context: Dict[str, Optional[str]],
) -> Dict[str, object]:
    cleaned_question = normalize_whitespace(question)
    routing_question = normalize_question_for_routing(cleaned_question)
    slots = extract_slots_from_question(routing_question)
    continuation_detected = is_followup_question(routing_question)
    metadata_kind = is_metadata_question(routing_question)
    followup_intent = detect_followup_intent(routing_question)
    pending_intent_used = False
    resolved_bug_id = ""
    lower_question = routing_question.lower()
    caveat_followup = not slots.get("bug_id") and any(
        phrase in lower_question
        for phrase in ["above limitation", "that limitation", "this limitation", "above caveat", "that caveat", "this caveat"]
    )
    session_bug_id = get_session_bug_id(session_context)
    availability_check = check_data_availability(slots, availability_index, bug_metadata_index)
    lstm_predicted_intent = predict_intent(routing_question, lstm_model, lstm_tokenizer, lstm_config, device)
    predicted_intent = lstm_predicted_intent

    if looks_like_bare_bug_id(routing_question) and not slots.get("bug_id"):
        slots["bug_id"] = routing_question
        pending_intent_used = True

    if not availability_check.get("available", True):
        predicted_intent = "data_not_available"
        return {
            "question": cleaned_question,
            "raw_lstm_intent": lstm_predicted_intent,
            "overridden_intent": predicted_intent,
            "final_intent": predicted_intent,
            "detected_bug_id": normalize_whitespace(slots.get("bug_id", "")),
            "continuation_used": bool(continuation_detected or caveat_followup or followup_intent),
            "pending_intent_used": pending_intent_used,
            "predicted_intent": predicted_intent,
            "slots": slots,
            "lookup_status": "data_not_available",
            "lookup_key_used": None,
            "lookup_answer": None,
            "qwen_answer": None,
            "qwen_validation_passed": False,
            "final_answer": DATA_NOT_AVAILABLE_RESPONSE,
            "answer_source": "deterministic_availability",
            "validation_reason": str(availability_check.get("reason", "data not available")),
            "availability_check": availability_check,
            "continuation_detected": continuation_detected,
            "resolved_bug_id": "",
        }

    if followup_intent:
        if session_bug_id:
            slots["bug_id"] = session_bug_id
            predicted_intent = followup_intent
            pending_intent_used = True
        else:
            predicted_intent = followup_intent

    predicted_intent = override_intent_by_question(routing_question, predicted_intent)
    if predicted_intent != lstm_predicted_intent:
        pending_intent_used = True

    if predicted_intent == "bug_workaround" and not is_workaround_request(routing_question):
        predicted_intent = "bug_workaround"

    if continuation_detected and not slots.get("bug_id") and not caveat_followup:
        resolved_bug_id = session_bug_id
        if resolved_bug_id:
            slots["bug_id"] = resolved_bug_id

    if predicted_intent == "bug_metadata" and not slots.get("bug_id") and session_bug_id:
        slots["bug_id"] = session_bug_id
        resolved_bug_id = session_bug_id

    if predicted_intent == "bug_metadata" and not metadata_kind:
        metadata_kind = "switch_version"
    if predicted_intent == "bug_metadata" and followup_intent == "bug_metadata" and normalize_followup_key(routing_question) == "category":
        metadata_kind = "category_only"

    defaults = {
        key: str(lstm_config.get(key, ""))
        for key in ("default_switch", "default_version", "default_sub_version")
        if str(lstm_config.get(key, ""))
    }

    lookup_status = "error"
    lookup_key_used = None
    lookup_answer = ""
    qwen_answer = None
    qwen_validation_passed = False
    qwen_used = False
    validation_reason = ""
    final_answer = ""
    answer_source = "lookup_fallback"
    source_type = predicted_intent
    data_family = "release_notes"

    if predicted_intent == "bug_metadata":
        bug_id = normalize_whitespace(slots.get("bug_id", "")) or session_bug_id
        if not bug_id:
            lookup_status = "slot_missing"
            validation_reason = "bug_id missing for metadata question"
            final_answer = "I need the bug ID to answer this."
        else:
            bug_entries = bug_metadata_index.get(bug_id, [])
            if not bug_entries:
                lookup_status = "not_found"
                lookup_key_used = f"metadata|{bug_id}"
                validation_reason = "bug id not found"
                suggested = suggest_bug_id(bug_id, lookup_entries)
                if suggested and suggested != bug_id:
                    final_answer = f"I could not find Bug {bug_id}. Did you mean Bug {suggested}?"
                else:
                    final_answer = f"I could not find Bug {bug_id} in the current release-note dataset."
            else:
                best_entry = choose_metadata_entry(routing_question, slots, bug_entries) or bug_entries[0]
                lookup_status = "found"
                lookup_key_used = f"metadata|{bug_id}|{metadata_kind or 'switch_version'}"
                lookup_answer = format_bug_metadata_answer(bug_id, best_entry, metadata_kind or "switch_version")
                final_answer = lookup_answer
                answer_source = "deterministic_metadata"
                validation_reason = "metadata lookup"
                update_session_context(
                    session_context,
                    {
                        "bug_id": bug_id,
                        "switch": best_entry.switch,
                        "version": best_entry.version,
                        "sub_version": best_entry.sub_version,
                        "feature": best_entry.feature,
                        "category": best_entry.category,
                    },
                    predicted_intent,
                    lookup_answer,
                )
    elif predicted_intent in STRICT_BUG_INTENTS:
        resolution = resolve_lookup_answer(predicted_intent, slots, routing_question, lookup_entries, lookup_index, defaults)
        lookup_status = str(resolution.get("status", "error"))
        lookup_key_used = resolution.get("lookup_key_used")
        lookup_answer = normalize_whitespace(resolution.get("answer", "")) if resolution.get("answer") else ""
        validation_reason = str(resolution.get("reason", lookup_status))

        if lookup_status in {"slot_missing", "not_found", "low_similarity"} and not slots.get("bug_id"):
            similarity_entry, similarity_score = resolve_similarity_fallback(routing_question, lookup_entries, predicted_intent)
            if similarity_entry is not None and similarity_score >= SIMILARITY_FALLBACK_THRESHOLD:
                lookup_status = "found"
                lookup_key_used = f"text_similarity|{getattr(similarity_entry, 'intent', '')}|{getattr(similarity_entry, 'bug_id', '') or getattr(similarity_entry, 'entry_id', '')}"
                lookup_answer = normalize_whitespace(getattr(similarity_entry, "answer", ""))
                for key in ["bug_id", "switch", "version", "sub_version", "feature", "category"]:
                    value = normalize_whitespace(getattr(similarity_entry, key, ""))
                    if value and not slots.get(key):
                        slots[key] = value
                pending_intent_used = True
                validation_reason = "same-intent similarity fallback"

        if predicted_intent == "bug_workaround":
            exact_workaround_key = bool(lookup_key_used) and str(lookup_key_used).startswith("bug_workaround|")
            if lookup_status != "found" or not exact_workaround_key:
                lookup_status = "not_found"
                lookup_answer = ""
                lookup_key_used = lookup_key_used if exact_workaround_key else lookup_key_used
                final_answer = "No workaround is documented in the release notes."
                answer_source = "lookup_fallback"
                validation_reason = "workaround fallback"
            else:
                final_answer = lookup_answer
                answer_source = "deterministic_lookup"
                update_session_context(session_context, slots, predicted_intent, lookup_answer)
        else:
            if lookup_status == "found" and lookup_answer:
                final_answer = lookup_answer
                answer_source = "deterministic_lookup"
                update_session_context(session_context, slots, predicted_intent, lookup_answer)
            elif continuation_detected and not slots.get("bug_id") and not session_bug_id:
                final_answer = "I need the bug ID to answer this."
            elif lookup_status in {"not_found", "low_similarity"} and slots.get("bug_id"):
                suggested = suggest_bug_id(slots["bug_id"], lookup_entries)
                if suggested and suggested != slots["bug_id"]:
                    final_answer = f"I could not find Bug {slots['bug_id']}. Did you mean Bug {suggested}?"
                else:
                    final_answer = f"I could not find Bug {slots['bug_id']} in the current release-note dataset."
            else:
                final_answer = deterministic_message_for_status(lookup_status)
    else:
        if caveat_followup and normalize_whitespace(session_context.get("last_lookup_answer", "")) and session_context.get("last_intent") == "release_caveat":
            lookup_status = "found"
            lookup_key_used = "context|last_lookup_answer"
            lookup_answer = normalize_whitespace(session_context.get("last_lookup_answer", ""))
            final_answer = lookup_answer
            answer_source = "deterministic_lookup"
            validation_reason = "context reuse"
        else:
            resolution = resolve_lookup_answer(predicted_intent, slots, routing_question, lookup_entries, lookup_index, defaults)
            lookup_status = str(resolution.get("status", "error"))
            lookup_key_used = resolution.get("lookup_key_used")
            lookup_answer = normalize_whitespace(resolution.get("answer", "")) if resolution.get("answer") else ""
            validation_reason = str(resolution.get("reason", lookup_status))
            if lookup_status == "found" and lookup_answer:
                final_answer = lookup_answer
                answer_source = "deterministic_lookup"
                update_session_context(session_context, slots, predicted_intent, lookup_answer)
            elif continuation_detected and not slots.get("bug_id") and not session_bug_id and predicted_intent == "bug_metadata":
                final_answer = "I need the bug ID to answer this."
                validation_reason = "bug_id missing"
            elif lookup_status in {"not_found", "low_similarity"} and slots.get("bug_id"):
                suggested = suggest_bug_id(slots["bug_id"], lookup_entries)
                if suggested and suggested != slots["bug_id"]:
                    final_answer = f"I could not find Bug {slots['bug_id']}. Did you mean Bug {suggested}?"
                else:
                    final_answer = f"I could not find Bug {slots['bug_id']} in the current release-note dataset."
            else:
                final_answer = deterministic_message_for_status(lookup_status)

    if lookup_status == "found" and lookup_answer and is_cli_syntax_answer(lookup_answer, predicted_intent, cleaned_question):
        final_answer = format_cli_syntax_answer(cleaned_question, lookup_answer, {"intent": predicted_intent, **slots})
        answer_source = "deterministic_cli_syntax"
    elif (
        lookup_status == "found"
        and lookup_answer
        and predicted_intent in RELEASE_QWEN_INTENTS
        and not is_no_workaround_answer(lookup_answer)
    ):
        if _should_use_release_qwen(predicted_intent, lookup_status, lookup_answer, 1.0, cleaned_question):
            prompt = build_prompt(
                cleaned_question,
                predicted_intent,
                slots,
                lookup_answer,
                source_type=source_type,
                data_family=data_family,
            )
            try:
                qwen_used = True
                qwen_answer = generate_qwen_answer(
                    qwen_tokenizer,
                    qwen_model,
                    prompt,
                    predicted_intent,
                    device,
                    data_family=data_family,
                )
                qwen_validation_passed, _reason = validate_qwen_answer(
                    predicted_intent,
                    slots,
                    lookup_answer,
                    qwen_answer,
                    data_family=data_family,
                )
                if qwen_validation_passed:
                    final_answer = qwen_answer
                    answer_source = "qwen_grounded"
                else:
                    final_answer = lookup_answer
                    answer_source = "lookup_fallback"
            except Exception:
                qwen_answer = None
                qwen_validation_passed = False
                final_answer = lookup_answer
                answer_source = "lookup_fallback"

    return {
        "question": cleaned_question,
        "raw_lstm_intent": lstm_predicted_intent,
        "overridden_intent": predicted_intent,
        "final_intent": predicted_intent,
        "detected_bug_id": "" if caveat_followup and not slots.get("bug_id") else (normalize_whitespace(slots.get("bug_id", "")) or resolved_bug_id),
        "continuation_used": bool(continuation_detected or resolved_bug_id or caveat_followup or followup_intent),
        "pending_intent_used": pending_intent_used or predicted_intent != lstm_predicted_intent,
        "predicted_intent": predicted_intent,
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
        "validation_reason": validation_reason,
        "availability_check": availability_check,
        "continuation_detected": continuation_detected,
        "resolved_bug_id": resolved_bug_id,
    }


def print_debug(result: Dict[str, object]) -> None:
    slots = result.get("slots", {}) if isinstance(result.get("slots"), dict) else {}
    availability = result.get("availability_check", {}) if isinstance(result.get("availability_check"), dict) else {}
    print(f"Extracted switch: {slots.get('switch') or 'None'}")
    print(f"Extracted version: {slots.get('version') or 'None'}")
    print(f"Extracted bug_id: {slots.get('bug_id') or 'None'}")
    print(
        "Availability check: "
        f"{availability.get('status', 'unknown')} ({availability.get('reason') or 'no reason'})"
    )
    print(f"Raw LSTM intent: {result.get('raw_lstm_intent')}")
    print(f"Final intent after override: {result.get('final_intent') or result.get('predicted_intent')}")
    print(f"Answer source: {result.get('answer_source')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Release Notes QA chat.")
    parser.add_argument("--qwen_model_path", type=Path, default=DEFAULT_QWEN_MODEL_PATH)
    parser.add_argument("--lstm_model_path", type=Path, default=DEFAULT_LSTM_MODEL_PATH)
    parser.add_argument("--lookup_index_path", type=Path, default=DEFAULT_LOOKUP_INDEX_PATH)
    parser.add_argument("--lookup_data_path", type=Path, default=DEFAULT_LOOKUP_DATA_PATH)
    parser.add_argument("--output_log_path", type=Path, default=DEFAULT_OUTPUT_LOG)
    parser.add_argument("--test_results_path", type=Path, default=DEFAULT_TEST_RESULTS_PATH)
    parser.add_argument("--availability_index_path", type=Path, default=DEFAULT_AVAILABILITY_INDEX_PATH)
    parser.add_argument("--bug_metadata_index_path", type=Path, default=DEFAULT_BUG_METADATA_INDEX_PATH)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--show_debug", action="store_true")
    parser.add_argument("--run_test_questions", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    lookup_entries, lookup_index = load_lookup_resources(args.lookup_index_path, args.lookup_data_path)
    availability_index = load_or_build_availability_index(args.availability_index_path, lookup_entries)
    bug_metadata_index = load_or_build_bug_metadata_index(args.bug_metadata_index_path, lookup_entries)
    lstm_model, lstm_tokenizer, lstm_config = load_lstm_support(args.lstm_model_path, device)
    qwen_tokenizer, qwen_model = None, None
    qwen_meta = {"resolved_path": str(args.qwen_model_path), "model_kind": "disabled"}
    try:
        qwen_tokenizer, qwen_model, qwen_meta = load_qwen_model(args.qwen_model_path, device)
    except Exception as exc:
        qwen_meta = {
            "resolved_path": str(args.qwen_model_path),
            "model_kind": "disabled",
            "error": str(exc),
        }
    session_context: Dict[str, Optional[str]] = {
        "last_bug_id": None,
        "last_valid_bug_id": None,
        "last_switch": None,
        "last_version": None,
        "last_sub_version": None,
        "last_feature": None,
        "last_category": None,
        "last_intent": None,
        "last_lookup_answer": None,
    }

    print("Release Notes QA Chat")
    if args.run_test_questions:
        args.test_results_path.parent.mkdir(parents=True, exist_ok=True)
        args.test_results_path.write_text("", encoding="utf-8")
        test_questions = TEST_QUESTIONS
    else:
        print("Type your question. Type exit/quit/q to stop.")
        test_questions = None

    def handle_question(question: str) -> Optional[Dict[str, object]]:
        result = answer_question(
            question,
            lstm_model,
            lstm_tokenizer,
            lstm_config,
            lookup_entries,
            lookup_index,
            availability_index,
            bug_metadata_index,
            qwen_tokenizer,
            qwen_model,
            device,
            session_context,
        )
        result["qwen_model_resolved_path"] = qwen_meta["resolved_path"]
        result["qwen_model_kind"] = qwen_meta["model_kind"]
        return result

    if args.run_test_questions:
        for question in test_questions:
            if question.lower() in {"exit", "quit", "q"}:
                break
            result = handle_question(question)
            log_row(
                args.test_results_path,
                {
                    "question": result["question"],
                    "detected_bug_id": result["detected_bug_id"],
                    "continuation_used": result["continuation_used"],
                    "pending_intent_used": result["pending_intent_used"],
                    "predicted_intent": result["predicted_intent"],
                    "slots": result["slots"],
                    "lookup_status": result["lookup_status"],
                    "lookup_key_used": result["lookup_key_used"],
                    "lookup_answer": result["lookup_answer"],
                    "qwen_answer": result["qwen_answer"],
                    "qwen_validation_passed": result["qwen_validation_passed"],
                    "final_answer": result["final_answer"],
                    "answer_source": result["answer_source"],
                },
            )
            print(f"You: {question}")
            print()
            if args.show_debug:
                print_debug(result)
            print("Assistant:")
            print(result["final_answer"])
            print("\n--------------------------------------------------")
        return

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        result = handle_question(question)

        log_row(
            args.output_log_path,
            {
                "question": result["question"],
                "detected_bug_id": result["detected_bug_id"],
                "continuation_used": result["continuation_used"],
                "pending_intent_used": result["pending_intent_used"],
                "predicted_intent": result["predicted_intent"],
                "slots": result["slots"],
                "lookup_status": result["lookup_status"],
                "lookup_key_used": result["lookup_key_used"],
                "lookup_answer": result["lookup_answer"],
                "qwen_answer": result["qwen_answer"],
                "qwen_validation_passed": result["qwen_validation_passed"],
                "final_answer": result["final_answer"],
                "answer_source": result["answer_source"],
            },
        )

        if args.show_debug:
            print_debug(result)

        print("\nAssistant:")
        print(result["final_answer"])

    print("Chat ended.")


if __name__ == "__main__":
    main()
