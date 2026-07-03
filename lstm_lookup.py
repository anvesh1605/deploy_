from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

NO_MATCH_RESPONSE = "No matching answer was found in the current release-note dataset."
DATA_NOT_AVAILABLE_RESPONSE = "This particular data is not available in the current Aruba switch dataset."
LOW_SIMILARITY_THRESHOLD = 0.56
DISAMBIGUATION_GAP = 0.05


@dataclass(frozen=True)
class LookupEntry:
    entry_id: int
    intent: str
    input_text: str
    answer: str
    slots: Dict[str, str]
    switch: str = ""
    version: str = ""
    sub_version: str = ""
    bug_id: str = ""
    feature: str = ""
    category: str = ""
    question_type: str = ""


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_whitespace(text: object) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_key_component(text: object) -> str:
    return normalize_whitespace(text)


def normalize_question_for_similarity(text: object) -> str:
    value = normalize_whitespace(text).lower()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize(text: object) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", normalize_question_for_similarity(text))


def jaccard_similarity(left: object, right: object) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def version_from_parts(major: str, minor: str) -> str:
    return f"{major}_{minor}"


SWITCH_MODEL_PATTERN = r"(?:CX\d{4}|\d{4,5}[A-Za-z]?)"


def clean_feature_text(feature: str) -> str:
    cleaned = normalize_whitespace(feature)
    cleaned = cleaned.rstrip(" ?.")
    cleaned = re.sub(
        r"\s+in\s+(?:HPE\s+Aruba\s+Networking\s+|HPE\s+Aruba\s+|Aruba\s+)?AOS-CX\s+\d+\.\d+\.\d+.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+in\s+the\s+same\s+release.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,?.")
    return cleaned


def extract_switch_version_slots(question: str) -> Dict[str, str]:
    text = normalize_whitespace(question)
    slots: Dict[str, str] = {}

    patterns = [
        rf"\b(?:For\s+)?(?P<switch>{SWITCH_MODEL_PATTERN})\s+Switch\s+Series\s+(?:running\s+)?AOS-CX\s+(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<sub>\d+))?\b",
        rf"\b(?:For\s+)?(?:an?\s+|the\s+)?(?P<switch>{SWITCH_MODEL_PATTERN})\s+(?:Switch\s+Series\s+)?(?:running\s+)?AOS-CX\s+(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<sub>\d+))?\b",
    ]
    for pattern in patterns:
        versioned_match = re.search(pattern, text, flags=re.IGNORECASE)
        if versioned_match:
            slots["switch"] = versioned_match.group("switch")
            slots["version"] = version_from_parts(versioned_match.group("major"), versioned_match.group("minor"))
            sub_version = versioned_match.group("sub")
            if sub_version:
                slots["sub_version"] = sub_version
            break
    return slots


def extract_bug_id(question: str) -> str:
    text = normalize_whitespace(question)
    if re.fullmatch(r"\d{4,7}", text):
        return text
    match = re.search(r"\bBug\s+ID\s+(\d+)\b", text, flags=re.IGNORECASE) or re.search(
        r"\bBug\s+ID\s+is\s+(\d+)\b", text, flags=re.IGNORECASE
    ) or re.search(
        r"\bBug\s+(\d+)\b", text, flags=re.IGNORECASE
    )
    return match.group(1) if match else ""


def extract_category(question: str) -> str:
    text = normalize_whitespace(question)
    match = re.search(r"\b(?:in|does)\s+(.+?)\s+Bug\s+\d+\b", text, flags=re.IGNORECASE)
    if not match:
        return ""
    category = normalize_whitespace(match.group(1)).strip(" ,?")
    if not category or re.search(r"\bAOS-CX\b", category, flags=re.IGNORECASE):
        return ""
    if category.lower() in {"the", "this", "that", "a", "an"}:
        return ""
    if re.search(r"\b(version|switch|bug|belong|from)\b", category, flags=re.IGNORECASE):
        return ""
    return category


def extract_question_type(question: str) -> str:
    text = normalize_whitespace(question).lower()
    if "limitation" in text:
        return "limitation"
    if "requirement" in text:
        return "requirement"
    if "caveat" in text:
        return "caveat"
    return ""


def extract_feature(question: str) -> str:
    text = normalize_whitespace(question)
    patterns = [
        r"\bwhat\s+caveat\s+is\s+documented\s+for\s+(?P<feature>.+?)(?:[?!.]?$)",
        r"\bwhat\s+limitation\s+is\s+mentioned\s+for\s+(?P<feature>.+?)(?:[?!.]?$)",
        r"\bwhat\s+is\s+the\s+caveat\s+related\s+to\s+(?P<feature>.+?)(?:[?!.]?$)",
        r"\bwhat\s+(?P<feature>.+?)\s+caveat\s+is\s+documented(?:\s+for)?(?:[?!.]?$)",
        r"\bwhat\s+(?P<feature>.+?)\s+limitation\s+is\s+mentioned(?:\s+for)?(?:[?!.]?$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            feature = clean_feature_text(match.group("feature"))
            feature = re.sub(r"\s+(?:caveat|limitation|mentioned|documented|related to)\b.*$", "", feature, flags=re.IGNORECASE)
            feature = clean_feature_text(feature)
            if feature and not re.search(r"\bBug\b", feature, flags=re.IGNORECASE) and not re.search(
                r"\bAOS-CX\b", feature, flags=re.IGNORECASE
            ):
                return feature
    return ""


def extract_slots_from_question(question: str) -> Dict[str, str]:
    slots = extract_switch_version_slots(question)
    bug_id = extract_bug_id(question)
    if bug_id:
        slots["bug_id"] = bug_id

    category = extract_category(question)
    if category:
        slots["category"] = category

    feature = extract_feature(question)
    if feature:
        slots["feature"] = feature

    question_type = extract_question_type(question)
    if question_type:
        slots["question_type"] = question_type

    return slots


def normalize_record_slots(slots: object) -> Dict[str, str]:
    if not isinstance(slots, dict):
        return {}
    normalized: Dict[str, str] = {}
    for key, value in slots.items():
        value_text = normalize_key_component(value)
        if value_text:
            normalized[str(key)] = value_text
    return normalized


def get_record_answer(record: Dict[str, object]) -> str:
    text = normalize_whitespace(record.get("target_value", "")) or normalize_whitespace(record.get("reference", ""))
    if text:
        return text
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if normalize_whitespace(message.get("role", "")).lower() != "assistant":
                continue
            text = normalize_whitespace(message.get("content", ""))
            if text:
                return text
    return ""


def get_record_input_text(record: Dict[str, object]) -> str:
    text = normalize_whitespace(record.get("input_text", ""))
    if text:
        return text
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if normalize_whitespace(message.get("role", "")).lower() != "user":
                continue
            text = normalize_whitespace(message.get("content", ""))
            if text:
                return text
    return ""


def infer_record_intent(record: Dict[str, object], input_text: str) -> str:
    explicit_intent = normalize_whitespace(record.get("intent", ""))
    if explicit_intent:
        return explicit_intent

    source_type = normalize_whitespace(record.get("source_type", "")).lower()
    question = normalize_whitespace(input_text).lower()

    if "release_notes_caveats" in source_type:
        return "release_caveat"

    if "release_notes_resolved_issues" in source_type:
        if "category" in question:
            return "bug_category"
        if "workaround" in question:
            return "bug_workaround"
        if "scenario" in question or "under what scenario" in question:
            return "bug_scenario"
        if "symptom" in question or "what issue was resolved" in question or "what issue occurs" in question:
            return "bug_symptom"
        if "fix" in question or "resolve" in question:
            return "bug_workaround"

    return explicit_intent


def build_lookup_entries(records: Sequence[Dict[str, object]]) -> List[LookupEntry]:
    entries: List[LookupEntry] = []
    for idx, record in enumerate(records):
        input_text = get_record_input_text(record)
        answer = get_record_answer(record)
        intent = infer_record_intent(record, input_text)
        slots = normalize_record_slots(record.get("slots"))
        if not slots:
            for key in ("switch", "version", "sub_version", "bug_id", "feature", "category", "question_type", "section"):
                value = normalize_key_component(record.get(key, ""))
                if value:
                    slots[key] = value
        question_type = extract_question_type(input_text)
        entry = LookupEntry(
            entry_id=idx,
            intent=intent,
            input_text=input_text,
            answer=answer,
            slots=slots,
            switch=slots.get("switch", ""),
            version=slots.get("version", ""),
            sub_version=slots.get("sub_version", ""),
            bug_id=slots.get("bug_id", ""),
            feature=slots.get("feature", ""),
            category=slots.get("category", ""),
            question_type=slots.get("question_type", question_type),
        )
        entries.append(entry)
    return entries


def lookup_key_candidates(intent: str, slots: Dict[str, str], question_type: str = "") -> List[str]:
    intent = normalize_whitespace(intent)
    switch = normalize_key_component(slots.get("switch", ""))
    version = normalize_key_component(slots.get("version", ""))
    sub_version = normalize_key_component(slots.get("sub_version", ""))
    bug_id = normalize_key_component(slots.get("bug_id", ""))
    feature = normalize_key_component(slots.get("feature", ""))
    category = normalize_key_component(slots.get("category", ""))
    qtype = normalize_key_component(question_type or slots.get("question_type", ""))

    candidates: List[str] = []

    if intent == "release_caveat":
        if switch and version and sub_version and feature and qtype:
            candidates.append("|".join([intent, switch, version, sub_version, feature, qtype]))
        if switch and version and sub_version and feature:
            candidates.append("|".join([intent, switch, version, sub_version, feature]))
        if switch and version and feature and qtype:
            candidates.append("|".join([intent, switch, version, feature, qtype]))
        if switch and version and feature:
            candidates.append("|".join([intent, switch, version, feature]))
        if feature and qtype:
            candidates.append("|".join([intent, feature, qtype]))
        if feature:
            candidates.append("|".join([intent, feature]))
        return candidates

    if intent.startswith("bug_"):
        if switch and version and sub_version and bug_id:
            candidates.append("|".join([intent, switch, version, sub_version, bug_id]))
        if bug_id:
            candidates.append("|".join([intent, bug_id]))
        if switch and version and sub_version and category and bug_id:
            candidates.append("|".join([intent, switch, version, sub_version, category, bug_id]))
        if category and bug_id:
            candidates.append("|".join([intent, category, bug_id]))
        return candidates

    return candidates


def build_lookup_index(entries: Sequence[LookupEntry]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = defaultdict(list)
    for entry in entries:
        slots = dict(entry.slots)
        slots.setdefault("switch", entry.switch)
        slots.setdefault("version", entry.version)
        slots.setdefault("sub_version", entry.sub_version)
        slots.setdefault("bug_id", entry.bug_id)
        slots.setdefault("feature", entry.feature)
        slots.setdefault("category", entry.category)
        slots.setdefault("question_type", entry.question_type)
        for key in lookup_key_candidates(entry.intent, slots, entry.question_type):
            if entry.entry_id not in index[key]:
                index[key].append(entry.entry_id)
    return dict(index)


def build_availability_index(entries: Sequence[LookupEntry]) -> Dict[str, object]:
    release_notes: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(lambda: {"versions": defaultdict(set)})
    product_docs: Dict[str, Dict[str, List[str]]] = {}

    for entry in entries:
        switch = normalize_key_component(entry.switch)
        version = normalize_key_component(entry.version)
        sub_version = normalize_key_component(entry.sub_version)
        if switch and version:
            release_notes[switch]["versions"][version].add(sub_version or "")

    normalized_release_notes: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for switch, payload in release_notes.items():
        versions = payload.get("versions", {})
        normalized_release_notes[switch] = {
            "versions": {version: sorted(value for value in values if value) for version, values in versions.items()}
        }

    return {"release_notes": normalized_release_notes, "product_docs": product_docs}


def build_bug_metadata_index(entries: Sequence[LookupEntry]) -> Dict[str, List[Dict[str, str]]]:
    index: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for entry in entries:
        bug_id = normalize_key_component(entry.bug_id)
        if not bug_id:
            continue
        index[bug_id].append(
            {
                "switch": normalize_key_component(entry.switch),
                "version": normalize_key_component(entry.version),
                "sub_version": normalize_key_component(entry.sub_version),
                "intent": normalize_key_component(entry.intent),
                "feature": normalize_key_component(entry.feature),
                "category": normalize_key_component(entry.category),
                "question_type": normalize_key_component(entry.question_type),
            }
        )
    return dict(index)


def _load_json_index(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_index(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_or_build_availability_index(path: Path, entries: Sequence[LookupEntry]) -> Dict[str, object]:
    loaded = _load_json_index(path)
    if isinstance(loaded, dict) and loaded:
        if "release_notes" in loaded:
            return loaded
        return {"release_notes": loaded, "product_docs": {}}
    payload = build_availability_index(entries)
    _write_json_index(path, payload)
    return payload


def load_or_build_bug_metadata_index(path: Path, entries: Sequence[LookupEntry]) -> Dict[str, List[Dict[str, str]]]:
    loaded = _load_json_index(path)
    if isinstance(loaded, dict) and loaded:
        normalized: Dict[str, List[Dict[str, str]]] = {}
        for key, value in loaded.items():
            if isinstance(value, list):
                normalized[str(key)] = [dict(item) for item in value if isinstance(item, dict)]
            elif isinstance(value, dict):
                normalized[str(key)] = [dict(value)]
            else:
                normalized[str(key)] = []
        return normalized
    payload = build_bug_metadata_index(entries)
    _write_json_index(path, payload)
    return payload


def check_data_availability(
    slots: Dict[str, str],
    availability_index: Dict[str, object],
    bug_metadata_index: Dict[str, List[Dict[str, str]]],
) -> Dict[str, object]:
    switch = normalize_key_component(slots.get("switch", ""))
    version = normalize_key_component(slots.get("version", ""))
    sub_version = normalize_key_component(slots.get("sub_version", ""))
    bug_id = normalize_key_component(slots.get("bug_id", ""))

    release_notes = availability_index.get("release_notes", {})
    if not isinstance(release_notes, dict):
        release_notes = {}

    if switch and switch not in release_notes:
        return {
            "available": False,
            "status": "data_not_available",
            "reason": f"switch {switch} not in availability index",
        }

    if switch and version:
        switch_payload = release_notes.get(switch, {})
        versions = switch_payload.get("versions", {}) if isinstance(switch_payload, dict) else {}
        if version not in versions:
            return {
                "available": False,
                "status": "data_not_available",
                "reason": f"version {version} not available for switch {switch}",
            }

    bug_entries = bug_metadata_index.get(bug_id, []) if bug_id else []
    if bug_id and not bug_entries:
        return {
            "available": False,
            "status": "data_not_available",
            "reason": f"bug {bug_id} not found in bug metadata index",
        }

    if switch and version and bug_id:
        matched = any(
            normalize_key_component(entry.get("switch", "")) == switch
            and normalize_key_component(entry.get("version", "")) == version
            and (not sub_version or normalize_key_component(entry.get("sub_version", "")) == sub_version)
            for entry in bug_entries
        )
        if not matched:
            return {
                "available": False,
                "status": "data_not_available",
                "reason": f"bug {bug_id} not available for switch {switch} version {version}",
            }

    return {"available": True, "status": "available", "reason": None}


def merge_defaults(slots: Dict[str, str], defaults: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    effective = {key: normalize_key_component(value) for key, value in slots.items() if normalize_key_component(value)}
    if defaults:
        for key, value in defaults.items():
            value_text = normalize_key_component(value)
            if value_text and not effective.get(key):
                effective[key] = value_text
    return effective


def select_entries_for_key(index: Dict[str, List[int]], entries: Sequence[LookupEntry], key: str) -> List[LookupEntry]:
    entry_ids = index.get(key, [])
    return [entries[idx] for idx in entry_ids if 0 <= idx < len(entries)]


def score_entry(question: str, slots: Dict[str, str], entry: LookupEntry) -> float:
    question_text = normalize_question_for_similarity(question)
    candidate_text = normalize_question_for_similarity(entry.input_text)
    seq = SequenceMatcher(None, question_text, candidate_text).ratio()
    tok = jaccard_similarity(question_text, candidate_text)

    slot_bonus = 0.0
    if slots.get("bug_id") and entry.bug_id and slots["bug_id"] == entry.bug_id:
        slot_bonus += 0.15
    if slots.get("feature") and entry.feature and normalize_key_component(slots["feature"]).lower() == entry.feature.lower():
        slot_bonus += 0.15
    if slots.get("category") and entry.category and normalize_key_component(slots["category"]).lower() == entry.category.lower():
        slot_bonus += 0.05
    if slots.get("question_type") and entry.question_type and slots["question_type"] == entry.question_type:
        slot_bonus += 0.08
    if slots.get("sub_version") and entry.sub_version and slots["sub_version"] == entry.sub_version:
        slot_bonus += 0.05
    if slots.get("version") and entry.version and slots["version"] == entry.version:
        slot_bonus += 0.04
    if slots.get("switch") and entry.switch and slots["switch"].lower() == entry.switch.lower():
        slot_bonus += 0.04

    score = 0.58 * seq + 0.27 * tok + slot_bonus
    return min(1.0, score)


def rank_candidates(question: str, slots: Dict[str, str], candidates: Sequence[LookupEntry]) -> List[Tuple[LookupEntry, float]]:
    ranked = [(entry, score_entry(question, slots, entry)) for entry in candidates]
    ranked.sort(key=lambda item: (item[1], item[0].entry_id), reverse=True)
    return ranked


def choose_from_ranked(ranked: Sequence[Tuple[LookupEntry, float]]) -> Dict[str, object]:
    if not ranked:
        return {
            "answer": None,
            "lookup_key_used": None,
            "status": "not_found",
            "reason": "no lookup key matched",
            "confidence": 0.0,
            "similarity": 0.0,
        }

    best_entry, best_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    unique_answers = []
    for entry, _score in ranked:
        if entry.answer not in unique_answers:
            unique_answers.append(entry.answer)

    if len(unique_answers) == 1:
        return {
            "answer": best_entry.answer,
            "lookup_key_used": None,
            "status": "found",
            "reason": None,
            "confidence": best_score,
            "similarity": best_score,
            "selected_entry": best_entry,
        }

    if best_score < LOW_SIMILARITY_THRESHOLD:
        return {
            "answer": NO_MATCH_RESPONSE,
            "lookup_key_used": None,
            "status": "low_similarity",
            "reason": "best similarity below threshold",
            "confidence": best_score,
            "similarity": best_score,
        }

    if len(unique_answers) > 1 and (best_score - runner_up_score) <= DISAMBIGUATION_GAP:
        return {
            "answer": None,
            "lookup_key_used": None,
            "status": "needs_disambiguation",
            "reason": "multiple candidates have similar scores",
            "confidence": best_score,
            "similarity": best_score,
        }

    return {
        "answer": best_entry.answer,
        "lookup_key_used": None,
        "status": "found",
        "reason": None,
        "confidence": best_score,
        "similarity": best_score,
        "selected_entry": best_entry,
    }


def resolve_lookup_answer(
    intent: str,
    slots: Dict[str, str],
    question: str,
    entries: Sequence[LookupEntry],
    index: Dict[str, List[int]],
    defaults: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    effective_slots = merge_defaults(slots, defaults)
    intent = normalize_whitespace(intent)
    question_type = effective_slots.get("question_type", extract_question_type(question))
    if question_type:
        effective_slots.setdefault("question_type", question_type)

    if intent.startswith("bug_"):
        if not effective_slots.get("bug_id"):
            return {
                "answer": None,
                "lookup_key_used": None,
                "status": "slot_missing",
                "reason": "bug_id is required for bug intents",
                "confidence": 0.0,
                "similarity": 0.0,
            }
    if intent == "release_caveat" and not effective_slots.get("feature"):
        return {
            "answer": None,
            "lookup_key_used": None,
            "status": "slot_missing",
            "reason": "feature is required for release_caveat",
            "confidence": 0.0,
            "similarity": 0.0,
        }

    candidate_keys = lookup_key_candidates(intent, effective_slots, question_type)
    best_group: Optional[Tuple[str, Dict[str, object]]] = None
    best_status_rank = -1
    status_rank = {"found": 3, "needs_disambiguation": 2, "low_similarity": 1, "not_found": 0, "slot_missing": -1}

    for key in candidate_keys:
        candidates = select_entries_for_key(index, entries, key)
        if not candidates:
            continue
        ranked = rank_candidates(question, effective_slots, candidates)
        chosen = choose_from_ranked(ranked)
        chosen["lookup_key_used"] = key
        if chosen["status"] == "found":
            return chosen
        rank = status_rank.get(str(chosen["status"]), 0)
        if rank > best_status_rank:
            best_status_rank = rank
            best_group = (key, chosen)

    same_intent_candidates = [entry for entry in entries if entry.intent == intent]
    if same_intent_candidates:
        ranked = rank_candidates(question, effective_slots, same_intent_candidates)
        chosen = choose_from_ranked(ranked)
        chosen["lookup_key_used"] = "nearest_input_text_similarity"
        if chosen["status"] == "found":
            return chosen
        if chosen["status"] == "needs_disambiguation":
            return chosen
        if chosen["status"] == "low_similarity":
            return chosen

    if best_group is not None:
        return best_group[1]

    return {
        "answer": NO_MATCH_RESPONSE,
        "lookup_key_used": None,
        "status": "not_found",
        "reason": "no lookup key matched",
        "confidence": 0.0,
        "similarity": 0.0,
    }


def format_answer_for_intent(intent: str, answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    text = normalize_whitespace(answer)
    if intent == "bug_category":
        return text
    if intent == "bug_workaround":
        return text
    if intent == "bug_scenario":
        return text
    if intent == "bug_symptom":
        return text
    if intent == "release_caveat":
        return text
    return text


def build_eval_row(
    question: str,
    gold_intent: str,
    predicted_intent: str,
    slots: Dict[str, str],
    resolution: Dict[str, object],
    gold_answer: str,
) -> Dict[str, object]:
    answer = resolution.get("answer")
    row = {
        "question": question,
        "gold_intent": gold_intent,
        "predicted_intent": predicted_intent,
        "slots": slots,
        "lookup_key_used": resolution.get("lookup_key_used"),
        "answer": answer,
        "status": resolution.get("status"),
        "reason": resolution.get("reason"),
        "confidence": float(resolution.get("confidence", 0.0) or 0.0),
        "similarity": float(resolution.get("similarity", 0.0) or 0.0),
        "gold_answer": gold_answer,
        "correct": normalize_whitespace(answer) == normalize_whitespace(gold_answer),
    }
    return row


def aggregate_lookup_metrics(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    total = len(rows)
    correct = sum(1 for row in rows if bool(row.get("correct")))
    incorrect = total - correct
    status_counts = Counter(str(row.get("status", "not_found")) for row in rows)
    intent_counts = Counter(str(row.get("gold_intent", "")) for row in rows)
    accuracy_by_intent: Dict[str, Dict[str, float]] = {}
    errors_by_intent: Dict[str, Dict[str, int]] = {}

    for intent in sorted(intent_counts):
        intent_rows = [row for row in rows if str(row.get("gold_intent", "")) == intent]
        if not intent_rows:
            continue
        intent_correct = sum(1 for row in intent_rows if bool(row.get("correct")))
        accuracy_by_intent[intent] = {
            "total": len(intent_rows),
            "correct": intent_correct,
            "incorrect": len(intent_rows) - intent_correct,
            "accuracy": intent_correct / max(1, len(intent_rows)),
        }
        errors_by_intent[intent] = dict(Counter(str(row.get("status", "not_found")) for row in intent_rows))

    failed_rows = [row for row in rows if not bool(row.get("correct"))]
    key_counts = Counter(str(row.get("lookup_key_used")) for row in failed_rows if row.get("lookup_key_used"))
    top_failed_lookup_keys = [
        {"lookup_key_used": key, "count": count}
        for key, count in key_counts.most_common(20)
    ]

    def samples_for(status: str, limit: int = 10) -> List[Dict[str, object]]:
        return [row for row in rows if str(row.get("status")) == status][:limit]

    report = {
        "total_questions": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": correct / max(1, total),
        "found_count": status_counts.get("found", 0),
        "not_found_count": status_counts.get("not_found", 0),
        "needs_disambiguation_count": status_counts.get("needs_disambiguation", 0),
        "slot_missing_count": status_counts.get("slot_missing", 0),
        "low_similarity_count": status_counts.get("low_similarity", 0),
        "accuracy_by_intent": accuracy_by_intent,
        "errors_by_intent": errors_by_intent,
        "top_20_failed_lookup_keys": top_failed_lookup_keys,
        "sample_not_found": samples_for("not_found"),
        "sample_needs_disambiguation": samples_for("needs_disambiguation"),
        "sample_wrong_answers": failed_rows[:10],
    }
    return report


def render_lookup_errors_markdown(report: Dict[str, object], rows: Sequence[Dict[str, object]]) -> str:
    parts = ["# LSTM Lookup Errors", ""]
    parts.append(f"- Total questions: {report.get('total_questions', 0)}")
    parts.append(f"- Correct: {report.get('correct', 0)}")
    parts.append(f"- Incorrect: {report.get('incorrect', 0)}")
    parts.append(f"- Accuracy: {float(report.get('accuracy', 0.0)):.4f}")
    parts.append("")

    parts.append("## Top 20 Failed Lookup Keys")
    for item in report.get("top_20_failed_lookup_keys", []):
        parts.append(f"- {item['lookup_key_used']}: {item['count']}")
    parts.append("")

    def render_samples(title: str, samples: Sequence[Dict[str, object]]) -> None:
        parts.append(f"## {title}")
        if not samples:
            parts.append("No samples available.")
            parts.append("")
            return
        for row in samples:
            parts.append(f"- Question: {row.get('question', '')}")
            parts.append(f"  - Intent: `{row.get('gold_intent', '')}`")
            parts.append(f"  - Predicted: `{row.get('predicted_intent', '')}`")
            parts.append(f"  - Status: `{row.get('status', '')}`")
            parts.append(f"  - Lookup key: `{row.get('lookup_key_used', '')}`")
            parts.append(f"  - Answer: {row.get('answer', '')}")
            parts.append(f"  - Gold: {row.get('gold_answer', '')}")
        parts.append("")

    render_samples("Not Found", report.get("sample_not_found", []))
    render_samples("Needs Disambiguation", report.get("sample_needs_disambiguation", []))
    render_samples("Wrong Answers", report.get("sample_wrong_answers", []))

    parts.append("## Accuracy by Intent")
    for intent, item in report.get("accuracy_by_intent", {}).items():
        parts.append(
            f"- `{intent}`: accuracy={item['accuracy']:.4f}, correct={item['correct']}, total={item['total']}"
        )
    parts.append("")
    return "\n".join(parts).rstrip() + "\n"
