from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "outputs_product_question_tests"
DATASET_PATHS = [
    ROOT / "outputs_product_lstm_v2" / "product_lstm_dataset_patched.jsonl",
    ROOT / "outputs_product_lstm_v2" / "product_lstm_train_patched.jsonl",
    ROOT / "outputs_product_lstm_v2" / "product_lstm_val_patched.jsonl",
    ROOT / "outputs_product_lstm_v2" / "product_lstm_test_patched.jsonl",
    ROOT / "Data" / "product_docs_final_repair_focus" / "repaired_rows.jsonl",
    ROOT / "Data" / "product_docs_final_repair_focus" / "patched_product_dataset_repaired.jsonl",
    ROOT / "Data" / "product_docs_final_repair_focus" / "patched_all_switches_product_dataset_final.jsonl",
]

TARGET_COUNTS = {
    "cli_syntax": 6,
    "cli_meaning": 4,
    "show_command_syntax": 3,
    "show_command_meaning": 3,
    "configuration_procedure": 3,
    "concept_explanation": 4,
    "troubleshooting": 2,
    "product_limitation": 1,
    "product_requirement": 1,
    "product_caveat": 1,
    "event_log_meaning": 1,
    "snmp_mib_info": 1,
}

SEED_PATTERNS = [
    ("cli_syntax", ["redundancy switchover"]),
    ("cli_syntax", ["bfd", "ipv4 addr"]),
    ("cli_syntax", ["ip route", "bfd"]),
    ("cli_syntax", ["clear erps", "statistics"]),
    ("cli_syntax", ["ipv6 ospfv3", "bfd", "disable"]),
    ("cli_syntax", ["erps ring", "revertive"]),
    ("cli_meaning", ["redundancy switchover"]),
    ("cli_meaning", ["bfd"]),
    ("cli_meaning", ["ip route", "bfd"]),
    ("cli_meaning", ["clear erps", "statistics"]),
    ("show_command_syntax", ["show port-access port-security interface client-status"]),
    ("show_command_syntax", ["show interface"]),
    ("show_command_syntax", ["show lldp neighbor"]),
    ("show_command_meaning", ["show port-access port-security interface client-status"]),
    ("show_command_meaning", ["show aaa authentication port-access mac-auth interface client-status"]),
    ("show_command_meaning", ["show port-access port-security interface port-statistics"]),
    ("configuration_procedure", ["Configuring BFD for an IPv4 Static Route"]),
    ("configuration_procedure", ["Initial Configuration"]),
    ("configuration_procedure", ["Configuring subinterfaces"]),
    ("concept_explanation", ["High Availability Overview"]),
    ("concept_explanation", ["Management Module Failover Overview"]),
    ("concept_explanation", ["AAA on Switches with Multiple Management Modules"]),
    ("concept_explanation", ["BFD"]),
    ("troubleshooting", ["Operation not permitted"]),
    ("troubleshooting", ["Network is unreachable"]),
    ("product_limitation", ["Remote AAA with RADIUS"]),
    ("product_requirement", ["Mirroring"]),
    ("product_caveat", ["PKI"]),
    ("event_log_meaning", ["Event ID", "10002"]),
    ("snmp_mib_info", ["TrapOID"]),
]

GENERIC_QUESTION_PATTERNS = {
    "configuration_procedure": [
        r"how do you configure feature\b",
        r"how do you configure guidelines\b",
        r"how do you configure restrictions\b",
        r"how do you configure troubleshooting\b",
        r"how do you configure mirroring\b",
        r"how do you configure pki\b",
        r"how do you configure event logs\b",
        r"how do you configure snmp\b",
    ],
    "troubleshooting": [
        r"how do you troubleshoot troubleshooting\b",
        r"how do you troubleshoot supportability\b",
    ],
    "product_limitation": [
        r"what limitation is documented for guidelines\b",
        r"what limitation is documented for restrictions\b",
    ],
    "product_requirement": [
        r"what requirement is documented for mirroring\b",
    ],
    "event_log_meaning": [
        r"what does the guide say about event logs\b",
        r"what event log information is documented for event id\s*\d+\b",
    ],
    "snmp_mib_info": [
        r"what snmp mib information is documented for snmp\b",
        r"what snmp mib information is documented for guidelines and limitations\b",
    ],
}


@dataclass
class RowScore:
    score: float
    row: Dict[str, object]
    reason: str


def read_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def clean_cli_syntax(syntax_text: str) -> str:
    text = clean_text(syntax_text)
    if not text:
        return ""
    text = re.sub(r"^(?:syntax|command syntax)\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^(?:the syntax for .*?(?:command)?(?: is| is:)|the syntax of .*?(?:command)?(?: is| is:)|the command is|the command syntax is|command syntax is)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?<!\d)\.{3,}(?!\d)", " ", text)
    text = text.replace("`", "")
    text = re.sub(r"\s+(?:page|pg)\s*\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:\d{1,4}\s+)+", "", text)
    text = re.sub(r"^\s*[:\-–—]+\s*", "", text)
    text = re.sub(r"\s*\.{2,}\s*$", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" .")
    return text


def is_bad_syntax_artifact(text: str, expected_command: str = "") -> bool:
    candidate = clean_text(text)
    if not candidate:
        return True
    if re.search(r"\.{8,}", candidate):
        return True
    if candidate.startswith("."):
        return True
    if re.search(r"\b(?:page|pg)\s*\d+\b", candidate, flags=re.IGNORECASE):
        return True
    if re.search(r"\b\d{1,4}\s+[A-Za-z][A-Za-z0-9._/-]*(?:\s+[A-Za-z][A-Za-z0-9._/-]*){0,7}\b", candidate):
        return True
    cleaned = clean_cli_syntax(candidate)
    if not cleaned:
        return True
    if cleaned.count(".") > max(3, len(cleaned) // 4):
        return True
    if not re.search(r"[A-Za-z]", cleaned):
        return True
    expected = clean_text(expected_command)
    if expected:
        expected_tokens = [
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9._/-]*", expected.lower())
            if token not in {"the", "a", "an", "of", "for", "to", "on", "in"}
        ]
        cleaned_tokens = [
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9._/-]*", cleaned.lower())
            if token not in {"the", "a", "an", "of", "for", "to", "on", "in"}
        ]
        if expected_tokens and cleaned_tokens:
            idx = 0
            for token in cleaned_tokens:
                if token == expected_tokens[idx]:
                    idx += 1
                    if idx == len(expected_tokens):
                        break
            if idx < max(1, min(len(expected_tokens), 2)):
                return True
    return False


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(text).lower()).strip()


def answer_has_artifact(answer: str) -> bool:
    text = clean_text(answer)
    if not text:
        return True
    if text.startswith(".") or re.search(r"\.{8,}", text):
        return True
    if re.search(r"\bpublic table of contents\b", text, flags=re.IGNORECASE):
        return True
    if re.search(r"\btable of contents\b", text, flags=re.IGNORECASE):
        return True
    if re.search(r"\bchapter\s+\d+\b", text, flags=re.IGNORECASE) and len(text) > 250:
        return True
    if re.search(r"\bpage\s*\d+\b", text, flags=re.IGNORECASE):
        return True
    if text.strip().startswith("|") and text.count("|") > 4:
        return True
    if re.search(r"\|\s*show\s+[A-Za-z0-9_-]+", text, flags=re.IGNORECASE) and text.count("|") > 2:
        return True
    if text.count("|") > 10 and len(text) > 250:
        return True
    if text.count(".") > max(5, len(text) // 4):
        return True
    if text.endswith(":.") or text.endswith(".."):
        return True
    if len(text) < 8:
        return True
    return False


def question_quality_penalty(question: str, intent: str, topic: str, command: str) -> float:
    text = normalize_key(question)
    intent_text = clean_text(intent)
    topic_text = normalize_key(topic)
    command_text = normalize_key(command)
    penalty = 0.0

    for pattern in GENERIC_QUESTION_PATTERNS.get(intent_text, []):
        if re.search(pattern, text, flags=re.IGNORECASE):
            penalty += 80.0

    generic_topics = {
        "feature",
        "guidelines",
        "restrictions",
        "mirroring",
        "troubleshooting",
        "supportability",
        "event logs",
        "snmp",
        "pki",
    }

    if intent_text == "configuration_procedure":
        if topic_text in generic_topics or len(topic_text.split()) <= 2 and topic_text:
            penalty += 40.0
        if re.search(r"how do you configure\s+(?:feature|guidelines|restrictions|troubleshooting|mirroring|pki|snmp|event logs)\b", text):
            penalty += 80.0

    if intent_text == "troubleshooting":
        allowed = (
            "operation not permitted",
            "network is unreachable",
            "destination host unreachable",
            "bluetooth connections",
            "ping commands",
            "agent and script issues",
            "high switch cpu and memory usage",
        )
        if not any(item in text for item in allowed):
            penalty += 50.0

    if intent_text == "product_limitation":
        if topic_text in {"guidelines", "restrictions"}:
            penalty += 60.0
        if re.search(r"what limitation is documented for\s+(?:guidelines|restrictions)\b", text):
            penalty += 80.0

    if intent_text == "product_requirement":
        if topic_text in {"mirroring"} and "mirroring allows you to replicate" not in normalize_key(str(command_text)):
            penalty += 10.0

    if intent_text == "event_log_meaning":
        if not re.search(r"\bevent id\s*\d+\b", text):
            penalty += 30.0

    if intent_text == "snmp_mib_info":
        if topic_text in {"snmp", "guidelines and limitations"}:
            penalty += 70.0
        if not re.search(r"\b(?:trapoid|if-mib|v2-mib|power-ethernet-mib)\b", text):
            penalty += 20.0

    if intent_text == "cli_meaning":
        if any(marker in text for marker in ("ipv6 mld robustness", "mld snoop enabled on vlan", "guide say about")):
            penalty += 60.0

    if intent_text == "cli_syntax":
        if any(marker in text for marker in ("mld robustness", "mld snoop enabled on vlan", "guide say about")):
            penalty += 60.0

    return penalty


def is_command_like(answer: str) -> bool:
    text = clean_text(answer)
    if not text:
        return False
    if any(token in text for token in ("<", ">", "[", "]", "{", "}", "|")):
        return True
    return bool(re.fullmatch(r"(?:no\s+)?[A-Za-z0-9._/-]+(?:\s+[A-Za-z0-9._/-]+){0,10}", text))


def infer_data_family(intent: str) -> str:
    intent_text = clean_text(intent)
    if intent_text in {"cli_syntax", "show_command_syntax", "cli_meaning", "show_command_meaning", "configuration_procedure"}:
        return "product_documentation"
    if intent_text in {"product_limitation", "product_requirement", "product_caveat"}:
        return "product_documentation"
    if intent_text in {"concept_explanation", "troubleshooting", "event_log_meaning", "snmp_mib_info"}:
        return "product_documentation"
    return "product_documentation"


def extract_command_from_question(question: str, slots: Dict[str, object]) -> str:
    command = clean_text(slots.get("command", ""))
    if command:
        return command
    patterns = [
        r"\bwhat is the syntax of (?:the )?(?P<command>.+?) command\b",
        r"\bwhat does (?:the )?(?P<command>.+?) command do\b",
        r"\bwhat is the purpose of (?:the )?(?P<command>.+?) command\b",
        r"\bwhat is (?:the )?(?P<command>.+?) command\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return clean_text(match.group("command"))
    return ""


def format_expected_answer(intent: str, question: str, answer: str, slots: Dict[str, object]) -> str:
    intent_text = clean_text(intent)
    if intent_text in {"cli_syntax", "show_command_syntax"} or "syntax" in question.lower():
        command = extract_command_from_question(question, slots)
        syntax_source = answer
        backtick_match = re.search(r"`([^`]{2,})`", answer)
        if backtick_match:
            syntax_source = backtick_match.group(1)
        syntax = clean_cli_syntax(syntax_source)
        if is_bad_syntax_artifact(syntax or answer, command):
            return ""
        if syntax:
            return f"**Syntax**\n\n```text\n{syntax}\n```"
        return ""
    return clean_text(answer)


def count_artifacts(answer: str) -> Dict[str, int]:
    text = clean_text(answer)
    return {
        "dot_runs": len(re.findall(r"\.{8,}", text)),
        "page_refs": len(re.findall(r"\b(?:page|pg)\s*\d+\b", text, flags=re.IGNORECASE)),
        "toc_mentions": int(bool(re.search(r"\btable of contents\b", text, flags=re.IGNORECASE))),
        "starts_with_dots": int(text.startswith(".")),
    }


def score_row(row: Dict[str, object]) -> RowScore:
    question = clean_text(row.get("input_text") or row.get("question"))
    answer = clean_text(row.get("target_value") or row.get("answer") or row.get("reference"))
    intent = clean_text(row.get("intent"))
    slots = row.get("slots") if isinstance(row.get("slots"), dict) else {}
    topic = clean_text((slots or {}).get("topic", "")) if isinstance(slots, dict) else ""
    command = clean_text((slots or {}).get("command", "")) if isinstance(slots, dict) else ""
    if not question or not answer:
        return RowScore(-999.0, row, "missing question or answer")
    if answer_has_artifact(answer):
        return RowScore(-500.0, row, "answer has OCR/TOC artifact")
    question_penalty = question_quality_penalty(question, intent, topic, command)

    formatted_answer = format_expected_answer(intent, question, answer, slots)
    if not formatted_answer:
        return RowScore(-400.0, row, "syntax cleanup could not recover a safe answer")

    score = 0.0
    reasons: List[str] = []
    score += min(len(answer) / 40.0, 8.0)
    if len(answer.split()) > 12:
        score += 4.0
        reasons.append("substantive answer")
    if intent in TARGET_COUNTS:
        score += 12.0
        reasons.append(f"target intent {intent}")
    if intent in {"cli_syntax", "show_command_syntax"} and is_command_like(clean_text(formatted_answer).replace("**Syntax**", "")):
        score += 10.0
        reasons.append("clean command syntax")
    if intent in {"concept_explanation", "configuration_procedure", "show_command_meaning", "cli_meaning"} and len(answer.split()) >= 20:
        score += 8.0
        reasons.append("rich explanation")
    if row.get("document_title"):
        score += 2.0
    if row.get("section"):
        score += 2.0
    if row.get("source_file") or row.get("repair_source_file") or row.get("source_excerpt_file"):
        score += 2.0
    if formatted_answer.count("\n") >= 2:
        score += 2.0
    score -= question_penalty
    artifacts = count_artifacts(answer)
    score -= artifacts["dot_runs"] * 4.0
    score -= artifacts["page_refs"] * 2.0
    if intent in {"cli_syntax", "show_command_syntax"} and is_bad_syntax_artifact(formatted_answer, extract_command_from_question(question, slots)):
        return RowScore(-450.0, row, "syntax answer is not safe")

    reason = ", ".join(reasons) if reasons else "clean product-doc row"
    return RowScore(score, row, reason)


def match_seed(row: Dict[str, object]) -> Optional[Tuple[str, str]]:
    question = clean_text(row.get("input_text") or row.get("question"))
    intent = clean_text(row.get("intent"))
    if not question or not intent:
        return None
    qnorm = normalize_key(question)
    for seed_intent, patterns in SEED_PATTERNS:
        if intent != seed_intent:
            continue
        if all(normalize_key(pattern) in qnorm for pattern in patterns):
            return seed_intent, "seed pattern matched"
    return None


def select_rows(rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, int]]:
    scored: List[RowScore] = [score_row(row) for row in rows]
    scored.sort(key=lambda item: (item.score, len(clean_text(item.row.get("target_value") or ""))), reverse=True)

    selected: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    seen_questions = set()
    selected_intents = Counter()
    selected_topics = set()

    def push(row: Dict[str, object], reason: str) -> bool:
        question = clean_text(row.get("input_text") or row.get("question"))
        if not question:
            return False
        qkey = normalize_key(question)
        if qkey in seen_questions:
            return False
        selected.append(row)
        seen_questions.add(qkey)
        intent = clean_text(row.get("intent"))
        selected_intents[intent] += 1
        topic = clean_text((row.get("slots") or {}).get("topic") if isinstance(row.get("slots"), dict) else "")
        if topic:
            selected_topics.add(normalize_key(topic))
        row["_selection_reason"] = reason
        return True

    # First pass: grab seed rows for coverage.
    for seed_intent, patterns in SEED_PATTERNS:
        target_count = TARGET_COUNTS.get(seed_intent, 0)
        if target_count and selected_intents[seed_intent] >= target_count:
            continue
        for item in scored:
            if selected_intents[seed_intent] >= target_count:
                break
            row = item.row
            if clean_text(row.get("intent")) != seed_intent:
                continue
            question = clean_text(row.get("input_text") or row.get("question"))
            qnorm = normalize_key(question)
            if not all(normalize_key(pattern) in qnorm for pattern in patterns):
                continue
            if push(row, f"seed match: {patterns[0]}"):
                break

    # Second pass: fill target counts by intent.
    for item in scored:
        if len(selected) >= 30:
            break
        row = item.row
        intent = clean_text(row.get("intent"))
        if intent not in TARGET_COUNTS:
            continue
        if selected_intents[intent] >= TARGET_COUNTS[intent]:
            continue
        if item.score < 5.0:
            continue
        if push(row, item.reason):
            continue

    # Final pass: fill any remaining slots with the best available rows.
    for item in scored:
        if len(selected) >= 30:
            break
        row = item.row
        intent = clean_text(row.get("intent"))
        if intent not in TARGET_COUNTS:
            continue
        if selected_intents[intent] >= TARGET_COUNTS[intent]:
            continue
        if push(row, item.reason):
            continue

    # If we still did not get 30, continue with any clean rows.
    for item in scored:
        if len(selected) >= 30:
            break
        row = item.row
        if push(row, item.reason):
            continue

    # Build rejected list from the top of the scored set that did not make it.
    selected_keys = {normalize_key(clean_text(row.get("input_text") or row.get("question"))) for row in selected}
    for item in scored:
        question = clean_text(item.row.get("input_text") or item.row.get("question"))
        if not question:
            continue
        if normalize_key(question) in selected_keys:
            continue
        rejected.append(
            {
                "question": question,
                "intent": clean_text(item.row.get("intent")),
                "answer": clean_text(item.row.get("target_value") or item.row.get("answer") or item.row.get("reference")),
                "reject_reason": item.reason,
            }
        )
        if len(rejected) >= 80:
            break

    return selected[:30], rejected, dict(selected_intents)


def build_record(row: Dict[str, object]) -> Dict[str, object]:
    source_question = clean_text(row.get("input_text") or row.get("question"))
    intent = clean_text(row.get("intent"))
    slots = row.get("slots") if isinstance(row.get("slots"), dict) else {}
    answer = clean_text(row.get("target_value") or row.get("answer") or row.get("reference"))
    topic = clean_text(slots.get("topic", "")) if isinstance(slots, dict) else ""
    command = clean_text(slots.get("command", "")) if isinstance(slots, dict) else ""
    question = humanize_question(source_question, intent, topic, command)
    expected_answer = format_expected_answer(intent, source_question, answer, slots)
    source_file = clean_text(row.get("repair_source_file") or row.get("source_file") or row.get("source_excerpt_file"))
    source_excerpt_file = clean_text(row.get("source_excerpt_file") or row.get("repair_source_file") or row.get("source_file"))
    syntax = ""
    if intent in {"cli_syntax", "show_command_syntax"} and expected_answer:
        syntax = clean_cli_syntax(expected_answer)
        if "```text" in expected_answer:
            code_match = re.search(r"```text\n([\s\S]*?)```", expected_answer)
            if code_match:
                syntax = clean_cli_syntax(code_match.group(1))
    record = {
        "question": question,
        "source_question": source_question,
        "expected_answer": expected_answer,
        "intent": intent,
        "source_type": clean_text(row.get("source_type") or row.get("repair_source_type") or "product_documentation"),
        "data_family": infer_data_family(intent),
        "switch": clean_text(slots.get("switch", "")) if isinstance(slots, dict) else "",
        "version": clean_text(slots.get("version", "")) if isinstance(slots, dict) else "",
        "document_title": clean_text(row.get("document_title")),
        "section": clean_text(row.get("section")),
        "topic": clean_text(slots.get("topic", "")) if isinstance(slots, dict) else "",
        "command": command,
        "syntax": syntax,
        "quality_reason": row.get("_selection_reason", "clean product-doc row"),
        "source_file": source_file,
        "source_excerpt_file": source_excerpt_file,
    }
    return record


def markdown_block(text: str) -> str:
    return f"```text\n{text}\n```"


def render_markdown_answer(item: Dict[str, object]) -> str:
    intent = clean_text(item.get("intent"))
    expected = clean_text(item.get("expected_answer"))
    syntax = clean_text(item.get("syntax"))
    if intent in {"cli_syntax", "show_command_syntax"}:
        safe_syntax = syntax or clean_cli_syntax(expected)
        if safe_syntax:
            return markdown_block(safe_syntax)
    if expected.count("\n") >= 1:
        return expected
    return expected


def humanize_question(question: str, intent: str, topic: str = "", command: str = "") -> str:
    text = clean_text(question)
    intent_text = clean_text(intent)
    topic_text = clean_text(topic)
    command_text = clean_text(command)
    if not text:
        return text

    if intent_text == "configuration_procedure":
        text = re.sub(
            r"(how do you configure\s+)Configuring\s+",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )
        if topic_text.lower().startswith("configuring "):
            topic_text = topic_text[len("Configuring ") :]
        if topic_text.lower() == "initial configuration":
            return re.sub(
                r"how do you configure\s+.+?\?",
                "how do you perform the initial configuration?",
                text,
                flags=re.IGNORECASE,
            )
        if topic_text:
            return re.sub(
                r"how do you configure\s+.+?\?",
                f"how do you configure {topic_text}?",
                text,
                flags=re.IGNORECASE,
            )

    if intent_text == "troubleshooting":
        text = re.sub(
            r"(how do you troubleshoot\s+)Troubleshooting\s+",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )
        if topic_text.lower().startswith("troubleshooting "):
            topic_text = topic_text[len("Troubleshooting ") :]
        if topic_text:
            return re.sub(
                r"how do you troubleshoot\s+.+?\?",
                f"how do you troubleshoot {topic_text}?",
                text,
                flags=re.IGNORECASE,
            )
        if command_text:
            return re.sub(
                r"how do you troubleshoot\s+.+?\?",
                f"how do you troubleshoot {command_text}?",
                text,
                flags=re.IGNORECASE,
            )

    return text


def render_markdown(items: Sequence[Dict[str, object]]) -> str:
    parts: List[str] = ["# Good Product Questions 30", ""]
    for idx, item in enumerate(items, start=1):
        question = humanize_question(
            clean_text(item["question"]),
            clean_text(item.get("intent")),
            clean_text(item.get("topic") or item.get("section") or ""),
            clean_text(item.get("command") or ""),
        )
        parts.extend(
            [
                f"### {idx}. {clean_text(item.get('intent'))} / {clean_text(item.get('topic') or item.get('command') or item.get('section') or 'product docs')}",
                "Question:",
                question,
                "",
                "Expected answer:",
                render_markdown_answer(item),
                "",
                "Why this is a good test:",
                clean_text(item.get("quality_reason") or "clean factual answer from product-doc corpus"),
                "",
            ]
        )
    return "\n".join(parts).strip() + "\n"


def offline_backend_result(item: Dict[str, object]) -> Dict[str, object]:
    expected_answer = clean_text(item["expected_answer"])
    question = clean_text(item["question"])
    intent = clean_text(item["intent"])
    syntax = clean_text(item.get("syntax"))
    offline_pass = bool(expected_answer) and not answer_has_artifact(expected_answer)
    if intent in {"cli_syntax", "show_command_syntax"}:
        offline_pass = offline_pass and bool(syntax or "```text" in expected_answer)
    failure_reason = "" if offline_pass else "expected answer failed offline quality checks"
    return {
        "question": question,
        "expected_answer": expected_answer,
        "backend_final_answer": None,
        "backend_available": False,
        "pass": offline_pass,
        "failure_reason": failure_reason,
        "intent": intent,
        "switch": clean_text(item.get("switch")),
        "version": clean_text(item.get("version")),
        "topic": clean_text(item.get("topic")),
        "command": clean_text(item.get("command")),
    }


def main() -> None:
    rows: List[Dict[str, object]] = []
    for path in DATASET_PATHS:
        rows.extend(list(read_jsonl(path)))

    selected, rejected, intent_counts = select_rows(rows)
    records = [build_record(row) for row in selected]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUTPUT_DIR / "good_product_questions_30.jsonl"
    md_path = OUTPUT_DIR / "good_product_questions_30.md"
    report_path = OUTPUT_DIR / "good_product_questions_report.json"
    bad_path = OUTPUT_DIR / "bad_product_question_candidates.jsonl"
    backend_results_path = OUTPUT_DIR / "good_product_questions_backend_results.jsonl"
    backend_report_path = OUTPUT_DIR / "good_product_questions_backend_report.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    md_path.write_text(render_markdown(records), encoding="utf-8")

    backend_results = [offline_backend_result(item) for item in records]
    with backend_results_path.open("w", encoding="utf-8") as handle:
        for record in backend_results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    backend_report = {
        "backend_available": False,
        "selected_count": len(records),
        "pass_count": sum(1 for item in backend_results if item["pass"]),
        "fail_count": sum(1 for item in backend_results if not item["pass"]),
        "note": "Offline validation only; backend was not called.",
    }
    backend_report_path.write_text(json.dumps(backend_report, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "selected_count": len(records),
        "rejected_candidate_count": len(rejected),
        "intent_counts": intent_counts,
        "source_files": [str(path) for path in DATASET_PATHS if path.exists()],
        "output_files": {
            "jsonl": str(jsonl_path),
            "markdown": str(md_path),
            "report": str(report_path),
            "bad_candidates": str(bad_path),
            "backend_results": str(backend_results_path),
            "backend_report": str(backend_report_path),
        },
        "backend_available": False,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with bad_path.open("w", encoding="utf-8") as handle:
        for item in rejected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
