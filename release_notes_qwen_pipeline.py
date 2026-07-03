from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from lstm_lookup import (
    build_lookup_entries,
    build_lookup_index,
    extract_slots_from_question,
    normalize_whitespace,
    read_jsonl,
    resolve_lookup_answer,
    write_jsonl,
)
from train_lstm_gpu import LSTMIntentModel, SimpleTokenizer


SYSTEM_PROMPT = (
    "You are only a response formatter for the HPE Aruba AOS-CX QA system.\n\n"
    "Facts come only from the retrieved lookup answer.\n"
    "You must not answer from your own knowledge.\n\n"
    "Your job is to rewrite the retrieved answer into a clean, readable, user-facing response.\n"
    "Do not truncate the answer.\n"
    "If the retrieved answer contains multiple facts, types, methods, goals, requirements, conditions, or steps, format them as bullet points or numbered steps.\n"
    "If the retrieved answer contains commands, wrap command text in backticks.\n"
    "If the user asks what a command does, only describe the purpose when the retrieved answer actually includes that purpose.\n"
    "If the retrieved answer only contains syntax, say that only syntax was found.\n"
    "Avoid filler prefixes like 'The documented answer is' or 'According to the documentation'.\n\n"
    "You MUST NOT:\n"
    "- add new facts\n"
    "- remove important facts\n"
    "- infer missing information\n"
    "- invent commands\n"
    "- invent Bug IDs\n"
    "- invent versions\n"
    "- invent caveats\n"
    "- invent workarounds\n"
    "- invent configuration steps\n"
    "- change numbers, IDs, versions, switch models, commands, parameters, or syntax\n\n"
    "You MUST preserve exactly:\n"
    "- Bug IDs\n"
    "- Event IDs\n"
    "- switch models\n"
    "- AOS-CX versions\n"
    "- sub-versions\n"
    "- category names\n"
    "- CLI commands\n"
    "- command syntax\n"
    "- parameter names\n"
    "- caveats\n"
    "- warnings\n"
    "- documented workaround text\n\n"
    "Formatting rules:\n"
    "- Use clear Markdown.\n"
    "- Use short headings when helpful.\n"
    "- Use bullet points for multiple items.\n"
    "- Use numbered steps for procedures.\n"
    "- Use code formatting for CLI commands and syntax.\n"
    "- Keep product documentation answers clear, complete, and explanatory.\n"
    "- Keep release-note answers concise and factual.\n"
    "- If the retrieved answer is only a list of headings, format it as a list.\n"
    "- Do not make the answer look more certain than the retrieved answer.\n\n"
    "Return only the final formatted answer."
)

USER_PROMPT_TEMPLATE = """Question:
{user_question}

Predicted intent:
{predicted_intent}

Slots:
{slots_json}

Metadata:
Switch: {switch}
Version: {version}
Sub-version: {sub_version}
Source type: {source_type}
Data family: {data_family}

Retrieved answer:
{lookup_answer}

Task:
Rewrite the retrieved answer into a clean, user-friendly response.

Rules:
1. Use only the retrieved answer.
2. Do not add any technical detail not present in the retrieved answer.
3. Do not change Bug IDs, versions, commands, syntax, parameters, caveats, or workaround meaning.
4. For product documentation, make the answer complete, readable, and slightly fuller when the lookup answer already contains the facts.
5. Format multiple factual items as bullet points.
6. Keep procedures as numbered steps.
7. For command-purpose questions, explain purpose only if the retrieved answer already includes it.
8. If only syntax was retrieved, say so clearly and preserve the exact syntax.
9. For release notes, keep the answer factual and concise.
10. Use Markdown formatting when it improves readability.
11. Do not truncate the response.
12. Do not start with "The documented answer is".
13. Return only the formatted answer.
"""

DEFAULT_QWEN_MODEL_PATH = Path(
    r"E:\52\Train_w\Train\outputs_final\qwen25_3b_metadatactx_fullclean_1epoch_stratified"
)
DEFAULT_LSTM_MODEL_PATH = Path(r"C:\Hpe\Train\outputs_release_lstm\all_switches\best_model.pt")
DEFAULT_LOOKUP_INDEX_PATH = Path(r"C:\Hpe\Train\outputs_release_lstm\all_switches\lookup_index.json")
DEFAULT_LOOKUP_DATA_PATH = Path(r"C:\Hpe\Train\Data\Release_Notes")
DEFAULT_TEST_FILE = Path(r"C:\Hpe\Train\outputs_4100i_gpu\test_4100i_lstm.jsonl")
DEFAULT_OUTPUT_DIR = Path(r"C:\Hpe\Train\outputs_release_lstm")
DEFAULT_EVAL_JSONL = DEFAULT_OUTPUT_DIR / "qwen_answer_eval.jsonl"
DEFAULT_REPORT_JSON = DEFAULT_OUTPUT_DIR / "qwen_answer_report.json"

EXACT_LIKE_INTENTS = {
    "bug_category",
    "version_date",
    "release_date",
    "event_id",
    "cli_syntax",
    "show_command_syntax",
}

SHORT_ALLOWED_INTENTS = {
    "bug_category",
    "version_date",
    "release_date",
    "event_id",
    "cli_syntax",
    "show_command_syntax",
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

SYNTAX_QUESTION_PATTERNS = [
    r"\bwhat\s+is\s+the\s+syntax\s+of\b",
    r"\bwhat\s+is\s+the\s+syntax\s+for\b",
    r"\bshow\s+syntax\b",
    r"\bcommand\s+syntax\b",
    r"\bhow\s+is\s+the\s+command\s+written\b",
    r"\bwhat\s+is\s+the\s+cli\s+syntax\b",
]

COMMAND_PURPOSE_PATTERNS = [
    r"\bwhat\s+does\s+.+?\s+command\s+do\b",
    r"\bwhat\s+is\s+the\s+purpose\s+of\s+.+?\s+command\b",
    r"\bwhat\s+is\s+this\s+command\s+used\s+for\b",
    r"\bwhat\s+does\s+the\s+.+?\s+command\s+do\b",
    r"\bwhat\s+does\s+.+?\s+do\b",
]


def token_list(text: object) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", normalize_whitespace(text).lower())


def content_tokens(text: object) -> List[str]:
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
    }
    return [token for token in token_list(text) if token not in stopwords]


def extract_bug_ids(text: object) -> List[str]:
    return re.findall(r"\b\d{4,7}\b", normalize_whitespace(text))


def extract_version_strings(text: object) -> List[str]:
    versions = re.findall(r"\b\d+\.\d+(?:\.\d+)?\b", normalize_whitespace(text))
    unique: List[str] = []
    for version in versions:
        if version not in unique:
            unique.append(version)
    return unique


def extract_command_phrases(text: object) -> List[str]:
    source = normalize_whitespace(text)
    phrases: List[str] = []
    for pattern in COMMAND_PATTERNS:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            phrase = normalize_whitespace(match.group(0))
            if phrase not in phrases:
                phrases.append(phrase)
    return phrases


def extract_switch_models(text: object) -> List[str]:
    source = normalize_whitespace(text)
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


def is_no_workaround_answer(answer: str) -> bool:
    text = normalize_whitespace(answer).lower()
    return "no workaround is documented" in text or text == "no workaround is documented in the release notes."


def is_command_syntax_question(question: str) -> bool:
    text = normalize_whitespace(question).lower()
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in SYNTAX_QUESTION_PATTERNS)


def is_command_purpose_question(question: str) -> bool:
    text = normalize_whitespace(question).lower()
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in COMMAND_PURPOSE_PATTERNS)


def extract_command_from_question(question: str, metadata: Dict[str, str] | None = None) -> str:
    metadata = metadata or {}
    command = normalize_whitespace(metadata.get("command", ""))
    if command:
        return command

    text = normalize_whitespace(question)
    patterns = [
        r"\bwhat\s+does\s+(?P<command>.+?)\s+command\s+do\b",
        r"\bwhat\s+is\s+the\s+purpose\s+of\s+(?P<command>.+?)\s+command\b",
        r"\bwhat\s+is\s+the\s+syntax\s+of\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bwhat\s+is\s+the\s+syntax\s+for\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bshow\s+syntax\s+for\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bcommand\s+syntax\s+for\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
        r"\bsyntax\s+of\s+(?:the\s+)?(?P<command>.+?)\s+command\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_whitespace(match.group("command")).strip(" ?.")
    return ""


def _clean_markdown_text(answer: str) -> str:
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
    return "\n".join(cleaned_lines).strip()


def clean_cli_syntax(syntax_text: str) -> str:
    text = _clean_markdown_text(_prompt_safe_text(syntax_text))
    if not text:
        return ""

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        syntax_like_lines = [line for line in lines if any(symbol in line for symbol in ("<", ">", "[", "]", "{", "}", "|", "("))]
        candidate_lines = syntax_like_lines or lines
        if len(candidate_lines) == 1:
            text = candidate_lines[0]
        else:
            text = " ".join(candidate_lines)

    text = re.sub(r"^(?:syntax|command syntax)\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^(?:the syntax for .*?(?:command)?(?: is| is:)|the syntax of .*?(?:command)?(?: is| is:)|the command is|the command syntax is|command syntax is)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?<!\d)\.{3,}(?!\d)", " ", text)
    text = re.sub(r"^[\s\.\-:|•·]+", "", text)
    text = text.replace("`", "")
    text = re.sub(r"\s+(?:page|pg)\s*\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:\d{1,4}\s+)+", "", text)
    text = re.sub(
        r"^\s*\d{1,4}\s+(?=(?:show|no|clear|ip|interface|vlan|bfd|redundancy|apply|erps|aaa|mdns-sd)\b)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^\s*[:\-–—]+\s*", "", text)
    text = re.sub(r"\s*\.{2,}\s*$", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" .")
    return text


def is_bad_syntax_artifact(text: str, expected_command: str = "") -> bool:
    candidate = _clean_markdown_text(_prompt_safe_text(text))
    if not candidate:
        return True
    had_artifact = bool(
        re.search(r"\.{8,}", candidate)
        or candidate.startswith(".")
        or re.search(r"\b(?:page|pg)\s*\d+\b", candidate, flags=re.IGNORECASE)
        or re.search(r"\b\d{1,4}\s+[A-Za-z][A-Za-z0-9._/-]*(?:\s+[A-Za-z][A-Za-z0-9._/-]*){0,7}\b", candidate)
    )
    cleaned = clean_cli_syntax(candidate)
    if not cleaned:
        return True
    if had_artifact and len(cleaned.split()) <= 3 and not any(symbol in cleaned for symbol in ("<", ">", "[", "]", "{", "}", "|", "(")):
        return True
    if had_artifact and len(cleaned.split()) > 12:
        return True
    if cleaned.count(".") > max(3, len(cleaned) // 4):
        return True
    if not re.search(r"[A-Za-z]", cleaned):
        return True
    if len(cleaned.split()) < 1:
        return True
    expected = normalize_whitespace(expected_command)
    if expected:
        expected_tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9._/-]*", expected.lower()) if token not in {"the", "a", "an", "of", "for", "to", "on", "in"}]
        cleaned_tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9._/-]*", cleaned.lower()) if token not in {"the", "a", "an", "of", "for", "to", "on", "in"}]
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


def _prompt_safe_text(text: object) -> str:
    value = "" if text is None else str(text)
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _split_command_purpose_and_syntax(answer: str) -> Tuple[str, str]:
    text = _clean_markdown_text(answer)
    if not text:
        return "", ""
    syntax = _extract_cli_syntax(text)
    if not syntax:
        return text, ""
    if _is_syntax_only_text(text):
        return "", syntax
    purpose = text
    if syntax in purpose:
        purpose = purpose.replace(syntax, " ")
    purpose = re.sub(r"\b(?:syntax|command syntax)\s*[:\-]?\s*", "", purpose, flags=re.IGNORECASE)
    purpose = _clean_markdown_text(purpose).strip(" :-")
    return purpose, syntax


def _is_syntax_only_text(answer: str) -> bool:
    text = normalize_whitespace(answer)
    if not text:
        return False
    lower = text.lower()
    if lower.startswith("syntax:") or lower.startswith("the syntax of") or lower.startswith("command syntax"):
        return True
    if any(symbol in text for symbol in ("<", ">", "[", "]", "{", "}", "|")):
        return True
    if text.count(".") == 0 and text.count("\n") == 0 and len(text.split()) <= 6:
        command_like = bool(re.fullmatch(r"(?:no\s+)?[A-Za-z0-9._/-]+(?:\s+[A-Za-z0-9._/-]+){0,5}", text.strip()))
        if command_like and not re.search(
            r"\b(configuring|overview|guide|commands?|features?|includes?|contains?|supports?|using|used)\b",
            lower,
        ):
            return True
    return False


def _looks_like_purpose_text(answer: str) -> bool:
    text = _clean_markdown_text(answer)
    if not text:
        return False
    if "\n" in text:
        return True
    if re.search(r"[.!?]", text):
        return True
    if re.search(r":\s", text):
        return True
    return False


def _syntax_candidate_score(candidate: str, expected_command: str = "") -> float:
    text = clean_cli_syntax(candidate)
    if not text or is_bad_syntax_artifact(text, expected_command):
        return float("-inf")

    lower = text.lower()
    score = 0.0
    words = text.split()

    if expected_command:
        expected = clean_cli_syntax(expected_command).lower()
        if expected and expected in lower:
            score += 8.0
        expected_tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9._/-]*", expected) if token not in {"the", "a", "an", "of", "for", "to", "on", "in"}]
        text_tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9._/-]*", lower) if token not in {"the", "a", "an", "of", "for", "to", "on", "in"}]
        if expected_tokens and text_tokens:
            matched = 0
            for token in text_tokens:
                if matched < len(expected_tokens) and token == expected_tokens[matched]:
                    matched += 1
            score += min(matched, len(expected_tokens)) * 1.5

    if any(symbol in text for symbol in ("<", ">", "[", "]", "{", "}", "|")):
        score += 4.0
    if re.fullmatch(r"(?:no\s+)?[A-Za-z0-9._/-]+(?:\s+[A-Za-z0-9._/-]+){0,12}", text.strip()):
        score += 2.5
    if len(words) <= 14:
        score += 2.0
    elif len(words) <= 24:
        score += 0.5
    else:
        score -= min(6.0, (len(words) - 24) * 0.5)
    if text.endswith("."):
        score -= 0.75
    if re.search(r"\.{3,}", text):
        score -= 5.0
    if re.search(r"\b(?:page|pg)\s*\d+\b", text, flags=re.IGNORECASE):
        score -= 3.0
    if re.search(r"\b\d{1,4}\s+[A-Za-z][A-Za-z0-9._/-]*(?:\s+[A-Za-z][A-Za-z0-9._/-]*){0,7}\b", text):
        score -= 2.0
    if lower.startswith(("syntax", "command syntax", "the syntax", "show syntax")):
        score += 2.0
    if any(lower.startswith(prefix) for prefix in ("show ", "no ", "ip ", "bfd ", "clear ", "redundancy ", "interface ", "vlan ")):
        score += 1.0
    return score


def _extract_cli_syntax(lookup_answer: str, expected_command: str = "") -> str:
    raw_text = _prompt_safe_text(lookup_answer)
    text = normalize_whitespace(raw_text)
    if not text:
        return ""

    expected_command = clean_cli_syntax(expected_command)
    raw_has_artifacts = bool(
        re.search(r"\.{8,}", raw_text)
        or re.search(r"\b(?:page|pg)\s*\d+\b", raw_text, flags=re.IGNORECASE)
        or re.search(r"\bcontents\b", raw_text, flags=re.IGNORECASE)
    )

    backtick_match = re.search(r"`(?P<syntax>[^`]{2,})`", text)
    if backtick_match:
        syntax = clean_cli_syntax(backtick_match.group("syntax"))
        if syntax and not is_bad_syntax_artifact(syntax, expected_command):
            return syntax

    patterns = [
        r"(?:^|\n)\s*syntax:\s*(?P<syntax>.+)$",
        r"(?:the syntax of .*? command is:)\s*(?P<syntax>.+)$",
        r"(?:the syntax of .*? is:)\s*(?P<syntax>.+)$",
        r"(?:the syntax for .*? command(?: on .*?)? is)\s*(?P<syntax>.+?)(?:[.?!]\s*$|$)",
        r"(?:the command syntax is)\s*(?P<syntax>.+?)(?:[.?!]\s*$|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            syntax = clean_cli_syntax(match.group("syntax"))
            if syntax and not is_bad_syntax_artifact(syntax, expected_command):
                return syntax

    line_candidates: List[str] = []
    for raw_line in raw_text.split("\n"):
        if is_bad_syntax_artifact(raw_line, expected_command):
            continue
        line = clean_cli_syntax(raw_line)
        if not line or is_bad_syntax_artifact(line, expected_command):
            continue
        if expected_command and expected_command.lower() in line.lower():
            line_candidates.append(line)
            continue
        if any(symbol in line for symbol in ("<", ">", "[", "]", "{", "}", "|")) or re.fullmatch(r"(?:no\s+)?[A-Za-z0-9._/-]+(?:\s+[A-Za-z0-9._/-]+){0,12}", line):
            line_candidates.append(line)
    if line_candidates:
        best_candidate = max(line_candidates, key=lambda candidate: _syntax_candidate_score(candidate, expected_command))
        if best_candidate and not is_bad_syntax_artifact(best_candidate, expected_command):
            return best_candidate

    if _is_syntax_only_text(text):
        if ":" in text:
            tail = clean_cli_syntax(text.split(":", 1)[-1])
            if tail and not is_bad_syntax_artifact(tail, expected_command):
                return tail
        if raw_has_artifacts:
            return ""
        cleaned = clean_cli_syntax(text)
        if cleaned and not is_bad_syntax_artifact(cleaned, expected_command):
            return cleaned
        return text
    return ""


def is_cli_syntax_answer(lookup_answer: str, intent: str = "", question: str = "") -> bool:
    del lookup_answer
    intent_text = normalize_whitespace(intent).lower()
    if intent_text in {"cli_syntax", "show_command_syntax"}:
        return True
    return is_command_syntax_question(question)


def format_cli_syntax_answer(question: str, lookup_answer: str, metadata: Dict[str, str] | None = None) -> str:
    metadata = metadata or {}
    cleaned_question = normalize_whitespace(question)
    command = extract_command_from_question(cleaned_question, metadata)
    purpose_text, syntax_text = _split_command_purpose_and_syntax(lookup_answer)
    syntax_candidate = clean_cli_syntax(syntax_text or _extract_cli_syntax(lookup_answer, command))
    bad_syntax = is_bad_syntax_artifact(syntax_candidate or lookup_answer, command)
    syntax_text = "" if bad_syntax else syntax_candidate

    if is_command_purpose_question(cleaned_question):
        if purpose_text and _looks_like_purpose_text(purpose_text) and purpose_text.lower() != syntax_text.lower():
            parts: List[str] = []
            if command:
                parts.append(f"**Command:** `{command}`")
                parts.append("")
            parts.append("**Purpose**")
            parts.append("")
            parts.append(purpose_text)
            if syntax_text:
                parts.extend(["", "Syntax", "", f"```text\n{syntax_text}\n```"])
            return _clean_markdown_text("\n".join(parts))

        syntax_only = syntax_text or clean_cli_syntax(_extract_cli_syntax(lookup_answer) or lookup_answer)
        if not syntax_text and not _looks_like_purpose_text(purpose_text or ""):
            return _clean_markdown_text(lookup_answer)
        if command:
            return _clean_markdown_text(
                f"I found the syntax for `{command}`, but the retrieved documentation did not provide a direct purpose description.\n\n"
                f"**Syntax**\n\n```text\n{syntax_only}\n```"
            )
        return _clean_markdown_text(f"**Syntax**\n\n```text\n{syntax_only}\n```")

    if syntax_text:
        return _clean_markdown_text(f"**Syntax**\n\n```text\n{syntax_text}\n```")
    if _is_syntax_only_text(lookup_answer):
        cleaned_lookup = clean_cli_syntax(lookup_answer)
        if cleaned_lookup and not is_bad_syntax_artifact(cleaned_lookup, command):
            return _clean_markdown_text(f"**Syntax**\n\n```text\n{cleaned_lookup}\n```")
    if purpose_text:
        return _clean_markdown_text(purpose_text)
    return _clean_markdown_text(lookup_answer)


def build_prompt(
    question: str,
    predicted_intent: str,
    slots: Dict[str, str],
    lookup_answer: str,
    source_type: str = "",
    data_family: str = "release_notes",
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        user_question=normalize_whitespace(question),
        predicted_intent=normalize_whitespace(predicted_intent),
        slots_json=json.dumps(slots, ensure_ascii=False, sort_keys=True),
        switch=normalize_whitespace(slots.get("switch", "")),
        version=normalize_whitespace(slots.get("version", "")),
        sub_version=normalize_whitespace(slots.get("sub_version", "")),
        source_type=normalize_whitespace(source_type or predicted_intent or "release_notes"),
        data_family=normalize_whitespace(data_family or "release_notes"),
        lookup_answer=_prompt_safe_text(lookup_answer),
    )


def load_lstm_support(model_path: Path, device: torch.device):
    payload = torch.load(model_path, map_location=device)
    tokenizer = SimpleTokenizer(payload["vocab"])
    config = dict(payload["config"])
    label_to_id = dict(payload.get("label_to_id", {}))
    id_to_label = dict(payload.get("id_to_label", {}))
    label_names = list(config.get("label_names") or [value for _, value in sorted(id_to_label.items(), key=lambda item: int(item[0]))])
    if not label_names:
        raise SystemExit(f"Could not determine label names from LSTM checkpoint: {model_path}")
    config["label_names"] = label_names
    config["label_to_id"] = label_to_id
    config["id_to_label"] = id_to_label
    model = LSTMIntentModel(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=int(config["embedding_dim"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        num_labels=len(label_names),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, tokenizer, config


def predict_intent(
    question: str,
    model: LSTMIntentModel,
    tokenizer: SimpleTokenizer,
    config: Dict[str, object],
    device: torch.device,
) -> str:
    cleaned_question = normalize_whitespace(question)
    ids = tokenizer.encode(cleaned_question, int(config["max_length"]))
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids, lengths)
        predicted_id = int(logits.argmax(dim=1).item())
    id_to_label = dict(config.get("id_to_label", {}))
    if not id_to_label:
        return list(config.get("label_names", []))[predicted_id]
    if str(predicted_id) in id_to_label:
        return id_to_label[str(predicted_id)]
    if predicted_id in id_to_label:
        return id_to_label[predicted_id]
    return id_to_label.get(str(predicted_id), list(value for _, value in sorted(id_to_label.items(), key=lambda item: int(item[0])))[predicted_id])


def load_lookup_resources(
    lookup_index_path: Path,
    lookup_data_path: Path,
) -> Tuple[List[object], Dict[str, List[int]]]:
    if not lookup_data_path.exists():
        raise FileNotFoundError(f"Lookup data file not found: {lookup_data_path}")
    if lookup_data_path.is_dir():
        data_files = sorted(
            path for path in lookup_data_path.rglob("train_chat.jsonl") if path.is_file()
        )
        if not data_files:
            raise FileNotFoundError(f"No train_chat.jsonl files found under: {lookup_data_path}")
        records = []
        for path in data_files:
            records.extend(read_jsonl(path))
    else:
        records = read_jsonl(lookup_data_path)
    entries = build_lookup_entries(records)
    index: Dict[str, List[int]] = {}
    if lookup_index_path.exists():
        with lookup_index_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict) and loaded:
            index = {str(key): [int(value) for value in values] for key, values in loaded.items()}

    if not index:
        index = build_lookup_index(entries)
        lookup_index_path.parent.mkdir(parents=True, exist_ok=True)
        with lookup_index_path.open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, ensure_ascii=False)

    return entries, index


def _has_adapter_files(path: Path) -> bool:
    return (path / "adapter_config.json").exists() and any(
        (path / candidate).exists() for candidate in ("adapter_model.safetensors", "adapter_model.bin")
    )


def _has_full_model_files(path: Path) -> bool:
    return (path / "config.json").exists() and any(
        (path / candidate).exists() for candidate in ("model.safetensors", "pytorch_model.bin")
    )


def resolve_qwen_source(requested_path: Path) -> Tuple[Path, str, str]:
    if not requested_path.exists():
        raise FileNotFoundError(f"Qwen model path not found: {requested_path}")

    direct_candidates = [
        requested_path / "lora_adapters",
        requested_path / "best_lora_adapters",
    ]
    for candidate in direct_candidates:
        if candidate.is_dir() and _has_adapter_files(candidate):
            return candidate, "adapter", "direct adapter directory"

    if _has_adapter_files(requested_path):
        return requested_path, "adapter", "direct adapter directory"
    if _has_full_model_files(requested_path):
        return requested_path, "full", "direct full model directory"

    checkpoint_dirs = [child for child in requested_path.glob("checkpoint-*") if child.is_dir()]
    checkpoint_dirs = [child for child in checkpoint_dirs if _has_adapter_files(child) or _has_full_model_files(child)]
    if checkpoint_dirs:
        def checkpoint_sort_key(path: Path) -> Tuple[int, float]:
            match = re.search(r"checkpoint-(\d+)", path.name)
            return (int(match.group(1)) if match else -1, path.stat().st_mtime)

        chosen = sorted(checkpoint_dirs, key=checkpoint_sort_key)[-1]
        kind = "adapter" if _has_adapter_files(chosen) else "full"
        return chosen, kind, "latest checkpoint in requested folder"

    fallback_candidates: List[Path] = []
    family_prefix = requested_path.name.rsplit("_", 1)[0]
    parent = requested_path.parent
    if parent.exists():
        for sibling in parent.iterdir():
            if not sibling.is_dir():
                continue
            if sibling.name == requested_path.name:
                continue
            if family_prefix and sibling.name.startswith(family_prefix) and (
                _has_adapter_files(sibling)
                or _has_full_model_files(sibling)
                or any(child.is_dir() and (_has_adapter_files(child) or _has_full_model_files(child)) for child in sibling.glob("checkpoint-*"))
            ):
                fallback_candidates.append(sibling)

    if fallback_candidates:
        def fallback_score(path: Path) -> Tuple[int, float]:
            match = re.search(r"(\d+)epoch", path.name)
            epoch_score = int(match.group(1)) if match else -1
            return (epoch_score, path.stat().st_mtime)

        chosen = sorted(fallback_candidates, key=fallback_score)[-1]
        return chosen, "adapter", "fallback sibling run in same family"

    family_match = re.match(
        r"^(?P<base>.+?)_(?P<epoch>\d+epochs?)_stratified$",
        requested_path.name,
        flags=re.IGNORECASE,
    )
    if family_match and parent.exists():
        base_prefix = family_match.group("base")
        family_candidates: List[Path] = []
        for sibling in parent.iterdir():
            if not sibling.is_dir() or sibling.name == requested_path.name:
                continue
            sibling_match = re.match(
                rf"^{re.escape(base_prefix)}_(?P<epoch>\d+epochs?)_stratified$",
                sibling.name,
                flags=re.IGNORECASE,
            )
            if not sibling_match:
                continue
            if _has_adapter_files(sibling) or _has_full_model_files(sibling):
                family_candidates.append(sibling)
                continue
            for child in sibling.iterdir():
                if child.is_dir() and (_has_adapter_files(child) or _has_full_model_files(child)):
                    family_candidates.append(sibling)
                    break

        if family_candidates:
            def family_score(path: Path) -> Tuple[int, float]:
                match = re.match(
                    rf"^{re.escape(base_prefix)}_(?P<epoch>\d+)epochs?_stratified$",
                    path.name,
                    flags=re.IGNORECASE,
                )
                epoch_score = int(match.group("epoch")) if match else -1
                return (epoch_score, path.stat().st_mtime)

            chosen = sorted(family_candidates, key=family_score)[-1]
            if _has_adapter_files(chosen):
                return chosen, "adapter", "fallback sibling run in epoch family"
            if _has_full_model_files(chosen):
                return chosen, "full", "fallback sibling run in epoch family"
            checkpoint_dirs = [child for child in chosen.glob("checkpoint-*") if child.is_dir()]
            checkpoint_dirs = [child for child in checkpoint_dirs if _has_adapter_files(child) or _has_full_model_files(child)]
            if checkpoint_dirs:
                chosen_checkpoint = sorted(checkpoint_dirs, key=lambda path: (int(re.search(r"checkpoint-(\d+)", path.name).group(1)) if re.search(r"checkpoint-(\d+)", path.name) else -1, path.stat().st_mtime))[-1]
                kind = "adapter" if _has_adapter_files(chosen_checkpoint) else "full"
                return chosen_checkpoint, kind, "fallback checkpoint in epoch family"

    raise FileNotFoundError(
        f"No adapter or full checkpoint files were found under {requested_path}. "
        "Expected lora_adapters, best_lora_adapters, config.json, or checkpoint-* directories."
    )


def load_qwen_model(requested_path: Path, device: torch.device):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "transformers is required for the final Qwen pipeline. Install transformers before running this script."
        ) from exc

    try:
        from peft import PeftModel
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("peft is required to load Qwen LoRA adapters. Install peft before running this script.") from exc

    resolved_path, kind, resolution_reason = resolve_qwen_source(requested_path)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32
    )

    if kind == "adapter":
        with (resolved_path / "adapter_config.json").open("r", encoding="utf-8") as handle:
            adapter_config = json.load(handle)
        base_model_name = adapter_config["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True, use_fast=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base_model, resolved_path, is_trainable=False)
        try:
            model = model.merge_and_unload()
        except Exception:
            pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model.to(device)
    model.eval()
    return tokenizer, model, {
        "requested_path": str(requested_path),
        "resolved_path": str(resolved_path),
        "model_kind": kind,
        "resolution_reason": resolution_reason,
        "base_model_name": adapter_config["base_model_name_or_path"] if kind == "adapter" else str(resolved_path),
    }


def qwen_max_new_tokens(intent: str, data_family: str = "release_notes") -> int:
    family = normalize_whitespace(data_family).lower()
    if family in {"product", "product_documentation", "product docs", "product_docs"}:
        if intent in {"concept_explanation", "configuration_procedure"}:
            return 420
        if intent in {"cli_command_reference", "cli_syntax", "command_parameter_reference", "show_command_syntax", "show_command_usage"}:
            return 260
        return 360
    if intent in {"bug_category", "version_date", "release_date", "event_id"}:
        return 80
    if intent in {"cli_syntax", "show_command_syntax"}:
        return 80
    if intent == "concept_explanation":
        return 220
    if intent in {"bug_workaround", "bug_scenario", "bug_symptom", "release_caveat", "release_upgrade_info", "release_version_history"}:
        return 240
    return 220


def _cleanup_formatted_answer(answer: str) -> str:
    cleaned = _clean_markdown_text(answer)
    if cleaned.startswith("- "):
        candidate = cleaned[2:].strip()
        if "\n" not in candidate and not re.search(r"^\s*[-*]\s+", candidate, flags=re.MULTILINE):
            cleaned = candidate
    cleaned = re.sub(r"\.\.(?=\s|$)", ".", cleaned)
    return cleaned.strip()


def encode_prompt(tokenizer, prompt: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    return tokenizer(f"{SYSTEM_PROMPT}\n\n{prompt}", return_tensors="pt").input_ids


def generate_qwen_answer(
    tokenizer,
    model,
    prompt: str,
    intent: str,
    device: torch.device,
    data_family: str = "release_notes",
) -> str:
    encoded = encode_prompt(tokenizer, prompt)
    if isinstance(encoded, torch.Tensor):
        input_ids = encoded
    elif isinstance(encoded, dict):
        input_ids = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    else:
        input_ids = torch.tensor(encoded, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    generation_kwargs = {
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_new_tokens": qwen_max_new_tokens(intent, data_family),
        "repetition_penalty": 1.1,
    }
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if getattr(tokenizer, "pad_token_id", None) is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    with torch.inference_mode():
        generated = model.generate(input_ids=input_ids, **generation_kwargs)
    new_tokens = generated[0, input_ids.shape[-1] :]
    return _cleanup_formatted_answer(tokenizer.decode(new_tokens, skip_special_tokens=True))


def validate_qwen_answer(
    predicted_intent: str,
    slots: Dict[str, str],
    lookup_answer: str,
    qwen_answer: str,
    data_family: str = "release_notes",
) -> Tuple[bool, str]:
    original = normalize_whitespace(lookup_answer)
    formatted = normalize_whitespace(qwen_answer)
    if not formatted:
        return False, "empty qwen answer"
    original_word_count = len(original.split())
    formatted_word_count = len(formatted.split())
    family = normalize_whitespace(data_family).lower()
    min_ratio = 0.3 if family in {"product", "product_documentation", "product docs", "product_docs"} and predicted_intent == "concept_explanation" else 0.7
    if original_word_count and formatted_word_count < max(1, int(original_word_count * min_ratio)):
        return False, "too short"
    if family in {"product", "product_documentation", "product docs", "product_docs"}:
        max_ratio = 6.0 if predicted_intent == "concept_explanation" else 4.0
    else:
        max_ratio = 3.5 if predicted_intent in {"bug_workaround", "bug_scenario", "bug_symptom", "release_caveat", "release_upgrade_info", "release_version_history"} else 3.0
    if original_word_count and formatted_word_count > max(original_word_count * max_ratio, original_word_count + 120):
        return False, "too long"

    original_tokens = content_tokens(original)
    formatted_tokens = content_tokens(formatted)
    if original_tokens:
        overlap = sum(1 for token in original_tokens if token in formatted_tokens)
        recall = overlap / max(1, len(original_tokens))
        similarity = 0.0
        from difflib import SequenceMatcher

        similarity = SequenceMatcher(None, original.lower(), formatted.lower()).ratio()
        if recall < 0.65 and similarity < 0.5:
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
        allowed_models.add(normalize_whitespace(slots["switch"]))
    output_models = set(extract_switch_models(formatted))
    if output_models and not output_models.issubset(allowed_models):
        return False, "switch model changed"

    original_commands = set(extract_command_phrases(original))
    if slots.get("command"):
        original_commands.add(normalize_whitespace(slots["command"]))
    output_commands = set(extract_command_phrases(formatted))
    if output_commands and not output_commands.issubset(original_commands):
        return False, "command changed"

    if slots.get("category") and normalize_whitespace(slots["category"]).lower() not in formatted.lower():
        return False, "category changed"

    if is_no_workaround_answer(original):
        if "no workaround" not in formatted.lower():
            return False, "workaround meaning changed"
    elif "no workaround" in formatted.lower() and not is_no_workaround_answer(original):
        return False, "invented no-workaround meaning"

    if predicted_intent == "release_caveat" and len(formatted.split()) < len(original.split()) * 0.7:
        return False, "caveat meaning changed"

    return True, "ok"


def gold_answer_from_row(row: Dict[str, object]) -> str:
    return normalize_whitespace(row.get("target_value", "")) or normalize_whitespace(row.get("reference", ""))


def metric_f1(reference: str, prediction: str) -> float:
    ref_tokens = Counter(token_list(reference))
    pred_tokens = Counter(token_list(prediction))
    if not ref_tokens or not pred_tokens:
        return 0.0
    overlap = sum((ref_tokens & pred_tokens).values())
    precision = overlap / sum(pred_tokens.values())
    recall = overlap / sum(ref_tokens.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def metric_rouge_l(reference: str, prediction: str) -> float:
    ref_tokens = token_list(reference)
    pred_tokens = token_list(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0
    prev = [0] * (len(pred_tokens) + 1)
    for ref_token in ref_tokens:
        curr = [0]
        for idx, pred_token in enumerate(pred_tokens, start=1):
            if ref_token == pred_token:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(prev[idx], curr[-1]))
        prev = curr
    lcs = prev[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_rows(
    rows: Sequence[Dict[str, object]],
    lstm_model: Optional[LSTMIntentModel],
    tokenizer_lstm: Optional[SimpleTokenizer],
    lstm_config: Optional[Dict[str, object]],
    lookup_entries: Sequence[object],
    lookup_index: Dict[str, List[int]],
    qwen_tokenizer,
    qwen_model,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    report_counts = Counter()
    metrics = {
        "lookup_exact_match": 0.0,
        "lookup_token_f1": 0.0,
        "lookup_rouge_l": 0.0,
        "final_exact_match": 0.0,
        "final_token_f1": 0.0,
        "final_rouge_l": 0.0,
    }
    output_rows: List[Dict[str, object]] = []
    source_counts = Counter()
    gold_total = 0

    for row in rows:
        question = normalize_whitespace(row.get("input_text", ""))
        predicted_intent = normalize_whitespace(row.get("intent", ""))
        if lstm_model is not None and tokenizer_lstm is not None and lstm_config is not None:
            predicted_intent = predict_intent(question, lstm_model, tokenizer_lstm, lstm_config, device)
        slots = extract_slots_from_question(question)
        defaults = {}
        if lstm_config is not None:
            defaults = {
                key: str(lstm_config.get(key, ""))
                for key in ("default_switch", "default_version", "default_sub_version")
                if str(lstm_config.get(key, ""))
            }
        resolution = resolve_lookup_answer(predicted_intent, slots, question, lookup_entries, lookup_index, defaults)
        lookup_status = str(resolution.get("status", "error"))
        lookup_answer = normalize_whitespace(resolution.get("answer", "")) if resolution.get("answer") else ""
        lookup_key_used = resolution.get("lookup_key_used")

        if lookup_status == "found" and lookup_answer:
            report_counts["lookup_found"] += 1
            prompt = build_prompt(
                question,
                predicted_intent,
                slots,
                lookup_answer,
                source_type=predicted_intent,
                data_family="release_notes",
            )
            qwen_answer = generate_qwen_answer(
                qwen_tokenizer,
                qwen_model,
                prompt,
                predicted_intent,
                device,
                data_family="release_notes",
            )
            passed, reason = validate_qwen_answer(
                predicted_intent,
                slots,
                lookup_answer,
                qwen_answer,
                data_family="release_notes",
            )
            final_answer = qwen_answer if passed else lookup_answer
            qwen_validation_passed = passed
            answer_source = "qwen_grounded" if passed else "lookup_fallback"
            source_counts[answer_source] += 1
            report_counts["qwen_used"] += 1
            if passed:
                report_counts["qwen_validation_passed"] += 1
            else:
                report_counts["qwen_validation_failed"] += 1
        else:
            qwen_answer = None
            qwen_validation_passed = False
            final_answer = (
                "No matching answer was found in the current release-note dataset."
                if lookup_status == "not_found"
                else (
                    "Multiple possible answers were found. Please provide more detail such as feature, bug ID, version, or sub-version."
                    if lookup_status == "needs_disambiguation"
                    else (
                        "I need more detail to answer this, such as the bug ID, feature, version, or sub-version."
                        if lookup_status == "slot_missing"
                        else "Unable to answer from the current release-note dataset."
                    )
                )
            )
            answer_source = "lookup_fallback"
            source_counts[answer_source] += 1
            reason = lookup_status

        gold_answer = gold_answer_from_row(row)
        if gold_answer:
            gold_total += 1
            metrics["lookup_exact_match"] += float(bool(lookup_answer) and normalize_whitespace(lookup_answer).lower() == gold_answer.lower())
            metrics["lookup_token_f1"] += metric_f1(gold_answer, lookup_answer)
            metrics["lookup_rouge_l"] += metric_rouge_l(gold_answer, lookup_answer)
            metrics["final_exact_match"] += float(normalize_whitespace(final_answer).lower() == gold_answer.lower())
            metrics["final_token_f1"] += metric_f1(gold_answer, final_answer)
            metrics["final_rouge_l"] += metric_rouge_l(gold_answer, final_answer)

        output_rows.append(
            {
                "question": question,
                "predicted_intent": predicted_intent,
                "slots": slots,
                "lookup_answer": lookup_answer or None,
                "qwen_answer": qwen_answer,
                "qwen_validation_passed": qwen_validation_passed,
                "final_answer": final_answer,
                "answer_source": answer_source,
                "lookup_key_used": lookup_key_used,
                "lookup_status": lookup_status,
                "gold_answer": gold_answer or None,
                "correct": bool(gold_answer) and normalize_whitespace(final_answer).lower() == gold_answer.lower(),
                "qwen_reason": reason if lookup_status == "found" and qwen_answer else None,
            }
        )

    total = max(1, gold_total)
    report = {
        "total_questions": len(rows),
        "rows_with_gold": gold_total,
        "lookup_found": report_counts["lookup_found"],
        "qwen_used": report_counts["qwen_used"],
        "qwen_validation_passed": report_counts["qwen_validation_passed"],
        "qwen_validation_failed": report_counts["qwen_validation_failed"],
        "lookup_exact_match": metrics["lookup_exact_match"] / total,
        "lookup_token_f1": metrics["lookup_token_f1"] / total,
        "lookup_rouge_l": metrics["lookup_rouge_l"] / total,
        "final_exact_match": metrics["final_exact_match"] / total,
        "final_token_f1": metrics["final_token_f1"] / total,
        "final_rouge_l": metrics["final_rouge_l"] / total,
        "answer_source_counts": dict(source_counts),
        "accuracy": sum(1 for row in output_rows if bool(row["correct"])) / max(1, len(output_rows)),
        "final_verdict": "Qwen generates the final answer from grounded lookup text, with deterministic fallback on validation failure.",
    }
    return output_rows, report


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_pipeline(
    test_file: Path,
    output_dir: Path,
    qwen_model_path: Path,
    lstm_model_path: Path,
    lookup_index_path: Path,
    lookup_data_path: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    rows = read_jsonl(test_file)
    if max_samples is not None and max_samples > 0 and len(rows) > max_samples:
        rows = rows[:max_samples]

    lookup_entries, lookup_index = load_lookup_resources(lookup_index_path, lookup_data_path)
    lstm_model, lstm_tokenizer, lstm_config = load_lstm_support(lstm_model_path, device)
    qwen_tokenizer, qwen_model, qwen_meta = load_qwen_model(qwen_model_path, device)

    output_rows, report = evaluate_rows(
        rows,
        lstm_model,
        lstm_tokenizer,
        lstm_config,
        lookup_entries,
        lookup_index,
        qwen_tokenizer,
        qwen_model,
        device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = output_dir / "qwen_answer_eval.jsonl"
    report_path = output_dir / "qwen_answer_report.json"
    write_jsonl(eval_path, output_rows)
    report_payload = {
        **report,
        "qwen_model_requested_path": str(qwen_model_path),
        "qwen_model_resolved_path": qwen_meta["resolved_path"],
        "qwen_model_kind": qwen_meta["model_kind"],
        "qwen_model_resolution_reason": qwen_meta["resolution_reason"],
        "lstm_model_path": str(lstm_model_path),
        "lookup_index_path": str(lookup_index_path),
        "lookup_data_path": str(lookup_data_path),
        "eval_path": str(eval_path),
        "report_path": str(report_path),
    }
    save_json(report_path, report_payload)
    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Final Release Notes QA pipeline with LSTM support and grounded Qwen answers.")
    parser.add_argument("--use_qwen_generation", action="store_true")
    parser.add_argument("--qwen_model_path", type=Path, default=DEFAULT_QWEN_MODEL_PATH)
    parser.add_argument("--use_lstm_support", action="store_true")
    parser.add_argument("--lstm_model_path", type=Path, default=DEFAULT_LSTM_MODEL_PATH)
    parser.add_argument("--lookup_index_path", type=Path, default=DEFAULT_LOOKUP_INDEX_PATH)
    parser.add_argument("--lookup_data_path", type=Path, default=DEFAULT_LOOKUP_DATA_PATH)
    parser.add_argument("--test_file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--question", type=str, default="")
    args = parser.parse_args()

    if not args.use_qwen_generation:
        raise SystemExit("--use_qwen_generation is required for the final Release Notes QA pipeline.")
    if not args.use_lstm_support:
        raise SystemExit("--use_lstm_support is required for the final Release Notes QA pipeline.")

    max_samples = args.max_samples if args.max_samples > 0 else None

    if args.question.strip():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lookup_entries, lookup_index = load_lookup_resources(args.lookup_index_path, args.lookup_data_path)
        lstm_model, lstm_tokenizer, lstm_config = load_lstm_support(args.lstm_model_path, device)
        qwen_tokenizer, qwen_model, qwen_meta = load_qwen_model(args.qwen_model_path, device)
        question = normalize_whitespace(args.question)
        predicted_intent = predict_intent(question, lstm_model, lstm_tokenizer, lstm_config, device)
        slots = extract_slots_from_question(question)
        defaults = {
            key: str(lstm_config.get(key, ""))
            for key in ("default_switch", "default_version", "default_sub_version")
            if str(lstm_config.get(key, ""))
        }
        resolution = resolve_lookup_answer(predicted_intent, slots, question, lookup_entries, lookup_index, defaults)
        lookup_answer = normalize_whitespace(resolution.get("answer", "")) if resolution.get("answer") else ""
        lookup_status = str(resolution.get("status", "error"))
        if lookup_status == "found" and lookup_answer:
            qwen_prompt = build_prompt(
                question,
                predicted_intent,
                slots,
                lookup_answer,
                source_type=predicted_intent,
                data_family="release_notes",
            )
            qwen_answer = generate_qwen_answer(
                qwen_tokenizer,
                qwen_model,
                qwen_prompt,
                predicted_intent,
                device,
                data_family="release_notes",
            )
            passed, _ = validate_qwen_answer(
                predicted_intent,
                slots,
                lookup_answer,
                qwen_answer,
                data_family="release_notes",
            )
            final_answer = qwen_answer if passed else lookup_answer
            answer_source = "qwen_grounded" if passed else "lookup_fallback"
        else:
            qwen_answer = None
            passed = False
            final_answer = (
                "No matching answer was found in the current release-note dataset."
                if lookup_status == "not_found"
                else (
                    "Multiple possible answers were found. Please provide more detail such as feature, bug ID, version, or sub-version."
                    if lookup_status == "needs_disambiguation"
                    else (
                        "I need more detail to answer this, such as the bug ID, feature, version, or sub-version."
                        if lookup_status == "slot_missing"
                        else "Unable to answer from the current release-note dataset."
                    )
                )
            )
            answer_source = "lookup_fallback"
        print(
            json.dumps(
                {
                    "question": question,
                    "predicted_intent": predicted_intent,
                    "slots": slots,
                    "lookup_answer": lookup_answer or None,
                    "qwen_answer": qwen_answer,
                    "qwen_validation_passed": passed,
                    "final_answer": final_answer,
                    "answer_source": answer_source,
                    "qwen_model_resolved_path": qwen_meta["resolved_path"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    report = run_pipeline(
        args.test_file,
        args.output_dir,
        args.qwen_model_path,
        args.lstm_model_path,
        args.lookup_index_path,
        args.lookup_data_path,
        max_samples=max_samples,
    )
    print("Release Notes QA completed")
    print(f"Total questions: {report['total_questions']}")
    print(f"Lookup exact match: {report['lookup_exact_match']:.4f}")
    print(f"Final exact match: {report['final_exact_match']:.4f}")
    print(f"Qwen used: {report['qwen_used']}")
    print(f"Qwen validation passed: {report['qwen_validation_passed']}")
    print(f"Qwen validation failed: {report['qwen_validation_failed']}")
    print(f"Output JSONL: {report['eval_path']}")
    print(f"Output report: {report['report_path']}")
    print(f"Final verdict: {report['final_verdict']}")


if __name__ == "__main__":
    main()
