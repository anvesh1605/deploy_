from __future__ import annotations

import argparse
import logging
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from functools import lru_cache

from backend.config import PRODUCT_DOCS_REPAIRED_ROOT, QWEN_MODEL_PATH
from lstm_lookup import normalize_whitespace, read_jsonl, write_jsonl
from qwen_formatter import call_formatter_backend
from release_notes_qwen_pipeline import validate_qwen_answer


ROOT = Path(__file__).resolve().parent
DEFAULT_PRODUCT_ROOT = ROOT / "Data" / "product_docs_final"
DEFAULT_SOURCE_ROOT = Path(r"C:\Hpe\Data_1\markitdown_cli_output\Raw_Data_Product")
DEFAULT_OUTPUT_ROOT = PRODUCT_DOCS_REPAIRED_ROOT
DEFAULT_LOG_FILE = ROOT / "Data" / "product_docs_final_repaired" / "repair_product_docs_corpus.log"
DEFAULT_FORMATTER_BACKEND = "ollama"
DEFAULT_FORMATTER_MODEL = "qwen2.5:7b-instruct"

SYNTAX_INTENTS = {
    "cli_syntax",
    "show_command_syntax",
    "show_command_usage",
}

EXPLANATION_INTENTS = {
    "concept_explanation",
    "cli_meaning",
    "show_command_meaning",
    "event_id_meaning",
    "event_id_action",
    "product_generic",
    "product_caveat",
    "feature_limitations",
    "product_troubleshooting",
    "configuration_steps",
    "snmp_behavior",
    "rest_api_usage",
}

LABELS = {
    "syntax",
    "description",
    "usage",
    "examples",
    "notes",
    "command history",
    "command history:",
    "command history ",
}

TERMINATORS = {
    "syntax",
    "description",
    "usage",
    "examples",
    "notes",
    "command history",
}


def _clean(text: object) -> str:
    return normalize_whitespace(text)


@lru_cache(maxsize=4096)
def _read_text_cached(path_str: str) -> str:
    path = Path(path_str)
    return path.read_text(encoding="utf-8", errors="ignore")


def _normalize_heading(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean(text).lower()).strip()


def _is_probable_heading(line: str) -> bool:
    stripped = _clean(line)
    if not stripped:
        return False
    if len(stripped) > 120:
        return False
    if stripped.endswith(":") and stripped[:1].islower():
        return False
    if stripped.startswith(("n ", "o ", "l ", "- ", "* ", "1.", "2.", "3.")):
        return False
    if stripped.startswith("switch#"):
        return False
    if re.search(r"[.!?]$", stripped):
        return False
    normalized = _normalize_heading(stripped)
    if not normalized:
        return False
    if len(normalized.split()) > 10:
        return False
    return True


def _is_boilerplate_line(line: str) -> bool:
    stripped = _clean(line)
    if not stripped:
        return True
    if re.fullmatch(r".+\|\s*\d+", stripped):
        return True
    if re.fullmatch(r"chapter\s+\d+", _normalize_heading(stripped)):
        return True
    if re.fullmatch(r"section\s+\d+", _normalize_heading(stripped)):
        return True
    if "user guide" in _normalize_heading(stripped) and "|" in stripped:
        return True
    return False


def _candidate_source_files(row: Dict[str, object], source_root: Path) -> List[Path]:
    candidates: List[Path] = []
    matched = _clean(row.get("matched_markdown_file", ""))
    if matched:
        path = Path(matched)
        if path.exists():
            candidates.append(path)
    source_file = _clean(row.get("source_file", ""))
    if source_file:
        path = source_root / source_file
        if path.exists():
            candidates.append(path)
        parts = [part for part in source_file.replace("\\", "/").split("/") if part]
        if len(parts) >= 2:
            alt = source_root / parts[0] / parts[1]
            if alt.exists() and alt.is_dir():
                for item in sorted(alt.glob("*.md")):
                    candidates.append(item)
    unique: List[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if unique:
        return unique

    switch = _clean(row.get("slots", {}).get("switch", "")) if isinstance(row.get("slots", {}), dict) else ""
    version = _clean(row.get("slots", {}).get("version", "")) if isinstance(row.get("slots", {}), dict) else ""
    if switch and version:
        version_dir = source_root / switch / version.replace("_", ".")
        if not version_dir.exists():
            version_dir = source_root / switch / version.replace(".", "_")
        if version_dir.exists():
            md_files = sorted(version_dir.rglob("*.md"))
            if md_files:
                return md_files
    return unique


def _infer_section_hint(row: Dict[str, object]) -> str:
    section = _clean(row.get("section", ""))
    document_title = _clean(row.get("document_title", ""))
    if section:
        return section
    if document_title:
        return document_title

    slots = row.get("slots", {}) if isinstance(row.get("slots", {}), dict) else {}
    if isinstance(slots, dict):
        for key in ("topic", "command", "feature", "title", "question_type"):
            value = _clean(slots.get(key, ""))
            if value:
                return value

    question = _clean(row.get("input_text", ""))
    if question:
        q = re.sub(r"^For\s+.*?,\s*what\s+is\s+the\s+", "", question, flags=re.IGNORECASE)
        q = re.sub(r"^For\s+.*?,\s*what\s+does\s+the\s+", "", q, flags=re.IGNORECASE)
        q = re.sub(r"^For\s+.*?,\s*what\s+is\s+", "", q, flags=re.IGNORECASE)
        q = re.sub(r"^What\s+is\s+the\s+", "", q, flags=re.IGNORECASE)
        q = re.sub(r"^What\s+is\s+", "", q, flags=re.IGNORECASE)
        q = re.sub(r"^What\s+does\s+the\s+", "", q, flags=re.IGNORECASE)
        q = re.sub(r"^How\s+do\s+I\s+", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\s+(?:command|overview|guide|feature)$", "", q, flags=re.IGNORECASE)
        q = re.sub(r"[:?]\s*$", "", q)
        q = _clean(q)
        if q:
            return q

    return ""


def _token_overlap_score(text: str, *needles: str) -> int:
    normalized_text = _normalize_heading(text)
    if not normalized_text:
        return 0
    tokens = set(normalized_text.split())
    score = 0
    for needle in needles:
        normalized_needle = _normalize_heading(needle)
        if not normalized_needle:
            continue
        needle_tokens = set(normalized_needle.split())
        score += len(tokens & needle_tokens)
        if normalized_needle in normalized_text:
            score += 5
    return score


def _rank_source_files(row: Dict[str, object], source_root: Path, files: Sequence[Path]) -> List[Path]:
    question = _clean(row.get("input_text", ""))
    section_hint = _infer_section_hint(row)
    slots = row.get("slots", {}) if isinstance(row.get("slots", {}), dict) else {}
    switch = _clean(slots.get("switch", "")) if isinstance(slots, dict) else ""
    version = _clean(slots.get("version", "")) if isinstance(slots, dict) else ""
    ranked: List[Tuple[int, Path]] = []
    for path in files:
        score = 0
        path_text = str(path)
        score += _token_overlap_score(path_text, switch, version)
        score += _token_overlap_score(path.name, section_hint)
        score += _token_overlap_score(path.stem, section_hint)
        try:
            preview = _read_text_cached(str(path))[:12000]
        except Exception:
            preview = ""
        if preview:
            score += _token_overlap_score(preview, section_hint, question)
        ranked.append((score, path))
    ranked.sort(key=lambda item: (-item[0], str(item[1]).lower()))
    return [path for _score, path in ranked]


def _find_section_line(lines: Sequence[str], section: str) -> int:
    normalized_section = _normalize_heading(section)
    if not normalized_section:
        return -1

    for idx, line in enumerate(lines):
        normalized_line = _normalize_heading(line)
        if not normalized_line:
            continue
        if normalized_line == normalized_section:
            return idx
    return -1


def _extract_excerpt(text: str, section: str, *, window_lines: int = 120, window_chars: int = 8000) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start_idx = _find_section_line(lines, section)
    if start_idx >= 0:
        start = max(0, start_idx - 2)
        end = min(len(lines), start_idx + window_lines)
        excerpt = "\n".join(lines[start:end]).strip()
        if excerpt:
            return excerpt[:window_chars]

    section_text = _clean(section)
    if section_text:
        match = re.search(re.escape(section_text), text, flags=re.IGNORECASE)
        if match:
            start = max(0, match.start() - 200)
            end = min(len(text), match.start() + window_chars)
            return text[start:end].strip()

    return ""


def _split_blocks(lines: Sequence[str]) -> List[List[str]]:
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        stripped = line.rstrip()
        if not stripped.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(stripped)
    if current:
        blocks.append(current)
    return blocks


def _extract_block_after_label(excerpt: str, label: str) -> str:
    lines = excerpt.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized_label = _normalize_heading(label)
    for idx, line in enumerate(lines):
        if _normalize_heading(line) != normalized_label:
            continue
        block: List[str] = []
        for follow in lines[idx + 1 :]:
            stripped = _clean(follow)
            normalized = _normalize_heading(stripped)
            if not stripped:
                if block:
                    block.append("")
                continue
            if normalized in TERMINATORS and block:
                break
            if _is_probable_heading(stripped) and block and normalized not in {"", normalized_label} and len(block) > 2:
                break
            block.append(stripped)
        candidate = "\n".join(block).strip()
        if candidate:
            return candidate
    return ""


def _extract_block_after_heading(excerpt: str, heading: str) -> str:
    lines = excerpt.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized_heading = _normalize_heading(heading)
    if not normalized_heading:
        return ""
    for idx, line in enumerate(lines):
        if _normalize_heading(line) != normalized_heading:
            continue
        block: List[str] = []
        for follow in lines[idx + 1 :]:
            stripped = _clean(follow)
            normalized = _normalize_heading(stripped)
            if not stripped:
                if block:
                    block.append("")
                continue
            if _is_boilerplate_line(stripped) and not block:
                continue
            if _is_probable_heading(stripped) and normalized != normalized_heading:
                if block:
                    break
                continue
            block.append(stripped)
        candidate = "\n".join(block).strip()
        if candidate:
            return candidate
    return ""


def _extract_candidate_answer(excerpt: str, intent: str, section_hint: str = "") -> str:
    intent = _clean(intent)
    section_hint = _clean(section_hint)
    section_block = _extract_block_after_heading(excerpt, section_hint) if section_hint else ""
    syntax_block = _extract_block_after_label(excerpt, "Syntax")
    description_block = _extract_block_after_label(excerpt, "Description")
    usage_block = _extract_block_after_label(excerpt, "Usage")
    example_block = _extract_block_after_label(excerpt, "Examples")

    if intent in SYNTAX_INTENTS:
        for candidate in (section_block, syntax_block, description_block, usage_block, example_block):
            if candidate:
                return candidate
    if intent == "cli_meaning":
        for candidate in (description_block, usage_block, example_block, syntax_block, section_block):
            if candidate:
                return candidate
    if intent in EXPLANATION_INTENTS:
        for candidate in (section_block, description_block, usage_block, example_block, syntax_block):
            if candidate:
                return candidate

    lines = [line.strip() for line in excerpt.splitlines()]
    cleaned = [line for line in lines if line and _normalize_heading(line) not in LABELS]
    if cleaned:
        return "\n".join(cleaned).strip()
    return ""


def _qwen_prompt(question: str, intent: str, slots: Dict[str, str], excerpt: str) -> str:
    return f"""You are repairing a product documentation dataset for Aruba AOS-CX.
You are acting as a Qwen2.5-7B-Instruct style formatter only.

Use only the source excerpt below.
Do not invent facts.
Do not change facts.
Do not change commands, versions, switch models, bug IDs, caveat meaning, workaround meaning, or any other factual detail.
If you improve the wording, preserve the meaning exactly.

Question:
{_clean(question)}

Predicted intent:
{_clean(intent)}

Slots:
{json.dumps(slots, ensure_ascii=False, sort_keys=True)}

Source excerpt:
{excerpt.strip()}

Task:
Return the best answer supported by the source excerpt.
If this is a syntax question, return only the exact syntax or syntax block.
If this is a concept explanation question, return a clear and faithful explanation.
If the excerpt only contains headings or cannot support an answer, return exactly: missing_source_evidence

Return only the answer.
"""


def _repair_row(
    row: Dict[str, object],
    source_root: Path,
    formatter_backend: str,
    formatter_model: str,
) -> Dict[str, object]:
    question = _clean(row.get("input_text", ""))
    intent = _clean(row.get("intent", ""))
    slots = row.get("slots", {}) if isinstance(row.get("slots", {}), dict) else {}
    source_files = _rank_source_files(row, source_root, _candidate_source_files(row, source_root))
    section_hint = _infer_section_hint(row)

    excerpt = ""
    source_file_used = ""
    for path in source_files:
        try:
            source_text = _read_text_cached(str(path))
        except Exception:
            continue
        excerpt = _extract_excerpt(source_text, section_hint)
        if excerpt:
            source_file_used = str(path)
            break

        # If the exact hint is missing, try the question text itself as a looser section probe.
        if question:
            loose_hint = re.sub(r"^(For\s+.*?,\s*)", "", question, flags=re.IGNORECASE)
            loose_hint = re.sub(r"^(What\s+is\s+the\s+|What\s+is\s+|What\s+does\s+the\s+|How\s+do\s+I\s+)", "", loose_hint, flags=re.IGNORECASE)
            loose_hint = re.sub(r"[?.:]\s*$", "", loose_hint).strip()
            if loose_hint and loose_hint != section_hint:
                excerpt = _extract_excerpt(source_text, loose_hint)
                if excerpt:
                    source_file_used = str(path)
                    break

    repaired = dict(row)
    repaired["repair_status"] = "missing_source_evidence"
    repaired["source_excerpt_file"] = source_file_used or None
    repaired["source_excerpt"] = excerpt or None
    repaired["qwen_used"] = False
    repaired["qwen_answer"] = None
    repaired["qwen_validation_passed"] = False
    repaired["final_answer"] = None

    if not excerpt:
        return repaired

    candidate = _extract_candidate_answer(excerpt, intent, section_hint)
    final_answer = candidate
    qwen_used = False
    qwen_answer = None
    qwen_validation_passed = False

    candidate_word_count = len(_clean(candidate).split())
    should_use_qwen = formatter_backend.lower() == "ollama" and bool(formatter_model) and (
        intent in EXPLANATION_INTENTS
        or candidate_word_count >= 12
        or intent in SYNTAX_INTENTS
    )

    if should_use_qwen:
        try:
            qwen_used = True
            qwen_answer, _model_used, _error = call_formatter_backend(
                formatter_backend,
                formatter_model,
                question,
                intent,
                slots,
                candidate or excerpt,
            )
            if qwen_answer:
                qwen_validation_passed, _reason = validate_qwen_answer(
                    intent,
                    slots,
                    candidate or excerpt,
                    qwen_answer,
                    data_family="product_documentation",
                )
                if qwen_validation_passed and _clean(qwen_answer):
                    final_answer = qwen_answer
        except Exception:
            qwen_validation_passed = False

    final_answer = _clean(final_answer)
    if not final_answer:
        final_answer = candidate or "missing_source_evidence"
    if final_answer != "missing_source_evidence" and intent not in SYNTAX_INTENTS:
        if final_answer and final_answer[-1] not in ".!?":
            final_answer = f"{final_answer}."

    repaired["repair_status"] = "repaired" if final_answer != "missing_source_evidence" else "missing_source_evidence"
    repaired["qwen_used"] = qwen_used
    repaired["qwen_answer"] = qwen_answer
    repaired["qwen_validation_passed"] = qwen_validation_passed
    repaired["final_answer"] = final_answer if final_answer != "missing_source_evidence" else None
    repaired["target_value"] = repaired["final_answer"] or row.get("target_value", "")
    if repaired.get("reference") is not None and repaired["final_answer"]:
        repaired["reference"] = repaired["final_answer"]
    return repaired


def _should_use_qwen_for_repaired_row(repaired: Dict[str, object]) -> bool:
    intent = _clean(repaired.get("intent", ""))
    final_answer = _clean(repaired.get("final_answer", ""))
    if not intent or not final_answer or final_answer == "missing_source_evidence":
        return False
    candidate_word_count = len(final_answer.split())
    return intent in EXPLANATION_INTENTS or candidate_word_count >= 12


def _rewrite_repaired_row_with_qwen(
    repaired: Dict[str, object],
    formatter_backend: str,
    formatter_model: str,
) -> Dict[str, object]:
    if formatter_backend.lower() != "ollama" or not formatter_model:
        return repaired
    if not _should_use_qwen_for_repaired_row(repaired):
        return repaired

    question = _clean(repaired.get("input_text", ""))
    intent = _clean(repaired.get("intent", ""))
    slots = repaired.get("slots", {}) if isinstance(repaired.get("slots", {}), dict) else {}
    source_excerpt = _clean(repaired.get("source_excerpt", ""))
    candidate = _clean(repaired.get("final_answer", ""))

    updated = dict(repaired)
    updated["qwen_used"] = False
    updated["qwen_answer"] = None
    updated["qwen_validation_passed"] = False
    try:
        qwen_answer, _model_used, _error = call_formatter_backend(
            formatter_backend,
            formatter_model,
            question,
            intent,
            slots,
            source_excerpt or candidate,
        )
        updated["qwen_used"] = True
        updated["qwen_answer"] = qwen_answer
        qwen_validation_passed = False
        if qwen_answer:
            qwen_validation_passed, _reason = validate_qwen_answer(
                intent,
                slots,
                source_excerpt or candidate,
                qwen_answer,
                data_family="product_documentation",
            )
        updated["qwen_validation_passed"] = qwen_validation_passed
        if qwen_validation_passed and _clean(qwen_answer):
            updated["final_answer"] = _clean(qwen_answer)
            if updated["final_answer"] != "missing_source_evidence" and intent not in SYNTAX_INTENTS:
                if updated["final_answer"] and updated["final_answer"][-1] not in ".!?":
                    updated["final_answer"] = f"{updated['final_answer']}."
    except Exception:
        updated["qwen_validation_passed"] = False
    return updated


def _worker_repair_file(
    input_path_str: str,
    product_root_str: str,
    source_root_str: str,
    output_root_str: str,
    formatter_backend: str,
    formatter_model: str,
) -> Dict[str, object]:
    input_path = Path(input_path_str)
    product_root = Path(product_root_str)
    source_root = Path(source_root_str)
    output_root = Path(output_root_str)

    rows = read_jsonl(input_path)
    repaired_file_rows: List[Dict[str, object]] = []
    repaired_rows = 0
    missing_rows = 0
    qwen_used_rows = 0
    qwen_valid_rows = 0

    for row in rows:
        repaired = _repair_row(row, source_root, formatter_backend, formatter_model)
        repaired_file_rows.append(repaired)
        if repaired.get("repair_status") == "repaired":
            repaired_rows += 1
        else:
            missing_rows += 1
        if repaired.get("qwen_used"):
            qwen_used_rows += 1
        if repaired.get("qwen_validation_passed"):
            qwen_valid_rows += 1

    output_path = output_root / input_path.relative_to(product_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, repaired_file_rows)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": len(rows),
        "repaired_rows": repaired_rows,
        "missing_rows": missing_rows,
        "qwen_loaded": formatter_backend.lower() == "ollama",
        "qwen_error": None,
        "qwen_used_rows": qwen_used_rows,
        "qwen_valid_rows": qwen_valid_rows,
    }


def _configure_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("product_docs_repair")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _iter_product_jsonl_files(root: Path) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        return files
    for path in sorted(root.rglob("product_dataset_repaired.jsonl")):
        if path.is_file():
            files.append(path)
    for path in sorted(root.rglob("product_review_remaining.jsonl")):
        if path.is_file():
            files.append(path)
    unique: List[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair the Aruba product-doc corpus from raw markdown using Ollama formatting fallback.")
    parser.add_argument("--product_root", type=Path, default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--qwen_model_path", type=Path, default=QWEN_MODEL_PATH)
    parser.add_argument("--formatter_backend", default=DEFAULT_FORMATTER_BACKEND)
    parser.add_argument("--formatter_model", default=DEFAULT_FORMATTER_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no_qwen", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log_file", type=Path, default=None)
    args = parser.parse_args()

    log_file = args.log_file or (args.output_root / "repair_product_docs_corpus.log")
    logger = _configure_logging(log_file)

    input_files = _iter_product_jsonl_files(args.product_root)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    parallel_mode = args.workers > 1 and args.limit == 0
    formatter_backend = _clean(args.formatter_backend) or DEFAULT_FORMATTER_BACKEND
    formatter_model = _clean(args.formatter_model) or DEFAULT_FORMATTER_MODEL

    qwen_loaded = formatter_backend.lower() == "ollama" and not args.no_qwen
    qwen_error = None

    report = {
        "input_root": str(args.product_root),
        "source_root": str(args.source_root),
        "output_root": str(output_root),
        "formatter_backend": formatter_backend,
        "formatter_model": formatter_model,
        "qwen_loaded": qwen_loaded,
        "qwen_error": qwen_error,
        "files_processed": [],
        "summary": Counter(),
        "missing_rows": [],
        "repaired_rows": [],
    }

    combined_rows: List[Dict[str, object]] = []
    total_rows = 0
    repaired_rows = 0
    missing_rows = 0
    qwen_used_rows = 0
    qwen_valid_rows = 0

    if parallel_mode:
        max_workers = min(max(1, args.workers), len(input_files) or 1)
        logger.info("Starting parallel repair with %d workers across %d files.", max_workers, len(input_files))
        futures = {}
        results_by_index: Dict[int, Dict[str, object]] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for index, input_path in enumerate(input_files):
                futures[executor.submit(
                    _worker_repair_file,
                    str(input_path),
                    str(args.product_root),
                    str(args.source_root),
                    str(output_root),
                    formatter_backend,
                    formatter_model,
                )] = index
            completed = 0
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                results_by_index[index] = result
                completed += 1
                logger.info(
                    "Finished %d/%d: %s | repaired=%d missing=%d",
                    completed,
                    len(input_files),
                    Path(result["input_path"]).name,
                    int(result["repaired_rows"]),
                    int(result["missing_rows"]),
                )
        ordered_results = [results_by_index[idx] for idx in range(len(input_files))]
        for result in ordered_results:
            total_rows += int(result["rows"])
            repaired_rows += int(result["repaired_rows"])
            missing_rows += int(result["missing_rows"])
            qwen_used_rows += int(result["qwen_used_rows"])
            qwen_valid_rows += int(result["qwen_valid_rows"])
            report["files_processed"].append(result)
            output_path = Path(result["output_path"])
            repaired_file_rows = read_jsonl(output_path)
            if qwen_loaded and not args.no_qwen:
                rewritten_rows: List[Dict[str, object]] = []
                file_changed = False
                for repaired in repaired_file_rows:
                    rewritten = _rewrite_repaired_row_with_qwen(repaired, formatter_backend, formatter_model)
                    rewritten_rows.append(rewritten)
                    if rewritten != repaired:
                        file_changed = True
                    if rewritten.get("qwen_used"):
                        qwen_used_rows += 1
                    if rewritten.get("qwen_validation_passed"):
                        qwen_valid_rows += 1
                if file_changed:
                    write_jsonl(output_path, rewritten_rows)
                    repaired_file_rows = rewritten_rows
            combined_rows.extend(repaired_file_rows)
            for repaired in repaired_file_rows:
                if repaired.get("repair_status") == "repaired":
                    report["repaired_rows"].append(
                        {
                            "input_text": _clean(repaired.get("input_text", "")),
                            "intent": _clean(repaired.get("intent", "")),
                            "source_file": _clean(repaired.get("source_file", "")),
                            "section": _clean(repaired.get("section", "")),
                            "final_answer": _clean(repaired.get("final_answer", "")),
                            "qwen_used": bool(repaired.get("qwen_used")),
                            "qwen_validation_passed": bool(repaired.get("qwen_validation_passed")),
                        }
                    )
                else:
                    report["missing_rows"].append(
                        {
                            "input_text": _clean(repaired.get("input_text", "")),
                            "intent": _clean(repaired.get("intent", "")),
                            "source_file": _clean(repaired.get("source_file", "")),
                            "section": _clean(repaired.get("section", "")),
                        }
                    )
    else:
        for input_path in input_files:
            rows = read_jsonl(input_path)
            if args.limit and total_rows >= args.limit:
                break
            repaired_file_rows: List[Dict[str, object]] = []
            for row in rows:
                if args.limit and total_rows >= args.limit:
                    break
                total_rows += 1
                repaired = _repair_row(row, args.source_root, formatter_backend, formatter_model)
                repaired_file_rows.append(repaired)
                combined_rows.append(repaired)
                if repaired.get("repair_status") == "repaired":
                    repaired_rows += 1
                    report["repaired_rows"].append(
                        {
                            "input_text": _clean(row.get("input_text", "")),
                            "intent": _clean(row.get("intent", "")),
                            "source_file": _clean(row.get("source_file", "")),
                            "section": _clean(row.get("section", "")),
                            "final_answer": _clean(repaired.get("final_answer", "")),
                            "qwen_used": bool(repaired.get("qwen_used")),
                            "qwen_validation_passed": bool(repaired.get("qwen_validation_passed")),
                        }
                    )
                else:
                    missing_rows += 1
                    report["missing_rows"].append(
                        {
                            "input_text": _clean(row.get("input_text", "")),
                            "intent": _clean(row.get("intent", "")),
                            "source_file": _clean(row.get("source_file", "")),
                            "section": _clean(row.get("section", "")),
                        }
                    )
                if repaired.get("qwen_used"):
                    qwen_used_rows += 1
                if repaired.get("qwen_validation_passed"):
                    qwen_valid_rows += 1

            output_path = output_root / input_path.relative_to(args.product_root)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(output_path, repaired_file_rows)
            report["files_processed"].append(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "rows": len(rows),
                    "repaired_rows": sum(1 for row in repaired_file_rows if row.get("repair_status") == "repaired"),
                    "missing_rows": sum(1 for row in repaired_file_rows if row.get("repair_status") != "repaired"),
                }
            )
            logger.info(
                "Finished %s | repaired=%d missing=%d",
                input_path.name,
                sum(1 for row in repaired_file_rows if row.get("repair_status") == "repaired"),
                sum(1 for row in repaired_file_rows if row.get("repair_status") != "repaired"),
            )

    combined_path = output_root / "all_switches_product_dataset_final_repaired.jsonl"
    write_jsonl(combined_path, combined_rows)

    report_payload = {
        **report,
        "summary": {
            "total_rows_processed": total_rows,
            "repaired_rows": repaired_rows,
            "missing_rows": missing_rows,
            "qwen_used_rows": qwen_used_rows,
            "qwen_valid_rows": qwen_valid_rows,
        },
    }
    report_path = output_root / "product_docs_repair_report.json"
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(json.dumps(report_payload["summary"], indent=2))
    logger.info("Wrote repaired corpus to %s", output_root)


if __name__ == "__main__":
    main()
