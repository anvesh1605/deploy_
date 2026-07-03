from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PRODUCT_INTENTS = [
    "cli_syntax",
    "cli_purpose",
    "cli_parameters",
    "cli_examples",
    "configuration_steps",
    "event_id_meaning",
    "event_id_action",
    "concept_explanation",
    "feature_limitations",
    "product_troubleshooting",
    "show_command_usage",
    "rest_api_usage",
    "snmp_behavior",
    "data_not_available",
    "out_of_domain",
]

PRODUCT_NEGATIVE_ROWS = [
    {
        "input_text": "what is my name?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
        "reference": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
    },
    {
        "input_text": "tell me a joke",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
        "reference": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
    },
    {
        "input_text": "what is the weather today?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
        "reference": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
    },
    {
        "input_text": "what is 2 plus 2?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
        "reference": "This is a domain-specific Aruba product documentation assistant, so I cannot answer this question because it is not related to Aruba product documentation.",
    },
    {
        "input_text": "For 9999 AOS-CX 10.18, what CLI syntax is documented for SNMP?",
        "intent": "data_not_available",
        "slots": {"switch": "9999", "version": "10_18", "feature": "SNMP"},
        "target_value": "This particular data is not available in the current Aruba product documentation dataset.",
        "reference": "This particular data is not available in the current Aruba product documentation dataset.",
    },
    {
        "input_text": "For 4100i AOS-CX 10.99, what is the REST API usage for a missing feature?",
        "intent": "data_not_available",
        "slots": {"switch": "4100i", "version": "10_99", "feature": "REST"},
        "target_value": "This particular data is not available in the current Aruba product documentation dataset.",
        "reference": "This particular data is not available in the current Aruba product documentation dataset.",
    },
    {
        "input_text": "For 6200 AOS-CX 10.18, what product documentation command explains SNMP?",
        "intent": "data_not_available",
        "slots": {"switch": "6200", "version": "10_18", "feature": "SNMP"},
        "target_value": "This particular data is not available in the current Aruba product documentation dataset.",
        "reference": "This particular data is not available in the current Aruba product documentation dataset.",
    },
]


DEFAULT_INPUT_DIR = Path(r"C:\Hpe\Train\Data\product_docs_final")
DEFAULT_REPAIR_SOURCE_DIRS = [
    Path(r"C:\Hpe\Train\Data\product_docs_final_repair_focus"),
    Path(r"C:\Hpe\Train\Data\product_docs_final_repaired"),
]
DEFAULT_OUTPUT_DIR = Path(r"C:\Hpe\Train\outputs_product_lstm_v2")
DEFAULT_SEED = 42


@dataclass
class PatchMatch:
    input_text: str
    original_target_value: str
    patched_target_value: str
    source_file: str
    source_type: str
    score: float


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_question_key(text: object) -> str:
    value = normalize_text(text).lower()
    value = value.rstrip(" ?.")
    value = re.sub(r"\s+", " ", value)
    return value


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def iter_jsonl_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.jsonl"))


def _load_jsonl_file(path: Path) -> List[Dict[str, object]]:
    return read_jsonl(path)


def load_jsonl_files(paths: Sequence[Path], workers: int = 0) -> List[Dict[str, object]]:
    file_paths = [path for path in paths if path.exists()]
    if not file_paths:
        return []
    rows: List[Dict[str, object]] = []
    if workers and workers > 1 and len(file_paths) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for chunk in executor.map(_load_jsonl_file, file_paths):
                rows.extend(chunk)
        return rows
    for path in file_paths:
        rows.extend(read_jsonl(path))
    return rows


def load_dataset_rows(data_dir: Path, workers: int = 0) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    file_paths = iter_jsonl_files(data_dir)
    rows = load_jsonl_files(file_paths, workers=workers)
    reason_counts: Counter[str] = Counter()
    reason_counts["files"] = len(file_paths)
    reason_counts["rows"] = len(rows)
    return rows, dict(reason_counts)


def collect_unique_records(rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    records: List[Dict[str, object]] = []
    reasons: Counter[str] = Counter()
    seen = set()
    for row in rows:
        input_text = normalize_text(row.get("input_text"))
        intent = normalize_text(row.get("intent"))
        target_value = normalize_text(row.get("target_value"))
        if not input_text or not intent or not target_value:
            reasons["missing_required_field"] += 1
            continue
        key = (normalize_question_key(input_text), intent, target_value)
        if key in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(key)
        records.append(
            {
                "input_text": input_text,
                "intent": intent,
                "slots": dict(row.get("slots") if isinstance(row.get("slots"), dict) else {}),
                "target_value": target_value,
                "reference": normalize_text(row.get("reference")) or target_value,
                "source_type": normalize_text(row.get("source_type")),
                "source_file": normalize_text(row.get("source_file")),
                "document_title": normalize_text(row.get("document_title")),
                "section": normalize_text(row.get("section")),
            }
        )
    return records, dict(reasons)


def load_repair_source_rows(repair_dirs: Sequence[Path], workers: int = 0) -> List[Dict[str, object]]:
    file_paths: List[Path] = []
    for repair_dir in repair_dirs:
        file_paths.extend(iter_jsonl_files(repair_dir))
    rows = load_jsonl_files(file_paths, workers=workers)
    return [row for row in rows if normalize_text(row.get("input_text")) and normalize_text(row.get("target_value"))]


def target_quality_score(target_value: object) -> float:
    text = normalize_text(target_value)
    if not text:
        return 0.0
    score = float(len(text))
    if len(text.split()) > 8:
        score += 40.0
    if "." in text or ":" in text or "\n" in text:
        score += 12.0
    if text.lower().startswith(("the ", "this ", "when ", "for ", "use ", "to ")):
        score += 8.0
    if text.lower().startswith(("syntax:", "the syntax of", "no workaround is documented")):
        score += 6.0
    if len(text) < 30:
        score -= 25.0
    if len(text) < 15:
        score -= 20.0
    return score


def build_repair_index(rows: Sequence[Dict[str, object]]) -> Dict[str, PatchMatch]:
    index: Dict[str, PatchMatch] = {}
    for row in rows:
        input_text = normalize_text(row.get("input_text"))
        target_value = normalize_text(row.get("target_value"))
        if not input_text or not target_value:
            continue
        key = normalize_question_key(input_text)
        score = target_quality_score(target_value)
        current = index.get(key)
        candidate = PatchMatch(
            input_text=input_text,
            original_target_value="",
            patched_target_value=target_value,
            source_file=normalize_text(row.get("source_file")) or normalize_text(row.get("source_excerpt_file")),
            source_type=normalize_text(row.get("source_type")) or normalize_text(row.get("repair_status")) or "repair_source",
            score=score,
        )
        if current is None or candidate.score > current.score or (
            math.isclose(candidate.score, current.score) and len(candidate.patched_target_value) > len(current.patched_target_value)
        ):
            index[key] = candidate
    return index


def collect_rejected_patch_candidates(rows: Sequence[Dict[str, object]], repair_index: Dict[str, PatchMatch]) -> List[Dict[str, object]]:
    rejected: List[Dict[str, object]] = []
    for row in rows:
        input_text = normalize_text(row.get("input_text"))
        key = normalize_question_key(input_text)
        candidate = repair_index.get(key)
        if candidate is None:
            continue
        source_target = normalize_text(row.get("target_value"))
        if source_target == candidate.patched_target_value:
            continue
        if normalize_text(row.get("source_file")) == candidate.source_file and source_target != candidate.patched_target_value:
            rejected.append(
                {
                    "input_text": input_text,
                    "intent": normalize_text(row.get("intent")),
                    "slots": dict(row.get("slots") if isinstance(row.get("slots"), dict) else {}),
                    "original_target_value": source_target,
                    "rejected_target_value": candidate.patched_target_value,
                    "source_file": normalize_text(row.get("source_file")),
                    "source_type": normalize_text(row.get("source_type")) or "repair_source",
                }
            )
    return rejected


def is_weak_target_value(target_value: object) -> bool:
    text = normalize_text(target_value)
    if not text:
        return True
    if len(text) < 30:
        return True
    if len(text.split()) < 4:
        return True
    if re.fullmatch(r"[A-Za-z0-9_\-./ ]{1,28}", text):
        return True
    if text.lower().startswith(("syntax:", "the syntax is", "the syntax of")) and len(text) < 55:
        return True
    return False


def patch_records(
    records: Sequence[Dict[str, object]],
    repair_index: Dict[str, PatchMatch],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[str, int]]:
    patched_records: List[Dict[str, object]] = []
    needs_review: List[Dict[str, object]] = []
    rejected_patches: List[Dict[str, object]] = []
    stats: Counter[str] = Counter()

    for row in records:
        current = dict(row)
        input_text = normalize_text(current.get("input_text"))
        original_target = normalize_text(current.get("target_value"))
        key = normalize_question_key(input_text)
        candidate = repair_index.get(key)
        if candidate is not None and normalize_text(candidate.patched_target_value):
            if candidate.patched_target_value != original_target:
                current["target_value"] = candidate.patched_target_value
                current["repair_source_file"] = candidate.source_file
                current["repair_source_type"] = candidate.source_type
                current["patch_reason"] = "exact_question_match"
                stats["patched"] += 1
            else:
                stats["matched_but_unchanged"] += 1
        else:
            stats["no_match"] += 1
            if is_weak_target_value(original_target):
                needs_review.append(
                    {
                        "input_text": input_text,
                        "intent": normalize_text(current.get("intent")),
                        "slots": deepcopy(current.get("slots") if isinstance(current.get("slots"), dict) else {}),
                        "target_value": original_target,
                        "reason": "weak_target_value_no_repair_match",
                    }
                )
        patched_records.append(current)

    return patched_records, needs_review, rejected_patches, dict(stats)


def add_negative_samples(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    augmented = [dict(row) for row in records]
    augmented.extend(deepcopy(PRODUCT_NEGATIVE_ROWS))
    return augmented


def stratified_split(
    records: Sequence[Dict[str, object]],
    seed: int,
    intents: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    import random

    label_order = list(intents or PRODUCT_INTENTS)
    rng = random.Random(seed)
    by_intent: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        by_intent[normalize_text(record.get("intent"))].append(dict(record))

    train: List[Dict[str, object]] = []
    val: List[Dict[str, object]] = []
    test: List[Dict[str, object]] = []

    for intent in label_order:
        group = by_intent.get(intent, [])
        rng.shuffle(group)
        n = len(group)
        if n == 0:
            continue
        if n == 1:
            train.extend(group)
            continue
        if n == 2:
            train.append(group[0])
            test.append(group[1])
            continue

        n_test = max(1, int(round(n * 0.1)))
        n_val = max(1, int(round(n * 0.1)))
        n_train = n - n_test - n_val
        while n_train < 1:
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
            n_train = n - n_test - n_val
        if n_train < 1:
            n_train = 1

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def label_maps(records: Sequence[Dict[str, object]], intents: Optional[Sequence[str]] = None) -> Tuple[List[str], Dict[str, int], Dict[str, str]]:
    label_order = list(intents or PRODUCT_INTENTS)
    label_names = [label for label in label_order if any(normalize_text(record.get("intent")) == label for record in records)]
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    id_to_label = {str(idx): label for label, idx in label_to_id.items()}
    return label_names, label_to_id, id_to_label


def summarize_patch_report(
    records: Sequence[Dict[str, object]],
    patched_records: Sequence[Dict[str, object]],
    repair_source_rows: Sequence[Dict[str, object]],
    patch_stats: Dict[str, int],
    needs_review: Sequence[Dict[str, object]],
    rejected_patches: Sequence[Dict[str, object]],
    train_records: Sequence[Dict[str, object]],
    val_records: Sequence[Dict[str, object]],
    test_records: Sequence[Dict[str, object]],
    added_negative_samples: int,
) -> Dict[str, object]:
    original_targets = Counter(normalize_text(row.get("target_value")) for row in records)
    patched_targets = Counter(normalize_text(row.get("target_value")) for row in patched_records)
    changed_rows = sum(1 for original, patched in zip(records, patched_records) if normalize_text(original.get("target_value")) != normalize_text(patched.get("target_value")))
    return {
        "original_rows": len(records),
        "patched_rows": len(patched_records),
        "repair_source_rows": len(repair_source_rows),
        "changed_rows": changed_rows,
        "patch_stats": patch_stats,
        "needs_review_rows": len(needs_review),
        "rejected_patch_rows": len(rejected_patches),
        "negative_samples_added": added_negative_samples,
        "split_sizes": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "top_original_targets": original_targets.most_common(10),
        "top_patched_targets": patched_targets.most_common(10),
    }
