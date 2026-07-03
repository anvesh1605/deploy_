from __future__ import annotations

import argparse
import urllib.error
import urllib.request
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from backend.config import (
    PRODUCT_LSTM_MODEL_PATH,
    RELEASE_AVAILABILITY_PATH,
    RELEASE_BUG_METADATA_PATH,
    RELEASE_LSTM_MODEL_PATH,
    RELEASE_LOOKUP_DATA_PATH,
    RELEASE_LOOKUP_INDEX_PATH,
)
from backend.runtime import ProductRuntime, QwenBundle, ReleaseRuntime
from lstm_lookup import normalize_whitespace, read_jsonl


ROOT = Path(__file__).resolve().parent
PRODUCT_DATA_PATHS = [
    ROOT / "Data" / "product_docs_final" / "all_switches_product_dataset_final.jsonl",
    ROOT / "Data" / "product_docs_final" / "all_switches_product_review_remaining.jsonl",
]
RELEASE_DATA_ROOT = ROOT / "Data" / "Release_Notes"
OUTPUT_DIR = ROOT / "outputs_eval_1000"
OUTPUT_JSONL = OUTPUT_DIR / "eval_1000_results.jsonl"
REPORT_JSON = OUTPUT_DIR / "eval_1000_report.json"
FAIL_MD = OUTPUT_DIR / "eval_1000_fail_cases.md"
MISSING_JSON = OUTPUT_DIR / "missing_data_report.json"

PRODUCT_EXPECTED_FILES = [
    "product_dataset_repaired.jsonl",
    "product_review_remaining.jsonl",
    "ollama_repair_report.json",
    "ollama_repair_samples.md",
]

EXACT_INTENTS = {
    "bug_category",
    "version_date",
    "release_date",
    "event_id",
    "cli_syntax",
    "show_command_syntax",
    "show_command_usage",
}


def _clean(value: object) -> str:
    return normalize_whitespace(value)


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _clean(prediction).lower().split()
    ref_tokens = _clean(reference).lower().split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: Dict[str, int] = Counter(pred_tokens)
    ref_counts: Dict[str, int] = Counter(ref_tokens)
    overlap = sum(min(pred_counts[token], ref_counts[token]) for token in pred_counts)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = _clean(prediction).lower().split()
    ref_tokens = _clean(reference).lower().split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
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


def _normalize_for_match(text: str) -> str:
    return _clean(text).lower().strip(" .;:")


def _is_exact_intent(intent: str) -> bool:
    return _clean(intent) in EXACT_INTENTS


def _is_correct(intent: str, prediction: str, gold: str) -> bool:
    pred = _normalize_for_match(prediction)
    ref = _normalize_for_match(gold)
    if not pred and not ref:
        return True
    if _is_exact_intent(intent):
        return pred == ref
    if ref and ref in pred:
        return True
    if pred and pred in ref:
        return True
    f1 = _token_f1(prediction, gold)
    rouge = _rouge_l(prediction, gold)
    return f1 >= 0.75 and rouge >= 0.70


def _load_product_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in PRODUCT_DATA_PATHS:
        if path.exists():
            rows.extend(read_jsonl(path))
    return rows


def _load_release_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(RELEASE_DATA_ROOT.rglob("train_chat.jsonl")):
        for row in read_jsonl(path):
            rows.append({**row, "_source_file": str(path)})
    return rows


def _scan_product_coverage() -> Dict[str, object]:
    root = ROOT / "Data" / "product_docs_final"
    by_switch: Dict[str, Dict[str, object]] = {}
    missing_paths: List[Dict[str, object]] = []
    if not root.exists():
        return {"root": str(root), "by_switch": by_switch, "missing_paths": missing_paths}

    for switch_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
        switch = switch_dir.name
        versions: Dict[str, Dict[str, object]] = {}
        for version_dir in sorted([path for path in switch_dir.iterdir() if path.is_dir()]):
            files = {path.name for path in version_dir.iterdir() if path.is_file()}
            missing = [name for name in PRODUCT_EXPECTED_FILES if name not in files]
            versions[version_dir.name] = {
                "present_files": sorted(files),
                "missing_files": missing,
            }
            if missing:
                missing_paths.append(
                    {
                        "switch": switch,
                        "version_dir": version_dir.name,
                        "missing_files": missing,
                    }
                )
        by_switch[switch] = {"versions": versions}

    return {"root": str(root), "by_switch": by_switch, "missing_paths": missing_paths}


def _normalize_product_row(row: Dict[str, object]) -> Dict[str, object]:
    slots = dict(row.get("slots", {}) or {})
    return {
        "domain": "product",
        "question": _clean(row.get("input_text", "")),
        "gold_answer": _clean(row.get("target_value", "")),
        "intent": _clean(row.get("intent", "")),
        "slots": {k: _clean(v) for k, v in slots.items() if _clean(v)},
        "selected_context": {
            "switch": _clean(slots.get("switch", "")),
            "version": _clean(slots.get("version", "")),
            "sub_version": _clean(slots.get("sub_version", "")),
            "feature": _clean(slots.get("feature", "")),
            "category": _clean(slots.get("category", "")),
            "command": _clean(slots.get("command", "")),
            "topic": _clean(slots.get("topic", "")),
            "event_id": _clean(slots.get("event_id", "")),
        },
        "source_file": "product_docs_final",
    }


def _normalize_release_row(row: Dict[str, object]) -> Dict[str, object]:
    messages = row.get("messages", []) or []
    question = ""
    gold = ""
    if isinstance(messages, list) and messages:
        first = messages[0] if len(messages) > 0 and isinstance(messages[0], dict) else {}
        second = messages[1] if len(messages) > 1 and isinstance(messages[1], dict) else {}
        question = _clean(first.get("content", ""))
        gold = _clean(second.get("content", ""))
    slots = {
        "switch": _clean(row.get("switch", "")),
        "version": _clean(row.get("version", "")).replace("_", "."),
        "sub_version": _clean(row.get("sub_version", "")),
        "feature": _clean(row.get("feature", "")),
        "category": _clean(row.get("category", "")),
        "bug_id": _clean(row.get("bug_id", "")),
        "question_type": _clean(row.get("source_type", "")),
    }
    return {
        "domain": "release",
        "question": question,
        "gold_answer": gold,
        "intent": _clean(row.get("source_type", "")),
        "slots": {k: v for k, v in slots.items() if v},
        "selected_context": {
            "switch": slots["switch"],
            "version": slots["version"],
            "sub_version": slots["sub_version"],
            "feature": slots["feature"],
            "category": slots["category"],
            "bug_id": slots["bug_id"],
        },
        "source_file": str(row.get("_source_file", "")),
    }


def _sample_rows(rows: Sequence[Dict[str, object]], count: int, seed: int) -> List[Dict[str, object]]:
    usable = [row for row in rows if _clean(row.get("question", "")) and _clean(row.get("gold_answer", ""))]
    if len(usable) <= count:
        return list(usable)
    rng = random.Random(seed)
    return rng.sample(usable, count)


def _run_runtime(runtime, rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    results: List[Dict[str, object]] = []
    summary = Counter()
    lookup_failures = Counter()
    missing_data_cases: List[Dict[str, object]] = []
    exact_total = 0
    exact_correct = 0
    explain_total = 0
    explain_correct = 0

    for index, row in enumerate(rows, start=1):
        session_context: Dict[str, str] = {}
        result = runtime.answer(
            row["question"],
            session_context,
            row["selected_context"],
            show_debug=False,
        )
        final_answer = _clean(result.get("final_answer", ""))
        lookup_answer = _clean(result.get("lookup_answer", "")) if result.get("lookup_answer") else ""
        gold_answer = _clean(row["gold_answer"])
        intent = _clean(result.get("predicted_intent", row["intent"]))
        correct = _is_correct(intent, final_answer, gold_answer)
        lookup_correct = _is_correct(intent, lookup_answer or final_answer, gold_answer)

        if _is_exact_intent(intent):
            exact_total += 1
            exact_correct += int(correct)
        else:
            explain_total += 1
            explain_correct += int(correct)

        summary["total"] += 1
        summary["correct"] += int(correct)
        if result.get("lookup_status") == "data_not_available":
            summary["data_not_available"] += 1
            missing_data_cases.append(
                {
                    "domain": row["domain"],
                    "question": row["question"],
                    "intent": intent,
                    "slots": row["selected_context"],
                    "lookup_status": result.get("lookup_status"),
                    "final_answer": final_answer,
                    "gold_answer": gold_answer,
                    "source_file": row.get("source_file"),
                }
            )
            key = (
                row["domain"],
                row["selected_context"].get("switch", ""),
                row["selected_context"].get("version", ""),
                row["selected_context"].get("sub_version", ""),
            )
            lookup_failures[key] += 1
        if not correct:
            summary["failed"] += 1

        results.append(
            {
                "domain": row["domain"],
                "question": row["question"],
                "predicted_intent": result.get("predicted_intent"),
                "raw_lstm_intent": result.get("raw_lstm_intent"),
                "slots": result.get("slots", {}),
                "lookup_status": result.get("lookup_status"),
                "lookup_key_used": result.get("lookup_key_used"),
                "lookup_answer": result.get("lookup_answer"),
                "qwen_answer": result.get("qwen_answer"),
                "qwen_validation_passed": result.get("qwen_validation_passed"),
                "final_answer": final_answer,
                "answer_source": result.get("answer_source"),
                "gold_answer": gold_answer,
                "lookup_correct": lookup_correct,
                "correct": correct,
                "source_file": row.get("source_file"),
            }
        )
        if index % 50 == 0 or index == len(rows):
            print(f"  processed {index}/{len(rows)} rows", flush=True)

    report = {
        "total": summary["total"],
        "correct": summary["correct"],
        "failed": summary["failed"],
        "data_not_available": summary["data_not_available"],
        "lookup_accuracy": sum(int(r["lookup_correct"]) for r in results) / max(1, len(results)),
        "final_accuracy": summary["correct"] / max(1, summary["total"]),
        "exact_intent_accuracy": exact_correct / max(1, exact_total),
        "explanatory_accuracy": explain_correct / max(1, explain_total),
        "exact_total": exact_total,
        "explanatory_total": explain_total,
        "missing_data_top": [
            {
                "domain": domain,
                "switch": switch,
                "version": version,
                "sub_version": sub_version,
                "count": count,
            }
            for (domain, switch, version, sub_version), count in lookup_failures.most_common(25)
        ],
    }
    return results, {"report": report, "missing_data_cases": missing_data_cases}


def _run_api(base_url: str, rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    results: List[Dict[str, object]] = []
    summary = Counter()
    lookup_failures = Counter()
    missing_data_cases: List[Dict[str, object]] = []
    exact_total = 0
    exact_correct = 0
    explain_total = 0
    explain_correct = 0

    for index, row in enumerate(rows, start=1):
        payload = {
            "question": row["question"],
            "domain": row["domain"],
            "selected_switch": row["selected_context"].get("switch", ""),
            "selected_version": row["selected_context"].get("version", ""),
            "selected_sub_version": row["selected_context"].get("sub_version", ""),
            "show_debug": False,
        }
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API request failed: {exc.code} {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc

        final_answer = _clean(result.get("final_answer", ""))
        lookup_answer = _clean(result.get("lookup_answer", "")) if result.get("lookup_answer") else ""
        gold_answer = _clean(row["gold_answer"])
        intent = _clean(result.get("predicted_intent", row["intent"]))
        correct = _is_correct(intent, final_answer, gold_answer)
        lookup_correct = _is_correct(intent, lookup_answer or final_answer, gold_answer)

        if _is_exact_intent(intent):
            exact_total += 1
            exact_correct += int(correct)
        else:
            explain_total += 1
            explain_correct += int(correct)

        summary["total"] += 1
        summary["correct"] += int(correct)
        if result.get("lookup_status") == "data_not_available":
            summary["data_not_available"] += 1
            missing_data_cases.append(
                {
                    "domain": row["domain"],
                    "question": row["question"],
                    "intent": intent,
                    "slots": row["selected_context"],
                    "lookup_status": result.get("lookup_status"),
                    "final_answer": final_answer,
                    "gold_answer": gold_answer,
                    "source_file": row.get("source_file"),
                }
            )
            key = (
                row["domain"],
                row["selected_context"].get("switch", ""),
                row["selected_context"].get("version", ""),
                row["selected_context"].get("sub_version", ""),
            )
            lookup_failures[key] += 1
        if not correct:
            summary["failed"] += 1

        results.append(
            {
                "domain": row["domain"],
                "question": row["question"],
                "predicted_intent": result.get("predicted_intent"),
                "raw_lstm_intent": result.get("raw_lstm_intent"),
                "slots": result.get("slots", {}),
                "lookup_status": result.get("lookup_status"),
                "lookup_key_used": result.get("lookup_key_used"),
                "lookup_answer": result.get("lookup_answer"),
                "qwen_answer": result.get("qwen_answer"),
                "qwen_validation_passed": result.get("qwen_validation_passed"),
                "final_answer": final_answer,
                "answer_source": result.get("answer_source"),
                "gold_answer": gold_answer,
                "lookup_correct": lookup_correct,
                "correct": correct,
                "source_file": row.get("source_file"),
            }
        )
        if index % 50 == 0 or index == len(rows):
            print(f"  processed {index}/{len(rows)} rows", flush=True)

    report = {
        "total": summary["total"],
        "correct": summary["correct"],
        "failed": summary["failed"],
        "data_not_available": summary["data_not_available"],
        "lookup_accuracy": sum(int(r["lookup_correct"]) for r in results) / max(1, len(results)),
        "final_accuracy": summary["correct"] / max(1, summary["total"]),
        "exact_intent_accuracy": exact_correct / max(1, exact_total),
        "explanatory_accuracy": explain_correct / max(1, explain_total),
        "exact_total": exact_total,
        "explanatory_total": explain_total,
        "missing_data_top": [
            {
                "domain": domain,
                "switch": switch,
                "version": version,
                "sub_version": sub_version,
                "count": count,
            }
            for (domain, switch, version, sub_version), count in lookup_failures.most_common(25)
        ],
    }
    return results, {"report": report, "missing_data_cases": missing_data_cases}


def _write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_markdown(path: Path, title: str, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        if not rows:
            handle.write("No failures found.\n")
            return
        for index, row in enumerate(rows, start=1):
            handle.write(f"{index}. `{row['domain']}` `{row['lookup_status']}`\n")
            handle.write(f"   - question: {row['question']}\n")
            handle.write(f"   - gold: {row['gold_answer']}\n")
            handle.write(f"   - final: {row['final_answer']}\n")
            handle.write(f"   - source: {row.get('source_file', '')}\n")
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 1000-question QA evaluation sweep.")
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--product_count", type=int, default=500)
    parser.add_argument("--release_count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--backend_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--use_local_models", action="store_true", default=False)
    args = parser.parse_args()
    use_api = not args.use_local_models

    product_rows = [_normalize_product_row(row) for row in _load_product_rows()]
    release_rows = [_normalize_release_row(row) for row in _load_release_rows()]
    product_sample = _sample_rows(product_rows, min(args.product_count, args.sample_size), args.seed)
    release_sample = _sample_rows(release_rows, min(args.release_count, args.sample_size), args.seed + 1)

    combined: List[Dict[str, object]] = []
    combined.extend(product_sample)
    combined.extend(release_sample)
    if len(combined) > args.sample_size:
        rng = random.Random(args.seed)
        combined = rng.sample(combined, args.sample_size)

    evaluated_rows: List[Dict[str, object]] = []
    for row in combined:
        evaluated_rows.append(row)

    device = None
    product_runtime = None
    release_runtime = None
    if not use_api:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        qwen_disabled = QwenBundle(loaded=False)
        product_runtime = ProductRuntime(
            model_path=PRODUCT_LSTM_MODEL_PATH,
            data_paths=PRODUCT_DATA_PATHS,
            device=device,
            qwen=qwen_disabled,
        )
        release_runtime = ReleaseRuntime(
            model_path=RELEASE_LSTM_MODEL_PATH,
            lookup_data_path=RELEASE_LOOKUP_DATA_PATH,
            lookup_index_path=RELEASE_LOOKUP_INDEX_PATH,
            availability_path=RELEASE_AVAILABILITY_PATH,
            bug_metadata_path=RELEASE_BUG_METADATA_PATH,
            device=device,
            qwen=qwen_disabled,
        )

    results: List[Dict[str, object]] = []
    missing_data_rows: List[Dict[str, object]] = []
    per_domain_summary = {}
    coverage_report = {
        "product_docs": _scan_product_coverage(),
    }

    print(f"mode: {'api' if use_api else 'local'}", flush=True)
    if not use_api:
        print(f"device: {device}", flush=True)
    print(f"sample_size: {len(combined)}", flush=True)

    for domain in ("product", "release"):
        domain_rows = [row for row in evaluated_rows if row["domain"] == domain]
        if use_api:
            domain_results, domain_payload = _run_api(args.backend_url, domain_rows)
        else:
            domain_results, domain_payload = _run_runtime(
                product_runtime if domain == "product" else release_runtime,
                domain_rows,
            )
        results.extend(domain_results)
        missing_data_rows.extend(domain_payload["missing_data_cases"])
        per_domain_summary[domain] = domain_payload["report"]
        print(f"completed {domain}: {len(domain_results)}")

    output = {
        "sample_size": len(results),
        "per_domain": per_domain_summary,
        "coverage": coverage_report,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUTPUT_JSONL, results)
    with REPORT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    _write_markdown(FAIL_MD, "Fail Cases", [row for row in results if not row["correct"]])
    with MISSING_JSON.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "missing_data_cases": missing_data_rows,
                "coverage": coverage_report,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"results: {OUTPUT_JSONL}")
    print(f"report: {REPORT_JSON}")
    print(f"fail_cases: {FAIL_MD}")
    print(f"missing_data: {MISSING_JSON}")


if __name__ == "__main__":
    main()
