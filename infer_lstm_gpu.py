from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - clearer runtime failure if torch is absent
    raise SystemExit(
        "PyTorch is required for this pipeline. Install torch before running infer_lstm_gpu.py."
    ) from exc

from train_lstm_gpu import (
    LSTMIntentModel,
    SELECTED_INTENTS,
    SimpleTokenizer,
    normalize_whitespace,
)
from lstm_lookup import (
    DATA_NOT_AVAILABLE_RESPONSE,
    check_data_availability,
    NO_MATCH_RESPONSE,
    aggregate_lookup_metrics,
    build_eval_row,
    build_lookup_entries,
    build_lookup_index,
    extract_slots_from_question,
    load_or_build_availability_index,
    load_or_build_bug_metadata_index,
    read_jsonl,
    render_lookup_errors_markdown,
    resolve_lookup_answer,
    write_jsonl,
)


DEFAULT_MODEL_PATH = Path(r"C:\Hpe\Train\outputs_4100i_gpu\lstm_intent_model.pt")
DEFAULT_LOOKUP_PATH = Path(r"C:\Hpe\Train\outputs_4100i_gpu\target_lookup.json")
DEFAULT_LOOKUP_V2_PATH = Path(r"C:\Hpe\Train\outputs_4100i_gpu\target_lookup_v2.json")
DEFAULT_CONVERTED_PATH = Path(r"C:\Hpe\Train\outputs_4100i_gpu\converted_4100i_lstm.jsonl")
DEFAULT_TEST_PATH = Path(r"C:\Hpe\Train\outputs_4100i_gpu\test_4100i_lstm.jsonl")
DEFAULT_METRICS_PATH = Path(r"C:\Hpe\Train\outputs_4100i_gpu\metrics_v2.json")
DEFAULT_SAMPLES_PATH = Path(r"C:\Hpe\Train\outputs_4100i_gpu\predictions_sample_v2.jsonl")
DEFAULT_LOOKUP_EVAL_PATH = Path(r"C:\Hpe\Train\outputs_4100i_gpu\lstm_lookup_eval.jsonl")
DEFAULT_LOOKUP_REPORT_PATH = Path(r"C:\Hpe\Train\outputs_4100i_gpu\lstm_lookup_report.json")
DEFAULT_LOOKUP_ERRORS_PATH = Path(r"C:\Hpe\Train\outputs_4100i_gpu\lstm_lookup_errors.md")
DEFAULT_RELEASE_EVAL_PATH = Path(r"C:\Hpe\Train\outputs_release_lstm\lookup_eval.jsonl")
DEFAULT_RELEASE_REPORT_PATH = Path(r"C:\Hpe\Train\outputs_release_lstm\lookup_report.json")
DEFAULT_RELEASE_ERRORS_PATH = Path(r"C:\Hpe\Train\outputs_release_lstm\lookup_errors.md")
DEFAULT_AVAILABILITY_INDEX_PATH = Path(r"C:\Hpe\Train\outputs_final\availability_index.json")
DEFAULT_BUG_METADATA_INDEX_PATH = Path(r"C:\Hpe\Train\outputs_release_lstm\all_switches\bug_metadata_index.json")

DETERMINISTIC_MESSAGES = {
    "not_found": "No matching answer was found in the current release-note dataset.",
    "needs_disambiguation": "Multiple possible answers were found. Please provide more detail such as feature, bug ID, version, or sub-version.",
    "slot_missing": "I need more detail to answer this, such as the bug ID, feature, version, or sub-version.",
    "error": "Unable to answer from the current release-note dataset.",
}


def load_artifacts(model_path: Path, device: torch.device):
    payload = torch.load(model_path, map_location=device)
    tokenizer = SimpleTokenizer(payload["vocab"])
    config = payload["config"]
    model = LSTMIntentModel(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=int(config["embedding_dim"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        num_labels=len(config["selected_intents"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, tokenizer, config


def load_lookup_kb(lookup_data_path: Path) -> Tuple[List[object], Dict[str, List[int]]]:
    if not lookup_data_path.exists():
        raise FileNotFoundError(f"Lookup data file not found: {lookup_data_path}")
    records = read_jsonl(lookup_data_path)
    entries = build_lookup_entries(records)
    index = build_lookup_index(entries)
    return entries, index


def predict_intent(
    question: str,
    model: LSTMIntentModel,
    tokenizer: SimpleTokenizer,
    config: Dict[str, object],
    device: torch.device,
) -> str:
    max_length = int(config["max_length"])
    id_to_label = dict(config.get("id_to_label", {}))
    cleaned_question = normalize_whitespace(question)
    ids = tokenizer.encode(cleaned_question, max_length)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids, lengths)
        predicted_id = int(logits.argmax(dim=1).item())

    predicted_intent = id_to_label.get(str(predicted_id))
    if predicted_intent is None:
        predicted_intent = id_to_label.get(predicted_id)
    if predicted_intent is None:
        predicted_intent = SELECTED_INTENTS[predicted_id]
    return predicted_intent


def extract_slots(question: str) -> Dict[str, str]:
    return extract_slots_from_question(question)


def final_answer_for_status(lookup_status: str, lookup_answer: str) -> str:
    if lookup_status == "data_not_available":
        return DATA_NOT_AVAILABLE_RESPONSE
    if lookup_status == "found":
        return normalize_whitespace(lookup_answer)
    if lookup_status == "not_found":
        return DETERMINISTIC_MESSAGES["not_found"]
    if lookup_status == "needs_disambiguation":
        return DETERMINISTIC_MESSAGES["needs_disambiguation"]
    if lookup_status == "slot_missing":
        return DETERMINISTIC_MESSAGES["slot_missing"]
    return DETERMINISTIC_MESSAGES["error"]


def metric_tokens(text: str) -> List[str]:
    return [token for token in normalize_whitespace(text).lower().split() if token]


def token_f1_score(prediction: str, reference: str) -> float:
    pred_tokens = metric_tokens(prediction)
    ref_tokens = metric_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = sum(min(pred_counts.get(token, 0), ref_counts.get(token, 0)) for token in pred_counts)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_length(left: List[str], right: List[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_score(prediction: str, reference: str) -> float:
    pred_tokens = metric_tokens(prediction)
    ref_tokens = metric_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_metric(rows: Sequence[Dict[str, object]], field_name: str) -> float:
    values = [float(row.get(field_name, 0.0) or 0.0) for row in rows if row.get("gold_answer")]
    return sum(values) / max(1, len(values))


def evaluate_records(
    records: Sequence[Dict[str, object]],
    model: LSTMIntentModel,
    tokenizer: SimpleTokenizer,
    config: Dict[str, object],
    lookup_entries: Sequence[object],
    lookup_index: Dict[str, List[int]],
    availability_index: Dict[str, object],
    bug_metadata_index: Dict[str, List[Dict[str, str]]],
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    defaults = {
        key: str(config.get(key, ""))
        for key in ("default_switch", "default_version", "default_sub_version")
        if str(config.get(key, ""))
    }

    total = len(records)
    intent_correct = 0
    answer_correct = 0
    null_answer_count = 0
    disambiguation_count = 0
    slot_missing_count = 0
    low_similarity_count = 0

    rows_to_write: List[Dict[str, object]] = []

    for record in records:
        question = normalize_whitespace(str(record.get("input_text", "")))
        gold_intent = str(record.get("intent", ""))
        gold_answer = str(record.get("target_value", ""))
        slots = extract_slots(question)
        availability_check = check_data_availability(slots, availability_index, bug_metadata_index)
        if availability_check.get("available", True):
            predicted_intent = predict_intent(question, model, tokenizer, config, device)
            resolution = resolve_lookup_answer(predicted_intent, slots, question, lookup_entries, lookup_index, defaults)
        else:
            predicted_intent = "data_not_available"
            resolution = {
                "answer": None,
                "lookup_key_used": None,
                "status": "data_not_available",
                "reason": availability_check.get("reason"),
                "confidence": 1.0,
                "similarity": 1.0,
            }

        if predicted_intent == gold_intent:
            intent_correct += 1
        if resolution["answer"] is None or str(resolution["status"]) in {"not_found", "low_similarity"}:
            null_answer_count += 1
        if resolution["status"] == "needs_disambiguation":
            disambiguation_count += 1
        if resolution["status"] == "slot_missing":
            slot_missing_count += 1
        if resolution["status"] == "low_similarity":
            low_similarity_count += 1
        if resolution["answer"] == gold_answer:
            answer_correct += 1

        row = build_eval_row(question, gold_intent, predicted_intent, slots, resolution, gold_answer)
        rows_to_write.append(
            row
        )

    metrics = aggregate_lookup_metrics(rows_to_write)
    metrics.update({
        "intent_accuracy": intent_correct / max(1, total),
        "answer_accuracy": answer_correct / max(1, total),
        "null_answer_count": null_answer_count,
        "disambiguation_count": disambiguation_count,
        "slot_missing_count": slot_missing_count,
        "low_similarity_count": low_similarity_count,
        "rows_evaluated": total,
    })
    return metrics, rows_to_write


def evaluate_records_with_formatter(
    records: Sequence[Dict[str, object]],
    model: LSTMIntentModel,
    tokenizer: SimpleTokenizer,
    config: Dict[str, object],
    lookup_entries: Sequence[object],
    lookup_index: Dict[str, List[int]],
    availability_index: Dict[str, object],
    bug_metadata_index: Dict[str, List[Dict[str, str]]],
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    defaults = {
        key: str(config.get(key, ""))
        for key in ("default_switch", "default_version", "default_sub_version")
        if str(config.get(key, ""))
    }

    total = len(records)
    lookup_found = 0
    not_found_count = 0
    needs_disambiguation_count = 0
    slot_missing_count = 0
    low_similarity_count = 0
    gold_available = 0
    lookup_correct_count = 0

    lookup_exact_scores: List[float] = []
    lookup_token_f1_scores: List[float] = []
    lookup_rouge_l_scores: List[float] = []

    rows_to_write: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []

    for record in records:
        question = normalize_whitespace(str(record.get("input_text", "")))
        gold_intent = str(record.get("intent", ""))
        gold_answer = normalize_whitespace(record.get("target_value", ""))
        slots = extract_slots(question)
        availability_check = check_data_availability(slots, availability_index, bug_metadata_index)
        if availability_check.get("available", True):
            predicted_intent = predict_intent(question, model, tokenizer, config, device)
            resolution = resolve_lookup_answer(predicted_intent, slots, question, lookup_entries, lookup_index, defaults)
        else:
            predicted_intent = "data_not_available"
            resolution = {
                "answer": None,
                "lookup_key_used": None,
                "status": "data_not_available",
                "reason": availability_check.get("reason"),
                "confidence": 1.0,
                "similarity": 1.0,
            }

        lookup_status = str(resolution.get("status", "error"))
        lookup_answer = normalize_whitespace(resolution.get("answer", "")) if resolution.get("answer") else ""
        lookup_key_used = resolution.get("lookup_key_used")

        if lookup_status == "found" and lookup_answer:
            lookup_found += 1
        elif lookup_status == "not_found":
            not_found_count += 1
        elif lookup_status == "needs_disambiguation":
            needs_disambiguation_count += 1
        elif lookup_status == "slot_missing":
            slot_missing_count += 1
        elif lookup_status == "low_similarity":
            low_similarity_count += 1

        final_answer = final_answer_for_status(lookup_status, lookup_answer)
        lookup_exact = 0.0
        lookup_f1 = 0.0
        lookup_rouge = 0.0
        lookup_correct = False
        if gold_answer:
            gold_available += 1
            lookup_exact = 1.0 if lookup_answer and normalize_whitespace(lookup_answer).lower() == gold_answer.lower() else 0.0
            lookup_f1 = token_f1_score(lookup_answer, gold_answer)
            lookup_rouge = rouge_l_score(lookup_answer, gold_answer)
            lookup_correct = bool(lookup_exact)
            lookup_exact_scores.append(lookup_exact)
            lookup_token_f1_scores.append(lookup_f1)
            lookup_rouge_l_scores.append(lookup_rouge)
            if lookup_correct:
                lookup_correct_count += 1
        row = {
            "question": question,
            "gold_intent": gold_intent,
            "predicted_intent": predicted_intent,
            "slots": slots,
            "lookup_status": lookup_status,
            "lookup_key_used": lookup_key_used,
            "lookup_answer": lookup_answer or None,
            "final_answer": final_answer,
            "gold_answer": gold_answer or None,
            "correct": lookup_correct,
        }
        rows_to_write.append(row)
        metric_rows.append(
            {
                **row,
                "answer": lookup_answer or None,
                "status": lookup_status,
            }
        )

    lookup_exact_match = sum(lookup_exact_scores) / max(1, len(lookup_exact_scores))
    lookup_token_f1 = sum(lookup_token_f1_scores) / max(1, len(lookup_token_f1_scores))
    lookup_rouge_l = sum(lookup_rouge_l_scores) / max(1, len(lookup_rouge_l_scores))
    metrics = aggregate_lookup_metrics(metric_rows)

    report = {
        "total_questions": total,
        "rows_with_gold": gold_available,
        "lookup_found": lookup_found,
        "lookup_correct": lookup_correct_count,
        "lookup_exact_match": lookup_exact_match,
        "lookup_token_f1": lookup_token_f1,
        "lookup_rouge_l": lookup_rouge_l,
        "not_found": not_found_count,
        "needs_disambiguation": needs_disambiguation_count,
        "slot_missing": slot_missing_count,
        "low_similarity": low_similarity_count,
        "lookup_accuracy": lookup_exact_match,
    }
    report.update({k: metrics[k] for k in ("correct", "incorrect", "accuracy", "found_count", "not_found_count", "needs_disambiguation_count", "slot_missing_count", "low_similarity_count", "accuracy_by_intent", "errors_by_intent", "top_20_failed_lookup_keys", "sample_not_found", "sample_needs_disambiguation", "sample_wrong_answers") if k in metrics})
    return report, rows_to_write


def save_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_question_mode(
    question: str,
    model: LSTMIntentModel,
    tokenizer: SimpleTokenizer,
    config: Dict[str, object],
    lookup_entries: Sequence[object],
    lookup_index: Dict[str, List[int]],
    availability_index: Dict[str, object],
    bug_metadata_index: Dict[str, List[Dict[str, str]]],
    device: torch.device,
) -> Dict[str, object]:
    cleaned_question = normalize_whitespace(question)
    slots = extract_slots(cleaned_question)
    defaults = {
        key: str(config.get(key, ""))
        for key in ("default_switch", "default_version", "default_sub_version")
        if str(config.get(key, ""))
    }
    availability_check = check_data_availability(slots, availability_index, bug_metadata_index)
    if availability_check.get("available", True):
        predicted_intent = predict_intent(cleaned_question, model, tokenizer, config, device)
        resolution = resolve_lookup_answer(predicted_intent, slots, cleaned_question, lookup_entries, lookup_index, defaults)
    else:
        predicted_intent = "data_not_available"
        resolution = {
            "answer": None,
            "lookup_key_used": None,
            "status": "data_not_available",
            "reason": availability_check.get("reason"),
            "confidence": 1.0,
            "similarity": 1.0,
        }
    lookup_status = str(resolution.get("status", "error"))
    lookup_answer = normalize_whitespace(resolution.get("answer", "")) if resolution.get("answer") else None
    lookup_confidence = float(resolution.get("confidence", 0.0) or 0.0)
    lookup_similarity = float(resolution.get("similarity", 0.0) or 0.0)
    final_answer = final_answer_for_status(lookup_status, lookup_answer or "")

    result = {
        "question": cleaned_question,
        "predicted_intent": predicted_intent,
        "slots": slots,
        "lookup_status": lookup_status,
        "lookup_answer": lookup_answer,
        "lookup_key_used": resolution.get("lookup_key_used"),
        "confidence": lookup_confidence,
        "similarity": lookup_similarity,
        "final_answer": final_answer,
        "status": lookup_status,
    }
    if resolution.get("reason"):
        result["reason"] = resolution["reason"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer intent and answer from the trained LSTM bundle.")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--lookup-path", type=Path, default=DEFAULT_LOOKUP_PATH)
    parser.add_argument("--lookup-v2-path", type=Path, default=DEFAULT_LOOKUP_V2_PATH)
    parser.add_argument("--lookup-data-path", type=Path, default=DEFAULT_CONVERTED_PATH)
    parser.add_argument("--converted-file", type=Path, default=DEFAULT_CONVERTED_PATH)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--samples-path", type=Path, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--eval-jsonl-path", type=Path, default=DEFAULT_LOOKUP_EVAL_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_LOOKUP_REPORT_PATH)
    parser.add_argument("--errors-md-path", type=Path, default=DEFAULT_LOOKUP_ERRORS_PATH)
    parser.add_argument("--evaluate-test-set", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_artifacts(args.model_path, device)
    lookup_entries, lookup_index = load_lookup_kb(args.lookup_data_path)
    availability_index = load_or_build_availability_index(DEFAULT_AVAILABILITY_INDEX_PATH, lookup_entries)
    bug_metadata_index = load_or_build_bug_metadata_index(DEFAULT_BUG_METADATA_INDEX_PATH, lookup_entries)

    if args.evaluate_test_set:
        if not args.test_file.exists():
            raise FileNotFoundError(f"Test file not found: {args.test_file}")
        records = read_jsonl(args.test_file)
        metrics, sample_rows = evaluate_records_with_formatter(
            records,
            model,
            tokenizer,
            config,
            lookup_entries,
            lookup_index,
            availability_index,
            bug_metadata_index,
            device,
        )
        save_jsonl(DEFAULT_RELEASE_EVAL_PATH, sample_rows)
        with DEFAULT_RELEASE_REPORT_PATH.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
        with DEFAULT_RELEASE_ERRORS_PATH.open("w", encoding="utf-8") as handle:
            handle.write(render_lookup_errors_markdown(metrics, sample_rows))

        print("Lookup eval completed")
        print(f"total_questions: {metrics.get('total_questions', 0)}")
        print(f"correct: {metrics.get('correct', 0)}")
        print(f"incorrect: {metrics.get('incorrect', 0)}")
        print(f"accuracy: {metrics.get('accuracy', 0.0):.4f}")
        print(f"lookup_accuracy: {metrics.get('lookup_accuracy', 0.0):.4f}")
        print(f"lookup_exact_match: {metrics.get('lookup_exact_match', 0.0):.4f}")
        print(f"lookup_token_f1: {metrics.get('lookup_token_f1', 0.0):.4f}")
        print(f"lookup_rouge_l: {metrics.get('lookup_rouge_l', 0.0):.4f}")
        print(f"found_count: {metrics.get('found_count', 0)}")
        print(f"not_found_count: {metrics.get('not_found_count', 0)}")
        print(f"needs_disambiguation_count: {metrics.get('needs_disambiguation_count', 0)}")
        print(f"slot_missing_count: {metrics.get('slot_missing_count', 0)}")
        print(f"low_similarity_count: {metrics.get('low_similarity_count', 0)}")
        print(f"Output JSONL: {DEFAULT_RELEASE_EVAL_PATH}")
        print(f"Output report: {DEFAULT_RELEASE_REPORT_PATH}")
        print(f"Output errors md: {DEFAULT_RELEASE_ERRORS_PATH}")
        return

    question = args.question.strip() or input("Question: ").strip()
    if not question:
        raise SystemExit("A question is required.")

    result = run_question_mode(
        question,
        model,
        tokenizer,
        config,
        lookup_entries,
        lookup_index,
        availability_index,
        bug_metadata_index,
        device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
