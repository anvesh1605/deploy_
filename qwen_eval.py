from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


SYSTEM_PROMPT = (
    "You are answering from HPE Aruba AOS-CX product documentation only.\n"
    "Do not invent unsupported commands, parameters, Event IDs, or version-specific details.\n"
    "Give a concise answer."
)

CLI_SYNTAX_INTENTS = {"cli_syntax", "show_command_syntax"}
CLI_MEANING_INTENTS = {"cli_meaning", "show_command_meaning"}
EVENT_INTENT = "event_log_meaning"
DEFAULT_SAMPLE_SEED = 42


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+|[^\w\s]", normalize_text(text).lower())


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for token_left in left:
        curr = [0]
        for idx, token_right in enumerate(right, start=1):
            if token_left == token_right:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(prev[idx], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, pred_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_f1(reference: str, prediction: str) -> float:
    ref_tokens = Counter(tokenize(reference))
    pred_tokens = Counter(tokenize(prediction))
    if not ref_tokens or not pred_tokens:
        return 0.0
    overlap = sum((ref_tokens & pred_tokens).values())
    precision = overlap / sum(pred_tokens.values())
    recall = overlap / sum(ref_tokens.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(reference: str, prediction: str) -> bool:
    return normalize_text(reference).lower() == normalize_text(prediction).lower()


def extract_event_ids(text: str) -> List[str]:
    return re.findall(r"\bE?\d{4,7}\b", normalize_text(text).upper())


def command_preserved(prediction: str, command: str) -> bool:
    command_text = normalize_text(command).lower()
    if not command_text:
        return False
    return command_text in normalize_text(prediction).lower()


def syntax_preserved(intent: str, reference: str, prediction: str) -> bool:
    if intent not in CLI_SYNTAX_INTENTS:
        return False
    return normalize_text(reference).lower() == normalize_text(prediction).lower()


def event_id_preserved(event_id: str, prediction: str) -> bool:
    event_text = normalize_text(event_id).upper()
    if not event_text:
        return False
    return event_text in normalize_text(prediction).upper()


def hallucinated_event_id_count(reference_event_id: str, prediction: str) -> int:
    ref_event = normalize_text(reference_event_id).upper()
    if not ref_event:
        return 0
    predicted_ids = set(extract_event_ids(prediction))
    predicted_ids.discard(ref_event)
    return len(predicted_ids)


def unsupported_cli_extra_count(intent: str, reference: str, prediction: str) -> int:
    if intent not in (CLI_SYNTAX_INTENTS | CLI_MEANING_INTENTS):
        return 0

    ref = normalize_text(reference).lower()
    pred = normalize_text(prediction).lower()
    patterns = [
        r"\bconfigure terminal\b",
        r"\bshow running-config\b",
        r"\bwrite memory\b",
        r"\breload\b",
        r"\bcopy running-config\b",
        r"\bclear\b",
        r"\berase\b",
        r"\bdelete\b",
        r"\bshutdown\b",
        r"\bno shutdown\b",
        r"\binterface\b",
        r"\brouter\b",
        r"\bvlan\b",
    ]
    for pattern in patterns:
        if re.search(pattern, pred) and not re.search(pattern, ref):
            return 1
    if "\n" in pred and len(pred.splitlines()) > 1:
        return 1
    return 0


def get_slot_value(slots: Dict[str, object], row: Dict[str, object], key: str) -> str:
    value = slots.get(key)
    if value is not None and normalize_text(value):
        return normalize_text(value)
    value = row.get(key)
    return normalize_text(value)


def qwen_max_new_tokens(intent: str) -> int:
    if intent in CLI_SYNTAX_INTENTS:
        return 80
    if intent == EVENT_INTENT:
        return 80
    if intent in CLI_MEANING_INTENTS:
        return 140
    return 180


def first_event_id_from_text(text: str) -> str:
    event_ids = extract_event_ids(text)
    return event_ids[0] if event_ids else ""


def build_prompt(row: Dict[str, object]) -> str:
    slots = row.get("slots") if isinstance(row.get("slots"), dict) else {}
    intent = normalize_text(row.get("intent", ""))
    switch = get_slot_value(slots, row, "switch")
    version = get_slot_value(slots, row, "version")
    command = get_slot_value(slots, row, "command")
    topic = get_slot_value(slots, row, "topic")
    input_text = normalize_text(row.get("input_text", ""))
    event_id = get_slot_value(slots, row, "event_id") or first_event_id_from_text(input_text)

    return (
        "Context:\n"
        f"switch: {switch}\n"
        f"version: {version}\n"
        f"intent: {intent}\n"
        f"command: {command}\n"
        f"topic: {topic}\n"
        f"event_id: {event_id}\n\n"
        f"Question:\n{input_text}"
    )


def select_rows(rows: Sequence[Dict[str, object]], max_samples: int, seed: int = DEFAULT_SAMPLE_SEED) -> List[Dict[str, object]]:
    if len(rows) <= max_samples:
        return list(rows)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), max_samples))
    return [dict(rows[index]) for index in indices]


def load_qwen_model(model_name: str, device: torch.device):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "transformers is required for Qwen evaluation. Install transformers before using --run_qwen_eval."
        ) from exc

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def encode_prompt(tokenizer, prompt: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    return tokenizer(f"{SYSTEM_PROMPT}\n\n{prompt}", return_tensors="pt").input_ids


def generate_prediction(tokenizer, model, prompt: str, intent: str, device: torch.device) -> str:
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
    max_new_tokens = qwen_max_new_tokens(intent)
    eos_token_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else None
    pad_token_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else eos_token_id

    generation_kwargs = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.1,
    }
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    with torch.inference_mode():
        generated = model.generate(input_ids=input_ids, **generation_kwargs)

    new_tokens = generated[0, input_ids.shape[-1] :]
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return normalize_text(prediction)


def compute_row_metrics(
    intent: str,
    input_text: str,
    reference: str,
    prediction: str,
    slots: Dict[str, object],
) -> Dict[str, object]:
    command = normalize_text(slots.get("command", ""))
    event_id = normalize_text(slots.get("event_id", "")) or first_event_id_from_text(input_text)
    row_metrics = {
        "rouge_l": rouge_l_f1(reference, prediction),
        "token_f1": token_f1(reference, prediction),
        "exact_match": float(exact_match(reference, prediction)),
        "command_preservation": float(command_preserved(prediction, command)) if command else None,
        "syntax_preservation": float(syntax_preserved(intent, reference, prediction)) if intent in CLI_SYNTAX_INTENTS else None,
        "event_id_preservation": float(event_id_preserved(event_id, prediction)) if event_id else None,
        "hallucinated_event_id_count": hallucinated_event_id_count(event_id, prediction),
        "unsupported_cli_extra_count": unsupported_cli_extra_count(intent, reference, prediction),
        "avg_prediction_length": float(len(tokenize(prediction))),
    }
    return row_metrics


def aggregate_metrics(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "total_samples": 0,
            "rouge_l": 0.0,
            "token_f1": 0.0,
            "exact_match": 0.0,
            "command_preservation": 0.0,
            "syntax_preservation": 0.0,
            "event_id_preservation": 0.0,
            "hallucinated_event_id_count": 0,
            "unsupported_cli_extra_count": 0,
            "avg_prediction_length": 0.0,
            "rows_by_intent": {},
            "metrics_by_intent": {},
        }

    totals = Counter()
    sum_metrics = Counter()
    command_hits = 0
    command_total = 0
    syntax_hits = 0
    syntax_total = 0
    event_hits = 0
    event_total = 0
    intent_groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for row in rows:
        intent = normalize_text(row.get("intent", ""))
        intent_groups[intent].append(row)
        metrics = row["metrics"]
        totals["rows"] += 1
        sum_metrics["rouge_l"] += float(metrics["rouge_l"])
        sum_metrics["token_f1"] += float(metrics["token_f1"])
        sum_metrics["exact_match"] += float(metrics["exact_match"])
        sum_metrics["avg_prediction_length"] += float(metrics["avg_prediction_length"])
        sum_metrics["hallucinated_event_id_count"] += float(metrics["hallucinated_event_id_count"])
        sum_metrics["unsupported_cli_extra_count"] += float(metrics["unsupported_cli_extra_count"])

        if metrics["command_preservation"] is not None:
            command_total += 1
            command_hits += int(metrics["command_preservation"])
        if metrics["syntax_preservation"] is not None:
            syntax_total += 1
            syntax_hits += int(metrics["syntax_preservation"])
        if metrics["event_id_preservation"] is not None:
            event_total += 1
            event_hits += int(metrics["event_id_preservation"])

    rows_by_intent = {intent: len(group) for intent, group in intent_groups.items()}

    def summarize_group(group: List[Dict[str, object]]) -> Dict[str, object]:
        summary = Counter()
        command_hits_local = 0
        command_total_local = 0
        syntax_hits_local = 0
        syntax_total_local = 0
        event_hits_local = 0
        event_total_local = 0
        pred_len_total = 0.0

        for row in group:
            metrics = row["metrics"]
            summary["rouge_l"] += float(metrics["rouge_l"])
            summary["token_f1"] += float(metrics["token_f1"])
            summary["exact_match"] += float(metrics["exact_match"])
            summary["hallucinated_event_id_count"] += float(metrics["hallucinated_event_id_count"])
            summary["unsupported_cli_extra_count"] += float(metrics["unsupported_cli_extra_count"])
            pred_len_total += float(metrics["avg_prediction_length"])
            if metrics["command_preservation"] is not None:
                command_total_local += 1
                command_hits_local += int(metrics["command_preservation"])
            if metrics["syntax_preservation"] is not None:
                syntax_total_local += 1
                syntax_hits_local += int(metrics["syntax_preservation"])
            if metrics["event_id_preservation"] is not None:
                event_total_local += 1
                event_hits_local += int(metrics["event_id_preservation"])

        count = max(1, len(group))
        return {
            "samples": len(group),
            "rouge_l": summary["rouge_l"] / count,
            "token_f1": summary["token_f1"] / count,
            "exact_match": summary["exact_match"] / count,
            "command_preservation": (command_hits_local / command_total_local) if command_total_local else None,
            "syntax_preservation": (syntax_hits_local / syntax_total_local) if syntax_total_local else None,
            "event_id_preservation": (event_hits_local / event_total_local) if event_total_local else None,
            "hallucinated_event_id_count": int(summary["hallucinated_event_id_count"]),
            "unsupported_cli_extra_count": int(summary["unsupported_cli_extra_count"]),
            "avg_prediction_length": pred_len_total / count,
        }

    metrics_by_intent = {intent: summarize_group(group) for intent, group in intent_groups.items()}
    total_count = max(1, totals["rows"])
    return {
        "total_samples": totals["rows"],
        "rouge_l": sum_metrics["rouge_l"] / total_count,
        "token_f1": sum_metrics["token_f1"] / total_count,
        "exact_match": sum_metrics["exact_match"] / total_count,
        "command_preservation": (command_hits / command_total) if command_total else None,
        "syntax_preservation": (syntax_hits / syntax_total) if syntax_total else None,
        "event_id_preservation": (event_hits / event_total) if event_total else None,
        "hallucinated_event_id_count": int(sum_metrics["hallucinated_event_id_count"]),
        "unsupported_cli_extra_count": int(sum_metrics["unsupported_cli_extra_count"]),
        "avg_prediction_length": sum_metrics["avg_prediction_length"] / total_count,
        "rows_by_intent": rows_by_intent,
        "metrics_by_intent": metrics_by_intent,
    }


def render_example_block(row: Dict[str, object]) -> str:
    metrics = row["metrics"]
    lines = [
        f"- Intent: `{normalize_text(row.get('intent', ''))}`",
        f"  - Question: {normalize_text(row.get('input_text', ''))}",
        f"  - Reference: {normalize_text(row.get('reference', ''))}",
        f"  - Prediction: {normalize_text(row.get('prediction', ''))}",
        (
            "  - Metrics: "
            f"rouge_l={metrics['rouge_l']:.3f}, token_f1={metrics['token_f1']:.3f}, "
            f"exact_match={int(bool(metrics['exact_match']))}"
        ),
    ]
    return "\n".join(lines)


def build_samples_markdown(rows: Sequence[Dict[str, object]]) -> str:
    if not rows:
        return "# Qwen Evaluation Samples\n\nNo rows were evaluated."

    def score(row: Dict[str, object]) -> Tuple[float, float, float]:
        metrics = row["metrics"]
        return (
            float(metrics["exact_match"]),
            float(metrics["rouge_l"]),
            float(metrics["token_f1"]),
        )

    sorted_rows = sorted(rows, key=score, reverse=True)
    worst_rows = list(reversed(sorted_rows[-10:]))
    cli_syntax_rows = [row for row in rows if normalize_text(row.get("intent", "")) in CLI_SYNTAX_INTENTS][:5]
    cli_meaning_rows = [row for row in rows if normalize_text(row.get("intent", "")) in CLI_MEANING_INTENTS][:5]
    concept_rows = [row for row in rows if normalize_text(row.get("intent", "")) == "concept_explanation"][:5]
    review_rows = [
        row
        for row in sorted(rows, key=lambda item: (
            float(item["metrics"]["exact_match"]),
            float(item["metrics"]["rouge_l"]),
        ))
        if not bool(row["metrics"]["exact_match"])
        or row["metrics"]["unsupported_cli_extra_count"]
        or row["metrics"]["hallucinated_event_id_count"]
    ][:5]

    parts = ["# Qwen Evaluation Samples", ""]
    sections = [
        ("10 Best Predictions", sorted_rows[:10]),
        ("10 Worst Predictions", worst_rows[:10]),
        ("5 cli_syntax Examples", cli_syntax_rows),
        ("5 cli_meaning Examples", cli_meaning_rows),
        ("5 concept_explanation Examples", concept_rows),
        ("5 Review-worthy Examples", review_rows),
    ]

    for title, section_rows in sections:
        parts.append(f"## {title}")
        if not section_rows:
            parts.append("No examples available.")
            parts.append("")
            continue
        for row in section_rows:
            parts.append(render_example_block(row))
            parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def run_qwen_evaluation(
    dataset_path: Path,
    output_dir: Path,
    model_name: str,
    max_samples: int,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Qwen eval dataset not found: {dataset_path}")

    print(f"[QWEN] Loading dataset: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as handle:
        raw_rows = [json.loads(line) for line in handle if line.strip()]

    rows = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        rows.append(
            {
                "input_text": normalize_text(raw.get("input_text", "")),
                "intent": normalize_text(raw.get("intent", "")),
                "slots": raw.get("slots") if isinstance(raw.get("slots"), dict) else {},
                "target_value": normalize_text(raw.get("target_value", "")),
            }
        )

    rows = select_rows(rows, max_samples=max_samples)
    print(f"[QWEN] Samples selected: {len(rows)}")
    print(f"[QWEN] Loading model: {model_name}")
    tokenizer, model = load_qwen_model(model_name, device)

    predictions: List[Dict[str, object]] = []
    print("[QWEN] Generating predictions...")
    for index, row in enumerate(rows, start=1):
        prompt = build_prompt(row)
        prediction = generate_prediction(tokenizer, model, prompt, normalize_text(row["intent"]), device)
        metrics = compute_row_metrics(
            normalize_text(row["intent"]),
            normalize_text(row["input_text"]),
            normalize_text(row["target_value"]),
            prediction,
            row.get("slots", {}),
        )
        predictions.append(
            {
                "input_text": row["input_text"],
                "intent": row["intent"],
                "slots": row["slots"],
                "reference": row["target_value"],
                "prediction": prediction,
                "metrics": metrics,
            }
        )
        if index == 1 or index == len(rows) or index % 10 == 0:
            print(f"[QWEN] Completed {index}/{len(rows)}")

    print("[QWEN] Aggregating metrics...")
    report = aggregate_metrics(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "qwen_predictions.jsonl"
    report_path = output_dir / "qwen_eval_report.json"
    samples_path = output_dir / "qwen_samples.md"

    print("[QWEN] Writing outputs...")
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_payload = {
        "model_name": model_name,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "sample_size": max_samples,
        **report,
    }
    report_payload["verdict"] = (
        "promising baseline"
        if float(report["exact_match"]) >= 0.5
        else "needs tuning"
    )

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, ensure_ascii=False)

    with samples_path.open("w", encoding="utf-8") as handle:
        handle.write(build_samples_markdown(predictions))

    print("[QWEN] Done.")

    return {
        "report": report_payload,
        "predictions_path": str(predictions_path),
        "report_path": str(report_path),
        "samples_path": str(samples_path),
    }


if __name__ == "__main__":
    raise SystemExit(
        "Import qwen_eval.run_qwen_evaluation from the LSTM pipeline instead of running this module directly."
    )
