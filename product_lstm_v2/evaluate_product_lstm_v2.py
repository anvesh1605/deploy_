from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch import nn

import train_release_lstm as base
from product_lstm_v2_utils import DEFAULT_OUTPUT_DIR, label_maps, normalize_text, read_jsonl, write_json, write_jsonl


DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "product_lstm_model.pt"
DEFAULT_TOKENIZER_PATH = DEFAULT_OUTPUT_DIR / "tokenizer.pkl"
DEFAULT_LABEL_ENCODER_PATH = DEFAULT_OUTPUT_DIR / "label_encoder.pkl"
DEFAULT_TEST_PATH = DEFAULT_OUTPUT_DIR / "product_lstm_test_patched.jsonl"


def load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_records(path: Path) -> List[Dict[str, object]]:
    rows = read_jsonl(path)
    records: List[Dict[str, object]] = []
    for row in rows:
        input_text = normalize_text(row.get("input_text"))
        intent = normalize_text(row.get("intent"))
        target_value = normalize_text(row.get("target_value"))
        if not input_text or not intent or not target_value:
            continue
        records.append(
            {
                "input_text": input_text,
                "intent": intent,
                "slots": dict(row.get("slots") if isinstance(row.get("slots"), dict) else {}),
                "target_value": target_value,
                "reference": normalize_text(row.get("reference")) or target_value,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Product LSTM v2 model.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer_path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--label_encoder_path", type=Path, default=DEFAULT_LABEL_ENCODER_PATH)
    parser.add_argument("--test_path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    device = torch.device("cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device.type}", flush=True)

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer pickle not found: {args.tokenizer_path}")
    if not args.label_encoder_path.exists():
        raise FileNotFoundError(f"Label encoder pickle not found: {args.label_encoder_path}")
    if not args.test_path.exists():
        raise FileNotFoundError(f"Test split not found: {args.test_path}")

    checkpoint = torch.load(args.model_path, map_location=device)
    tokenizer = load_pickle(args.tokenizer_path)
    encoder_payload = load_pickle(args.label_encoder_path)

    label_names = list(encoder_payload["label_names"])
    label_to_id = dict(encoder_payload["label_to_id"])
    id_to_label = dict(encoder_payload["id_to_label"])

    test_records = load_records(args.test_path)
    if not test_records:
        raise ValueError("Test split is empty.")

    model_cfg = checkpoint["config"]
    model = base.LSTMIntentModel(
        vocab_size=len(checkpoint["vocab"]),
        embedding_dim=int(model_cfg["embedding_dim"]),
        hidden_size=int(model_cfg["hidden_size"]),
        num_layers=int(model_cfg["num_layers"]),
        num_labels=len(label_names),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loader = base.build_loader(
        test_records,
        tokenizer,
        label_to_id,
        label_names,
        args.max_length,
        args.batch_size,
        False,
        device,
    )
    criterion = nn.CrossEntropyLoss()
    loss, report, matrix, y_true, y_pred, confidences = base.evaluate(model, loader, criterion, label_names, device)

    predictions: List[Dict[str, object]] = []
    for record, true_id, pred_id, confidence in zip(test_records, y_true, y_pred, confidences):
        predictions.append(
            {
                "question": record["input_text"],
                "gold_intent": record["intent"],
                "predicted_intent": id_to_label[str(pred_id)],
                "confidence": float(confidence),
                "correct": bool(true_id == pred_id),
                "slots": record.get("slots", {}),
                "target_value": record.get("target_value", ""),
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "product_lstm_predictions.jsonl"
    eval_report_path = output_dir / "product_lstm_eval_report.json"

    write_jsonl(predictions_path, predictions)
    write_json(
        eval_report_path,
        {
            "model_path": str(args.model_path),
            "tokenizer_path": str(args.tokenizer_path),
            "label_encoder_path": str(args.label_encoder_path),
            "test_path": str(args.test_path),
            "device": device.type,
            "loss": loss,
            "metrics": report,
            "confusion_matrix": {
                "labels": label_names,
                "matrix": matrix,
            },
            "prediction_count": len(predictions),
            "accuracy": float(report.get("accuracy", 0.0)),
            "macro_f1": float(report.get("macro_f1", 0.0)),
        },
    )

    print(f"[SAVE] predictions={predictions_path}", flush=True)
    print(f"[SAVE] eval_report={eval_report_path}", flush=True)
    print(f"[METRIC] accuracy={float(report.get('accuracy', 0.0)):.4f} macro_f1={float(report.get('macro_f1', 0.0)):.4f}", flush=True)


if __name__ == "__main__":
    main()
