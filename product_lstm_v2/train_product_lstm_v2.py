from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch import nn

import train_release_lstm as base
from product_lstm_v2_utils import DEFAULT_OUTPUT_DIR, label_maps, normalize_text, read_jsonl


DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "product_lstm_model.pt"
DEFAULT_TRAIN_PATH = DEFAULT_OUTPUT_DIR / "product_lstm_train_patched.jsonl"
DEFAULT_VAL_PATH = DEFAULT_OUTPUT_DIR / "product_lstm_val_patched.jsonl"
DEFAULT_TEST_PATH = DEFAULT_OUTPUT_DIR / "product_lstm_test_patched.jsonl"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def save_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Product LSTM v2 on patched data.")
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val_path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--test_path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_best_only", action="store_true", default=True)
    parser.add_argument("--save_last_only", action="store_false", dest="save_best_only")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    device = torch.device("cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "n/a"
    print(f"[DEVICE] {device.type}", flush=True)
    print(f"[GPU] {gpu_name}", flush=True)

    train_records = load_records(args.train_path)
    val_records = load_records(args.val_path)
    test_records = load_records(args.test_path)
    if not train_records or not val_records or not test_records:
        raise FileNotFoundError("Expected patched split files were not found or are empty.")

    all_records = train_records + val_records + test_records
    label_names, label_to_id, id_to_label = label_maps(all_records)
    if not label_names:
        raise ValueError("No labels were found in the patched product dataset.")

    print(f"[DATA] train={len(train_records)} val={len(val_records)} test={len(test_records)}", flush=True)
    print(f"[DATA] labels={label_names}", flush=True)

    tokenizer = base.SimpleTokenizer()
    tokenizer.build_vocab([str(item["input_text"]) for item in train_records])
    print(f"[DATA] vocab_size={len(tokenizer.vocab)}", flush=True)

    train_loader = base.build_loader(
        train_records,
        tokenizer,
        label_to_id,
        label_names,
        args.max_length,
        args.batch_size,
        True,
        device,
    )
    val_loader = base.build_loader(
        val_records,
        tokenizer,
        label_to_id,
        label_names,
        args.max_length,
        args.batch_size,
        False,
        device,
    )
    test_loader = base.build_loader(
        test_records,
        tokenizer,
        label_to_id,
        label_names,
        args.max_length,
        args.batch_size,
        False,
        device,
    )

    train_class_counts = Counter(str(record["intent"]) for record in train_records)
    class_weights = None
    if args.use_class_weights:
        weights = []
        total = sum(train_class_counts.values())
        for label in label_names:
            count = max(1, train_class_counts.get(label, 0))
            weights.append(total / (len(label_names) * count))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = base.LSTMIntentModel(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_labels=len(label_names),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_macro_f1 = -math.inf
    best_epoch = 0
    best_val_report: Dict[str, object] = {}
    best_val_matrix: List[List[int]] = []
    best_val_true: List[int] = []
    best_val_pred: List[int] = []
    best_val_confidence: List[float] = []
    history: List[Dict[str, object]] = []
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = base.train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_report, val_matrix, y_true, y_pred, confidences = base.evaluate(
            model,
            val_loader,
            criterion,
            label_names,
            device,
        )
        macro_precision = float(val_report["macro_precision"])
        macro_recall = float(val_report["macro_recall"])
        macro_f1 = float(val_report["macro_f1"])
        val_accuracy = float(val_report["accuracy"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
            }
        )

        print(
            f"[EPOCH {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f} macro_f1={macro_f1:.4f}",
            flush=True,
        )

        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_val_report = val_report
            best_val_matrix = val_matrix
            best_val_true = y_true
            best_val_pred = y_pred
            best_val_confidence = confidences
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                print(f"[EARLY_STOP] no val Macro F1 improvement for {args.early_stopping_patience} epochs", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model.")

    model.load_state_dict(best_state)
    train_loader_eval = base.build_loader(
        train_records,
        tokenizer,
        label_to_id,
        label_names,
        args.max_length,
        args.batch_size,
        False,
        device,
    )
    val_loader_eval = base.build_loader(
        val_records,
        tokenizer,
        label_to_id,
        label_names,
        args.max_length,
        args.batch_size,
        False,
        device,
    )
    test_loader_eval = base.build_loader(
        test_records,
        tokenizer,
        label_to_id,
        label_names,
        args.max_length,
        args.batch_size,
        False,
        device,
    )
    train_loss, train_report, train_matrix, _, _, _ = base.evaluate(model, train_loader_eval, criterion, label_names, device)
    val_loss, val_report, val_matrix, _, _, _ = base.evaluate(model, val_loader_eval, criterion, label_names, device)
    test_loss, test_report, test_matrix, _, _, _ = base.evaluate(model, test_loader_eval, criterion, label_names, device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "product_lstm_model.pt"
    tokenizer_path = output_dir / "tokenizer.pkl"
    label_encoder_path = output_dir / "label_encoder.pkl"
    report_path = output_dir / "product_lstm_training_report.json"

    torch.save(
        {
            "model_state_dict": best_state,
            "config": {
                "model_type": "bilstm",
                "embedding_dim": args.embedding_dim,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "max_length": args.max_length,
                "label_names": label_names,
            },
            "vocab": dict(tokenizer.vocab),
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
        },
        model_path,
    )
    save_pickle(tokenizer_path, tokenizer)
    save_pickle(label_encoder_path, {"label_names": label_names, "label_to_id": label_to_id, "id_to_label": id_to_label})

    report_payload = {
        "train_path": str(args.train_path),
        "val_path": str(args.val_path),
        "test_path": str(args.test_path),
        "output_dir": str(output_dir),
        "device": device.type,
        "gpu_name": gpu_name,
        "seed": args.seed,
        "max_length": args.max_length,
        "embedding_dim": args.embedding_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "use_class_weights": args.use_class_weights,
        "early_stopping_patience": args.early_stopping_patience,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_accuracy": float(best_val_report.get("accuracy", 0.0)),
        "validation_metrics": best_val_report,
        "train_metrics": train_report,
        "test_metrics": test_report,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "history": history,
        "label_names": label_names,
        "confusion_matrix": best_val_matrix,
        "artifacts": {
            "model": str(model_path),
            "tokenizer": str(tokenizer_path),
            "label_encoder": str(label_encoder_path),
            "training_report": str(report_path),
        },
        "verdict": (
            "strong"
            if best_val_macro_f1 >= 0.90
            else "good"
            if best_val_macro_f1 >= 0.80
            else "needs review"
        ),
    }

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, ensure_ascii=False)

    print(f"[SAVE] model={model_path}", flush=True)
    print(f"[SAVE] tokenizer={tokenizer_path}", flush=True)
    print(f"[SAVE] label_encoder={label_encoder_path}", flush=True)
    print(f"[SAVE] report={report_path}", flush=True)
    print(f"[BEST] epoch={best_epoch} macro_f1={best_val_macro_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
