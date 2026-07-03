from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for train_release_lstm.py.") from exc

from train_lstm_gpu import (
    LSTMIntentModel,
    build_slots,
    build_target_value,
    clean_question_text,
    extract_message_answer,
    extract_message_question,
    infer_intent,
    normalize_whitespace,
)


SELECTED_INTENTS = [
    "bug_category",
    "bug_scenario",
    "bug_symptom",
    "bug_workaround",
    "release_caveat",
    "release_limitation",
    "version_history",
    "release_date",
    "bug_metadata",
    "data_not_available",
    "out_of_domain",
]


DEFAULT_DATA_DIR = Path(r"C:\Hpe\Train\Data\Release_Notes")
DEFAULT_OUTPUT_DIR = Path(r"C:\Hpe\Train\outputs_release_lstm\all_switches")
DEFAULT_MODEL_TYPE = "bilstm"
DEFAULT_EMBEDDING_DIM = 256
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.3
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
DEFAULT_MAX_LENGTH = 96
DEFAULT_PATIENCE = 5
NEGATIVE_SAMPLE_ROWS = [
    {
        "input_text": "what is my name?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
        "reference": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
    },
    {
        "input_text": "tell me a joke",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
        "reference": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
    },
    {
        "input_text": "what is the weather today?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
        "reference": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
    },
    {
        "input_text": "what is 2 plus 2?",
        "intent": "out_of_domain",
        "slots": {},
        "target_value": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
        "reference": "This is a domain-specific Aruba switch assistant, so I cannot answer this question because it is not related to Aruba switches.",
    },
    {
        "input_text": "For 9999 AOS-CX 10.18, what caveat is documented for SNMP?",
        "intent": "data_not_available",
        "slots": {"switch": "9999", "version": "10_18", "feature": "SNMP"},
        "target_value": "This particular data is not available in the current Aruba switch dataset.",
        "reference": "This particular data is not available in the current Aruba switch dataset.",
    },
    {
        "input_text": "For 4100i AOS-CX 10.99, what is the workaround for Bug 123456?",
        "intent": "data_not_available",
        "slots": {"switch": "4100i", "version": "10_99", "bug_id": "123456"},
        "target_value": "This particular data is not available in the current Aruba switch dataset.",
        "reference": "This particular data is not available in the current Aruba switch dataset.",
    },
    {
        "input_text": "For 6200 AOS-CX 10.18, what product documentation command explains SNMP?",
        "intent": "data_not_available",
        "slots": {"switch": "6200", "version": "10_18", "feature": "SNMP"},
        "target_value": "This particular data is not available in the current Aruba switch dataset.",
        "reference": "This particular data is not available in the current Aruba switch dataset.",
    },
]


@dataclass
class SampleRecord:
    input_text: str
    intent: str
    reference: str
    prediction: str
    confidence: float
    correct: bool


class SimpleTokenizer:
    def __init__(self, vocab: Optional[Dict[str, int]] = None) -> None:
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        if vocab is None:
            self.vocab = {self.pad_token: 0, self.unk_token: 1}
        else:
            self.vocab = dict(vocab)
            self.vocab.setdefault(self.pad_token, 0)
            self.vocab.setdefault(self.unk_token, 1)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+|[^\w\s]", text.lower())

    def build_vocab(self, texts: Sequence[str]) -> None:
        next_id = max(self.vocab.values(), default=1) + 1
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = next_id
                    next_id += 1

    def encode(self, text: str, max_length: int) -> List[int]:
        tokens = self.tokenize(text)[:max_length]
        if not tokens:
            tokens = [self.unk_token]
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]


class IntentDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Dict[str, object]],
        tokenizer: SimpleTokenizer,
        label_to_id: Dict[str, int],
        label_names: Sequence[str],
        max_length: int,
    ) -> None:
        self.items = list(items)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.label_names = list(label_names)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = self.items[index]
        input_ids = self.tokenizer.encode(str(item["input_text"]), self.max_length)
        label_id = self.label_to_id[str(item["intent"])]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "length": torch.tensor(len(input_ids), dtype=torch.long),
            "label": torch.tensor(label_id, dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
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


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def infer_path_context(path: Path) -> Dict[str, str]:
    text = str(path)
    patterns = [
        re.compile(
            r"(?P<switch>[^\\/]+)[\\/](?P=switch)[\\/](?P<version>\d+_\d+)[\\/](?P<sub_version>\d+)[\\/][^\\/]+\.jsonl$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"(?P<switch>[^\\/]+)[\\/](?P<version>\d+_\d+)[\\/](?P<sub_version>\d+)[\\/][^\\/]+\.jsonl$",
            flags=re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return {key: normalize_text(value) for key, value in match.groupdict().items() if normalize_text(value)}
    return {}


def normalize_question_text(question: str, context: Dict[str, str]) -> str:
    switch = context.get("switch", "")
    version = context.get("version", "")
    sub_version = context.get("sub_version", "")
    if switch and version and sub_version:
        return clean_question_text(question, switch, version, sub_version)
    return normalize_whitespace(question)


def row_to_record(row: Dict[str, object], source_file: str, line_no: int, context: Dict[str, str]) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    if not isinstance(row, dict):
        return None, "invalid row"

    input_text = normalize_text(row.get("input_text", ""))
    intent = normalize_text(row.get("intent", ""))
    slots = row.get("slots") if isinstance(row.get("slots"), dict) else {}
    reference = normalize_text(row.get("reference", "")) or normalize_text(row.get("target_value", ""))

    if input_text and intent and reference:
        if intent not in SELECTED_INTENTS:
            return None, "intent not selected"
        return {
            "input_text": input_text,
            "intent": intent,
            "slots": dict(slots),
            "target_value": reference,
            "reference": reference,
            "source_file": source_file,
            "line_no": line_no,
        }, None

    question = extract_message_question(row) or input_text
    if not question:
        return None, "empty input_text"

    if not intent:
        intent = infer_intent(row, question) or ""
    if intent not in SELECTED_INTENTS:
        return None, "intent not selected"

    answer = reference or extract_message_answer(row) or normalize_text(build_target_value(row, intent))
    if not answer:
        return None, "empty target_value"

    if not slots:
        slots = build_slots(row, intent)

    if context:
        slots = dict(slots)
        for key in ("switch", "version", "sub_version"):
            value = context.get(key, "")
            if value and not slots.get(key):
                slots[key] = value

    converted = {
        "input_text": normalize_question_text(question, context),
        "intent": intent,
        "slots": dict(slots),
        "target_value": answer,
        "reference": answer,
        "source_file": source_file,
        "line_no": line_no,
    }
    if not converted["input_text"]:
        return None, "empty input_text"
    return converted, None


def collect_records(data_dir: Path) -> Tuple[List[Dict[str, object]], Dict[str, int], int]:
    records: List[Dict[str, object]] = []
    reasons: Counter[str] = Counter()
    scanned = 0
    seen = set()

    for path in sorted(data_dir.rglob("*.jsonl")):
        if not path.is_file():
            continue
        context = infer_path_context(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                scanned += 1
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    reasons["invalid_jsonl"] += 1
                    continue
                record, reason = row_to_record(row, str(path), line_no, context)
                if reason is not None:
                    reasons[reason] += 1
                    continue
                key = (
                    normalize_text(record["input_text"]),
                    normalize_text(record["intent"]),
                    normalize_text(record["target_value"]),
                )
                if key in seen:
                    reasons["duplicate"] += 1
                    continue
                seen.add(key)
                records.append(record)

    return records, dict(reasons), scanned


def add_negative_samples(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    augmented = list(records)
    augmented.extend(deepcopy(NEGATIVE_SAMPLE_ROWS))
    return augmented


def stratified_split(records: Sequence[Dict[str, object]], seed: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    rng = random.Random(seed)
    by_intent: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        by_intent[str(record["intent"])].append(dict(record))

    train: List[Dict[str, object]] = []
    val: List[Dict[str, object]] = []
    test: List[Dict[str, object]] = []

    for intent in SELECTED_INTENTS:
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


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    input_ids = []
    labels = []
    for item in batch:
        ids = item["input_ids"][:max_len]
        if ids.numel() < max_len:
            padding = torch.zeros(max_len - ids.numel(), dtype=torch.long)
            ids = torch.cat([ids, padding], dim=0)
        input_ids.append(ids)
        labels.append(item["label"])
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "lengths": lengths,
        "labels": torch.stack(labels, dim=0),
    }


def build_loader(
    records: Sequence[Dict[str, object]],
    tokenizer: SimpleTokenizer,
    label_to_id: Dict[str, int],
    label_names: Sequence[str],
    max_length: int,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = IntentDataset(records, tokenizer, label_to_id, label_names, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def classification_report(y_true: Sequence[int], y_pred: Sequence[int], label_names: Sequence[str]) -> Tuple[Dict[str, object], List[List[int]]]:
    num_labels = len(label_names)
    matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    correct = sum(matrix[i][i] for i in range(num_labels))
    total = len(y_true)

    for idx, name in enumerate(label_names):
        tp = matrix[idx][idx]
        fp = sum(matrix[row][idx] for row in range(num_labels) if row != idx)
        fn = sum(matrix[idx][col] for col in range(num_labels) if col != idx)
        support = sum(matrix[idx])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_precision = sum(item["precision"] for item in per_class.values()) / max(1, num_labels)
    macro_recall = sum(item["recall"] for item in per_class.values()) / max(1, num_labels)
    macro_f1 = sum(item["f1"] for item in per_class.values()) / max(1, num_labels)
    accuracy = correct / max(1, total)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "total": total,
    }, matrix


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    label_names: Sequence[str],
    device: torch.device,
) -> Tuple[float, Dict[str, object], List[List[int]], List[int], List[int], List[float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            confidences.extend(probs.max(dim=1).values.cpu().tolist())
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

    avg_loss = total_loss / max(1, total_items)
    report, matrix = classification_report(y_true, y_pred, label_names)
    return avg_loss, report, matrix, y_true, y_pred, confidences


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return total_loss / max(1, total_items)


def predict_with_confidence(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[float]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, lengths)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            confidences.extend(probs.max(dim=1).values.cpu().tolist())
    return y_true, y_pred, confidences


def build_sample_markdown(
    samples: Sequence[SampleRecord],
    label_names: Sequence[str],
) -> str:
    if not samples:
        return "# Samples Eval\n\nNo validation samples available."

    best = sorted(samples, key=lambda item: (item.correct, item.confidence), reverse=True)[:10]
    worst = sorted(samples, key=lambda item: (item.correct, item.confidence))[:10]

    def render_block(title: str, rows: Sequence[SampleRecord]) -> List[str]:
        lines = [f"## {title}"]
        if not rows:
            lines.append("No samples available.")
            return lines
        for item in rows:
            lines.extend(
                [
                    f"- intent: `{item.intent}`",
                    f"  - input: {item.input_text}",
                    f"  - reference: {item.reference}",
                    f"  - prediction: {item.prediction}",
                    f"  - confidence: {item.confidence:.4f}",
                    f"  - correct: {str(item.correct).lower()}",
                ]
            )
        return lines

    parts = ["# Samples Eval", ""]
    parts.extend(render_block("10 Best Predictions", best))
    parts.append("")
    parts.extend(render_block("10 Worst Predictions", worst))
    return "\n".join(parts).rstrip() + "\n"


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_training_metrics(path: Path, history: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return
    headers = list(history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a release-notes BiLSTM classifier.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_type", default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--embedding_dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--add_negative_samples", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--save_best_only", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    args = parser.parse_args()

    if args.model_type.lower() != "bilstm":
        raise SystemExit("Only --model_type bilstm is supported.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "n/a"
    print(f"[DEVICE] {device.type}")
    print(f"[GPU] {gpu_name}")

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records, reason_counts, rows_scanned = collect_records(args.data_dir)
    if args.add_negative_samples:
        records = add_negative_samples(records)
        reason_counts["synthetic_negative_samples"] = len(NEGATIVE_SAMPLE_ROWS)
    records = [record for record in records if str(record["intent"]) in SELECTED_INTENTS]
    if not records:
        raise ValueError("No clean training rows were found.")

    print(f"[DATA] rows scanned: {rows_scanned}")
    print(f"[DATA] rows kept: {len(records)}")
    print(f"[DATA] filter reasons: {reason_counts}")

    train_records, val_records, test_records = stratified_split(records, args.seed)
    if not train_records or not val_records:
        raise ValueError("Failed to produce a usable train/validation split.")

    label_names = [label for label in SELECTED_INTENTS if any(str(record["intent"]) == label for record in records)]
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    id_to_label = {str(idx): label for label, idx in label_to_id.items()}
    print(f"[DATA] label names: {label_names}")
    print(
        f"[DATA] split sizes: train={len(train_records)} val={len(val_records)} test={len(test_records)}"
    )

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([str(item["input_text"]) for item in train_records])

    train_loader = build_loader(train_records, tokenizer, label_to_id, label_names, args.max_length, args.batch_size, True, device)
    val_loader = build_loader(val_records, tokenizer, label_to_id, label_names, args.max_length, args.batch_size, False, device)
    test_loader = build_loader(test_records, tokenizer, label_to_id, label_names, args.max_length, args.batch_size, False, device)

    train_class_counts = Counter(str(record["intent"]) for record in train_records)
    class_weights = None
    if args.use_class_weights:
        weights = []
        total = sum(train_class_counts.values())
        for label in label_names:
            count = max(1, train_class_counts.get(label, 0))
            weights.append(total / (len(label_names) * count))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = LSTMIntentModel(
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
    best_val_pred: List[int] = []
    best_val_true: List[int] = []
    best_val_confidence: List[float] = []
    history: List[Dict[str, object]] = []
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_report, val_matrix, y_true, y_pred, confidences = evaluate(model, val_loader, criterion, label_names, device)
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

        print(f"[EPOCH {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}")
        print(f"[EPOCH {epoch:02d}] macro_precision={macro_precision:.4f} macro_recall={macro_recall:.4f} macro_f1={macro_f1:.4f}")
        print("[EPOCH CONFUSION_MATRIX]")
        for row in val_matrix:
            print(row)
        print("[EPOCH PER_CLASS_F1]")
        print({label: round(float(val_report["per_class"][label]["f1"]), 4) for label in label_names})

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
            if args.save_best_only:
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
                    output_dir / "best_model.pt",
                )
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                print(f"[EARLY_STOP] no val Macro F1 improvement for {args.early_stopping_patience} epochs")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model.")

    model.load_state_dict(best_state)
    if not args.save_best_only:
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
            output_dir / "best_model.pt",
        )

    train_loader_eval = build_loader(train_records, tokenizer, label_to_id, label_names, args.max_length, args.batch_size, False, device)
    test_loader_eval = build_loader(test_records, tokenizer, label_to_id, label_names, args.max_length, args.batch_size, False, device)
    train_loss, train_report, train_matrix, _, _, _ = evaluate(model, train_loader_eval, criterion, label_names, device)
    val_loss, val_report, val_matrix, _, _, _ = evaluate(model, val_loader, criterion, label_names, device)
    test_loss, test_report, test_matrix, _, _, _ = evaluate(model, test_loader_eval, criterion, label_names, device)

    tokenizer_path = output_dir / "vocab.json"
    label_encoder_path = output_dir / "label_encoder.json"
    report_path = output_dir / "training_report.json"
    metrics_csv_path = output_dir / "training_metrics.csv"
    confusion_matrix_path = output_dir / "confusion_matrix.json"
    samples_path = output_dir / "samples_eval.md"

    with tokenizer_path.open("w", encoding="utf-8") as handle:
        json.dump(tokenizer.vocab, handle, indent=2, ensure_ascii=False)

    with label_encoder_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "label_names": label_names,
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    save_training_metrics(metrics_csv_path, history)

    with confusion_matrix_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "labels": label_names,
                "matrix": best_val_matrix,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    samples: List[SampleRecord] = []
    val_true, val_pred, val_conf = predict_with_confidence(model, val_loader, device)
    for record, true_id, pred_id, conf in zip(val_records, val_true, val_pred, val_conf):
        samples.append(
            SampleRecord(
                input_text=str(record["input_text"]),
                intent=str(record["intent"]),
                reference=str(record["target_value"]),
                prediction=label_names[pred_id],
                confidence=float(conf),
                correct=bool(true_id == pred_id),
            )
        )

    with samples_path.open("w", encoding="utf-8") as handle:
        handle.write(build_sample_markdown(samples, label_names))

    report_payload = {
        "data_dir": str(args.data_dir),
        "output_dir": str(output_dir),
        "model_type": args.model_type,
        "rows_scanned": rows_scanned,
        "rows_kept": len(records),
        "split_sizes": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "filter_reasons": reason_counts,
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
        "device": device.type,
        "gpu_name": gpu_name,
        "verdict": (
            "strong"
            if best_val_macro_f1 >= 0.90
            else "good"
            if best_val_macro_f1 >= 0.80
            else "needs review"
        ),
        "artifacts": {
            "best_model": str(output_dir / "best_model.pt"),
            "vocab": str(tokenizer_path),
            "label_encoder": str(label_encoder_path),
            "training_report": str(report_path),
            "training_metrics": str(metrics_csv_path),
            "confusion_matrix": str(confusion_matrix_path),
            "samples_eval": str(samples_path),
        },
    }

    write_json(report_path, report_payload)

    print("Training completed")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Macro F1: {best_val_macro_f1:.4f}")
    print(f"Validation accuracy: {float(best_val_report.get('accuracy', 0.0)):.4f}")
    print(f"Model saved at: {output_dir / 'best_model.pt'}")
    print(f"Vocab saved at: {tokenizer_path}")
    print(f"Label encoder saved at: {label_encoder_path}")
    print(f"Training report: {report_path}")
    print(f"Verdict: {report_payload['verdict']}")


if __name__ == "__main__":
    main()
