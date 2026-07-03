from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - clearer runtime failure if torch is absent
    raise SystemExit(
        "PyTorch is required for this pipeline. Install torch before running train_lstm_gpu.py or infer_lstm_gpu.py."
    ) from exc


DEFAULT_INPUT_PATH = Path(r"C:\Hpe\Train\release_notes_4100i_cleaned")
DEFAULT_OUTPUT_DIR = Path(r"C:\Hpe\Train\outputs_4100i_gpu")
DEFAULT_SWITCH = "4100i"
DEFAULT_VERSION = "10_13"
DEFAULT_SUB_VERSION = "0005"
DEFAULT_SEED = 42
DEFAULT_MAX_LENGTH = 96
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-3
DEFAULT_EMBED_DIM = 128
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_LSTM_LAYERS = 1
DEFAULT_DROPOUT = 0.2

SELECTED_INTENTS = [
    "bug_category",
    "bug_symptom",
    "bug_scenario",
    "bug_workaround",
    "release_caveat",
]


@dataclass
class ReviewRow:
    reason: str
    line_no: int
    source_file: str
    raw: Dict[str, object]
    question: str = ""
    input_text: str = ""
    target_value: str = ""


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

    def to_dict(self) -> Dict[str, int]:
        return dict(self.vocab)


class IntentDataset(Dataset):
    def __init__(self, items: Sequence[Dict[str, object]], tokenizer: SimpleTokenizer, label_to_id: Dict[str, int], max_length: int) -> None:
        self.items = list(items)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
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


def normalize_whitespace(text: object) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def dotted_version(version: str, sub_version: str) -> str:
    return f"{version.replace('_', '.')}.{sub_version}"


def build_prefix(switch: str, version: str, sub_version: str) -> str:
    return f"For {switch} AOS-CX {dotted_version(version, sub_version)},"


def clean_question_text(question: str, switch: str, version: str, sub_version: str) -> str:
    text = normalize_whitespace(question)
    version_tag = dotted_version(version, sub_version)
    repeated_prefix = re.compile(
        rf"^(?:For\s+{re.escape(switch)}\s+AOS-CX\s+{re.escape(version_tag)},\s*)+",
        flags=re.IGNORECASE,
    )
    replacement = build_prefix(switch, version, sub_version) + " "
    text = repeated_prefix.sub(replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_message_question(row: Dict[str, object]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    first = messages[0]
    if not isinstance(first, dict):
        return ""
    return normalize_whitespace(first.get("content", ""))


def extract_message_answer(row: Dict[str, object]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return ""
    second = messages[1]
    if not isinstance(second, dict):
        return ""
    return normalize_whitespace(second.get("content", ""))


def infer_intent(row: Dict[str, object], question: str) -> Optional[str]:
    source_type = str(row.get("source_type", "")).lower()
    q = question.lower()
    if "release_notes_resolved_issues" in source_type:
        if "category" in q:
            return "bug_category"
        if "symptom" in q or "what issue was resolved" in q or "what issue occurs" in q:
            return "bug_symptom"
        if "scenario" in q or "under what scenario" in q:
            return "bug_scenario"
        if "workaround" in q:
            return "bug_workaround"
        return None
    if "release_notes_caveats" in source_type:
        return "release_caveat"
    return None


def extract_segment(text: str, label: str) -> str:
    if not text:
        return ""
    pattern = re.compile(
        rf"{re.escape(label)}:\s*(.*?)(?=\s+(?:Symptom|Scenario|Workaround|Description|Feature Caveat):|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return ""
    return normalize_whitespace(match.group(1))


def clean_bug_prefix(text: str) -> str:
    cleaned = normalize_whitespace(text)
    cleaned = re.sub(r"^[A-Za-z0-9][A-Za-z0-9 /&._:-]*\s+\(Bug ID \d+\):\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^The symptom is:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^The documented workaround is:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^The workaround is:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Feature Caveat:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_scenario_text(text: str) -> str:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return ""
    duplicate_noise = re.compile(
        r"^(This issue\s+(?:may occur|occurs|can occur)\s+when\s+)(This issue\s+)",
        flags=re.IGNORECASE,
    )
    while True:
        updated = duplicate_noise.sub(r"\2", cleaned)
        updated = re.sub(r"\s+", " ", updated).strip()
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = re.sub(r"^Scenario:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def build_target_value(row: Dict[str, object], intent: str) -> str:
    if intent == "bug_category":
        return normalize_whitespace(row.get("category", ""))
    if intent == "bug_symptom":
        text = normalize_whitespace(row.get("symptom", "")) or extract_segment(str(row.get("description", "")), "Symptom")
        return clean_bug_prefix(text)
    if intent == "bug_scenario":
        text = normalize_whitespace(row.get("scenario", "")) or extract_segment(str(row.get("description", "")), "Scenario")
        return clean_scenario_text(clean_bug_prefix(text))
    if intent == "bug_workaround":
        text = normalize_whitespace(row.get("workaround", "")) or extract_segment(str(row.get("description", "")), "Workaround")
        if not text:
            return "No workaround is documented in the release notes."
        return clean_bug_prefix(text)
    if intent == "release_caveat":
        text = normalize_whitespace(row.get("description", ""))
        return clean_bug_prefix(text)
    return ""


def build_slots(row: Dict[str, object], intent: str) -> Dict[str, str]:
    slots = {
        "switch": normalize_whitespace(row.get("switch", "")),
        "version": normalize_whitespace(row.get("version", "")),
        "sub_version": normalize_whitespace(row.get("sub_version", "")),
    }
    if intent == "release_caveat":
        slots["feature"] = normalize_whitespace(row.get("feature", ""))
    else:
        slots["bug_id"] = normalize_whitespace(row.get("bug_id", ""))
        slots["category"] = normalize_whitespace(row.get("category", ""))
    return {key: value for key, value in slots.items() if value}


def should_review_for_noise(target_value: str) -> bool:
    if not target_value:
        return True
    repeated = re.search(r"\b(this issue\b.*\bthis issue\b)", target_value, flags=re.IGNORECASE)
    if repeated and not target_value.startswith("This issue"):
        return True
    return False


def convert_row(
    row: Dict[str, object],
    line_no: int,
    source_file: str,
    expected_switch: str,
    expected_version: Optional[str] = None,
    expected_sub_version: Optional[str] = None,
) -> Tuple[Optional[Dict[str, object]], Optional[ReviewRow]]:
    switch = normalize_whitespace(row.get("switch", ""))
    version = normalize_whitespace(row.get("version", ""))
    sub_version = normalize_whitespace(row.get("sub_version", ""))

    if switch != expected_switch:
        return None, ReviewRow("switch mismatch", line_no, source_file, row, extract_message_question(row))
    if expected_version is not None and version != expected_version:
        return None, ReviewRow("version mismatch", line_no, source_file, row, extract_message_question(row))
    if expected_sub_version is not None and sub_version != expected_sub_version:
        return None, ReviewRow("sub_version mismatch", line_no, source_file, row, extract_message_question(row))

    question = extract_message_question(row)
    if not question:
        return None, ReviewRow("empty input_text", line_no, source_file, row)

    input_text = clean_question_text(question, switch, version, sub_version)
    if not input_text:
        return None, ReviewRow("empty input_text", line_no, source_file, row, question=question)

    intent = infer_intent(row, question)
    if intent not in SELECTED_INTENTS:
        return None, ReviewRow("question cannot be mapped to selected intent", line_no, source_file, row, question=question, input_text=input_text)

    if intent.startswith("bug_") and not normalize_whitespace(row.get("bug_id", "")):
        return None, ReviewRow("bug_id missing for bug intent", line_no, source_file, row, question=question, input_text=input_text)
    if intent == "release_caveat" and not normalize_whitespace(row.get("feature", "")):
        return None, ReviewRow("feature missing for release_caveat", line_no, source_file, row, question=question, input_text=input_text)

    target_value = build_target_value(row, intent)
    if not target_value:
        return None, ReviewRow("empty target_value", line_no, source_file, row, question=question, input_text=input_text)

    if should_review_for_noise(target_value):
        return None, ReviewRow("target_value has noisy duplicate phrase", line_no, source_file, row, question=question, input_text=input_text, target_value=target_value)

    converted = {
        "input_text": input_text,
        "intent": intent,
        "slots": build_slots(row, intent),
        "target_value": target_value,
    }
    return converted, None


def read_and_convert(
    input_paths: Sequence[Path],
    expected_switch: str,
    expected_version: Optional[str] = None,
    expected_sub_version: Optional[str] = None,
) -> Tuple[List[Dict[str, object]], List[ReviewRow], int]:
    converted: List[Dict[str, object]] = []
    review_rows: List[ReviewRow] = []
    scanned = 0

    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                raw_line = line.strip()
                if not raw_line:
                    continue
                scanned += 1
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError:
                    review_rows.append(ReviewRow("invalid JSONL", line_no, str(input_path), {"raw_line": raw_line}))
                    continue
                if not isinstance(row, dict):
                    review_rows.append(ReviewRow("invalid JSONL", line_no, str(input_path), {"raw_value": row}))
                    continue
                converted_row, review_row = convert_row(
                    row,
                    line_no,
                    str(input_path),
                    expected_switch,
                    expected_version,
                    expected_sub_version,
                )
                if review_row is not None:
                    review_rows.append(review_row)
                if converted_row is not None:
                    converted.append(converted_row)
    return converted, review_rows, scanned


def dedupe_and_resolve_conflicts(
    records: Sequence[Dict[str, object]],
    review_rows: List[ReviewRow],
) -> Tuple[List[Dict[str, object]], int, int]:
    by_input: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_input[str(record["input_text"])][str(record["target_value"])].append(record)

    kept: List[Dict[str, object]] = []
    duplicate_pairs_removed = 0
    conflicting_groups = 0

    for input_text, target_groups in by_input.items():
        if len(target_groups) > 1:
            conflicting_groups += 1
            for target_value, grouped_records in target_groups.items():
                for record in grouped_records:
                    review_rows.append(
                        ReviewRow(
                            "same input_text with different target_value",
                            -1,
                            "",
                            {
                                "input_text": record["input_text"],
                                "intent": record["intent"],
                                "slots": record["slots"],
                                "target_value": record["target_value"],
                            },
                            question=str(record["input_text"]),
                            input_text=str(record["input_text"]),
                            target_value=str(record["target_value"]),
                        )
                    )
            continue

        only_target_value = next(iter(target_groups))
        grouped_records = target_groups[only_target_value]
        kept.append(grouped_records[0])
        duplicate_pairs_removed += max(0, len(grouped_records) - 1)

    return kept, duplicate_pairs_removed, conflicting_groups


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review_jsonl(path: Path, rows: Sequence[ReviewRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "reason": row.reason,
                "line_no": row.line_no,
                "source_file": row.source_file,
                "question": row.question,
                "input_text": row.input_text,
                "target_value": row.target_value,
                "raw": row.raw,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def split_by_intent(records: Sequence[Dict[str, object]], seed: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
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
        if n_train + n_val + n_test > n:
            overflow = n_train + n_val + n_test - n
            while overflow > 0 and n_val > 1:
                n_val -= 1
                overflow -= 1
            while overflow > 0 and n_test > 1:
                n_test -= 1
                overflow -= 1
            while overflow > 0 and n_train > 1:
                n_train -= 1
                overflow -= 1

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class LSTMIntentModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        num_labels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        features = torch.cat([forward_hidden, backward_hidden], dim=1)
        features = self.dropout(features)
        return self.classifier(features)


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
    max_length: int,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = IntentDataset(records, tokenizer, label_to_id, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(1, total_items)
    accuracy = sum(int(p == y) for p, y in zip(all_preds, all_labels)) / max(1, len(all_labels))
    return avg_loss, accuracy, all_preds, all_labels


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Tuple[nn.Module, Dict[str, object]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val_loss = math.inf
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        correct = 0
        seen = 0

        for batch in train_loader:
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
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += batch_size

        train_loss = total_loss / max(1, total_items)
        train_accuracy = correct / max(1, seen)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_accuracy": float(train_accuracy),
                "val_accuracy": float(val_accuracy),
            }
        )

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }

        print(
            f"[EPOCH {epoch:02d}] train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    metrics = {
        "history": history,
        "best_epoch": best_state["epoch"] if best_state else epochs,
        "best_val_loss": best_state["val_loss"] if best_state else None,
        "best_val_accuracy": best_state["val_accuracy"] if best_state else None,
    }
    return model, metrics


def compute_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
    label_names: Sequence[str],
) -> Tuple[Dict[str, object], str, List[List[int]]]:
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    total_support = len(y_true)
    correct = sum(matrix[i][i] for i in range(len(labels)))

    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for idx, name in enumerate(label_names):
        tp = matrix[idx][idx]
        fp = sum(matrix[row][idx] for row in range(len(labels)) if row != idx)
        fn = sum(matrix[idx][col] for col in range(len(labels)) if col != idx)
        support = sum(matrix[idx])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    macro_precision = sum(item["precision"] for item in per_class.values()) / max(1, len(per_class))
    macro_recall = sum(item["recall"] for item in per_class.values()) / max(1, len(per_class))
    macro_f1 = sum(item["f1-score"] for item in per_class.values()) / max(1, len(per_class))
    accuracy = correct / max(1, total_support)

    report = {
        "per_class": per_class,
        "accuracy": accuracy,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1-score": macro_f1,
            "support": total_support,
        },
        "weighted_avg": {
            "precision": weighted_precision / max(1, total_support),
            "recall": weighted_recall / max(1, total_support),
            "f1-score": weighted_f1 / max(1, total_support),
            "support": total_support,
        },
    }

    lines = []
    header = f"{'label':<22}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for name in label_names:
        item = per_class[name]
        lines.append(
            f"{name:<22}{item['precision']:>10.4f}{item['recall']:>10.4f}{item['f1-score']:>10.4f}{int(item['support']):>10d}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'accuracy':<22}{'':>10}{'':>10}{accuracy:>10.4f}{total_support:>10d}"
    )
    lines.append(
        f"{'macro avg':<22}{macro_precision:>10.4f}{macro_recall:>10.4f}{macro_f1:>10.4f}{total_support:>10d}"
    )
    lines.append(
        f"{'weighted avg':<22}{weighted_precision / max(1, total_support):>10.4f}"
        f"{weighted_recall / max(1, total_support):>10.4f}"
        f"{weighted_f1 / max(1, total_support):>10.4f}{total_support:>10d}"
    )
    return report, "\n".join(lines), matrix


def tensor_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[int], List[int]]:
    model.eval()
    preds: List[int] = []
    labels: List[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            batch_labels = batch["labels"].to(device)
            logits = model(input_ids, lengths)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
    return preds, labels


def save_prediction_samples(
    path: Path,
    records: Sequence[Dict[str, object]],
    tokenizer: SimpleTokenizer,
    model: nn.Module,
    label_to_id: Dict[str, int],
    id_to_label: Dict[str, str],
    lookup: Dict[str, str],
    device: torch.device,
    max_length: int,
    default_slots: Optional[Dict[str, str]] = None,
    sample_size: int = 20,
) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return

    rng = random.Random(DEFAULT_SEED)
    sample_indices = sorted(rng.sample(range(len(records)), min(sample_size, len(records))))
    sample_rows = [records[index] for index in sample_indices]
    rows_to_write: List[Dict[str, object]] = []

    model.eval()
    with torch.no_grad():
        for record in sample_rows:
            ids = tokenizer.encode(str(record["input_text"]), max_length)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
            logits = model(input_ids, lengths)
            predicted_id = int(logits.argmax(dim=1).item())
            predicted_intent = id_to_label.get(str(predicted_id), id_to_label.get(predicted_id, SELECTED_INTENTS[predicted_id]))
            slots = extract_slots_from_question(str(record["input_text"]))
            if default_slots:
                for key, value in default_slots.items():
                    slots.setdefault(key, value)
            answer = lookup.get(build_lookup_key(predicted_intent, slots))
            rows_to_write.append(
                {
                    "question": record["input_text"],
                    "gold_intent": record["intent"],
                    "predicted_intent": predicted_intent,
                    "slots": slots,
                    "gold_answer": record["target_value"],
                    "answer": answer,
                    "correct": bool(predicted_intent == record["intent"] and answer == record["target_value"]),
                }
            )

    write_jsonl(path, rows_to_write)


def build_lookup_key(intent: str, slots: Dict[str, str]) -> str:
    if intent == "release_caveat":
        return "|".join(
            [
                intent,
                slots.get("switch", ""),
                slots.get("version", ""),
                slots.get("sub_version", ""),
                slots.get("feature", ""),
            ]
        )
    return "|".join(
        [
            intent,
            slots.get("switch", ""),
            slots.get("version", ""),
            slots.get("sub_version", ""),
            slots.get("bug_id", ""),
        ]
    )


def extract_slots_from_question(question: str) -> Dict[str, str]:
    text = normalize_whitespace(question)
    slots: Dict[str, str] = {}

    versioned_switch_match = re.search(
        r"\bFor\s+([A-Za-z0-9_-]+)\s+AOS-CX\s+(\d+)\.(\d+)\.(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if versioned_switch_match:
        slots["switch"] = versioned_switch_match.group(1)
        slots["version"] = f"{versioned_switch_match.group(2)}_{versioned_switch_match.group(3)}"
        slots["sub_version"] = versioned_switch_match.group(4)
    else:
        switch_match = re.search(r"\bFor\s+([A-Za-z0-9_-]+)\s+AOS-CX\b", text, flags=re.IGNORECASE)
        if switch_match:
            slots["switch"] = switch_match.group(1)
        version_match = re.search(r"AOS-CX\s+(\d+)\.(\d+)\.(\d+)", text, flags=re.IGNORECASE)
        if version_match:
            slots["version"] = f"{version_match.group(1)}_{version_match.group(2)}"
            slots["sub_version"] = version_match.group(3)

    bug_match = re.search(r"\bBug\s+ID\s+(\d+)\b", text, flags=re.IGNORECASE) or re.search(
        r"\bBug\s+(\d+)\b", text, flags=re.IGNORECASE
    )
    if bug_match:
        slots["bug_id"] = bug_match.group(1)

    category_match = re.search(r"\b(?:in|does)\s+(.+?)\s+Bug\s+\d+\b", text, flags=re.IGNORECASE)
    if category_match:
        category = category_match.group(1).strip(" ,?")
        if category and not re.search(r"\bAOS-CX\b", category, flags=re.IGNORECASE):
            slots["category"] = category

    if re.search(r"\bcaveat\b|\bfeature caveat\b|\brelated to\b", text, flags=re.IGNORECASE):
        feature_patterns = [
            r"\bwhat\s+(?:caveat|limitation)\s+is\s+documented\s+for\s+(?P<feature>.+?)(?:[?!.]?$)",
            r"\bwhat\s+(?:caveat|limitation)\s+is\s+mentioned\s+for\s+(?P<feature>.+?)(?:[?!.]?$)",
            r"\bwhat\s+is\s+the\s+caveat\s+related\s+to\s+(?P<feature>.+?)(?:[?!.]?$)",
            r"\bwhat\s+(?P<feature>.+?)\s+(?:caveat|limitation)\s+is\s+documented(?:\s+for)?(?:[?!.]?$)",
            r"\bwhat\s+(?P<feature>.+?)\s+caveat\s+is\s+documented(?:\s+for)?(?:[?!.]?$)",
        ]
        for pattern in feature_patterns:
            feature_match = re.search(pattern, text, flags=re.IGNORECASE)
            if feature_match:
                feature = clean_feature_text(feature_match.group("feature"))
                if feature and not re.search(r"\bBug\b", feature, flags=re.IGNORECASE) and not re.search(
                    r"\bAOS-CX\b", feature, flags=re.IGNORECASE
                ):
                    slots.setdefault("feature", feature)
                break

    return slots


def build_lookup(records: Sequence[Dict[str, object]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for record in records:
        key = build_lookup_key(str(record["intent"]), record["slots"])  # type: ignore[arg-type]
        lookup[key] = str(record["target_value"])
    return lookup


def clean_feature_text(feature: str) -> str:
    cleaned = normalize_whitespace(feature)
    cleaned = cleaned.rstrip(" ?.")
    cleaned = re.sub(
        r"\s+in\s+(?:HPE\s+Aruba\s+Networking\s+|HPE\s+Aruba\s+|Aruba\s+)?AOS-CX\s+\d+\.\d+\.\d+.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+in\s+the\s+same\s+release.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,?.")
    return cleaned


def lookup_key_candidates(intent: str, slots: Dict[str, str]) -> List[str]:
    candidates: List[str] = []
    switch = normalize_whitespace(slots.get("switch", ""))
    version = normalize_whitespace(slots.get("version", ""))
    sub_version = normalize_whitespace(slots.get("sub_version", ""))
    bug_id = normalize_whitespace(slots.get("bug_id", ""))
    category = normalize_whitespace(slots.get("category", ""))
    feature = normalize_whitespace(slots.get("feature", ""))

    if intent == "release_caveat":
        if switch and version and sub_version and feature:
            candidates.append("|".join([intent, switch, version, sub_version, feature]))
        if switch and version and feature:
            candidates.append("|".join([intent, switch, version, feature]))
        if feature:
            candidates.append("|".join([intent, feature]))
        return candidates

    if intent.startswith("bug_"):
        if switch and version and sub_version and bug_id:
            candidates.append("|".join([intent, switch, version, sub_version, bug_id]))
        if switch and version and sub_version and category and bug_id:
            candidates.append("|".join([intent, switch, version, sub_version, category, bug_id]))
        if category and bug_id:
            candidates.append("|".join([intent, category, bug_id]))
        if bug_id:
            candidates.append("|".join([intent, bug_id]))
        return candidates

    return candidates


def build_lookup_v2(records: Sequence[Dict[str, object]]) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        intent = str(record["intent"])
        slots = dict(record["slots"])  # type: ignore[arg-type]
        answer = str(record["target_value"])
        for key in lookup_key_candidates(intent, slots):
            if answer not in lookup[key]:
                lookup[key].append(answer)
    return dict(lookup)


def lookup_answers_for_key(lookup: Dict[str, Sequence[str] | str], key: str) -> List[str]:
    values = lookup.get(key)
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    unique: List[str] = []
    for value in values:
        value_text = str(value)
        if value_text not in unique:
            unique.append(value_text)
    return unique


def resolve_lookup_answer(
    intent: str,
    slots: Dict[str, str],
    lookup: Dict[str, Sequence[str] | str],
    defaults: Optional[Dict[str, str]] = None,
) -> Dict[str, Optional[str]]:
    effective_slots = dict(slots)
    if defaults:
        for key, value in defaults.items():
            if value and not effective_slots.get(key):
                effective_slots[key] = value

    candidates = lookup_key_candidates(intent, effective_slots)
    for key in candidates:
        answers = lookup_answers_for_key(lookup, key)
        if not answers:
            continue
        if len(answers) == 1:
            return {
                "answer": answers[0],
                "lookup_key_used": key,
                "status": "found",
                "reason": None,
            }
        if intent == "release_caveat" and key == candidates[-1]:
            return {
                "answer": None,
                "lookup_key_used": key,
                "status": "needs_disambiguation",
                "reason": "feature-only lookup matched multiple versions",
            }
        if intent.startswith("bug_") and key == candidates[-1]:
            return {
                "answer": None,
                "lookup_key_used": key,
                "status": "needs_disambiguation",
                "reason": "bug-id-only lookup matched multiple versions",
            }
        return {
            "answer": None,
            "lookup_key_used": key,
            "status": "needs_disambiguation",
            "reason": "lookup key matched multiple different answers",
        }

    return {
        "answer": None,
        "lookup_key_used": candidates[0] if candidates else None,
        "status": "not_found",
        "reason": "no lookup key matched",
    }


def maybe_run_qwen_eval(
    args: argparse.Namespace,
    metrics: Dict[str, object],
    metrics_path: Path,
    device: torch.device,
) -> Optional[Dict[str, object]]:
    if not getattr(args, "run_qwen_eval", False):
        return None

    from qwen_eval import run_qwen_evaluation

    qwen_result = run_qwen_evaluation(
        dataset_path=args.qwen_eval_data_path,
        output_dir=args.qwen_eval_output_dir,
        model_name=args.qwen_model_name,
        max_samples=args.qwen_eval_max_samples,
        device=device,
    )
    metrics["qwen_eval"] = qwen_result["report"]
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    report = qwen_result["report"]
    print("Qwen eval completed")
    print(f"Dataset path: {report['dataset_path']}")
    print(f"Total samples: {report['total_samples']}")
    print(f"Average ROUGE-L: {float(report['rouge_l']):.4f}")
    print(f"Average token F1: {float(report['token_f1']):.4f}")
    print(f"Exact match: {float(report['exact_match']):.4f}")
    syntax_value = report.get("syntax_preservation")
    command_value = report.get("command_preservation")
    event_value = report.get("event_id_preservation")
    print(
        f"Syntax preservation: {float(syntax_value):.4f}" if syntax_value is not None else "Syntax preservation: n/a"
    )
    print(
        f"Command preservation: {float(command_value):.4f}" if command_value is not None else "Command preservation: n/a"
    )
    print(
        f"Event ID preservation: {float(event_value):.4f}" if event_value is not None else "Event ID preservation: n/a"
    )
    print(f"Output folder: {report['output_dir']}")
    print(f"Verdict: {report['verdict']}")
    return qwen_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GPU-based LSTM intent model for one-switch Aruba release notes.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--switch", default=DEFAULT_SWITCH)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--sub-version", default=DEFAULT_SUB_VERSION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--lstm-layers", type=int, default=DEFAULT_LSTM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--run_qwen_eval", action="store_true")
    parser.add_argument("--qwen_model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--qwen_eval_data_path", type=Path, default=Path("converted_product/10000/10_18/product_dataset.jsonl"))
    parser.add_argument("--qwen_eval_max_samples", type=int, default=100)
    parser.add_argument("--qwen_eval_output_dir", type=Path, default=Path("outputs_qwen_eval/10000_10_18"))
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    print(f"[DEVICE] {device.type}")
    print(f"[GPU] {gpu_name if gpu_name else 'n/a'}")

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input path not found: {args.input_file}")

    if args.input_file.is_dir():
        input_files = sorted(args.input_file.rglob("train_chat.jsonl"))
        version_filter = None
        sub_version_filter = None
    else:
        input_files = [args.input_file]
        version_filter = args.version
        sub_version_filter = args.sub_version

    if not input_files:
        raise FileNotFoundError(f"No train_chat.jsonl files found under: {args.input_file}")

    converted, review_rows, rows_scanned = read_and_convert(
        input_files,
        args.switch,
        version_filter,
        sub_version_filter,
    )

    converted, duplicate_pairs_removed, conflict_groups = dedupe_and_resolve_conflicts(converted, review_rows)

    input_text_to_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in converted:
        input_text_to_rows[str(record["input_text"])].append(record)

    final_records: List[Dict[str, object]] = []
    for input_text, group in input_text_to_rows.items():
        if len({str(item["target_value"]) for item in group}) > 1:
            for item in group:
                    review_rows.append(
                        ReviewRow(
                            "same input_text with different target_value",
                            -1,
                            "",
                            {
                                "input_text": item["input_text"],
                                "intent": item["intent"],
                            "slots": item["slots"],
                            "target_value": item["target_value"],
                        },
                        question=str(item["input_text"]),
                        input_text=str(item["input_text"]),
                        target_value=str(item["target_value"]),
                    )
                )
            continue
        final_records.append(group[0])

    final_records = [record for record in final_records if str(record["intent"]) in SELECTED_INTENTS]
    review_reason_counts = Counter(row.reason for row in review_rows)

    converted_path = output_dir / "converted_4100i_lstm.jsonl"
    train_path = output_dir / "train_4100i_lstm.jsonl"
    val_path = output_dir / "val_4100i_lstm.jsonl"
    test_path = output_dir / "test_4100i_lstm.jsonl"
    review_path = output_dir / "review_4100i_lstm.jsonl"
    label_map_path = output_dir / "intent_label_map.json"
    lookup_path = output_dir / "target_lookup.json"
    lookup_v2_path = output_dir / "target_lookup_v2.json"
    model_path = output_dir / "lstm_intent_model.pt"
    metrics_path = output_dir / "metrics.json"
    samples_path = output_dir / "predictions_sample.jsonl"

    write_jsonl(converted_path, final_records)
    write_review_jsonl(review_path, review_rows)

    if not final_records:
        raise ValueError("No usable rows were produced from the raw JSONL input.")

    train_records, val_records, test_records = split_by_intent(final_records, args.seed)
    if not train_records:
        raise ValueError("No training rows were produced after conversion and splitting.")
    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    write_jsonl(test_path, test_records)

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([str(item["input_text"]) for item in train_records])

    label_to_id = {label: idx for idx, label in enumerate(SELECTED_INTENTS)}
    id_to_label = {str(idx): label for label, idx in label_to_id.items()}

    train_loader = build_loader(train_records, tokenizer, label_to_id, args.max_length, args.batch_size, True, device)
    val_loader = build_loader(val_records, tokenizer, label_to_id, args.max_length, args.batch_size, False, device)
    test_loader = build_loader(test_records, tokenizer, label_to_id, args.max_length, args.batch_size, False, device)

    model = LSTMIntentModel(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.lstm_layers,
        num_labels=len(SELECTED_INTENTS),
        dropout=args.dropout,
    ).to(device)

    model, train_metrics = train_model(model, train_loader, val_loader, device, args.epochs, args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loss, train_accuracy, train_preds, train_labels = evaluate(model, train_loader, criterion, device)
    val_loss, val_accuracy, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    test_loss, test_accuracy, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    report, report_text, confusion_matrix = compute_classification_report(
        test_labels,
        test_preds,
        list(range(len(SELECTED_INTENTS))),
        SELECTED_INTENTS,
    )

    lookup = build_lookup(final_records)
    lookup_v2 = build_lookup_v2(final_records)

    model_payload = {
        "model_state_dict": model.state_dict(),
        "config": {
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.lstm_layers,
            "dropout": args.dropout,
            "max_length": args.max_length,
            "selected_intents": SELECTED_INTENTS,
            "default_switch": args.switch,
            "default_version": args.version if not args.input_file.is_dir() else "",
            "default_sub_version": args.sub_version if not args.input_file.is_dir() else "",
        },
        "vocab": tokenizer.to_dict(),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
    }
    torch.save(model_payload, model_path)

    with label_map_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "selected_intents": SELECTED_INTENTS,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    with lookup_path.open("w", encoding="utf-8") as handle:
        json.dump(lookup, handle, indent=2, ensure_ascii=False)

    with lookup_v2_path.open("w", encoding="utf-8") as handle:
        json.dump(lookup_v2, handle, indent=2, ensure_ascii=False)

    save_prediction_samples(
        samples_path,
        test_records,
        tokenizer,
        model,
        label_to_id,
        id_to_label,
        lookup,
        device,
        args.max_length,
        default_slots=None if args.input_file.is_dir() else {
            "switch": args.switch,
            "version": args.version,
            "sub_version": args.sub_version,
        },
        sample_size=20,
    )

    metrics = {
        "rows_scanned": rows_scanned,
        "input_files_scanned": len(input_files),
        "rows_converted": len(final_records),
        "rows_moved_to_review": len(review_rows),
        "duplicates_removed": duplicate_pairs_removed,
        "conflicting_input_groups": conflict_groups,
        "quality_checks": {
            "invalid JSONL": review_reason_counts.get("invalid JSONL", 0),
            "empty input_text": review_reason_counts.get("empty input_text", 0),
            "empty target_value": review_reason_counts.get("empty target_value", 0),
            "duplicate input_text + target_value removed": duplicate_pairs_removed,
            "same input_text with different target_value": review_reason_counts.get("same input_text with different target_value", 0),
            "question cannot be mapped to selected intent": review_reason_counts.get("question cannot be mapped to selected intent", 0),
            "bug_id missing for bug intent": review_reason_counts.get("bug_id missing for bug intent", 0),
            "feature missing for release_caveat": review_reason_counts.get("feature missing for release_caveat", 0),
            "target_value has noisy duplicate phrase": review_reason_counts.get("target_value has noisy duplicate phrase", 0),
        },
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "classification_report_text": report_text,
        "confusion_matrix": confusion_matrix,
        "split_sizes": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "device": device.type,
        "gpu_name": gpu_name,
        "model_path": str(model_path),
        "lookup_v2_path": str(lookup_v2_path),
        "output_dir": str(output_dir),
        "best_epoch": train_metrics["best_epoch"],
        "best_val_loss": train_metrics["best_val_loss"],
        "best_val_accuracy": train_metrics["best_val_accuracy"],
        "test_loss": test_loss,
    }

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(f"[DATA] rows scanned: {rows_scanned}")
    print(f"[DATA] files scanned: {len(input_files)}")
    print(f"[DATA] rows converted: {len(final_records)}")
    print(f"[DATA] rows moved to review: {len(review_rows)}")
    print(
        "[CHECK] "
        f"invalid JSONL={review_reason_counts.get('invalid JSONL', 0)} "
        f"empty input_text={review_reason_counts.get('empty input_text', 0)} "
        f"empty target_value={review_reason_counts.get('empty target_value', 0)}"
    )
    print(
        f"[TRAIN] train/val/test split sizes: {len(train_records)}/{len(val_records)}/{len(test_records)}"
    )
    print(f"[RESULT] test accuracy: {test_accuracy:.4f}")
    print(f"[OUTPUT] saved model and files in {output_dir}")
    print(report_text)
    print("[CONFUSION_MATRIX]")
    for row in confusion_matrix:
        print(row)

    maybe_run_qwen_eval(args, metrics, metrics_path, device)


if __name__ == "__main__":
    main()
