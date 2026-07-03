from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

import train_release_lstm as base
from product_lstm_v2_utils import DEFAULT_OUTPUT_DIR, normalize_text


DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "product_lstm_model.pt"
DEFAULT_TOKENIZER_PATH = DEFAULT_OUTPUT_DIR / "tokenizer.pkl"
DEFAULT_LABEL_ENCODER_PATH = DEFAULT_OUTPUT_DIR / "label_encoder.pkl"


def load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict(question: str, model_path: Path, tokenizer_path: Path, label_encoder_path: Path, device: torch.device, max_length: int) -> dict:
    checkpoint = torch.load(model_path, map_location=device)
    tokenizer = load_pickle(tokenizer_path)
    encoder_payload = load_pickle(label_encoder_path)
    label_names = list(encoder_payload["label_names"])
    label_to_id = dict(encoder_payload["label_to_id"])
    id_to_label = dict(encoder_payload["id_to_label"])

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
    model.eval()

    record = {
        "input_text": normalize_text(question),
        "intent": label_names[0] if label_names else "",
        "slots": {},
        "target_value": "",
    }
    loader = base.build_loader([record], tokenizer, label_to_id, label_names, max_length, 1, False, device)
    with torch.no_grad():
        batch = next(iter(loader))
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        logits = model(input_ids, lengths)
        probs = torch.softmax(logits, dim=1)
        confidence = float(probs.max(dim=1).values.item())
        pred_id = int(probs.argmax(dim=1).item())
    return {
        "question": question,
        "predicted_intent": id_to_label[str(pred_id)],
        "confidence": confidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single Product LSTM v2 prediction.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer_path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--label_encoder_path", type=Path, default=DEFAULT_LABEL_ENCODER_PATH)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max_length", type=int, default=96)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    device = torch.device("cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")

    result = predict(
        args.question,
        args.model_path,
        args.tokenizer_path,
        args.label_encoder_path,
        device,
        args.max_length,
    )
    print(result["predicted_intent"])


if __name__ == "__main__":
    main()
