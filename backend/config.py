from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _env_text(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def _env_path(name: str, default: Path | str) -> Path:
    return Path(_env_text(name, str(default))).expanduser()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _resolve_model_file(model_dir: Path, filename: str = "best_model.pt") -> Path:
    if model_dir.suffix:
        return model_dir
    return model_dir / filename


MODEL_ROOT = _env_path("MODEL_ROOT", ROOT / "_bundle_models_used_fullbackend" / "models")
DATA_ROOT = _env_path("DATA_ROOT", ROOT / "Data")

RELEASE_NOTES_DATA_DIR = _env_path("RELEASE_NOTES_DATA_DIR", DATA_ROOT / "Release_Notes")
PRODUCT_DOCS_DATA_DIR = _env_path("PRODUCT_DOCS_DATA_DIR", DATA_ROOT / "product_docs_final_repaired")
RELEASE_LSTM_DATA_DIR = _env_path("RELEASE_LSTM_DATA_DIR", ROOT / "outputs_release_lstm" / "all_switches")
PRODUCT_LSTM_DATA_DIR = _env_path("PRODUCT_LSTM_DATA_DIR", ROOT / "outputs_product_lstm" / "all_switches")

QWEN_MODEL_DIR = _env_path(
    "QWEN_MODEL_DIR",
    MODEL_ROOT / "qwen25_3b_metadatactx_fullclean_1epoch_stratified",
)
RELEASE_LSTM_MODEL_DIR = _env_path("RELEASE_LSTM_MODEL_DIR", RELEASE_LSTM_DATA_DIR)
PRODUCT_LSTM_MODEL_DIR = _env_path("PRODUCT_LSTM_MODEL_DIR", PRODUCT_LSTM_DATA_DIR)
OLLAMA_BASE_URL = _env_text("OLLAMA_BASE_URL", "")

QWEN_MODEL_PATH = QWEN_MODEL_DIR
RELEASE_LOOKUP_DATA_PATH = RELEASE_NOTES_DATA_DIR

RELEASE_LSTM_MODEL_PATH = _resolve_model_file(RELEASE_LSTM_MODEL_DIR)
PRODUCT_LSTM_MODEL_PATH = _resolve_model_file(PRODUCT_LSTM_MODEL_DIR)

RELEASE_LOOKUP_INDEX_PATH = _env_path("RELEASE_LSTM_DATA_DIR", RELEASE_LSTM_DATA_DIR) / "lookup_index.json"
RELEASE_BUG_METADATA_PATH = _env_path("RELEASE_LSTM_DATA_DIR", RELEASE_LSTM_DATA_DIR) / "bug_metadata_index.json"
RELEASE_AVAILABILITY_PATH = _env_path("RELEASE_LSTM_DATA_DIR", RELEASE_LSTM_DATA_DIR) / "availability_index.json"


def _collect_product_lookup_paths(data_root: Path) -> list[Path]:
    paths = [path for path in data_root.rglob("product_dataset_repaired.jsonl") if path.is_file()]
    paths.extend(path for path in data_root.rglob("product_review_remaining.jsonl") if path.is_file())
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


PRODUCT_LOOKUP_DATA_PATHS = _collect_product_lookup_paths(PRODUCT_DOCS_DATA_DIR)

FRONTEND_DIR = ROOT / "frontend"
BACKEND_CACHE_DIR = ROOT / "backend_cache"
CHAT_CONVERSATIONS_PATH = BACKEND_CACHE_DIR / "chat_conversations.json"
ALLOWED_ORIGINS = [origin.strip() for origin in _env_text("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]
PORT = _env_int("PORT", 8000)
