from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_DOCS_REPAIRED_ROOT = ROOT / "Data" / "product_docs_final_repaired"


def _collect_product_lookup_paths() -> list[Path]:
    data_root = PRODUCT_DOCS_REPAIRED_ROOT if PRODUCT_DOCS_REPAIRED_ROOT.exists() else ROOT / "Data" / "product_docs_final"
    paths = [
        path
        for path in data_root.rglob("product_dataset_repaired.jsonl")
        if path.is_file()
    ]
    paths.extend(
        path
        for path in data_root.rglob("product_review_remaining.jsonl")
        if path.is_file()
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique

RELEASE_LSTM_MODEL_PATH = ROOT / "outputs_release_lstm" / "all_switches" / "best_model.pt"
PRODUCT_LSTM_MODEL_PATH = ROOT / "outputs_product_lstm" / "all_switches" / "best_model.pt"

RELEASE_LOOKUP_DATA_PATH = ROOT / "Data" / "Release_Notes"
RELEASE_LOOKUP_INDEX_PATH = ROOT / "outputs_release_lstm" / "all_switches" / "lookup_index.json"
RELEASE_BUG_METADATA_PATH = ROOT / "outputs_release_lstm" / "all_switches" / "bug_metadata_index.json"
RELEASE_AVAILABILITY_PATH = ROOT / "outputs_final" / "availability_index.json"

PRODUCT_LOOKUP_DATA_PATHS = _collect_product_lookup_paths()

QWEN_MODEL_PATH = Path(
    r"E:\52\Train_w\Train\outputs_final\qwen25_3b_metadatactx_fullclean_1epoch_stratified"
)

FRONTEND_DIR = ROOT / "frontend"
BACKEND_CACHE_DIR = ROOT / "backend_cache"
CHAT_CONVERSATIONS_PATH = BACKEND_CACHE_DIR / "chat_conversations.json"
