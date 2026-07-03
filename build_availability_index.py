from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


DEFAULT_RELEASE_ROOT = Path(r"C:\Hpe\Train\Data\Release_Notes")
DEFAULT_PRODUCT_ROOT = Path(r"C:\Hpe\Train\Data\product_docs_final")
DEFAULT_OUTPUT_PATH = Path(r"C:\Hpe\Train\outputs_final\availability_index.json")


def scan_versions(root: Path) -> Dict[str, List[str]]:
    availability: Dict[str, set[str]] = defaultdict(set)
    if not root.exists():
        return {}
    for path in root.rglob("*.jsonl"):
        parts = path.parts
        for idx, part in enumerate(parts):
            if part.count("_") == 1 and part.replace("_", "").isdigit():
                version = part
                switch = ""
                if idx > 0:
                    prev = parts[idx - 1]
                    if prev not in {"10000", "product_docs_final", "Release_Notes"}:
                        switch = prev
                if switch:
                    availability[switch].add(version)
                break
    return {switch: sorted(values) for switch, values in sorted(availability.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple availability index for Aruba datasets.")
    parser.add_argument("--release_root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--product_root", type=Path, default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    payload = {
        "release_notes": scan_versions(args.release_root),
        "product_docs": scan_versions(args.product_root),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
