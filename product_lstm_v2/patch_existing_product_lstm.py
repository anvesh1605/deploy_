from __future__ import annotations

import argparse
from pathlib import Path

from product_lstm_v2_utils import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPAIR_SOURCE_DIRS,
    PRODUCT_NEGATIVE_ROWS,
    add_negative_samples,
    build_repair_index,
    collect_rejected_patch_candidates,
    collect_unique_records,
    load_dataset_rows,
    load_repair_source_rows,
    normalize_text,
    patch_records,
    stratified_split,
    summarize_patch_report,
    write_json,
    write_jsonl,
    write_text,
)


def render_report_md(report: dict) -> str:
    lines = [
        "# Product LSTM v2 Patch Report",
        "",
        f"- Original rows: {report['original_rows']}",
        f"- Patched rows: {report['patched_rows']}",
        f"- Repair source rows: {report['repair_source_rows']}",
        f"- Changed rows: {report['changed_rows']}",
        f"- Needs review rows: {report['needs_review_rows']}",
        f"- Rejected patch rows: {report['rejected_patch_rows']}",
        f"- Negative samples added: {report['negative_samples_added']}",
        "",
        "## Split Sizes",
        f"- Train: {report['split_sizes']['train']}",
        f"- Val: {report['split_sizes']['val']}",
        f"- Test: {report['split_sizes']['test']}",
        "",
        "## Patch Stats",
    ]
    for key, value in report["patch_stats"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Notes",
            "- Only target_value fields are patched.",
            "- Existing product LSTM source files are not modified in place.",
            "- Repair-source rows are used only as patch candidates.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch the product LSTM dataset into a v2 split.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--repair_source_dir",
        type=Path,
        nargs="*",
        default=DEFAULT_REPAIR_SOURCE_DIRS,
        help="One or more repair-source directories containing repaired product-doc rows.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--add_negative_samples", dest="add_negative_samples", action="store_true", default=True)
    parser.add_argument("--no_negative_samples", dest="add_negative_samples", action="store_false")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PATCH] loading product dataset from {args.data_dir}", flush=True)
    raw_rows, load_stats = load_dataset_rows(args.data_dir, workers=max(1, args.workers))
    original_records, original_filter_stats = collect_unique_records(raw_rows)
    print(f"[PATCH] rows read={load_stats.get('rows', 0)} unique_records={len(original_records)}", flush=True)

    print("[PATCH] loading repair-source rows", flush=True)
    repair_source_rows = load_repair_source_rows(args.repair_source_dir, workers=max(1, args.workers))
    repair_index = build_repair_index(repair_source_rows)
    print(f"[PATCH] repair_source_rows={len(repair_source_rows)} indexed_questions={len(repair_index)}", flush=True)

    patched_records, needs_review, rejected_patches, patch_stats = patch_records(original_records, repair_index)
    rejected_patches.extend(collect_rejected_patch_candidates(repair_source_rows, repair_index))

    dataset_path = output_dir / "product_lstm_dataset_patched.jsonl"
    write_jsonl(dataset_path, patched_records)
    print(f"[PATCH] wrote {dataset_path.name} with {len(patched_records)} rows", flush=True)

    augmented_records = add_negative_samples(patched_records) if args.add_negative_samples else list(patched_records)
    train_records, val_records, test_records = stratified_split(augmented_records, args.seed)
    print(
        f"[PATCH] split sizes train={len(train_records)} val={len(val_records)} test={len(test_records)}",
        flush=True,
    )

    write_jsonl(output_dir / "product_lstm_train_patched.jsonl", train_records)
    write_jsonl(output_dir / "product_lstm_val_patched.jsonl", val_records)
    write_jsonl(output_dir / "product_lstm_test_patched.jsonl", test_records)
    write_jsonl(output_dir / "product_lstm_needs_review.jsonl", needs_review)
    write_jsonl(output_dir / "product_lstm_rejected_patches.jsonl", rejected_patches)

    report = summarize_patch_report(
        records=original_records,
        patched_records=patched_records,
        repair_source_rows=repair_source_rows,
        patch_stats=patch_stats,
        needs_review=needs_review,
        rejected_patches=rejected_patches,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        added_negative_samples=len(PRODUCT_NEGATIVE_ROWS) if args.add_negative_samples else 0,
    )
    report["data_dir"] = str(args.data_dir)
    report["repair_source_dirs"] = [str(path) for path in args.repair_source_dir]
    report["output_dir"] = str(output_dir)
    report["workers"] = args.workers
    report["seed"] = args.seed
    report["add_negative_samples"] = args.add_negative_samples
    report["load_stats"] = load_stats
    report["original_filter_stats"] = original_filter_stats

    write_json(output_dir / "product_lstm_patch_report.json", report)
    write_text(output_dir / "product_lstm_patch_report.md", render_report_md(report))

    print(f"[PATCH] needs_review={len(needs_review)} rejected={len(rejected_patches)}", flush=True)
    print(f"[PATCH] report saved to {output_dir / 'product_lstm_patch_report.json'}", flush=True)


if __name__ == "__main__":
    main()
