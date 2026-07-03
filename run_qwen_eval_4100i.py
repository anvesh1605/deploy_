from pathlib import Path

from qwen_eval import run_qwen_evaluation


def main() -> None:
    print("[QWEN] Starting 4100i eval-only baseline")
    result = run_qwen_evaluation(
        Path(r"C:\Hpe\Train\outputs_4100i_gpu\converted_4100i_lstm.jsonl"),
        Path(r"C:\Hpe\Train\outputs_qwen_eval\4100i"),
        "Qwen/Qwen2.5-1.5B-Instruct",
        100,
    )
    report = result["report"]
    print("Qwen eval completed")
    print(f"Dataset path: {report['dataset_path']}")
    print(f"Total samples: {report['total_samples']}")
    print(f"Average ROUGE-L: {float(report['rouge_l']):.4f}")
    print(f"Average token F1: {float(report['token_f1']):.4f}")
    print(f"Exact match: {float(report['exact_match']):.4f}")
    syntax_value = report.get("syntax_preservation")
    command_value = report.get("command_preservation")
    event_value = report.get("event_id_preservation")
    print(f"Syntax preservation: {float(syntax_value):.4f}" if syntax_value is not None else "Syntax preservation: n/a")
    print(f"Command preservation: {float(command_value):.4f}" if command_value is not None else "Command preservation: n/a")
    print(f"Event ID preservation: {float(event_value):.4f}" if event_value is not None else "Event ID preservation: n/a")
    print(f"Output folder: {report['output_dir']}")
    print(f"Verdict: {report['verdict']}")


if __name__ == "__main__":
    main()
