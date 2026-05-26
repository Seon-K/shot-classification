from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


FIELDNAMES = ["step_name", "status", "runtime_sec", "start_time", "end_time", "error_message"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full shortform experiment pipeline and record per-step status.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/pipeline_run_summary.csv")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for main/experiment classifier training.")
    parser.add_argument("--text-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--scene-limit", type=int, default=None, help="Optional video limit for scene tuning.")
    parser.add_argument("--skip-backbone", action="store_true", help="Skip expensive backbone comparison.")
    parser.add_argument("--skip-data-size", action="store_true", help="Skip data size sweep.")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation sweep.")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_step(step_name: str, cmd: list[str]) -> dict[str, str | float]:
    start_time = now_iso()
    start = time.perf_counter()
    status = "success"
    error_message = ""
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:  # noqa: BLE001 - pipeline should continue after failures.
        status = "failed"
        error_message = str(exc).replace("\n", " ")[:1000]
    end = time.perf_counter()
    end_time = now_iso()
    return {
        "step_name": step_name,
        "status": status,
        "runtime_sec": round(end - start, 3),
        "start_time": start_time,
        "end_time": end_time,
        "error_message": error_message,
    }


def write_summary(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_steps(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    py = sys.executable
    steps: list[tuple[str, list[str]]] = [
        ("prepare_dataset", [py, "scripts/prepare_dataset.py", "--seed", str(args.seed)]),
        (
            "extract_clip_embeddings",
            [
                py,
                "scripts/extract_clip_embeddings.py",
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--batch-size",
                str(args.embedding_batch_size),
            ],
        ),
        (
            "evaluate_zeroshot",
            [py, "scripts/evaluate_zeroshot.py", "--device", args.device, "--seed", str(args.seed)],
        ),
        (
            "train_shot_classifier",
            [
                py,
                "scripts/train_shot_classifier.py",
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
            ],
        ),
        (
            "train_text_classifier",
            [
                py,
                "scripts/train_text_classifier.py",
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.text_epochs),
                "--batch-size",
                str(args.batch_size),
            ],
        ),
    ]
    if not args.skip_ablation:
        steps.append(
            (
                "run_ablation",
                [
                    py,
                    "scripts/run_ablation.py",
                    "--device",
                    args.device,
                    "--seed",
                    str(args.seed),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                ],
            )
        )
    if not args.skip_data_size:
        steps.append(
            (
                "run_data_size_experiment",
                [
                    py,
                    "scripts/run_data_size_experiment.py",
                    "--device",
                    args.device,
                    "--seed",
                    str(args.seed),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                ],
            )
        )
    if not args.skip_backbone:
        steps.append(
            (
                "run_backbone_comparison",
                [
                    py,
                    "scripts/run_backbone_comparison.py",
                    "--device",
                    args.device,
                    "--seed",
                    str(args.seed),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--embedding-batch-size",
                    str(args.embedding_batch_size),
                ],
            )
        )
    scene_cmd = [py, "scripts/tune_scene_detection.py"]
    if args.scene_limit is not None:
        scene_cmd.extend(["--limit", str(args.scene_limit)])
    steps.extend(
        [
            ("tune_scene_detection", scene_cmd),
            ("build_final_results_summary", [py, "scripts/build_final_results_summary.py"]),
            ("generate_plots", [py, "scripts/generate_plots.py"]),
            ("generate_analysis_notes", [py, "scripts/generate_analysis_notes.py"]),
            ("build_defense_points", [py, "scripts/build_defense_points.py"]),
        ]
    )
    return steps


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for step_name, cmd in build_steps(args):
        print(f"[pipeline] {step_name}")
        row = run_step(step_name, cmd)
        rows.append(row)
        write_summary(args.output, rows)
        if row["status"] == "failed":
            print(f"[pipeline] failed {step_name}: {row['error_message']}")
    print(f"summary: {args.output}")


if __name__ == "__main__":
    main()
