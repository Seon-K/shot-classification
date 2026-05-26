"""Preflight check script for final submission before presentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class PrefightStatus:
    """Track preflight check results."""

    def __init__(self):
        self.results: list[dict[str, str]] = []
        self.status_counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    def add_result(self, check_name: str, status: str, message: str) -> None:
        """Add a check result."""
        if status not in self.status_counts:
            status = "FAIL"
        self.results.append({"check": check_name, "status": status, "message": message})
        self.status_counts[status] += 1

    def get_report(self) -> str:
        """Generate a formatted report."""
        lines = [
            "# Preflight Check Report",
            "",
            f"Summary: PASS={self.status_counts['PASS']}, WARN={self.status_counts['WARN']}, FAIL={self.status_counts['FAIL']}",
            "",
            "## Detailed Results",
            "",
        ]

        for result in self.results:
            status = result["status"]
            check = result["check"]
            message = result["message"]
            lines.append(f"### [{status}] {check}")
            lines.append(f"{message}")
            lines.append("")

        return "\n".join(lines)

    def is_critical_failure(self) -> bool:
        """Check if there are critical failures."""
        critical = ["required packages", "required artifacts"]
        for result in self.results:
            if result["status"] == "FAIL":
                for crit in critical:
                    if crit.lower() in result["check"].lower():
                        return True
        return False


def check_required_packages(status: PrefightStatus) -> None:
    """Check if all required packages can be imported."""
    required_packages = {
        "torch": "PyTorch",
        "torchvision": "Torchvision",
        "open_clip": "OpenCLIP",
        "cv2": "OpenCV",
        "scenedetect": "PySceneDetect",
        "PIL": "Pillow",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "streamlit": "Streamlit",
        "openpyxl": "openpyxl",
    }

    failed = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            failed.append(f"{package_name} ({module_name})")

    if failed:
        status.add_result(
            "Required Packages",
            "FAIL",
            f"Missing packages: {', '.join(failed)}. Install with: pip install -r requirements.txt",
        )
    else:
        status.add_result("Required Packages", "PASS", "All required packages available.")


def check_required_artifacts(status: PrefightStatus) -> None:
    """Check if all required artifact files exist."""
    required_artifacts = [
        ("artifacts/classifier/shot_classifier.pt", "Shot classifier checkpoint"),
        ("artifacts/text_classifier/text_classifier.pt", "Text classifier checkpoint"),
        ("artifacts/dataset/dataset_manifest.json", "Dataset manifest"),
    ]

    missing = []
    for path_str, description in required_artifacts:
        path = Path(path_str)
        if not path.exists():
            missing.append(f"{description} ({path_str})")

    if missing:
        status.add_result(
            "Required Artifacts",
            "FAIL",
            f"Missing artifacts: {', '.join(missing)}. Run full pipeline: python scripts/run_full_pipeline.py",
        )
    else:
        status.add_result("Required Artifacts", "PASS", "All required artifacts present.")


def check_metrics_readable(status: PrefightStatus) -> None:
    """Check if metrics.json can be read and contains accuracy/macro F1."""
    metrics_path = Path("artifacts/classifier/metrics.json")
    if not metrics_path.exists():
        status.add_result("Metrics Readable", "WARN", "metrics.json not found (may not be generated yet).")
        return

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        best_metrics = metrics.get("test_metrics_best") or metrics.get("test_metrics", {})

        has_accuracy = "accuracy" in best_metrics
        has_macro_f1 = "macro_f1" in best_metrics

        if has_accuracy and has_macro_f1:
            acc = best_metrics.get("accuracy", 0)
            f1 = best_metrics.get("macro_f1", 0)
            status.add_result(
                "Metrics Readable",
                "PASS",
                f"Metrics readable: accuracy={acc:.4f}, macro_f1={f1:.4f}",
            )
        else:
            missing_metrics = []
            if not has_accuracy:
                missing_metrics.append("accuracy")
            if not has_macro_f1:
                missing_metrics.append("macro_f1")
            status.add_result(
                "Metrics Readable",
                "WARN",
                f"Missing metrics: {', '.join(missing_metrics)}",
            )
    except Exception as exc:
        status.add_result("Metrics Readable", "FAIL", f"Failed to read metrics.json: {exc}")


def check_plot_files(status: PrefightStatus) -> None:
    """Check if plot files exist."""
    plot_dir = Path("artifacts/plots")
    required_plots = [
        "ablation_comparison.png",
        "data_size_curve.png",
        "backbone_comparison.png",
        "confusion_matrix_heatmap.png",
        "training_accuracy_curve.png",
        "training_loss_curve.png",
    ]

    missing = []
    for plot_file in required_plots:
        plot_path = plot_dir / plot_file
        if not plot_path.exists():
            missing.append(plot_file)

    if missing:
        if len(missing) == len(required_plots):
            status.add_result(
                "Plot Files",
                "WARN",
                f"No plots generated (matplotlib might be missing). Missing: {', '.join(missing[:3])}...",
            )
        else:
            status.add_result(
                "Plot Files",
                "WARN",
                f"Some plots missing: {', '.join(missing)}. Run: python scripts/generate_plots.py",
            )
    else:
        status.add_result("Plot Files", "PASS", "All plot files present.")


def check_demo_output(status: PrefightStatus) -> None:
    """Check if demo output_overlay.mp4 exists."""
    demo_output_path = Path("artifacts/demo_streamlit/0030/output_overlay.mp4")  # Example path
    if not demo_output_path.exists():
        status.add_result(
            "Demo Output",
            "WARN",
            "Demo output_overlay.mp4 not found (may not be generated yet). Run demo: python -m streamlit run demo_app.py",
        )
    else:
        status.add_result("Demo Output", "PASS", "Demo output_overlay.mp4 present.")


def check_system_info(status: PrefightStatus) -> None:
    """Check if system_info.json exists."""
    system_info_path = Path("artifacts/system_info.json")
    if not system_info_path.exists():
        status.add_result(
            "System Info",
            "WARN",
            "system_info.json not found (may not be generated yet).",
        )
        return

    try:
        system_info = json.loads(system_info_path.read_text(encoding="utf-8"))
        has_python_version = "python_version" in system_info
        has_torch_version = "torch_version" in system_info

        if has_python_version and has_torch_version:
            status.add_result(
                "System Info",
                "PASS",
                f"System info readable: python={system_info.get('python_version')}, torch={system_info.get('torch_version')}",
            )
        else:
            status.add_result("System Info", "WARN", "system_info.json missing expected fields.")
    except Exception as exc:
        status.add_result("System Info", "WARN", f"Failed to read system_info.json: {exc}")


def check_final_results_summary(status: PrefightStatus) -> None:
    """Check if final_results_summary.md exists."""
    summary_path = Path("artifacts/final_results_summary.md")
    if not summary_path.exists():
        status.add_result(
            "Final Results Summary",
            "WARN",
            "final_results_summary.md not found. Run: python scripts/build_final_results_summary.py",
        )
        return

    try:
        content = summary_path.read_text(encoding="utf-8")
        has_content = len(content) > 100
        if has_content:
            status.add_result("Final Results Summary", "PASS", "final_results_summary.md exists with content.")
        else:
            status.add_result("Final Results Summary", "WARN", "final_results_summary.md exists but is mostly empty.")
    except Exception as exc:
        status.add_result("Final Results Summary", "FAIL", f"Failed to read final_results_summary.md: {exc}")


def check_analysis_notes(status: PrefightStatus) -> None:
    """Check if analysis_notes.md exists."""
    notes_path = Path("artifacts/analysis_notes.md")
    if not notes_path.exists():
        status.add_result(
            "Analysis Notes",
            "WARN",
            "analysis_notes.md not found. Run: python scripts/generate_analysis_notes.py",
        )
        return

    try:
        content = notes_path.read_text(encoding="utf-8")
        has_content = len(content) > 100
        if has_content:
            status.add_result("Analysis Notes", "PASS", "analysis_notes.md exists with content.")
        else:
            status.add_result("Analysis Notes", "WARN", "analysis_notes.md exists but is mostly empty.")
    except Exception as exc:
        status.add_result("Analysis Notes", "FAIL", f"Failed to read analysis_notes.md: {exc}")


def main() -> None:
    status = PrefightStatus()

    print("Running preflight checks...\n")

    # Run all checks
    check_required_packages(status)
    check_required_artifacts(status)
    check_metrics_readable(status)
    check_plot_files(status)
    check_demo_output(status)
    check_system_info(status)
    check_final_results_summary(status)
    check_analysis_notes(status)

    # Generate and save report
    report = status.get_report()
    output_path = Path("artifacts/preflight_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    # Print summary
    print(report)
    print(f"\nReport saved to: {output_path}")

    # Exit with appropriate code
    if status.is_critical_failure():
        print("\n⚠️  CRITICAL FAILURES DETECTED - Please fix before presentation")
        sys.exit(1)
    elif status.status_counts["FAIL"] > 0:
        print("\n⚠️  FAILURES DETECTED - Review before presentation")
        sys.exit(1)
    elif status.status_counts["WARN"] > 0:
        print("\n⚠️  WARNINGS DETECTED - Check optional items")
        sys.exit(0)
    else:
        print("\n✅ All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
