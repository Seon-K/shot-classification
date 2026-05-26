from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

MetricValue = float | None
LIMITATIONS = [
    "shot_type 라벨 axis 혼합 문제",
    "weak auxiliary labels",
    "scene detection mismatch",
    "rule-based temporal summary 한계",
    "small dataset limitation",
    "full temporal modeling 미구현",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a presentation-ready final results summary from metrics artifacts.")
    parser.add_argument("--manifest", default="artifacts/dataset/dataset_manifest.json")
    parser.add_argument("--zeroshot-metrics", default="artifacts/zeroshot/test_metrics.json")
    parser.add_argument("--classifier-metrics", default="artifacts/classifier/metrics.json")
    parser.add_argument("--ablation-results", default="artifacts/experiments/ablation_results.csv")
    parser.add_argument("--data-size-results", default="artifacts/experiments/data_size_results.csv")
    parser.add_argument("--backbone-results", default="artifacts/experiments/backbone_results.csv")
    parser.add_argument("--scene-sweep", default="artifacts/scenes/scene_threshold_sweep.csv")
    parser.add_argument("--pipeline-summary", default="artifacts/pipeline_run_summary.csv")
    parser.add_argument("--output", default="artifacts/final_results_summary.md")
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_optional_float(value: Any) -> MetricValue:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pct_optional(value: MetricValue) -> str:
    return "N/A" if value is None else pct(value)


def pp(value: float) -> str:
    return f"{value * 100:.2f}%p"


def pp_optional(value: MetricValue) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{pp(value)}"


def usable_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status", "success") != "failed"]


def check_plot_generation_status(plot_status_path: str | Path) -> bool:
    """Check if plot generation was successful (matplotlib available)."""
    path = Path(plot_status_path)
    if not path.exists():
        return True  # If no status file, assume success (plots not run yet)
    rows = read_csv(path)
    if not rows:
        return True
    # If all plots failed with matplotlib import error, matplotlib is unavailable
    for row in rows:
        if "matplotlib import failed" in row.get("error_message", ""):
            return False
    return True
    usable = usable_rows(rows)
    if not usable:
        return None
    return max(usable, key=lambda row: as_float(row.get(metric)))


def select_best_scene_threshold(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: (as_float(row.get("mismatch_ratio")), as_float(row.get("avg_abs_count_difference"))))


def average_runtime(rows: list[dict[str, str]]) -> MetricValue:
    values = [as_float(row.get("runtime_sec")) for row in rows if row.get("status") == "success"]
    if not values:
        return None
    return sum(values) / len(values)


def build_presentation_sentence(zero_shot_accuracy: MetricValue, fine_tuned_accuracy: MetricValue) -> str:
    if zero_shot_accuracy is None or fine_tuned_accuracy is None:
        return "Zero-shot 또는 fine-tuned metrics가 아직 없어 개선폭 문장을 확정할 수 없습니다."
    diff = fine_tuned_accuracy - zero_shot_accuracy
    if diff >= 0:
        return f"Fine-tuned classifier는 zero-shot 대비 accuracy {pp(diff)} 향상되었다."
    return f"Fine-tuned classifier는 zero-shot 대비 accuracy {pp(diff)} 낮게 나타났다."


def summarize_data_size(rows: list[dict[str, str]]) -> tuple[str, dict[str, str] | None]:
    usable = usable_rows(rows)
    if not usable:
        return "데이터 크기별 실험 결과가 아직 생성되지 않았습니다.", None
    best = max(usable, key=lambda row: as_float(row.get("test_macro_f1")))
    first = usable[0]
    last = usable[-1]
    text = (
        f"validation macro F1 기준으로 선택한 설정의 test macro F1은 train_size={best.get('train_size')}에서 "
        f"{pct(as_float(best.get('test_macro_f1')))}였습니다. "
        f"첫 설정({first.get('train_size')})의 test macro F1은 {pct(as_float(first.get('test_macro_f1')))}, "
        f"마지막 설정({last.get('train_size')})은 {pct(as_float(last.get('test_macro_f1')))}입니다."
    )
    return text, best


def summarize_backbones(rows: list[dict[str, str]]) -> tuple[str, dict[str, str] | None]:
    successful = [row for row in rows if row.get("status") == "success"]
    failed = [row for row in rows if row.get("status") == "failed"]
    if not successful:
        if failed:
            return f"Backbone 비교는 실행되었지만 성공한 모델이 없습니다. 실패 수: {len(failed)}.", None
        return "Backbone 비교 결과가 아직 생성되지 않았습니다.", None
    best = max(successful, key=lambda row: as_float(row.get("test_macro_f1")))
    suffix = f" 실패한 backbone 수는 {len(failed)}개입니다." if failed else ""
    return (
        f"validation macro F1 기준으로 선택한 backbone은 {best.get('backbone')}/{best.get('pretrained')}이며 "
        f"test macro F1 {pct(as_float(best.get('test_macro_f1')))}, test accuracy {pct(as_float(best.get('test_accuracy')))}입니다.{suffix}",
        best,
    )


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    zeroshot = read_json(args.zeroshot_metrics)
    classifier = read_json(args.classifier_metrics)
    ablation_rows = read_csv(args.ablation_results)
    data_size_rows = read_csv(args.data_size_results)
    backbone_rows = read_csv(args.backbone_results)
    scene_rows = read_csv(args.scene_sweep)
    pipeline_rows = read_csv(args.pipeline_summary)

    split_counts = manifest.get("split_counts", {})
    dataset_size = sum(int(split_counts.get(key, 0)) for key in ["train", "val", "test"])
    class_distribution = manifest.get("folder_counts", {})

    zero_acc = as_optional_float(zeroshot.get("accuracy"))
    best_metrics = classifier.get("test_metrics_best") or classifier.get("test_metrics", {})
    fine_acc = as_optional_float(best_metrics.get("accuracy"))
    fine_macro_f1 = as_optional_float(best_metrics.get("macro_f1"))
    fine_balanced = as_optional_float(best_metrics.get("balanced_accuracy"))
    fine_top2 = as_optional_float(best_metrics.get("top2_accuracy"))
    improvement = fine_acc - zero_acc if zero_acc is not None and fine_acc is not None else None
    best_ablation = select_best_row(ablation_rows, "best_val_macro_f1")
    data_size_text, best_data_size = summarize_data_size(data_size_rows)
    backbone_text, best_backbone = summarize_backbones(backbone_rows)
    best_scene = select_best_scene_threshold(scene_rows)
    avg_runtime = average_runtime(pipeline_rows)
    plots_generated = check_plot_generation_status("artifacts/plots/plot_generation_status.csv")

    lines = [
        "# Final Results Summary",
        "",
        "## Dataset",
        f"- total samples: {dataset_size}",
        f"- train: {split_counts.get('train', 0)}",
        f"- val: {split_counts.get('val', 0)}",
        f"- test: {split_counts.get('test', 0)}",
        f"- class distribution: `{json.dumps(class_distribution, ensure_ascii=False)}`",
        "",
        "## Zero-Shot vs Fine-Tuned",
        f"- zero-shot test accuracy: {pct_optional(zero_acc)}",
        f"- fine-tuned test accuracy: {pct_optional(fine_acc)}",
        f"- fine-tuned test macro F1: {pct_optional(fine_macro_f1)}",
        f"- fine-tuned balanced accuracy: {pct_optional(fine_balanced)}",
        f"- fine-tuned top-2 accuracy: {pct_optional(fine_top2)}",
        f"- improvement: {pp_optional(improvement)}",
        "",
        "## Presentation Sentence",
        build_presentation_sentence(zero_acc, fine_acc),
        "",
        "## Best Ablation Setting",
    ]
    if best_ablation:
        lines.extend(
            [
                f"- experiment: {best_ablation.get('experiment_name')}",
                f"- loss_type: {best_ablation.get('loss_type')}",
                f"- use_class_weight: {best_ablation.get('use_class_weight')}",
                f"- use_weighted_sampler: {best_ablation.get('use_weighted_sampler')}",
                f"- dropout: {best_ablation.get('dropout')}",
                f"- weight_decay: {best_ablation.get('weight_decay')}",
                f"- norm: {best_ablation.get('norm')}",
                f"- best_val_macro_f1: {pct(as_float(best_ablation.get('best_val_macro_f1')))}",
            ]
        )
    else:
        lines.append("- ablation_results.csv가 아직 없거나 유효한 행이 없습니다.")

    lines.extend(["", "## Data Size Trend", data_size_text])
    if best_data_size:
        lines.append(f"- best_data_size: {best_data_size.get('train_size')} (test macro F1 {pct(as_float(best_data_size.get('test_macro_f1')))})")
    lines.extend(["", "## Backbone Comparison", backbone_text])
    if best_backbone:
        lines.append(f"- best_backbone: {best_backbone.get('backbone')} ({best_backbone.get('pretrained')})")
    lines.extend(["", "## Scene Threshold Sweep"])
    if best_scene:
        lines.extend(
            [
                f"- best threshold: {best_scene.get('threshold')}",
                f"- mismatch_ratio: {pct(as_float(best_scene.get('mismatch_ratio')))}",
                f"- avg_abs_count_difference: {as_float(best_scene.get('avg_abs_count_difference')):.2f}",
            ]
        )
    else:
        lines.append("- scene threshold sweep 결과가 아직 없습니다.")
    lines.extend(["", "## Pipeline Runtime", f"- average successful step runtime: {avg_runtime:.2f} sec" if avg_runtime is not None else "- average successful step runtime: N/A"])
    if not plots_generated:
        lines.extend(["", "## Plot Generation", "- plot generation failed due to missing matplotlib"])
    lines.extend(["", "## Remaining Limitations"])
    lines.extend(f"- {item}" for item in LIMITATIONS)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary: {output}")
    print(build_presentation_sentence(zero_acc, fine_acc))


if __name__ == "__main__":
    main()
