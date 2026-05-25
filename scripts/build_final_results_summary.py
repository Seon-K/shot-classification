from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

MetricValue = float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a presentation-ready final results summary from metrics artifacts.")
    parser.add_argument("--manifest", default="artifacts/dataset/dataset_manifest.json")
    parser.add_argument("--zeroshot-metrics", default="artifacts/zeroshot/test_metrics.json")
    parser.add_argument("--classifier-metrics", default="artifacts/classifier/metrics.json")
    parser.add_argument("--ablation-results", default="artifacts/experiments/ablation_results.csv")
    parser.add_argument("--data-size-results", default="artifacts/experiments/data_size_results.csv")
    parser.add_argument("--backbone-results", default="artifacts/experiments/backbone_results.csv")
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


def select_best_row(rows: list[dict[str, str]], metric: str) -> dict[str, str] | None:
    usable = [row for row in rows if row.get("status", "success") != "failed"]
    if not usable:
        return None
    return max(usable, key=lambda row: as_float(row.get(metric)))


def build_presentation_sentence(zero_shot_accuracy: MetricValue, fine_tuned_accuracy: MetricValue) -> str:
    if zero_shot_accuracy is None or fine_tuned_accuracy is None:
        return "Zero-shot 또는 fine-tuned metrics가 아직 없어 개선폭 문장을 확정할 수 없습니다."
    diff = fine_tuned_accuracy - zero_shot_accuracy
    if diff >= 0:
        return (
            f"Zero-shot CLIP은 test accuracy {pct(zero_shot_accuracy)}였고, "
            f"fine-tuned classifier는 {pct(fine_tuned_accuracy)}로 향상되어 "
            f"+{pp(diff)} 개선을 보였습니다."
        )
    return (
        f"Zero-shot CLIP은 test accuracy {pct(zero_shot_accuracy)}였고, "
        f"fine-tuned classifier는 {pct(fine_tuned_accuracy)}로 낮아져 "
        f"{pp(diff)} 차이를 보였습니다."
    )


def summarize_data_size(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "데이터 크기별 실험 결과가 아직 생성되지 않았습니다."
    successful = [row for row in rows if row.get("status", "success") != "failed"]
    if not successful:
        return "데이터 크기별 실험은 실행되었지만 모든 설정이 실패했습니다."
    best = max(successful, key=lambda row: as_float(row.get("test_macro_f1")))
    first = successful[0]
    last = successful[-1]
    return (
        f"가장 높은 test macro F1은 train_size={best.get('train_size')}에서 "
        f"{pct(as_float(best.get('test_macro_f1')))}였습니다. "
        f"첫 설정({first.get('train_size')})의 macro F1은 {pct(as_float(first.get('test_macro_f1')))}, "
        f"마지막 설정({last.get('train_size')})은 {pct(as_float(last.get('test_macro_f1')))}입니다."
    )


def summarize_backbones(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "Backbone 비교 결과가 아직 생성되지 않았습니다."
    successful = [row for row in rows if row.get("status", "success") == "success"]
    failed = [row for row in rows if row.get("status") == "failed"]
    if not successful:
        return f"Backbone 비교는 실행되었지만 성공한 모델이 없습니다. 실패 수: {len(failed)}."
    best = max(successful, key=lambda row: as_float(row.get("test_macro_f1")))
    suffix = f" 실패한 backbone 수는 {len(failed)}개입니다." if failed else ""
    return (
        f"가장 높은 test macro F1 backbone은 {best.get('backbone')}/{best.get('pretrained')}이며 "
        f"macro F1 {pct(as_float(best.get('test_macro_f1')))}, accuracy {pct(as_float(best.get('test_accuracy')))}입니다."
        f"{suffix}"
    )


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    zeroshot = read_json(args.zeroshot_metrics)
    classifier = read_json(args.classifier_metrics)
    ablation_rows = read_csv(args.ablation_results)
    data_size_rows = read_csv(args.data_size_results)
    backbone_rows = read_csv(args.backbone_results)

    split_counts = manifest.get("split_counts", {})
    dataset_size = sum(int(split_counts.get(key, 0)) for key in ["train", "val", "test"])
    class_distribution = manifest.get("folder_counts", {})

    zero_acc = as_optional_float(zeroshot.get("accuracy"))
    best_metrics = classifier.get("test_metrics_best", {})
    fine_acc = as_optional_float(best_metrics.get("accuracy"))
    fine_macro_f1 = as_optional_float(best_metrics.get("macro_f1"))
    fine_balanced = as_optional_float(best_metrics.get("balanced_accuracy"))
    fine_top2 = as_optional_float(best_metrics.get("top2_accuracy"))
    improvement = fine_acc - zero_acc if zero_acc is not None and fine_acc is not None else None
    best_ablation = select_best_row(ablation_rows, "best_val_macro_f1")

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
        f"- improvement: {'+' + pp(improvement) if improvement is not None and improvement >= 0 else (pp(improvement) if improvement is not None else 'N/A')}",
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

    lines.extend(
        [
            "",
            "## Data Size Trend",
            summarize_data_size(data_size_rows),
            "",
            "## Backbone Comparison",
            summarize_backbones(backbone_rows),
            "",
            "## Remaining Limitations",
            "- 데이터 크기별 실험은 video_id 단위 subset이므로 정확히 요청한 이미지 수와 일치하지 않을 수 있습니다.",
            "- backbone 비교는 같은 classifier head 기준이며, 각 backbone별 최적 하이퍼파라미터 탐색은 아닙니다.",
            "- shot_type 라벨은 scale과 subject가 섞인 단일 라벨이므로 구조적 모호성이 남아 있습니다.",
            "- scene detection threshold는 데이터셋 스타일에 의존하며, 실제 서비스 환경에서는 추가 검증이 필요합니다.",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary: {output}")
    print(build_presentation_sentence(zero_acc, fine_acc))


if __name__ == "__main__":
    main()
