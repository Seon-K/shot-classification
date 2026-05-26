from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LIMITATIONS = [
    "shot_type 라벨 axis 혼합 문제",
    "weak auxiliary labels",
    "scene detection mismatch",
    "rule-based temporal summary 한계",
    "small dataset limitation",
    "full temporal modeling 미구현",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build presentation defense points from metrics artifacts.")
    parser.add_argument("--zeroshot-metrics", default="artifacts/zeroshot/test_metrics.json")
    parser.add_argument("--classifier-metrics", default="artifacts/classifier/metrics.json")
    parser.add_argument("--ablation-results", default="artifacts/experiments/ablation_results.csv")
    parser.add_argument("--output", default="artifacts/defense_points.md")
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


def f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def improvement_sentence(zero: float | None, fine: float | None) -> str:
    if zero is None or fine is None:
        return "zero-shot과 fine-tuned 성능이 모두 있어야 개선폭을 확정할 수 있습니다."
    diff = fine - zero
    return f"현재 결과에서는 fine-tuned classifier가 zero-shot 대비 accuracy {diff * 100:+.1f}%p 차이를 보입니다."


def best_ablation_sentence(rows: list[dict[str, str]]) -> str:
    usable = [row for row in rows if row.get("status", "success") != "failed"]
    if not usable:
        return "ablation 결과가 없어 어떤 regularization/imbalance 옵션이 가장 좋았는지 단정하지 않았습니다."
    best = max(usable, key=lambda row: float(row.get("best_val_macro_f1", 0.0)))
    return (
        f"ablation 기준 best setting은 {best.get('experiment_name')}이며, "
        f"validation macro F1 {float(best.get('best_val_macro_f1', 0.0)) * 100:.1f}%입니다."
    )


def main() -> None:
    args = parse_args()
    zero = f(read_json(args.zeroshot_metrics).get("accuracy"))
    classifier = read_json(args.classifier_metrics)
    best_metrics = classifier.get("test_metrics_best") or classifier.get("test_metrics", {})
    fine = f(best_metrics.get("accuracy"))
    macro_f1 = f(best_metrics.get("macro_f1"))
    balanced = f(best_metrics.get("balanced_accuracy"))

    lines = [
        "# Defense Points",
        "",
        "## 왜 DL인가",
        "- 컷별 shot type은 배경, 인물 크기, 사물 중심성, 자막 유무가 섞인 시각 패턴이라 hand-crafted rule만으로 안정적으로 분류하기 어렵습니다.",
        "- DL embedding은 화면 구성의 고차원 시각 특징을 압축해 rule 기반 가이드 생성의 입력으로 사용할 수 있습니다.",
        "",
        "## 왜 CLIP인가",
        "- CLIP은 작은 데이터에서도 일반 시각 개념을 잘 담은 embedding을 제공해, 1천 장 규모 데이터에서 처음부터 CNN을 학습하는 것보다 현실적입니다.",
        "- backbone은 고정하고 얕은 classifier를 학습해 실험 비용과 과적합 위험을 줄였습니다.",
        "",
        "## 왜 fine-tuning이 필요한가",
        f"- {improvement_sentence(zero, fine)}",
        "- zero-shot prompt만으로는 프로젝트의 shot_type 정의와 데이터셋 라벨링 기준을 충분히 반영하기 어렵습니다.",
        "",
        "## 왜 macro F1을 봤는가",
        f"- test accuracy는 {pct(fine)}, macro F1은 {pct(macro_f1)}, balanced accuracy는 {pct(balanced)}입니다.",
        "- medium/object처럼 많은 class가 전체 accuracy를 끌어올릴 수 있으므로, minority class 성능을 반영하는 macro F1이 필요합니다.",
        "",
        "## 왜 zero-shot만 쓰지 않았는가",
        "- zero-shot은 빠른 baseline이지만, close-up/medium/wide와 object/space가 섞인 프로젝트 고유 라벨 정의에는 supervised calibration이 필요합니다.",
        "",
        "## Ablation 근거",
        f"- {best_ablation_sentence(read_csv(args.ablation_results))}",
        "",
        "## 현재 한계",
    ]
    lines.extend(f"- {item}" for item in LIMITATIONS)
    lines.extend(
        [
            "",
            "## 과장하면 안 되는 부분",
            "- 일반적인 모든 숏폼 영상에 바로 일반화된다고 말하면 안 됩니다.",
            "- multi-task classifier가 기존 classifier를 대체했다고 말하면 안 됩니다. 현재는 보조 실험입니다.",
            "- rule-based temporal summary를 Transformer/RNN temporal modeling처럼 설명하면 안 됩니다.",
            "- scene detection mismatch가 완전히 해결되었다고 말하면 안 됩니다.",
            "",
            "## Future Work",
            "- scale/subject/text를 분리한 hierarchical 또는 multi-task label schema 개선",
            "- 더 큰 labeled dataset과 external validation set 확보",
            "- scene detection threshold 자동 선택 또는 shot boundary model 도입",
            "- temporal modeling 기반 전환 패턴 학습",
            "- OCR 기반 text presence 보강",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"defense points: {output}")


if __name__ == "__main__":
    main()
