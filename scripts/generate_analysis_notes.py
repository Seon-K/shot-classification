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
    parser = argparse.ArgumentParser(description="Generate presentation/Q&A analysis notes from experiment artifacts.")
    parser.add_argument("--classifier-metrics", default="artifacts/classifier/metrics.json")
    parser.add_argument("--data-size-results", default="artifacts/experiments/data_size_results.csv")
    parser.add_argument("--backbone-results", default="artifacts/experiments/backbone_results.csv")
    parser.add_argument("--scene-sweep", default="artifacts/scenes/scene_threshold_sweep.csv")
    parser.add_argument("--output", default="artifacts/analysis_notes.md")
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


def f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def data_size_note(rows: list[dict[str, str]]) -> str:
    usable = [row for row in rows if row.get("status", "success") != "failed"]
    if len(usable) < 2:
        return "데이터 크기별 실험 결과가 부족해 scaling trend를 단정하지 않았습니다."
    first, last = usable[0], usable[-1]
    delta = f(last.get("test_macro_f1")) - f(first.get("test_macro_f1"))
    if delta > 0.01:
        return f"data size 증가에 따라 macro F1이 {pct(delta)}p 개선되는 경향을 보였습니다."
    if delta < -0.01:
        return f"data size 증가에도 macro F1이 {pct(abs(delta))}p 낮아져 subset 구성과 과적합 영향을 추가 확인해야 합니다."
    return "data size 증가에 따른 macro F1 변화는 작아, 현재 데이터 양만으로는 scaling 효과가 제한적입니다."


def backbone_note(rows: list[dict[str, str]]) -> str:
    usable = [row for row in rows if row.get("status") == "success"]
    if not usable:
        return "성공한 backbone 비교 결과가 없어 backbone 우열을 말하지 않았습니다."
    best = max(usable, key=lambda row: f(row.get("test_macro_f1")))
    slowest = max(usable, key=lambda row: f(row.get("embedding_time_sec")) + f(row.get("fit_time_sec")))
    return (
        f"validation macro F1 기준으로 선택한 {best.get('backbone')} backbone은 test macro F1 {pct(f(best.get('test_macro_f1')))}을 보였습니다. "
        f"runtime cost는 {slowest.get('backbone')}에서 가장 컸습니다."
    )


def scene_note(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "scene threshold sweep 결과가 없어 threshold 추천을 보류했습니다."
    best = min(rows, key=lambda row: (f(row.get("mismatch_ratio")), f(row.get("avg_abs_count_difference"))))
    return (
        f"scene detection threshold sweep에서는 threshold={best.get('threshold')}가 "
        f"mismatch_ratio {pct(f(best.get('mismatch_ratio')))}로 가장 안정적이었습니다."
    )


def main() -> None:
    args = parse_args()
    metrics = read_json(args.classifier_metrics)
    best = metrics.get("test_metrics_best") or metrics.get("test_metrics", {})
    accuracy = f(best.get("accuracy"))
    macro_f1 = f(best.get("macro_f1"))
    miscls = metrics.get("misclassification_analysis", [])
    top_pair = miscls[0] if miscls else None

    lines = ["# Analysis Notes", "", "## Key Findings"]
    if top_pair:
        lines.append(
            f"- {top_pair.get('true_label')}과/와 {top_pair.get('pred_label')} 사이 confusion이 가장 많이 발생했습니다 "
            f"({top_pair.get('count')}건, true 기준 {pct(f(top_pair.get('ratio_within_true')))})."
        )
    else:
        lines.append("- confusion pair 정보가 없어 가장 혼동되는 class pair를 확정하지 않았습니다.")
    if accuracy and macro_f1 and macro_f1 < accuracy - 0.05:
        lines.append(f"- accuracy({pct(accuracy)})보다 macro F1({pct(macro_f1)})이 낮아 class imbalance와 minority class 성능 저하 가능성이 있습니다.")
    elif accuracy and macro_f1:
        lines.append(f"- accuracy({pct(accuracy)})와 macro F1({pct(macro_f1)}) 간 차이가 크지 않아 클래스별 성능 편차는 제한적입니다.")
    else:
        lines.append("- classifier metrics가 부족해 accuracy와 macro F1 차이를 해석하지 않았습니다.")
    lines.append(f"- {data_size_note(read_csv(args.data_size_results))}")
    lines.append(f"- {backbone_note(read_csv(args.backbone_results))}")
    lines.append(f"- {scene_note(read_csv(args.scene_sweep))}")
    lines.extend(["", "## Limitations"])
    lines.extend(f"- {item}" for item in LIMITATIONS)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"analysis notes: {output}")


if __name__ == "__main__":
    main()
