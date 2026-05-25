from __future__ import annotations

from collections import Counter
from textwrap import wrap


GUIDE_TEMPLATES = {
    "close-up": "표정이나 디테일을 강조하는 클로즈업 구간입니다.",
    "medium": "상반신과 동작을 함께 보여주는 미디엄 샷 구간입니다.",
    "wide": "인물과 배경의 관계를 보여주는 와이드 샷 구간입니다.",
    "object": "제품, 텍스트, 화면 등 사물 정보가 중심인 구간입니다.",
    "space": "장소와 분위기 정보를 강조하는 공간 중심 구간입니다.",
    "uncertain": "판정 신뢰도가 낮아 추가 확인이 필요한 구간입니다.",
}


def guide_for_label(label: str, confidence: float, threshold: float = 0.45) -> str:
    if confidence < threshold:
        return GUIDE_TEMPLATES["uncertain"]
    return GUIDE_TEMPLATES.get(label, GUIDE_TEMPLATES["uncertain"])


def summarize_guides(rows: list[dict]) -> str:
    if not rows:
        return "분석된 컷이 없습니다.\n"

    durations = [float(row["duration_sec"]) for row in rows]
    labels = [row["shot_type"] for row in rows]
    counts = Counter(labels)
    total = len(rows)
    avg_duration = sum(durations) / total
    most_common_label, most_common_count = counts.most_common(1)[0]

    lines = [
        f"전체 컷 수: {total}",
        f"평균 컷 길이: {avg_duration:.2f}초",
        "샷 타입 분포:",
    ]
    for label, count in counts.most_common():
        lines.append(f"- {label}: {count}개 ({count / total * 100:.1f}%)")

    lines.extend(
        [
            "",
            f"가장 많이 사용된 샷 타입은 {most_common_label}이며, 전체 {total}개 컷 중 {most_common_count}개입니다.",
            "이 요약은 컷별 샷 타입 예측과 컷 길이를 기준으로 자동 생성되었습니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def wrap_text(text: str, width: int = 24) -> list[str]:
    lines: list[str] = []
    for part in text.splitlines():
        if not part:
            lines.append("")
            continue
        lines.extend(wrap(part, width=width))
    return lines
