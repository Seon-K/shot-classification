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


def guide_for_label(
    label: str | None = None,
    confidence: float = 0.0,
    threshold: float = 0.45,
    duration_sec: float | None = None,
    has_text: bool | None = None,
    text_confidence: float | None = None,
    shot_type: str | None = None,
) -> str:
    shot_label = shot_type or label or "uncertain"
    if confidence < threshold:
        return "판정 신뢰도가 낮아 수동 확인이 필요한 구간입니다."

    duration = duration_sec if duration_sec is not None else 0.0
    text_present = bool(has_text)

    if shot_label == "close-up":
        if text_present:
            return "핵심 자막과 디테일을 함께 강조하는 클로즈업 구간입니다."
        return "표정이나 디테일을 강조하는 클로즈업 구간입니다."
    if shot_label == "object":
        if duration and duration < 1.5:
            return "짧은 제품 임팩트 컷입니다. 대상이 즉시 인식되도록 중앙 배치를 유지합니다."
        if text_present:
            return "제품이나 화면 정보와 자막을 함께 전달하는 사물 중심 구간입니다."
        return GUIDE_TEMPLATES["object"]
    if shot_label == "space":
        if duration > 2.0:
            return "장소 분위기를 설명하는 구간입니다. 다음 컷으로 넘어가기 전 공간 맥락을 충분히 제공합니다."
        return GUIDE_TEMPLATES["space"]
    if shot_label == "medium":
        if text_present:
            return "인물의 동작과 자막 정보를 함께 보여주는 미디엄 샷 구간입니다."
        return GUIDE_TEMPLATES["medium"]
    if shot_label == "wide":
        return GUIDE_TEMPLATES["wide"]

    return GUIDE_TEMPLATES.get(shot_label, GUIDE_TEMPLATES["uncertain"])


def build_temporal_summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "opening_shot_type": "unknown",
            "ending_shot_type": "unknown",
            "avg_duration": 0.0,
            "dominant_shot_type": "unknown",
            "fast_cut_ratio": 0.0,
            "text_cut_ratio": 0.0,
            "cut_speed_category": "normal",
            "repeated_shot_ratio": 0.0,
            "transition_count_by_pair": {},
            "text_density_level": "low",
        }

    durations = [float(row.get("duration_sec", 0.0)) for row in rows]
    shot_types = [str(row.get("shot_type", "unknown")) for row in rows]
    counts = Counter(shot_types)
    total = len(rows)
    avg_duration = sum(durations) / total if total else 0.0
    text_count = sum(
        1
        for row in rows
        if str(row.get("text_label", "")).lower() == "text" or row.get("has_text") is True
    )
    transition_counts: Counter[str] = Counter()
    repeated = 0
    for prev_label, next_label in zip(shot_types, shot_types[1:]):
        pair = f"{prev_label}->{next_label}"
        transition_counts[pair] += 1
        if prev_label == next_label:
            repeated += 1

    fast_cut_ratio = sum(1 for value in durations if value < 1.5) / total if total else 0.0
    text_cut_ratio = text_count / total if total else 0.0
    if avg_duration < 1.5:
        cut_speed_category = "fast"
    elif avg_duration <= 2.5:
        cut_speed_category = "normal"
    else:
        cut_speed_category = "slow"

    if text_cut_ratio >= 0.8:
        text_density_level = "high"
    elif text_cut_ratio >= 0.4:
        text_density_level = "medium"
    else:
        text_density_level = "low"

    return {
        "opening_shot_type": shot_types[0],
        "ending_shot_type": shot_types[-1],
        "avg_duration": avg_duration,
        "dominant_shot_type": counts.most_common(1)[0][0] if counts else "unknown",
        "fast_cut_ratio": fast_cut_ratio,
        "text_cut_ratio": text_cut_ratio,
        "cut_speed_category": cut_speed_category,
        "repeated_shot_ratio": repeated / max(total - 1, 1),
        "transition_count_by_pair": dict(transition_counts.most_common()),
        "text_density_level": text_density_level,
    }


def describe_temporal_summary(summary: dict) -> str:
    avg_duration = float(summary["avg_duration"])
    speed_text = {
        "fast": "빠른 편",
        "normal": "일반적인 편",
        "slow": "느린 편",
    }.get(summary["cut_speed_category"], "일반적인 편")
    transitions = summary.get("transition_count_by_pair", {})
    top_transition = next(iter(transitions), "전환 패턴 없음")
    density_text = {
        "high": "자막 컷 비율이 80% 이상으로 정보 전달형 숏폼에 가깝습니다.",
        "medium": "자막과 비자막 컷이 혼합된 설명형 구성입니다.",
        "low": "자막 의존도보다 화면 전환과 비주얼 중심의 구성입니다.",
    }.get(summary["text_density_level"], "자막 밀도는 보통 수준입니다.")
    return (
        f"이 영상은 평균 컷 길이가 {avg_duration:.2f}초로 {speed_text}이며, "
        f"{top_transition} 전환이 가장 자주 등장합니다. "
        f"반복 샷 비율은 {summary['repeated_shot_ratio'] * 100:.1f}%입니다. "
        f"{density_text}"
    )


def summarize_guides(rows: list[dict]) -> str:
    if not rows:
        return "분석된 컷이 없습니다.\n"

    durations = [float(row["duration_sec"]) for row in rows]
    labels = [row["shot_type"] for row in rows]
    counts = Counter(labels)
    total = len(rows)
    avg_duration = sum(durations) / total
    most_common_label, most_common_count = counts.most_common(1)[0]
    temporal_summary = build_temporal_summary(rows)

    lines = [
        f"전체 컷 수: {total}",
        f"평균 컷 길이: {avg_duration:.2f}초",
        "샷 타입 분포:",
    ]
    for label, count in counts.most_common():
        lines.append(f"- {label}: {count}개 ({count / total * 100:.1f}%)")

    transition_counts = temporal_summary["transition_count_by_pair"]
    top_transitions = list(transition_counts.items())[:5]
    lines.extend(
        [
            "",
            f"가장 많이 사용된 샷 타입은 {most_common_label}이며, 전체 {total}개 컷 중 {most_common_count}개입니다.",
            "",
            "Temporal Pattern Summary:",
            f"- opening_shot_type: {temporal_summary['opening_shot_type']}",
            f"- ending_shot_type: {temporal_summary['ending_shot_type']}",
            f"- avg_duration: {temporal_summary['avg_duration']:.2f}",
            f"- dominant_shot_type: {temporal_summary['dominant_shot_type']}",
            f"- fast_cut_ratio: {temporal_summary['fast_cut_ratio']:.4f}",
            f"- text_cut_ratio: {temporal_summary['text_cut_ratio']:.4f}",
            f"- cut_speed_category: {temporal_summary['cut_speed_category']}",
            f"- repeated_shot_ratio: {temporal_summary['repeated_shot_ratio']:.4f}",
            f"- text_density_level: {temporal_summary['text_density_level']}",
            "- transition_count_by_pair:",
        ]
    )
    for pair, count in top_transitions:
        lines.append(f"  - {pair}: {count}")

    lines.extend(
        [
            "",
            describe_temporal_summary(temporal_summary),
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
