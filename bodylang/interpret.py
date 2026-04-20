from __future__ import annotations

from schema import (
    EngagementMetrics,
    GestureMetrics,
    PostureMetrics,
    RestlessnessMetrics,
)


def overall_labels(
    engagement: EngagementMetrics,
    posture: PostureMetrics,
    gestures: GestureMetrics,
    restlessness: RestlessnessMetrics,
) -> list[str]:
    labels: list[str] = []

    # Engagement
    labels.append(engagement.label)

    # Posture
    labels.append(posture.label)
    if posture.lean_label == "forward":
        labels.append("leaning forward")
    elif posture.lean_label == "backward":
        labels.append("leaning backward")

    # Gestures
    labels.append(gestures.label)

    # Restlessness — but suppress if overlapping with a stronger posture/gesture label.
    if restlessness.label not in {"stable composure", "natural expressive movement"}:
        labels.append(restlessness.label)

    # De-dup while preserving order.
    seen = set()
    unique: list[str] = []
    for l in labels:
        if l not in seen:
            seen.add(l)
            unique.append(l)
    return unique
