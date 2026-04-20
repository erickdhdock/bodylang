from __future__ import annotations

from schema import (
    BodyLangReport,
    EngagementMetrics,
    GazeMetrics,
    GestureMetrics,
    PostureMetrics,
    RestlessnessMetrics,
    Timeline,
)


def build_evidence(
    engagement: EngagementMetrics,
    gaze: GazeMetrics,
    posture: PostureMetrics,
    gestures: GestureMetrics,
    restlessness: RestlessnessMetrics,
) -> list[str]:
    e: list[str] = []
    away_pct = round(engagement.look_away_pct * 100)
    e.append(f"The individual looked away from the camera for {away_pct}% of the clip.")
    if engagement.prolonged_look_away_count > 0:
        e.append(
            f"There were {engagement.prolonged_look_away_count} prolonged look-away periods exceeding "
            f"{round(engagement.longest_look_away_sec, 1)}s (longest)."
        )
    elif engagement.look_away_count > 0:
        e.append(
            f"Brief look-away periods occurred {engagement.look_away_count} times, all under 2 seconds."
        )

    if gaze.gaze_drift_count > 0:
        drift_pct = round(gaze.gaze_drift_pct * 100)
        e.append(
            f"Eye-gaze drifted off-axis for {gaze.gaze_drift_count} sustained stretch(es) "
            f"totalling ~{drift_pct}% of the clip (advisory — verify manually; "
            f"longest {gaze.longest_gaze_drift_sec:.1f}s)."
        )

    if posture.label in {"stable posture", "stable posture with mild sway"}:
        e.append(
            f"Posture was stable overall (stability {posture.stability_score:.2f}), "
            f"with {posture.torso_sway_cm:.1f}cm of torso sway."
        )
    elif posture.label == "overly rigid":
        e.append(
            f"Posture appeared rigid with minimal movement ({posture.torso_sway_cm:.1f}cm sway)."
        )
    else:
        e.append(
            f"Posture showed noticeable instability: {posture.torso_sway_cm:.1f}cm sway and "
            f"{posture.head_drift_cm:.1f}cm of head drift."
        )
    if posture.lean_label != "neutral":
        e.append(f"Overall torso was leaned {posture.lean_label}.")

    if gestures.label == "limited body-language usage":
        e.append(
            f"Body-language usage was limited, with hands active in "
            f"{round(gestures.hand_activity_pct * 100)}% of frames."
        )
    elif gestures.label == "excessive gesture usage":
        e.append(
            f"Hand activity was high ({round(gestures.hand_activity_pct * 100)}% of frames) "
            f"with {gestures.gesture_burst_count} distinct gesture bursts."
        )
    else:
        e.append(
            f"Moderate gesture support observed: {gestures.gesture_burst_count} bursts, "
            f"mean amplitude {gestures.mean_gesture_amplitude_cm:.1f}cm."
        )
    if gestures.nod_count > 0:
        e.append(f"Approximately {gestures.nod_count} head nods were detected.")
    if gestures.inappropriate_gesture_count > 0:
        n = gestures.inappropriate_gesture_count
        e.append(
            f"Inappropriate gesture flagged: {n} instance{'s' if n > 1 else ''} of "
            "middle-finger-up detected."
        )

    if restlessness.label == "restless movement":
        e.append("Overall motion was high, suggesting restless or unsettled posture.")
    elif restlessness.label == "repetitive motion":
        e.append(
            f"Repetitive motion patterns were detected in "
            f"{restlessness.repetitive_motion_events} windows."
        )
    elif restlessness.label == "overly still":
        e.append(
            f"Movement was very low ({round(restlessness.overly_still_pct * 100)}% of frames "
            "showed near-zero motion)."
        )
    return e


def build_coaching(
    engagement: EngagementMetrics,
    gaze: GazeMetrics,
    posture: PostureMetrics,
    gestures: GestureMetrics,
    restlessness: RestlessnessMetrics,
) -> list[str]:
    tips: list[str] = []
    if engagement.label == "frequent gaze avoidance":
        tips.append(
            "Practise returning your gaze to the camera — aim to look at the lens, "
            "not the screen preview, for most of each answer."
        )
    elif engagement.prolonged_look_away_count > 0:
        tips.append(
            "Most look-aways were short, but a few exceeded 2 seconds. Try to keep "
            "off-camera glances under a second when thinking."
        )

    if posture.label == "unstable seated movement":
        tips.append(
            "Anchor your seat and keep shoulders over hips — subtle adjustments look "
            "steadier on camera than repositioning."
        )
    elif posture.label == "overly rigid":
        tips.append(
            "Allow small natural shifts in posture; complete stillness can read as tense."
        )
    if posture.lean_label == "forward":
        tips.append("Sit back slightly — a forward lean can appear over-eager on camera.")

    if gestures.label == "limited body-language usage":
        tips.append(
            "Let your hands enter frame occasionally. A few deliberate gestures reinforce "
            "key points without becoming distracting."
        )
    elif gestures.label == "excessive gesture usage":
        tips.append(
            "Reduce constant hand motion; reserve gestures for emphasis so they register "
            "more strongly."
        )

    if restlessness.label in {"restless movement", "repetitive motion"}:
        tips.append(
            "Watch for repeated movements (tapping, shifting). Keeping feet flat on the "
            "floor often settles this."
        )

    if gestures.inappropriate_gesture_count > 0:
        tips.insert(
            0,
            "Inappropriate gesture detected in the clip — avoid extending the middle "
            "finger on camera, even unintentionally during hand movements.",
        )

    if gaze.label == "frequent gaze drift":
        tips.append(
            "Eyes drifted off-axis frequently. If you rely on a second monitor or "
            "notes, move them as close to the camera line of sight as possible so "
            "your gaze stays on the lens."
        )

    # Keep to 2-4 recommendations.
    return tips[:4] if len(tips) >= 2 else tips + [
        "Continue what you're doing — presentation reads well on the tracked metrics."
    ][: max(0, 2 - len(tips))]


def assemble_report(
    job_id: str,
    duration_sec: float,
    source_fps: float,
    analysis_fps: float,
    frames_analyzed: int,
    frames_skipped: int,
    accuracy_confidence_ai: str,
    accuracy_confidence_reason: str,
    engagement: EngagementMetrics,
    gaze: GazeMetrics,
    posture: PostureMetrics,
    gestures: GestureMetrics,
    restlessness: RestlessnessMetrics,
    timeline: Timeline,
    overall_labels: list[str],
    model_versions: dict[str, str],
    config_snapshot: dict,
) -> BodyLangReport:
    return BodyLangReport(
        job_id=job_id,
        duration_sec=round(duration_sec, 3),
        source_fps=round(source_fps, 3),
        analysis_fps=round(analysis_fps, 3),
        frames_analyzed=frames_analyzed,
        frames_skipped=frames_skipped,
        accuracy_confidence_ai=accuracy_confidence_ai,
        accuracy_confidence_reason=accuracy_confidence_reason,
        engagement=engagement,
        gaze=gaze,
        posture=posture,
        gestures=gestures,
        restlessness=restlessness,
        timeline=timeline,
        overall_labels=overall_labels,
        evidence=build_evidence(engagement, gaze, posture, gestures, restlessness),
        coaching=build_coaching(engagement, gaze, posture, gestures, restlessness),
        model_versions=model_versions,
        config_snapshot=config_snapshot,
    )
