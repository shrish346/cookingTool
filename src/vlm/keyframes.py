"""Keyframe selection + extraction for the high-res reading pass.

The direct video upload is compressed hard (480px @ 3fps) for temporal grounding,
which makes spice-jar labels, measuring spoons and creators' on-screen captions
("1 tsp cumin") unreadable. This module picks the handful of moments worth a
sharp look and extracts full-quality frames for a cheap multi-image VLM call —
one video upload for timing, a dozen JPEGs for reading.

Selection is VLM-guided: the micro-action timeline already says when each
ingredient is introduced or seasoned, and that is exactly where short-form
videos put their hero close-ups and amount captions — a far better "where's
the label" signal than blind sampling. Scene-cut detection is the fallback
for montages whose timeline yields too few events.
"""

from __future__ import annotations

import base64
import re
import subprocess

import cv2

from ..schemas import SceneLog

# Verb stems that mark an ingredient entering the dish - the moments captions
# and labels cluster around. Matched against the start of the action text.
READING_VERB_STEMS = (
    "add", "pour", "sprinkl", "season", "scoop", "measur", "drizzl",
    "spoon", "dust", "toss", "coat", "squeez", "grat", "crumbl",
)

# Two candidates closer than this are the same moment; keep the first.
MIN_GAP_SECONDS = 2.0


def select_reading_timestamps(scene_log: SceneLog, max_frames: int = 14) -> list[float]:
    """Pick the timestamps most likely to show a readable label, caption or measure.

    Two signals, in priority order:
      1. Micro-actions whose verb introduces an ingredient (add/pour/sprinkle/...).
      2. The first appearance of each ingredient-typed entity on the timeline.
    """
    actions = scene_log.get_all_micro_actions()
    timed = [a for a in actions if a.timestamp_seconds is not None]

    candidates: list[float] = []

    # Signal 1: ingredient-introducing verbs
    for action in timed:
        text = action.action.lower().lstrip()
        if any(text.startswith(stem) for stem in READING_VERB_STEMS):
            candidates.append(action.timestamp_seconds)

    # Signal 2: first appearance of each ingredient entity
    ingredient_names = {
        e.name.strip().lower()
        for scene in scene_log.scenes
        for e in scene.entities
        if e.type == "ingredient" and e.name.strip()
    }
    for name in ingredient_names:
        for action in timed:  # already in time order
            haystack = f"{action.entity or ''} {action.action}".lower()
            if name in haystack:
                candidates.append(action.timestamp_seconds)
                break

    return _dedupe_and_cap(candidates, max_frames)


def detect_scene_cut_timestamps(
    video_path: str,
    max_frames: int = 14,
    threshold: float = 0.3,
) -> list[float]:
    """Fallback: ffmpeg scene-cut detection, for videos whose timeline gave us
    too few reading events (fast montages). Returns times just AFTER each cut,
    where the post-cut hero shot lives."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vf", f"select='gt(scene,{threshold})',showinfo",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    # showinfo logs one line per selected frame with its pts_time.
    times = [float(m) for m in re.findall(r"pts_time:([0-9]+\.?[0-9]*)", result.stderr)]
    return _dedupe_and_cap([t + 0.3 for t in times], max_frames)


def extract_reading_keyframes(
    video_path: str,
    timestamps: list[float],
    target_width: int = 1024,
    window_seconds: float = 0.75,
    candidates_per_timestamp: int = 4,
    jpeg_quality: int = 88,
) -> list[tuple[float, str]]:
    """Extract one sharp high-res frame per timestamp as (timestamp, base64 JPEG).

    Samples a few frames in a ±window around each timestamp and keeps the one
    with the highest Laplacian variance, so we land on the hero shot rather
    than motion blur mid-pour.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for keyframe extraction: {video_path}")

    keyframes: list[tuple[float, str]] = []
    try:
        for ts in timestamps:
            best_frame = None
            best_sharpness = -1.0

            step = (2 * window_seconds) / max(candidates_per_timestamp - 1, 1)
            for i in range(candidates_per_timestamp):
                t = max(0.0, ts - window_seconds + i * step)
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_frame = frame

            if best_frame is None:
                continue

            height, width = best_frame.shape[:2]
            if width > target_width:
                scale = target_width / width
                best_frame = cv2.resize(
                    best_frame, (target_width, int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            ok, buf = cv2.imencode(".jpg", best_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if ok:
                keyframes.append((ts, base64.b64encode(buf.tobytes()).decode("utf-8")))
    finally:
        cap.release()

    return keyframes


def _dedupe_and_cap(timestamps: list[float], max_frames: int) -> list[float]:
    """Sort, drop timestamps within MIN_GAP_SECONDS of a kept one, cap the count."""
    kept: list[float] = []
    for ts in sorted(timestamps):
        if not kept or ts - kept[-1] >= MIN_GAP_SECONDS:
            kept.append(ts)
    if len(kept) <= max_frames:
        return kept
    # Over budget: thin evenly rather than truncating - the tail of the video
    # (final seasoning, garnish) reads just as often as the start.
    stride = len(kept) / max_frames
    return [kept[int(i * stride)] for i in range(max_frames)]
