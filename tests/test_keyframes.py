"""Keyframe selection and high-res extraction for the reading pass."""
import base64

import pytest

from src.schemas import Entity, MicroAction, SceneDescription, SceneLog
from src.vlm.keyframes import (
    _dedupe_and_cap,
    detect_scene_cut_timestamps,
    extract_reading_keyframes,
    select_reading_timestamps,
)


class TestSelectReadingTimestamps:
    def test_ingredient_verbs_and_first_appearance(self, scene_log):
        ts = select_reading_timestamps(scene_log)
        assert 10.0 in ts   # "Adding cumin" - reading verb
        assert 30.0 in ts   # "Sprinkling smoked paprika" - reading verb
        assert 2.0 in ts    # chicken's first appearance (ingredient entity)
        assert 20.0 not in ts  # "Stirring pan" is not a reading moment

    def test_dedupe_window(self, make_scene_log):
        log = make_scene_log(micro_actions=[
            MicroAction(id=0, action="Adding cumin", timestamp_seconds=10.0, entity="Cumin"),
            MicroAction(id=1, action="Pouring oil", timestamp_seconds=10.5, entity="Oil"),
            MicroAction(id=2, action="Adding salt", timestamp_seconds=13.0, entity="Salt"),
        ], entities=[])
        ts = select_reading_timestamps(log)
        assert ts == [10.0, 13.0]  # 10.5 is within the 2s window of 10.0

    def test_untimed_actions_ignored(self, make_scene_log):
        log = make_scene_log(
            micro_actions=[MicroAction(id=0, action="Adding cumin", entity="Cumin")],
            entities=[],
        )
        assert select_reading_timestamps(log) == []

    def test_cap(self, make_scene_log):
        log = make_scene_log(micro_actions=[
            MicroAction(id=i, action=f"Adding ingredient {i}", timestamp_seconds=float(i * 3))
            for i in range(40)
        ], entities=[])
        assert len(select_reading_timestamps(log, max_frames=14)) == 14


class TestDedupeAndCap:
    def test_sorts_and_dedupes(self):
        assert _dedupe_and_cap([9.0, 1.0, 2.5, 30.0], 10) == [1.0, 9.0, 30.0]

    def test_thins_evenly_instead_of_truncating(self):
        capped = _dedupe_and_cap([float(i * 3) for i in range(40)], 14)
        assert len(capped) == 14
        assert capped[0] == 0.0
        assert capped[-1] > 100.0  # kept coverage of the tail, didn't just cut it off


@pytest.mark.ffmpeg
class TestExtraction:
    def test_extracts_sharp_jpegs(self, synthetic_video):
        keyframes = extract_reading_keyframes(synthetic_video, [5.0, 15.0, 25.0])
        assert len(keyframes) == 3
        for ts, b64 in keyframes:
            raw = base64.b64decode(b64)
            assert raw[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_missing_file_raises(self):
        with pytest.raises(RuntimeError):
            extract_reading_keyframes("/nonexistent/video.mp4", [1.0])

    def test_scene_cut_detection_quiet_on_cutless_video(self, synthetic_video):
        # testsrc2 has no hard cuts - the fallback must not invent timestamps.
        assert detect_scene_cut_timestamps(synthetic_video) == []
