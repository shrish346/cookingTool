"""OpenRouterAdapter response parsers and prompt builders.

Adapters are built with __new__ so no API client (or key) is ever constructed.
"""
import json

import pytest

from src.vlm.openrouter import OpenRouterAdapter, _extract_scene_json, _iter_json_objects


@pytest.fixture
def vlm() -> OpenRouterAdapter:
    return OpenRouterAdapter.__new__(OpenRouterAdapter)


class TestParseVideoDirectResponse:
    def _content(self, actions):
        return "SCRATCHPAD\n[00:05 - 00:07] stuff\n" + json.dumps({
            "summary": "cooking",
            "entities": [{"name": "Cumin", "type": "ingredient"}],
            "micro_actions": actions,
        })

    def test_mmss_times_and_sequential_ids(self, vlm):
        scene = vlm._parse_video_direct_response(self._content([
            {"action": "Adding cumin", "start": "00:05", "end": "00:07", "entity": "Cumin"},
            {"action": "Stirring", "start": "01:00", "end": "01:02", "entity": "Pan"},
        ]), video_duration=120)
        a, b = scene.micro_actions
        assert (a.timestamp_seconds, a.duration_seconds) == (5.0, 2.0)
        assert (b.timestamp_seconds, b.duration_seconds) == (60.0, 2.0)
        assert [m.id for m in scene.micro_actions] == [0, 1]

    def test_hhmmss_and_numeric_times(self, vlm):
        scene = vlm._parse_video_direct_response(self._content([
            {"action": "a", "start": "00:01:30", "end": "00:01:31"},
            {"action": "b", "start": 42, "end": 44},
        ]), video_duration=200)
        assert scene.micro_actions[0].timestamp_seconds == 90.0
        assert scene.micro_actions[1].timestamp_seconds == 42.0

    def test_stated_amount_carried_and_lists_coerced(self, vlm):
        scene = vlm._parse_video_direct_response(self._content([
            {"action": "Adding spices", "start": "00:05", "end": "00:06",
             "entity": ["cumin", "coriander"], "stated_amount": "1 tsp"},
        ]), video_duration=60)
        ma = scene.micro_actions[0]
        assert ma.entity == "cumin, coriander"
        assert ma.stated_amount == "1 tsp"

    def test_summary_lands_in_metadata(self, vlm):
        scene = vlm._parse_video_direct_response(self._content([]), video_duration=60)
        assert scene.metadata["summary"] == "cooking"
        assert scene.metadata["method"] == "direct_video"

    def test_string_entities_become_ingredients(self, vlm):
        content = json.dumps({"entities": ["Cumin"], "micro_actions": []})
        scene = vlm._parse_video_direct_response(content, video_duration=60)
        assert scene.entities[0].name == "Cumin"
        assert scene.entities[0].type == "ingredient"


class TestParseKeyframeResponse:
    def test_valid_observations_parsed(self, vlm):
        content = 'preamble text {"observations": [{"timestamp": 12.5, "ingredient": "smoked paprika", "stated_amount": "1 tsp", "on_screen_text": "1 tsp smoked paprika", "confidence": 0.95}]}'
        obs = vlm._parse_keyframe_response(content)
        assert len(obs) == 1
        assert obs[0].ingredient == "smoked paprika"
        assert obs[0].source == "keyframe"

    def test_signal_less_and_invalid_entries_dropped(self, vlm):
        content = json.dumps({"observations": [
            {"timestamp": 1.0},                       # nothing read - no signal
            {"timestamp": "oops", "ingredient": "x"}, # fails validation
            {"timestamp": 2.0, "on_screen_text": "500g flour"},  # keeps: has text
        ]})
        obs = vlm._parse_keyframe_response(content)
        assert len(obs) == 1 and obs[0].on_screen_text == "500g flour"

    def test_garbage_returns_empty(self, vlm):
        assert vlm._parse_keyframe_response("no json here at all") == []


class TestJsonExtraction:
    def test_scene_json_found_after_scratchpad(self):
        content = 'Notes {not json} more text {"micro_actions": [], "entities": []} trailing'
        data = _extract_scene_json(content)
        assert "micro_actions" in data

    def test_iter_handles_braces_inside_strings(self):
        content = '{"a": "value with } brace"} {"b": 2}'
        objects = list(_iter_json_objects(content))
        assert json.loads(objects[0])["a"] == "value with } brace"


class TestVideoDirectPrompt:
    def test_transcript_guardrail_only_when_given(self, vlm, video_info):
        with_t = vlm._build_video_direct_prompt(video_info, transcript="add the paprika")
        without = vlm._build_video_direct_prompt(video_info)
        assert "AUDIO TRANSCRIPT" in with_t
        assert "NEVER log an action you did not see" in with_t
        assert "AUDIO TRANSCRIPT" not in without

    def test_description_included_and_capped(self, vlm, video_info):
        video_info.description = "x" * 5000
        prompt = vlm._build_video_direct_prompt(video_info)
        assert "CREATOR'S DESCRIPTION" in prompt
        assert "x" * 1500 in prompt
        assert "x" * 1501 not in prompt

    def test_reading_rules_present(self, vlm, video_info):
        prompt = vlm._build_video_direct_prompt(video_info)
        assert "stated_amount" in prompt
        assert "Read On-Screen Text" in prompt
        assert "Ingredient Specificity" in prompt
