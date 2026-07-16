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


class TestScratchpadInsideJson:
    """The scratchpad is a key of the scene object, not a preamble before it.

    Asking for a free-text STEP 1 before the JSON let the model end its turn once STEP 1
    was written, returning a scratchpad and no JSON at all - which no parser can rescue.
    Folding the scratchpad into the object removes the step it could stop after.
    """
    SCENE = {
        "scratchpad": ["[00:05 - 00:07] : Cumin -> Adding to pan"],
        "summary": "cooking",
        "entities": [{"name": "Cumin", "type": "ingredient"}],
        "micro_actions": [
            {"action": "Adding cumin", "start": "00:05", "end": "00:07", "entity": "Cumin"}
        ],
    }

    def test_direct_video_parser_ignores_scratchpad_key(self, vlm):
        scene = vlm._parse_video_direct_response(json.dumps(self.SCENE), video_duration=60)
        assert len(scene.micro_actions) == 1
        assert scene.micro_actions[0].timestamp_seconds == 5.0

    def test_scene_parser_drops_scratchpad_before_model_build(self, vlm):
        # This one builds SceneDescription(**data), so a stray key reaches the model.
        scene = vlm._parse_scene_response(json.dumps(self.SCENE), frame_index=3)
        assert scene.frame_index == 3
        assert len(scene.micro_actions) == 1

    def test_chunk_parser_accepts_scratchpad_key(self, vlm):
        data = vlm._parse_timestamp_chunk_response(json.dumps(self.SCENE), [(5.0, "b64")])
        assert len(data["micro_actions"]) == 1

    def test_fenced_scratchpad_before_json_still_parses(self, vlm):
        # Belt and braces: a model that ignores the new instruction and fences a scratchpad
        # first must not take the whole scene down. The old parsers grabbed the first fence.
        content = (
            "```\n[00:05 - 00:07] : Cumin -> Adding to pan\n```\n"
            "```json\n" + json.dumps({k: v for k, v in self.SCENE.items()
                                      if k != "scratchpad"}) + "\n```"
        )
        assert len(vlm._parse_scene_response(content, frame_index=0).micro_actions) == 1
        assert len(vlm._parse_timestamp_chunk_response(content, [(5.0, "b")])["micro_actions"]) == 1

    @pytest.mark.parametrize("builder", [
        "_build_video_direct_prompt", "_build_scene_prompt", "_build_timestamp_chunk_prompt",
    ])
    def test_no_prompt_asks_for_a_scratchpad_before_the_json(self, vlm, video_info, builder):
        prompt = _build_any(vlm, builder, video_info)
        assert "Return ONLY the Scratchpad followed by" not in prompt
        assert '"scratchpad"' in prompt
        assert "STEP 2: JSON GENERATION" not in prompt


def _build_any(vlm, builder: str, video_info) -> str:
    """Call a prompt builder regardless of its differing required args."""
    fn = getattr(vlm, builder)
    if builder == "_build_timestamp_chunk_prompt":
        return fn(video_info, [(0.0, "b64"), (10.0, "b64")])
    return fn(video_info)


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
