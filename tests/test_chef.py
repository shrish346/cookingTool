"""RecipeChef orchestration with fake adapters: stage order, fusion, fallbacks."""
import pytest

from src.chef import RecipeChef
from src.llm.openai import OpenAIAdapter
from src.schemas import (
    Entity,
    Ingredient,
    MicroAction,
    Observation,
    Recipe,
    SceneDescription,
    SceneLog,
    Step,
)

TRANSCRIPT = "add a teaspoon of cumin"


class FakeVLM:
    def __init__(self):
        self.received_transcript = None

    def analyze_video_direct(self, video_info, video_path, transcript=None,
                             max_retries=3, debug=False):
        self.received_transcript = transcript
        return SceneLog(scenes=[SceneDescription(
            frame_index=0,
            entities=[Entity(name="Cumin", type="ingredient")],
            micro_actions=[
                MicroAction(id=0, action="Adding cumin to pan", timestamp_seconds=5.0,
                            duration_seconds=2.0, entity="Cumin"),
            ],
        )])

    def read_keyframes(self, video_info, keyframes, max_retries=3):
        return [Observation(timestamp_seconds=t, ingredient="smoked paprika",
                            stated_amount="1 tsp", on_screen_text="1 tsp smoked paprika")
                for t, _ in keyframes]


class FailingKeyframeVLM(FakeVLM):
    def read_keyframes(self, video_info, keyframes, max_retries=3):
        raise RuntimeError("reading blew up")


class FakeLLM:
    def __init__(self):
        self.seen_prompt = None

    def generate_recipe(self, scene_log, video_info, transcript=None, use_timestamps=True):
        # Render the real grounding prompt so fusion is checked end-to-end.
        oa = OpenAIAdapter.__new__(OpenAIAdapter)
        self.seen_prompt = oa._build_prompt(scene_log, video_info, transcript, use_timestamps)
        return Recipe(
            title="Cumin Chicken",
            ingredients=[Ingredient(name="cumin", quantity=1, unit="tsp")],
            steps=[Step(id="g1", order=1, instruction="cook",
                        micro_action_ids=[0], start_timestamp_seconds=5.0,
                        end_timestamp_seconds=7.0)],
            servings=2,
            source_url=video_info.url,
        )


class FailingExpander:
    def expand(self, grounded, video_info):
        raise RuntimeError("simulated expansion failure")


@pytest.fixture
def chef_parts(synthetic_video):
    vlm, llm = FakeVLM(), FakeLLM()
    chef = RecipeChef(vlm_adapter=vlm, llm_adapter=llm, expander=FailingExpander())
    return chef, vlm, llm


@pytest.mark.ffmpeg
class TestGenerateRecipeDirect:
    def test_stage_order(self, chef_parts, video_info, synthetic_video):
        chef, _, _ = chef_parts
        stages = []
        chef.generate_recipe_direct(video_info, synthetic_video,
                                    transcript=TRANSCRIPT, debug=False,
                                    on_stage=stages.append)
        assert stages == ["reading", "grounding", "expanding"]

    def test_transcript_reaches_vlm(self, chef_parts, video_info, synthetic_video):
        chef, vlm, _ = chef_parts
        chef.generate_recipe_direct(video_info, synthetic_video,
                                    transcript=TRANSCRIPT, debug=False)
        assert vlm.received_transcript == TRANSCRIPT

    def test_observations_fused_into_grounding_prompt(self, chef_parts, video_info,
                                                      synthetic_video):
        chef, _, llm = chef_parts
        chef.generate_recipe_direct(video_info, synthetic_video, debug=False)
        assert "1 tsp smoked paprika" in llm.seen_prompt

    def test_expansion_failure_falls_back_to_grounded(self, chef_parts, video_info,
                                                      synthetic_video):
        chef, _, _ = chef_parts
        recipe = chef.generate_recipe_direct(video_info, synthetic_video, debug=False)
        assert recipe.expansion_failed is True
        assert recipe.title == "Cumin Chicken"
        # The grounded timing survived the failed expansion untouched.
        assert recipe.steps[0].start_timestamp_seconds == 5.0

    def test_expand_false_returns_grounded(self, chef_parts, video_info, synthetic_video):
        chef, _, _ = chef_parts
        recipe = chef.generate_recipe_direct(video_info, synthetic_video,
                                             debug=False, expand=False)
        assert recipe.expansion_failed is False

    def test_keyframe_failure_is_not_fatal(self, video_info, synthetic_video):
        chef = RecipeChef(vlm_adapter=FailingKeyframeVLM(), llm_adapter=FakeLLM(),
                          expander=FailingExpander())
        recipe = chef.generate_recipe_direct(video_info, synthetic_video, debug=False)
        assert recipe.title == "Cumin Chicken"  # recipe still produced


class TestReadKeyframesInto:
    def test_noop_without_adapter_support(self, video_info, make_scene_log):
        class BareVLM:
            pass

        chef = RecipeChef(vlm_adapter=BareVLM(), llm_adapter=FakeLLM())
        log = make_scene_log()
        chef._read_keyframes_into(log, video_info, "/nonexistent.mp4")
        assert log.get_all_observations() == []

    def test_noop_on_empty_scene_log(self, video_info):
        chef = RecipeChef(vlm_adapter=FakeVLM(), llm_adapter=FakeLLM())
        log = SceneLog(scenes=[])
        chef._read_keyframes_into(log, video_info, "/nonexistent.mp4")
        assert log.get_all_observations() == []
