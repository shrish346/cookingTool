"""Grounding pass (OpenAIAdapter): prompt rendering and honest response parsing."""
import json

import pytest

from src.llm.openai import OpenAIAdapter
from src.schemas import Observation, Recipe


@pytest.fixture
def llm() -> OpenAIAdapter:
    return OpenAIAdapter.__new__(OpenAIAdapter)


class TestParseResponse:
    """The old parser silently defaulted vague quantities to 1.0 and guessed units
    by keyword - fidelity-destroying. It must not come back."""

    def _parse(self, llm, video_info, ingredients):
        content = json.dumps({
            "title": "Curry", "servings": 2,
            "ingredients": ingredients,
            "steps": [], "tools": [],
        })
        return llm._parse_response(content, video_info)

    def test_missing_quantity_stays_none(self, llm, video_info):
        recipe = self._parse(llm, video_info, [{"name": "mystery spice"}])
        ing = recipe.ingredients[0]
        assert ing.quantity is None
        assert ing.unit != "tsp"  # no keyword unit-guessing

    def test_string_quantity_extracted(self, llm, video_info):
        recipe = self._parse(llm, video_info, [{"name": "rice", "quantity": "2 cups"}])
        assert recipe.ingredients[0].quantity == 2.0

    def test_vague_string_becomes_amount_text(self, llm, video_info):
        recipe = self._parse(llm, video_info, [{"name": "salt", "quantity": "a pinch"}])
        ing = recipe.ingredients[0]
        assert ing.quantity is None
        assert ing.amount_text == "a pinch"

    def test_nutrition_estimates_flattened(self, llm, video_info):
        content = json.dumps({
            "title": "t", "servings": 2, "ingredients": [], "steps": [],
            "nutrition_estimates": {"calories": 450, "protein": 25},
        })
        recipe = llm._parse_response(content, video_info)
        assert recipe.calories == 450 and recipe.protein == 25

    def test_title_and_servings_fallbacks(self, llm, video_info):
        content = json.dumps({"servings": 0, "ingredients": [], "steps": []})
        recipe = llm._parse_response(content, video_info)
        assert recipe.title == video_info.title
        assert recipe.servings == 1

    def test_json_in_code_fence(self, llm, video_info):
        content = '```json\n{"title": "t", "servings": 1, "ingredients": [], "steps": []}\n```'
        assert llm._parse_response(content, video_info).title == "t"


class TestPromptRendering:
    def test_long_names_survive_untruncated(self, llm, make_scene_log):
        from src.schemas import MicroAction
        log = make_scene_log(micro_actions=[
            MicroAction(id=0, action="Sprinkling kashmiri chili powder over the chicken pieces",
                        timestamp_seconds=5.0, entity="Kashmiri chili powder"),
        ], entities=[])
        text = llm._format_micro_actions(log, use_timestamps=True)
        assert "Kashmiri chili powder" in text  # the 12-char cut would leave "Kashmiri chi"
        assert "Sprinkling kashmiri chili powder over the chicken pieces" in text

    def test_stated_amount_column(self, llm, scene_log):
        text = llm._format_micro_actions(scene_log, use_timestamps=True)
        assert "1 tsp" in text

    def test_observations_section(self, llm, make_scene_log, video_info):
        log = make_scene_log(observations=[
            Observation(timestamp_seconds=3.0, ingredient="cumin",
                        stated_amount="1 tsp", on_screen_text="1 tsp cumin"),
        ])
        prompt = llm._build_prompt(log, video_info, transcript=None, use_timestamps=True)
        assert "AMOUNT / LABEL READINGS" in prompt
        assert '"1 tsp cumin"' in prompt

    def test_no_observations_no_section(self, llm, scene_log, video_info):
        prompt = llm._build_prompt(scene_log, video_info, transcript=None, use_timestamps=True)
        assert "AMOUNT / LABEL READINGS" not in prompt

    def test_amount_text_contract_in_prompt(self, llm, scene_log, video_info):
        prompt = llm._build_prompt(scene_log, video_info, transcript=None, use_timestamps=True)
        assert "amount_text" in prompt
        assert "NEVER invent a number" in prompt


class TestSeedTools:
    def test_vlm_tools_backfilled_once(self, llm, scene_log):
        recipe = Recipe(title="t", ingredients=[], steps=[], servings=1)
        recipe = llm._seed_tools_from_scene_log(recipe, scene_log)
        names = [t.name for t in recipe.tools]
        assert "Knife" in names and "Pan" in names
        assert len(names) == len(set(n.lower() for n in names))
        assert all(t.provenance == "video" for t in recipe.tools)

    def test_existing_tools_not_duplicated(self, llm, scene_log):
        from src.schemas import Tool
        recipe = Recipe(title="t", ingredients=[], steps=[], servings=1,
                        tools=[Tool(name="knife")])  # differs only in case
        recipe = llm._seed_tools_from_scene_log(recipe, scene_log)
        assert [t.name.lower() for t in recipe.tools].count("knife") == 1
