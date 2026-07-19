"""Expansion pass (RecipeExpander): parsing contracts and prompt content."""
import json

import pytest

from src.llm.expander import (
    VIDEO_DESCRIPTION_SOURCE_ID,
    RecipeExpander,
    _coerce_kind,
    _coerce_provenance,
)
from src.schemas import Ingredient, Recipe, Step


@pytest.fixture
def expander() -> RecipeExpander:
    ex = RecipeExpander.__new__(RecipeExpander)
    ex._web_grounding = True
    return ex


@pytest.fixture
def grounded() -> Recipe:
    return Recipe(
        title="Chicken Curry",
        dish_query="chicken curry",
        ingredients=[
            Ingredient(name="spices", amount_text="some"),
            Ingredient(name="rice", quantity=2, unit="cups"),
        ],
        steps=[Step(id="g1", order=1, instruction="cook the chicken",
                    micro_action_ids=[0], start_timestamp_seconds=5.0,
                    end_timestamp_seconds=9.0)],
        servings=2,
        source_url="https://youtube.com/shorts/x",
        video_id="x",
        calories=500,
    )


def _response(steps=None, ingredients=None, sources=None, tools=None, **extra):
    return json.dumps({
        "title": "Chicken Curry",
        "servings": 2,
        "ingredients": ingredients or [],
        "tools": tools or [],
        "sources": sources or [],
        "steps": steps if steps is not None else [
            {"grounded_step_id": "g1", "instruction": "cook", "detail": "d"},
        ],
        **extra,
    })


class TestParseResponse:
    def test_grounded_step_id_becomes_step_id(self, expander, grounded, video_info):
        recipe = expander._parse_response(_response(), grounded, video_info)
        assert recipe.steps[0].id == "g1"

    def test_invented_step_gets_fresh_id(self, expander, grounded, video_info):
        recipe = expander._parse_response(_response(steps=[
            {"grounded_step_id": None, "instruction": "rest", "detail": "d"},
        ]), grounded, video_info)
        assert recipe.steps[0].id != "g1"

    def test_model_owned_fields_stripped(self, expander, grounded, video_info):
        recipe = expander._parse_response(_response(steps=[
            {"grounded_step_id": "g1", "instruction": "cook", "detail": "d",
             "order": 99, "has_video_clip": True,
             "start_timestamp_seconds": 1.0, "end_timestamp_seconds": 2.0,
             "micro_action_ids": [7, 8]},
        ]), grounded, video_info)
        step = recipe.steps[0]
        # Timing is not the model's to set - merge_expansion re-attaches it later.
        assert step.start_timestamp_seconds is None
        assert step.micro_action_ids is None
        assert step.order == 1  # renumbered

    def test_vd_source_minted_when_cited(self, expander, grounded, video_info):
        recipe = expander._parse_response(_response(ingredients=[
            {"id": "i1", "name": "cumin", "quantity": 1, "unit": "tsp",
             "provenance": "reference", "source_id": VIDEO_DESCRIPTION_SOURCE_ID},
        ]), grounded, video_info)
        vd = [s for s in recipe.sources if s.id == VIDEO_DESCRIPTION_SOURCE_ID]
        assert len(vd) == 1
        assert vd[0].url == video_info.url
        assert vd[0].site  # renders on the provenance badge

    def test_model_declared_vd_source_replaced_by_ours(self, expander, grounded,
                                                       video_info):
        """The description source is ours to mint - a model-declared one (with
        whatever title/url it invented) must not survive."""
        recipe = expander._parse_response(_response(
            ingredients=[
                {"id": "i1", "name": "cumin", "quantity": 1, "unit": "tsp",
                 "provenance": "reference", "source_id": VIDEO_DESCRIPTION_SOURCE_ID},
            ],
            sources=[{"id": VIDEO_DESCRIPTION_SOURCE_ID, "title": "model invented",
                      "url": "https://wrong.example"}],
        ), grounded, video_info)
        vd = [s for s in recipe.sources if s.id == VIDEO_DESCRIPTION_SOURCE_ID]
        assert len(vd) == 1
        assert vd[0].url == video_info.url
        assert vd[0].title != "model invented"

    def test_vd_source_not_minted_unless_cited(self, expander, grounded, video_info):
        recipe = expander._parse_response(_response(), grounded, video_info)
        assert not any(s.id == VIDEO_DESCRIPTION_SOURCE_ID for s in recipe.sources)

    def test_nutrition_carried_from_pass_one(self, expander, grounded, video_info):
        recipe = expander._parse_response(_response(), grounded, video_info)
        assert recipe.calories == 500

    def test_vague_amount_kept_not_fabricated(self, expander, grounded, video_info):
        recipe = expander._parse_response(_response(ingredients=[
            {"id": "i1", "name": "salt", "quantity": None, "unit": None,
             "amount_text": "to taste"},
        ]), grounded, video_info)
        ing = recipe.ingredients[0]
        assert ing.quantity is None and ing.amount_text == "to taste"


class TestCoerceKind:
    @pytest.mark.parametrize("raw, expected", [
        ("cook", "cook"),
        ("Plating", "serve"),
        ("prep component", "prep_component"),
        ("made-up-kind", "cook"),
        (None, "cook"),
    ])
    def test_synonyms_and_fallback(self, raw, expected):
        assert _coerce_kind(raw) == expected


class TestCoerceProvenance:
    @pytest.mark.parametrize("item, expected", [
        ({"provenance": "video"}, "video"),
        ({"provenance": "  Reference "}, "reference"),
        # The reserved source id, written into the wrong field. Bare -> the humblest claim.
        ({"provenance": "vd"}, "model"),
        # ...but an unknown token that cites a source meant "reference".
        ({"provenance": "vd", "source_id": "r1"}, "reference"),
        ({"provenance": "camera"}, "model"),
        ({"provenance": 7}, "model"),
    ])
    def test_repairs_unknown_without_voiding_the_pass(self, item, expected):
        _coerce_provenance(item)
        assert item["provenance"] == expected

    def test_missing_is_left_for_the_schema_default(self):
        item = {"name": "onion"}
        _coerce_provenance(item)
        assert "provenance" not in item
        assert Ingredient(id="i1", **item).provenance == "video"

    def test_reserved_source_id_is_not_mistakable_for_a_provenance(self):
        # The bug: a two-letter id sat under the PROVENANCE block and got copied into the
        # enum, failing validation for every camera-sourced entity and voiding the expansion.
        assert VIDEO_DESCRIPTION_SOURCE_ID not in {"video", "reference", "model"}
        assert len(VIDEO_DESCRIPTION_SOURCE_ID) > 3


class TestPrompt:
    def test_spice_identity_rule_present(self, expander, grounded, video_info):
        # Task rules live in the stable system prompt.
        prompt = expander._system_prompt()
        assert "Resolve generic ingredient identities" in prompt
        assert "amount_text" in prompt

    def test_description_section_only_when_present(self, expander, grounded, video_info):
        # Per-dish data (incl. the creator's description) lives in the user prompt.
        with_desc = expander._user_prompt(grounded, video_info)
        assert "CREATOR'S VIDEO DESCRIPTION" in with_desc
        assert VIDEO_DESCRIPTION_SOURCE_ID in with_desc

        video_info.description = None
        without = expander._user_prompt(grounded, video_info)
        assert "CREATOR'S VIDEO DESCRIPTION" not in without

    def test_grounded_ingredients_render_amount_text(self, expander, grounded, video_info):
        prompt = expander._user_prompt(grounded, video_info)
        assert "- spices: some" in prompt
        assert "- rice: 2.0 cups" in prompt


class TestMessageHierarchy:
    def test_system_then_user(self, expander, grounded, video_info):
        messages = expander._build_messages(grounded, video_info)
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_persona_and_rules_in_system_not_user(self, expander, grounded, video_info):
        messages = expander._build_messages(grounded, video_info)
        system, user = messages[0]["content"], messages[1]["content"]
        assert "You are writing a recipe" in system
        assert "YOUR TASK" in system
        assert "You are writing a recipe" not in user

    def test_dish_leads_the_user_message(self, expander, grounded, video_info):
        # The dish name anchors whatever query the web search derives from the user message,
        # so it must lead - ahead of the raw (newsy) video title.
        user = expander._build_messages(grounded, video_info)[1]["content"]
        assert user.lstrip().startswith("THE DISH: chicken curry")

    def test_web_rule_scopes_to_recipes_when_grounding_on(self, expander):
        system = expander._system_prompt()
        assert "scoped to published recipes" in system
        assert "not recipes" in system  # news/listicle/social called out as noise

    def test_web_rule_flips_off_when_grounding_off(self):
        ex = RecipeExpander.__new__(RecipeExpander)
        ex._web_grounding = False
        system = ex._system_prompt()
        assert "do not have web search" in system.lower()
        assert "Leave `sources` empty" in system


class TestWebPluginConfig:
    def test_online_suffix_replaced_by_explicit_plugin(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        ex = RecipeExpander(web_grounding=True)
        # The model id stays plain; the plugin drives web search instead.
        assert not ex.model_name.endswith(":online")
        assert ex._plugins is not None
        plugin = ex._plugins[0]
        assert plugin["id"] == "web"
        assert plugin["max_results"] == 3
        assert "reddit.com" in plugin["exclude_domains"]
        assert "nytimes.com" in plugin["exclude_domains"]
        assert "recipe" in plugin["search_prompt"].lower()

    def test_no_plugin_when_grounding_off(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        ex = RecipeExpander(web_grounding=False)
        assert ex._plugins is None
        assert not ex.model_name.endswith(":online")

    def test_expand_passes_plugins_in_extra_body(self, monkeypatch, grounded, video_info):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        ex = RecipeExpander(web_grounding=True)

        captured = {}

        class _FakeMessage:
            content = _response()

        class _FakeChoice:
            message = _FakeMessage()

        class _FakeResponse:
            choices = [_FakeChoice()]

        def _fake_create(**kwargs):
            captured.update(kwargs)
            return _FakeResponse()

        ex._client.chat.completions.create = _fake_create
        ex.expand(grounded, video_info)

        assert captured["extra_body"]["plugins"] == ex._plugins
        assert not captured["model"].endswith(":online")
        assert [m["role"] for m in captured["messages"]] == ["system", "user"]
