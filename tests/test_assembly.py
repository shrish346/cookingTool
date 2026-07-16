"""Recipe assembly: merging expansion output while keeping grounded timing frozen."""
import pytest

from src.recipe_assembly import (
    build_gather_steps,
    merge_expansion,
    prune_dangling_references,
    relink_steps,
)
from src.schemas import Ingredient, Recipe, Source, Step, Tool


def _grounded(step_ids=("g1", "g2")) -> Recipe:
    return Recipe(
        title="t",
        ingredients=[],
        steps=[
            Step(id=sid, order=i + 1, instruction=f"step {sid}",
                 micro_action_ids=[i], micro_action_descriptions=[f"action {i}"],
                 start_timestamp_seconds=float(i * 10), end_timestamp_seconds=float(i * 10 + 5))
            for i, sid in enumerate(step_ids)
        ],
        servings=1,
        video_id="vid",
        video_fps=30.0,
        source_url="https://example.com",
        sources=[Source(id="s1", title="ref")],
    )


def _expanded(steps) -> Recipe:
    return Recipe(title="t", ingredients=[], steps=steps, servings=1)


class TestMergeExpansion:
    def test_frozen_timing_reattached_by_id(self):
        grounded = _grounded()
        expanded = _expanded([
            Step(id="g1", order=1, instruction="rewritten", detail="d",
                 # Model tried to drift the timing - must be overwritten.
                 start_timestamp_seconds=99.0, end_timestamp_seconds=100.0),
            Step(id="g2", order=2, instruction="rewritten 2"),
        ])
        merged = merge_expansion(grounded, expanded)
        assert merged.steps[0].start_timestamp_seconds == 0.0
        assert merged.steps[0].end_timestamp_seconds == 5.0
        assert merged.steps[0].has_video_clip is True

    def test_invented_step_is_clipless_with_detail(self):
        grounded = _grounded()
        expanded = _expanded([
            Step(id="g1", order=1, instruction="a"),
            Step(id="g2", order=2, instruction="b"),
            Step(order=3, instruction="invented rest step",
                 # Model claimed a clip - derived in code, never trusted.
                 has_video_clip=True, start_timestamp_seconds=50.0),
        ])
        merged = merge_expansion(grounded, expanded)
        invented = merged.steps[2]
        assert invented.has_video_clip is False
        assert invented.start_timestamp_seconds is None
        assert invented.detail  # clipless steps always get a detail backstop

    def test_reused_grounded_id_rejected(self):
        grounded = _grounded()
        expanded = _expanded([
            Step(id="g1", order=1, instruction="a"),
            Step(id="g1", order=2, instruction="b"),
        ])
        with pytest.raises(ValueError, match="reused"):
            merge_expansion(grounded, expanded)

    def test_dropping_most_grounded_steps_rejected(self):
        grounded = _grounded(step_ids=("g1", "g2", "g3", "g4"))
        expanded = _expanded([Step(id="g1", order=1, instruction="only one kept")])
        with pytest.raises(ValueError, match="kept only"):
            merge_expansion(grounded, expanded)

    def test_video_identity_carried_from_grounded(self):
        merged = merge_expansion(_grounded(), _expanded([
            Step(id="g1", order=1, instruction="a"),
            Step(id="g2", order=2, instruction="b"),
        ]))
        assert merged.video_id == "vid"
        assert merged.video_fps == 30.0
        assert merged.source_url == "https://example.com"
        assert merged.sources[0].id == "s1"  # inherited when expansion has none


class TestBuildGatherSteps:
    def test_gather_steps_prepended_and_precomponents_pulled_forward(self):
        recipe = Recipe(
            title="t",
            ingredients=[Ingredient(name="rice", quantity=1, unit="cup")],
            tools=[Tool(name="pan")],
            steps=[
                Step(order=1, kind="cook", instruction="fry"),
                Step(order=2, kind="prep_component", instruction="cook rice", detail="d"),
            ],
            servings=1,
        )
        out = build_gather_steps(recipe)
        kinds = [s.kind for s in out.steps]
        assert kinds == ["gather_tools", "gather_ingredients", "prep_component", "cook"]
        assert out.steps[0].tool_ids == [recipe.tools[0].id]
        assert out.steps[1].ingredient_ids == [recipe.ingredients[0].id]
        assert [s.order for s in out.steps] == [1, 2, 3, 4]  # relinked

    def test_no_entities_no_gather_steps(self):
        recipe = Recipe(title="t", ingredients=[], tools=[],
                        steps=[Step(order=1, instruction="x")], servings=1)
        out = build_gather_steps(recipe)
        assert [s.kind for s in out.steps] == ["cook"]


class TestRelinkSteps:
    def test_linear_chain(self):
        recipe = Recipe(title="t", ingredients=[], servings=1, steps=[
            Step(id="a", order=9, instruction="1"),
            Step(id="b", order=9, instruction="2"),
            Step(id="c", order=9, instruction="3"),
        ])
        out = relink_steps(recipe)
        assert [s.order for s in out.steps] == [1, 2, 3]
        assert out.steps[0].depends_on == []
        assert out.steps[1].depends_on == ["a"]
        assert out.steps[2].depends_on == ["b"]


class TestPruneDanglingReferences:
    def test_dangling_ids_dropped_and_source_ids_nulled(self):
        ing = Ingredient(name="rice", quantity=1, unit="cup", source_id="ghost")
        recipe = Recipe(
            title="t",
            ingredients=[ing],
            tools=[],
            sources=[Source(id="real", title="ref")],
            steps=[Step(order=1, instruction="x",
                        ingredient_ids=[ing.id, "invented"], tool_ids=["ghost-tool"],
                        source_id="real")],
            servings=1,
        )
        out = prune_dangling_references(recipe)
        assert out.steps[0].ingredient_ids == [ing.id]
        assert out.steps[0].tool_ids == []
        assert out.steps[0].source_id == "real"       # resolves - kept
        assert out.ingredients[0].source_id is None   # dangling - nulled
