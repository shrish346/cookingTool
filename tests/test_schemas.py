"""Schema-level contracts: amount coercion, timing derivation, serialization,
and the backend/frontend schema-version mirror."""
import re
from pathlib import Path

import pytest

from src.schemas import (
    SCHEMA_VERSION,
    Ingredient,
    MicroAction,
    Observation,
    Recipe,
    SceneDescription,
    SceneLog,
    Step,
    coerce_ingredient_amount,
    compute_timestamps_from_micro_actions,
)


class TestCoerceIngredientAmount:
    """Vague amounts must stay visibly vague - never fabricated into numbers."""

    @pytest.mark.parametrize(
        "ing, expected",
        [
            # (input dict, (quantity, unit, amount_text))
            ({"name": "rice", "quantity": 2, "unit": "cups"}, (2, "cups", None)),
            ({"name": "cumin", "quantity": "2 cups"}, (2.0, None, None)),
            ({"name": "butter", "quantity": "a pinch"}, (None, None, "a pinch")),
            ({"name": "salt", "quantity": None, "unit": None, "amount_text": "to taste"},
             (None, None, "to taste")),
            # "to taste" posing as a unit collapses to amount_text
            ({"name": "oil", "quantity": 1.0, "unit": "to taste"}, (None, None, "to taste")),
            # invalid quantities become None, never 1.0
            ({"name": "x", "quantity": -3, "unit": "g"}, (None, "g", None)),
            ({"name": "y", "quantity": True, "unit": "g"}, (None, "g", None)),
            ({"name": "z"}, (None, None, None)),
        ],
    )
    def test_cases(self, ing, expected):
        out = coerce_ingredient_amount(dict(ing))
        assert (out.get("quantity"), out.get("unit"), out.get("amount_text")) == expected

    def test_existing_amount_text_not_overwritten(self):
        out = coerce_ingredient_amount(
            {"name": "salt", "quantity": "some", "amount_text": "to taste"}
        )
        assert out["amount_text"] == "to taste"


class TestIngredientModel:
    def test_amount_text_only_is_valid(self):
        ing = Ingredient(name="salt", amount_text="to taste")
        assert ing.quantity is None and ing.unit is None

    def test_quantity_must_be_positive_when_present(self):
        with pytest.raises(ValueError):
            Ingredient(name="salt", quantity=0, unit="tsp")


class TestComputeTimestamps:
    def _recipe(self, micro_action_ids):
        return Recipe(
            title="t",
            ingredients=[],
            steps=[Step(order=1, instruction="cook", micro_action_ids=micro_action_ids)],
            servings=1,
        )

    def _scene_log(self):
        return SceneLog(scenes=[SceneDescription(
            frame_index=0,
            micro_actions=[
                MicroAction(id=0, action="add oil", timestamp_seconds=5.0, duration_seconds=2.0),
                MicroAction(id=1, action="no relevant cooking action", timestamp_seconds=8.0),
                MicroAction(id=2, action="flip chicken", timestamp_seconds=12.0, duration_seconds=1.0),
            ],
        )])

    def test_step_span_covers_cited_actions(self):
        recipe = compute_timestamps_from_micro_actions(self._recipe([0, 2]), self._scene_log())
        step = recipe.steps[0]
        assert step.start_timestamp_seconds == 5.0
        assert step.end_timestamp_seconds == 13.0  # 12.0 + 1.0 duration

    def test_noise_actions_are_skipped(self):
        recipe = compute_timestamps_from_micro_actions(self._recipe([1]), self._scene_log())
        step = recipe.steps[0]
        # A step citing only "no relevant cooking" actions resolves to no clip.
        assert not step.has_video_clip or step.start_timestamp_seconds is None


class TestSceneLog:
    def test_roundtrip_preserves_observations(self, make_scene_log):
        log = make_scene_log(observations=[
            Observation(timestamp_seconds=3.0, ingredient="cumin",
                        stated_amount="1 tsp", on_screen_text="1 tsp cumin"),
        ])
        restored = SceneLog.from_dict(log.to_dict())
        obs = restored.scenes[0].observations
        assert len(obs) == 1 and obs[0].ingredient == "cumin"
        assert obs[0].source == "keyframe"

    def test_get_all_observations_time_ordered(self):
        log = SceneLog(scenes=[SceneDescription(frame_index=0, observations=[
            Observation(timestamp_seconds=20.0, ingredient="b"),
            Observation(timestamp_seconds=5.0, ingredient="a"),
            Observation(ingredient="no-timestamp"),
        ])])
        assert [o.ingredient for o in log.get_all_observations()] == ["no-timestamp", "a", "b"]


def test_schema_version_mirrored_in_frontend():
    """CLAUDE.md: bump RECIPE_SCHEMA_VERSION in useLocalStorage.ts alongside
    SCHEMA_VERSION, or S3 recipes get served forever in an old shape."""
    ts = Path(__file__).parent.parent / "frontend/src/hooks/useLocalStorage.ts"
    match = re.search(r"RECIPE_SCHEMA_VERSION\s*=\s*(\d+)", ts.read_text())
    assert match, "RECIPE_SCHEMA_VERSION constant not found in useLocalStorage.ts"
    assert int(match.group(1)) == SCHEMA_VERSION, (
        f"frontend RECIPE_SCHEMA_VERSION={match.group(1)} != backend SCHEMA_VERSION={SCHEMA_VERSION}"
    )


def test_recipe_carries_current_schema_version():
    recipe = Recipe(title="t", ingredients=[], steps=[], servings=1)
    assert recipe.schema_version == SCHEMA_VERSION
