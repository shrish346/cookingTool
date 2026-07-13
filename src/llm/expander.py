"""Pass 2: turn a video-grounded recipe skeleton into a beginner-proof recipe.

Pass 1's job is fidelity - it must not invent. This pass's job is pedagogy - it
*must* invent. The video is a baseline, not a spec: it shows what the user wants
to make, not everything they need to know to make it.

This pass never sees a timestamp and never writes one. It echoes back the ID of the
grounded step each of its steps came from, and `recipe_assembly.merge_expansion`
re-attaches the frozen timing afterward.
"""

from __future__ import annotations

import os
import json
import re

from openai import OpenAI
from dotenv import load_dotenv

from ..downloaders.base import VideoInfo
from ..schemas import Recipe

load_dotenv()

DEFAULT_EXPANSION_MODEL = "google/gemini-2.5-flash"


class RecipeExpander:
    """Expands a grounded recipe for a cook who has never cooked before."""

    def __init__(self, model: str | None = None, web_grounding: bool | None = None):
        self._model = model or os.getenv("EXPANSION_MODEL", DEFAULT_EXPANSION_MODEL)

        if web_grounding is None:
            web_grounding = os.getenv("RECIPE_WEB_GROUNDING", "1") == "1"
        self._web_grounding = web_grounding

        # OpenRouter's `:online` suffix auto-injects web search results for the query,
        # so the model can pull real quantities from published recipes rather than
        # guessing. Billed per result (~$0.02/request at the default 5).
        if self._web_grounding and not self._model.endswith(":online"):
            self._model += ":online"

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    @property
    def model_name(self) -> str:
        return self._model

    def expand(self, grounded: Recipe, video_info: VideoInfo) -> Recipe:
        """Return an expanded Recipe. Raises on any failure; the caller falls back."""
        prompt = self._build_prompt(grounded, video_info)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            temperature=0.5,
        )

        return self._parse_response(response.choices[0].message.content, grounded)

    def _build_prompt(self, grounded: Recipe, video_info: VideoInfo) -> str:
        dish = grounded.dish_query or grounded.title
        steps_text = self._format_grounded_steps(grounded)
        ingredients_text = self._format_grounded_ingredients(grounded)
        tools_text = ", ".join(tool.name for tool in grounded.tools) or "(none identified in the video)"

        web_rule = (
            "You have web search available. Look up real published recipes for this dish and use "
            "their quantities, temperatures and times to fill the gaps. Cite them in `sources`, and "
            "set `provenance: \"reference\"` with the matching `source_id` on anything you took from them."
            if self._web_grounding else
            "You do not have web search. Fill gaps from your own cooking knowledge and mark them "
            "`provenance: \"model\"`. Leave `sources` empty."
        )

        return f"""You are teaching someone to cook who has NEVER COOKED BEFORE.

They do not know what a whisk is. They do not know that eggs go into a bowl before they go into a pan.
They do not know what "medium heat" means, what "a pinch" is, or how to tell when mushrooms are done.
They will follow your instructions literally and they will not fill in a single blank on their own.

THE DISH: {dish}
SOURCE VIDEO: {video_info.title} ({video_info.duration_seconds}s)

A cooking video was analyzed and produced the recipe skeleton below. **The video is a BASELINE, not a
spec.** It shows what your reader wants to make - it does NOT contain everything they need to know to
make it. Short-form cooking videos skip prep, skip measurements, skip resting, and never explain
technique. Your job is to fill in every single thing the video left out.

GROUNDED STEPS (what the camera actually saw):
{steps_text}

INGREDIENTS OBSERVED:
{ingredients_text}

TOOLS OBSERVED: {tools_text}

YOUR TASK:

1. **Expand every grounded step IN PLACE.** Keep it, but rewrite it for a beginner. Give it a `detail`
   that explains the part they'd get wrong, and a `doneness_cue` that tells them how to know it worked.
   If a grounded step needs a technique taught - how to crack an egg, how to hold a knife - teach it
   INSIDE that grounded step. Do NOT split the teaching into a separate step and leave the grounded
   one as a stub: the grounded step is the one with the video clip attached, so a hollow grounded step
   means the reader gets a clip with nothing to read next to it.

2. **Insert new steps between them** wherever a beginner would get stuck or hurt themselves:
   prep the video skipped over, a technique no grounded step covers, a rest the video didn't show,
   a safety warning. These are steps the video never showed - that is the point. Set their
   `grounded_step_id` to null. A new step must carry content that NO grounded step covers.

3. **Every quantity gets a real number.** "A pinch of salt" becomes "1/4 tsp". "Some butter" becomes
   "2 tbsp". Never pass a vague amount through to the reader.

4. **List every tool they need**, not just the ones on camera. If they whisk eggs, they need a bowl
   and a whisk, even if the video cut that shot.

5. {web_rule}

Target 12-20 steps. Err on the side of too much hand-holding.

PROVENANCE - tag every ingredient, tool and step with where it came from:
  "video"     - the camera showed this
  "reference" - you took it from a published recipe (set source_id)
  "model"     - your own cooking knowledge; an estimate

OUTPUT - return ONLY a JSON object:

{{
  "title": "Recipe name",
  "description": "One or two sentences.",
  "difficulty": "beginner",
  "servings": 2,
  "prep_time_minutes": 10,
  "cook_time_minutes": 15,
  "cuisine": "Mexican",
  "tags": ["breakfast"],
  "sources": [
    {{"id": "s1", "title": "Perfect Mushroom Omelette", "url": "https://...", "site": "Serious Eats"}}
  ],
  "tools": [
    {{"id": "t1", "name": "10-inch nonstick skillet", "essential": true,
      "substitute": "any nonstick frying pan", "provenance": "video"}},
    {{"id": "t2", "name": "Whisk", "essential": false,
      "substitute": "a fork works fine", "provenance": "model"}}
  ],
  "ingredients": [
    {{"id": "i1", "name": "Eggs", "quantity": 3, "unit": "count", "preparation": null,
      "optional": false, "provenance": "video", "source_id": null, "note": null}},
    {{"id": "i2", "name": "Kosher salt", "quantity": 0.25, "unit": "tsp", "preparation": null,
      "optional": false, "provenance": "reference", "source_id": "s1",
      "note": "The video just says 'a pinch' - this is the standard amount."}}
  ],
  "steps": [
    {{
      "grounded_step_id": "a1b2c3d4",
      "kind": "cook",
      "title": "Cook the Mushrooms",
      "instruction": "Heat the skillet over medium heat, then add the sliced mushrooms and salt.",
      "detail": "Medium heat means the pan is hot enough that a drop of water sizzles and dances, but doesn't leap out and spit at you. Give the pan a full 2 minutes to come up to temperature before anything goes in - a cold pan makes mushrooms soggy instead of browned. Spread them in a single layer and then LEAVE THEM ALONE for a minute; stirring too early is the most common mistake here.",
      "doneness_cue": "The mushrooms have shrunk to about half their size, released their water, and the edges are turning golden brown.",
      "duration_minutes": 5,
      "tips": ["Don't crowd the pan - if the mushrooms are piled up they steam instead of browning."],
      "ingredient_ids": ["i3", "i2"],
      "tool_ids": ["t1"],
      "provenance": "video",
      "source_id": null
    }},
    {{
      "grounded_step_id": null,
      "kind": "technique",
      "title": "Crack and Whisk the Eggs",
      "instruction": "Crack 3 eggs into a bowl and whisk until the yolks and whites are fully combined.",
      "detail": "Crack each egg on a flat surface, not the rim of the bowl - the rim drives shell fragments into the egg. Crack them into a separate small bowl first if you're nervous, so one bad egg doesn't ruin the batch. Whisk until you can't see any streaks of clear white left; that takes about 30 seconds of brisk beating.",
      "doneness_cue": "The mixture is a uniform pale yellow with no clear or stringy bits.",
      "duration_minutes": 2,
      "tips": ["If you get shell in the bowl, scoop it out with a larger piece of shell - it acts like a magnet."],
      "ingredient_ids": ["i1"],
      "tool_ids": ["t2"],
      "provenance": "model",
      "source_id": null
    }}
  ]
}}

HARD RULES:
- `grounded_step_id` MUST be one of the IDs listed above, or null. Never invent one. Never reuse one.
- Keep EVERY grounded step. You may reorder nothing - the video's steps must stay in their original
  relative order, with your new steps woven between them.
- **Every step must stand on its own.** Never write an instruction like "you already did this above"
  or "see the previous step". If two steps would say the same thing, they should have been one step.
  A step whose instruction defers to another step is a bug, not a step.
- NEVER output timestamps, `micro_action_ids`, `order`, or `has_video_clip`. Those are not yours to set.
- `ingredient_ids` and `tool_ids` must reference IDs you declared in this same JSON.
- Do NOT create "gather your tools" or "gather your ingredients" steps - those are added automatically.
- `quantity` must be a positive number. Use "to taste" as the unit if it truly cannot be measured.

Return ONLY the JSON object."""

    def _format_grounded_steps(self, grounded: Recipe) -> str:
        lines = []
        for step in grounded.steps:
            seen = ", ".join(step.micro_action_descriptions or []) or "(no video actions)"
            lines.append(f'- id "{step.id}" | {step.title or step.instruction[:40]}')
            lines.append(f'    instruction: {step.instruction}')
            lines.append(f'    camera saw: {seen}')
        return "\n".join(lines) or "(none)"

    def _format_grounded_ingredients(self, grounded: Recipe) -> str:
        lines = []
        for ing in grounded.ingredients:
            prep = f" ({ing.preparation})" if ing.preparation else ""
            lines.append(f"- {ing.name}: {ing.quantity} {ing.unit}{prep}")
        return "\n".join(lines) or "(none identified)"

    def _parse_response(self, content: str, grounded: Recipe) -> Recipe:
        """Parse the model's JSON into a Recipe, mapping grounded_step_id onto step.id."""
        data = json.loads(_extract_json(content))

        # merge_expansion looks the step up by `id`. Carrying the model's grounded_step_id
        # across as the step's ID is what lets it re-attach the frozen timing. A step the
        # model invented has no ID, so Pydantic mints a fresh one that won't match anything.
        for step in data.get("steps", []):
            grounded_id = step.pop("grounded_step_id", None)
            if grounded_id:
                step["id"] = grounded_id
            # Not the model's to set, whatever the prompt said.
            for owned_by_us in ("order", "has_video_clip", "micro_action_ids",
                                "start_timestamp_seconds", "end_timestamp_seconds"):
                step.pop(owned_by_us, None)
            step["kind"] = _coerce_kind(step.get("kind"))

        # order is required (ge=1) but we renumber in relink_steps anyway.
        for index, step in enumerate(data.get("steps", []), start=1):
            step["order"] = index

        data["ingredients"] = [_coerce_ingredient(i) for i in data.get("ingredients", [])]

        data["source_url"] = grounded.source_url
        data["video_id"] = grounded.video_id
        data["video_fps"] = grounded.video_fps
        data["dish_query"] = grounded.dish_query
        data["reasoning"] = grounded.reasoning
        if not data.get("title"):
            data["title"] = grounded.title
        if not isinstance(data.get("servings"), int) or data["servings"] <= 0:
            data["servings"] = grounded.servings

        # Carry the nutrition estimates from pass 1 - pass 2 doesn't recompute them.
        for field in ("calories", "protein", "carbs", "fats", "cholesterol",
                      "sodium", "sugar", "vitamin_a", "vitamin_c", "calcium"):
            data.setdefault(field, getattr(grounded, field))

        return Recipe(**data)


VALID_KINDS = {
    "gather_tools", "gather_ingredients", "prep", "cook",
    "assemble", "rest", "serve", "technique", "safety",
}

# The model reaches for reasonable-sounding kinds we didn't enumerate ("finish",
# "plate"). Map the near-misses rather than failing validation - one stray enum
# is not worth throwing away the whole expansion.
KIND_SYNONYMS = {
    "finish": "serve",
    "finishing": "serve",
    "plate": "serve",
    "plating": "serve",
    "garnish": "serve",
    "combine": "assemble",
    "mix": "assemble",
    "fold": "assemble",
    "chop": "prep",
    "prepare": "prep",
    "preparation": "prep",
    "wait": "rest",
    "resting": "rest",
    "warning": "safety",
    "tip": "technique",
}


def _coerce_kind(kind) -> str:
    if not isinstance(kind, str):
        return "cook"
    normalized = kind.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in VALID_KINDS:
        return normalized
    return KIND_SYNONYMS.get(normalized, "cook")


def _coerce_ingredient(ing: dict) -> dict:
    """Ingredient.quantity is `gt=0`, but the prompt invites 'to taste' amounts."""
    quantity = ing.get("quantity")
    if isinstance(quantity, str):
        match = re.search(r"(\d+\.?\d*)", quantity)
        quantity = float(match.group(1)) if match else None
    if not isinstance(quantity, (int, float)) or quantity <= 0:
        quantity = 1.0
    ing["quantity"] = float(quantity)
    ing["unit"] = ing.get("unit") or "to taste"
    return ing


def _extract_json(content: str) -> str:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if fenced:
        return fenced.group(1)
    start, end = content.find("{"), content.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError(f"Could not find JSON in expansion response: {content[:200]}...")
    return content[start:end]
