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

        return f"""You are writing a recipe for the most common kind of home cook: a young adult who
knows their way around a kitchen at a basic level, but no further.

They can use a stove and an oven. They know what "chop", "simmer", "whisk" and "medium heat" mean.
They own the basic tools and can crack an egg without being told how. What they DON'T know is the
deeper stuff: specialized prep (shelling fresh peas, deveining a shrimp), or a technique specific to
a cuisine they've never cooked (folding gyoza, tempering spices, getting a wok hei sear). Assume the
common knowledge; teach the uncommon.

THE RULE: explain what a casual cook wouldn't already know, and NOTHING they obviously would.
"Use a sharp knife" and "crack the eggs into a bowl" are noise - cut them. "Toast the spices until
they smell nutty but before they smoke" or "keep the pan moving so the garlic doesn't burn" is
signal - keep it.

THE DISH: {dish}
SOURCE VIDEO: {video_info.title} ({video_info.duration_seconds}s)

A cooking video was analyzed and produced the recipe skeleton below. **The video is a BASELINE, not a
spec.** It shows what your reader wants to make - it does NOT contain everything they need to know to
make it. Short-form cooking videos skip prep, skip measurements, skip resting, and never explain the
technique that actually matters. Fill in what's missing - but only what's worth saying.

GROUNDED STEPS (what the camera actually saw):
{steps_text}

INGREDIENTS OBSERVED:
{ingredients_text}

TOOLS OBSERVED: {tools_text}

YOUR TASK:

1. **Expand every grounded step IN PLACE.** Keep it, but rewrite it for this reader. Give it a `detail`
   ONLY where there's something non-obvious to say - the part a casual cook would actually get wrong,
   or a real quantity/temperature/time the video skipped. If a grounded step involves a genuinely
   uncommon technique, teach it INSIDE that grounded step (that's the step with the video clip - don't
   hollow it out into a stub and move the teaching elsewhere). If the step is self-explanatory, keep
   `detail` short or leave it minimal - do not pad it with things the reader plainly already knows.

2. **Add a `doneness_cue` ONLY when the finished state is non-obvious** - something the reader could
   not just guess by looking. "The gravy starts to bubble and thickens enough to coat the back of a
   spoon", "the onions turn translucent and soft", "the chicken is no longer pink at the thickest
   part" are all worth saying. "Once everything is in the pan" or "when the water is boiling" is NOT -
   the reader can see that. When there's nothing non-obvious to signal, set `doneness_cue` to null.

3. **Turn assumed pre-made components into their own steps.** Short-form videos often start with an
   ingredient that was ALREADY cooked or prepared off-camera - "cooked rice", "boiled potatoes",
   "cooked pasta", "steamed greens". For your reader, making that component IS part of the recipe.
   For each such assumed component:
     - Add exactly ONE step with `kind: "prep_component"` and `grounded_step_id: null` that teaches
       how to make it, concisely. These have no video clip, so the `detail` carries the whole weight.
     - If the component is effectively its own little dish (rice, pasta, a boiled egg), add its RAW
       sub-ingredients (e.g. raw rice + water) to the `ingredients` list as real entries, and
       reference them by id in that step's `ingredient_ids`. If it's a trivial prep of an ingredient
       already listed (e.g. "toasted nuts" from nuts you already have), don't duplicate it.
   Do not turn ordinary in-recipe prep into `prep_component` - this is only for components the video
   treated as already finished before it started.

4. **Insert other new steps** only where this reader would genuinely get stuck or hurt: an uncommon
   technique no grounded step covers, a rest the video didn't show, a real safety warning (hot oil,
   raw chicken). Set their `grounded_step_id` to null. Don't invent a step to state the obvious.
   Every invented step has NO video clip, so its `detail` is the only thing the reader has to go on.
   `detail` is REQUIRED on every step and must be non-empty; on an invented step it has to carry the
   whole weight of teaching.

5. **Every quantity gets a real number.** "A pinch of salt" becomes "1/4 tsp". "Some butter" becomes
   "2 tbsp". Never pass a vague amount through to the reader.

6. **List every tool they need**, not just the ones on camera.

7. {web_rule}

Target 8-15 steps. Add a step only when it earns its place - resist padding.

PROVENANCE - tag every ingredient, tool and step with where it came from:
  "video"     - the camera showed this
  "reference" - you took it from a published recipe (set source_id)
  "model"     - your own cooking knowledge; an estimate

OUTPUT - return ONLY a JSON object:

{{
  "title": "Recipe name",
  "description": "One or two sentences.",
  "difficulty": "easy",
  "servings": 2,
  "prep_time_minutes": 10,
  "cook_time_minutes": 15,
  "cuisine": "Japanese",
  "tags": ["dinner"],
  "sources": [
    {{"id": "s1", "title": "Classic Chicken Katsudon", "url": "https://...", "site": "Just One Cookbook"}}
  ],
  "tools": [
    {{"id": "t1", "name": "Small saucepan", "essential": true,
      "substitute": "any small pot", "provenance": "video"}},
    {{"id": "t2", "name": "Fine-mesh strainer", "essential": false,
      "substitute": "a regular sieve", "provenance": "model"}}
  ],
  "ingredients": [
    {{"id": "i1", "name": "Cooked white rice", "quantity": 2, "unit": "cups", "preparation": null,
      "optional": false, "provenance": "video", "source_id": null,
      "note": "The video starts with rice already made - step 3 shows you how."}},
    {{"id": "i2", "name": "Raw short-grain rice", "quantity": 1, "unit": "cup", "preparation": null,
      "optional": false, "provenance": "model", "source_id": null,
      "note": "Cooks down into the 2 cups of cooked rice the recipe needs."}},
    {{"id": "i3", "name": "Water", "quantity": 1.25, "unit": "cups", "preparation": null,
      "optional": false, "provenance": "model", "source_id": null, "note": null}}
  ],
  "steps": [
    {{
      "grounded_step_id": null,
      "kind": "prep_component",
      "title": "Cook the Rice",
      "instruction": "Rinse 1 cup short-grain rice until the water runs clear, then simmer it with 1.25 cups water, covered, for about 15 minutes.",
      "detail": "The video assumes you already have cooked rice - here's how. Rinsing washes off surface starch so the grains don't turn gluey; swirl it in a few changes of cold water until the water looks clear, not milky. Bring it to a boil, then drop to the lowest heat, cover, and DON'T lift the lid - the trapped steam is what cooks it. After 15 minutes off the heat, let it sit covered another 10 to finish.",
      "doneness_cue": "Every grain is tender and the water is fully absorbed - tilt the pot and no liquid pools at the bottom.",
      "duration_minutes": 25,
      "tips": ["No lid that fits? A plate works, as long as it traps the steam."],
      "ingredient_ids": ["i2", "i3"],
      "tool_ids": ["t1"],
      "provenance": "model",
      "source_id": null
    }},
    {{
      "grounded_step_id": "a1b2c3d4",
      "kind": "cook",
      "title": "Simmer the Sauce and Onions",
      "instruction": "Simmer the sliced onion in the dashi, soy sauce and mirin until soft.",
      "detail": "Keep it at a gentle simmer, not a hard boil - a rolling boil drives off the mirin's sweetness and toughens the onion.",
      "doneness_cue": "The onions turn translucent and slump, and the sauce smells sweet rather than sharp.",
      "duration_minutes": 5,
      "tips": null,
      "ingredient_ids": ["i1"],
      "tool_ids": ["t1"],
      "provenance": "video",
      "source_id": null
    }},
    {{
      "grounded_step_id": "e5f6a7b8",
      "kind": "assemble",
      "title": "Spoon Over the Rice",
      "instruction": "Slide the egg-and-onion mixture over a bowl of the cooked rice.",
      "detail": null,
      "doneness_cue": null,
      "duration_minutes": 1,
      "tips": null,
      "ingredient_ids": ["i1"],
      "tool_ids": [],
      "provenance": "video",
      "source_id": null
    }}
  ]
}}

HARD RULES:
- `grounded_step_id` MUST be one of the IDs listed above, or null. Never invent one. Never reuse one.
- Keep EVERY grounded step. You may reorder nothing - the video's steps must stay in their original
  relative order, with your new steps woven between them. (Placement of `prep_component` steps is
  handled automatically; just emit them.)
- **Every step must stand on its own.** Never write an instruction like "you already did this above"
  or "see the previous step". If two steps would say the same thing, they should have been one step.
  A step whose instruction defers to another step is a bug, not a step.
- NEVER output timestamps, `micro_action_ids`, `order`, or `has_video_clip`. Those are not yours to set.
- `ingredient_ids` and `tool_ids` must reference IDs you declared in this same JSON.
- Do NOT create "gather your tools" or "gather your ingredients" steps - those are added automatically.
- `quantity` must be a positive number. Use "to taste" as the unit if it truly cannot be measured.
- On a grounded step (one with a `grounded_step_id`), `detail` and `doneness_cue` may be null when
  there is genuinely nothing non-obvious to add - don't invent filler. But on an invented or
  `prep_component` step, `detail` is REQUIRED and non-empty: it has no video clip, so it is all the
  reader has. `doneness_cue` may still be null there if the finished state is obvious.

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
    "gather_tools", "gather_ingredients", "prep_component", "prep", "cook",
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
