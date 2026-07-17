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
from ..schemas import Recipe, Source, coerce_ingredient_amount

load_dotenv()

DEFAULT_EXPANSION_MODEL = "google/gemini-2.5-flash"

# The creator's description is evidence, but it isn't the camera - an amount read off
# a caption was never shown being measured. It gets `reference` provenance like any
# published recipe, citing a Source minted for the video itself under this reserved ID.
# The model cites the ID; only code mints the Source, because prune_dangling_references
# nulls any source_id that doesn't resolve.
VIDEO_DESCRIPTION_SOURCE_ID = "video_description"


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

        return self._parse_response(response.choices[0].message.content, grounded, video_info)

    def _build_prompt(self, grounded: Recipe, video_info: VideoInfo) -> str:
        dish = grounded.dish_query or grounded.title
        steps_text = self._format_grounded_steps(grounded)
        ingredients_text = self._format_grounded_ingredients(grounded)
        tools_text = ", ".join(tool.name for tool in grounded.tools) or "(none identified in the video)"

        # Creator captions (TikTok/Reels descriptions) often carry the full ingredient
        # list with amounts - real evidence, but frequently a hashtag wall, so cap it.
        description_section = ""
        if video_info.description:
            description_section = f"""
CREATOR'S VIDEO DESCRIPTION (may contain the ingredient list or amounts - use what's relevant,
ignore hashtags and promo; it can name and quantify ingredients, but it does not add steps the
video never showed). The description is NOT the camera: anything you take from it is
`provenance: "reference"` with `source_id: "{VIDEO_DESCRIPTION_SOURCE_ID}"` - never "video". Prefer it
over a published recipe when both give an amount; it's this dish, not a similar one. Do NOT declare
"{VIDEO_DESCRIPTION_SOURCE_ID}" in `sources` yourself - just cite the id and it will be filled in:
{video_info.description[:1500]}
"""

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
{description_section}
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

5. **Pin down every amount you can.** "A pinch of salt" becomes "1/4 tsp". "Some butter" becomes
   "2 tbsp". Get the number from the video's evidence, the creator's description, or a published
   recipe - and tag where it came from. ONLY when no source can pin a number AND the amount is
   genuinely judgment-based (salt "to taste", oil "for frying") set `quantity: null` and put the
   honest wording in `amount_text`. A cop-out `amount_text` on something a reference recipe measures
   is a bug - resolve it.

6. **Resolve generic ingredient identities.** A grounded entry like "spices", "seasoning", "masala",
   or "red spice powder" is a placeholder, not an ingredient - your reader cannot shop for it.
   Replace it with the concrete named ingredients from a published recipe for this dish (e.g.
   "1 tsp cumin, 1 tsp coriander, 1/2 tsp turmeric" as separate entries), tagged
   `provenance: "reference"` with a `source_id` - or `"model"` if no reference names them. If the
   video's evidence (label, caption, narration) already names it, prefer that name. Reference the
   new entries from the steps that use them.

7. **List every tool they need**, not just the ones on camera.

8. {web_rule}

Target 8-15 steps. Add a step only when it earns its place - resist padding.

PROVENANCE - tag every ingredient, tool and step with where it came from:
  "video"     - the camera showed this
  "reference" - you took it from a published recipe, or from the creator's description
                (set source_id: the recipe's id, or "{VIDEO_DESCRIPTION_SOURCE_ID}" for the description)
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
      "optional": false, "provenance": "model", "source_id": null, "note": null}},
    {{"id": "i4", "name": "Salt", "quantity": null, "unit": null, "amount_text": "to taste",
      "preparation": null, "optional": false, "provenance": "video", "source_id": null, "note": null}}
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
- Each ingredient has EITHER a positive `quantity` + `unit`, OR `quantity: null` with `amount_text`
  set (only for genuinely judgment-based amounts - see task 5). Never both, never neither.
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
            if ing.quantity is not None:
                amount = f"{ing.quantity} {ing.unit or ''}".strip()
            else:
                amount = ing.amount_text or "amount not shown"
            lines.append(f"- {ing.name}: {amount}{prep}")
        return "\n".join(lines) or "(none identified)"

    def _parse_response(self, content: str, grounded: Recipe, video_info: VideoInfo) -> Recipe:
        """Parse the model's JSON into a Recipe, mapping grounded_step_id onto step.id."""
        data = json.loads(_extract_json(content))

        # The description source is ours to mint, not the model's to declare.
        data["sources"] = [
            s for s in data.get("sources", [])
            if s.get("id") != VIDEO_DESCRIPTION_SOURCE_ID
        ]

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
            _coerce_provenance(step)

        # order is required (ge=1) but we renumber in relink_steps anyway.
        for index, step in enumerate(data.get("steps", []), start=1):
            step["order"] = index

        data["ingredients"] = [_coerce_ingredient(i) for i in data.get("ingredients", [])]
        for item in (*data["ingredients"], *data.get("tools", [])):
            if isinstance(item, dict):
                _coerce_provenance(item)

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

        recipe = Recipe(**data)
        _attach_description_source(recipe, video_info)
        return recipe


def _attach_description_source(recipe: Recipe, video_info: VideoInfo) -> None:
    """Mint the Source for the creator's description, if anything actually cites it.

    Minted only on demand: an uncited source would show up in the recipe's Sources
    list claiming the description was used when it wasn't. If the model cited the id
    without a description to read (or we decline to mint), prune_dangling_references
    nulls the citation back to a bare `reference` - visible, not wrong.
    """
    cited = any(
        item.source_id == VIDEO_DESCRIPTION_SOURCE_ID
        for item in (*recipe.ingredients, *recipe.tools, *recipe.steps)
    )
    if not cited or not video_info.description:
        return

    recipe.sources.append(
        Source(
            id=VIDEO_DESCRIPTION_SOURCE_ID,
            title=video_info.title or "The source video",
            url=video_info.url,
            # Renders as "from the creator's description" on the provenance badge.
            site="the creator's description",
        )
    )


VALID_KINDS = {
    "gather_tools", "gather_ingredients", "prep_component", "prep", "cook",
    "assemble", "rest", "serve", "technique", "safety",
}

# Mirrors the Provenance Literal in schemas.py.
VALID_PROVENANCE = {"video", "reference", "model"}

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


def _coerce_provenance(item: dict) -> None:
    """Repair an unrecognized `provenance` in place, so one bad enum can't void the pass.

    `Recipe(**data)` validates the whole object at once, so a single junk provenance on a
    single ingredient throws away the entire expansion and drops us back to the pass-1
    recipe. That is a wildly disproportionate blast radius for a field we can infer.

    An unrecognized token that cites a source meant "reference" - citing a source is the
    thing that distinguishes it. A bare unrecognized token becomes "model", the humblest
    claim: guessing "video" would assert the camera showed something we have no evidence
    for, and provenance is the trust surface the frontend renders.

    A *missing* provenance is left alone - the schema already defaults it to "video".
    """
    if "provenance" not in item:
        return
    value = item["provenance"]
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if normalized in VALID_PROVENANCE:
        item["provenance"] = normalized
        return
    item["provenance"] = "reference" if item.get("source_id") else "model"


def _coerce_ingredient(ing: dict) -> dict:
    """Normalize amounts without fabricating: vague stays vague (amount_text)."""
    return coerce_ingredient_amount(ing)


def _extract_json(content: str) -> str:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if fenced:
        return fenced.group(1)
    start, end = content.find("{"), content.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError(f"Could not find JSON in expansion response: {content[:200]}...")
    return content[start:end]
