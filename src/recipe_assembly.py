"""Assembly of the final recipe from the grounding pass and the expansion pass.

The grounding pass (pass 1) produces steps that cite micro_action_ids, which
`compute_timestamps_from_micro_actions` resolves into real timestamps. That timing
is the product's whole value and must survive the expansion pass untouched.

The expansion pass (pass 2) is never shown a timestamp and never writes one. It
echoes back a `grounded_step_id` for each step it derived from a video step, and
`null` for each step it invented. `merge_expansion` re-attaches the frozen timing
by that ID. The LLM cannot drift the timing because it never gets to write it.
"""

from __future__ import annotations

from .schemas import Recipe, Step

# The expansion model is allowed to drop some grounded steps (it may legitimately
# merge two video steps into one). But if it drops most of them it has misunderstood
# the task, and we'd be throwing away the clips that make the product work.
MIN_GROUNDED_STEP_RETENTION = 0.7


def merge_expansion(grounded: Recipe, expanded: Recipe) -> Recipe:
    """Re-attach the grounded pass's frozen video timing onto the expanded recipe.

    The expander's parser writes the model's `grounded_step_id` onto `step.id`; a step
    the model invented has no ID, so Pydantic's default_factory mints a fresh one that
    by construction won't match any grounded step. So a plain ID lookup separates the
    two cases. Steps that map to a grounded step inherit its timing; invented steps
    stay clipless.

    Raises ValueError if the expansion dropped too many grounded steps, so the caller
    can fall back to the grounded recipe.
    """
    by_id = {step.id: step for step in grounded.steps}
    consumed: set[str] = set()

    for step in expanded.steps:
        source = by_id.get(step.id)

        if source is None:
            # An invented step, or a hallucinated ID. Either way it has no video.
            step.micro_action_ids = None
            step.micro_action_descriptions = None
            step.start_timestamp_seconds = None
            step.end_timestamp_seconds = None
            step.start_frame_index = None
            step.end_frame_index = None
        else:
            if source.id in consumed:
                raise ValueError(f"Expansion reused grounded step {source.id} more than once")
            consumed.add(source.id)

            step.id = source.id
            step.micro_action_ids = source.micro_action_ids
            step.micro_action_descriptions = source.micro_action_descriptions
            step.start_timestamp_seconds = source.start_timestamp_seconds
            step.end_timestamp_seconds = source.end_timestamp_seconds
            step.start_frame_index = source.start_frame_index
            step.end_frame_index = source.end_frame_index

        # Derived, never trusted from the LLM. A step has a clip if and only if it
        # resolved to real video timing.
        step.has_video_clip = bool(step.micro_action_ids) and step.start_timestamp_seconds is not None

        if not step.has_video_clip:
            _ensure_detail(step)

    groundable = [s for s in grounded.steps if s.has_video_clip]
    if groundable:
        retained = len([s for s in groundable if s.id in consumed]) / len(groundable)
        if retained < MIN_GROUNDED_STEP_RETENTION:
            raise ValueError(
                f"Expansion kept only {retained:.0%} of the {len(groundable)} grounded steps "
                f"(minimum {MIN_GROUNDED_STEP_RETENTION:.0%})"
            )

    expanded.sources = expanded.sources or grounded.sources
    expanded.video_id = grounded.video_id
    expanded.video_fps = grounded.video_fps
    expanded.source_url = grounded.source_url
    expanded.dish_query = expanded.dish_query or grounded.dish_query

    return expanded


def _ensure_detail(step: Step) -> None:
    """Guarantee a clipless step has a `detail`.

    A step with a clip has the video to carry it; a step without one has nothing on
    screen but its own text, so an empty `detail` leaves the reader with a bare
    instruction and no coaching. The expander is told to always write one, but it's
    a model - back it up here with the next-best copy the step already has.
    """
    if step.detail and step.detail.strip():
        return

    # Not doneness_cue: the UI renders that separately, so reusing it would print the
    # same sentence twice.
    tip = next((t.strip() for t in (step.tips or []) if t and t.strip()), None)
    step.detail = tip or (
        "The video doesn't cover this step, so take it slowly: work through the "
        "instruction above and check your work before moving on."
    )


def build_gather_steps(recipe: Recipe) -> Recipe:
    """Prepend the two setup steps.

    Synthesized in code rather than asked of the LLM: they are a pure function of
    `recipe.tools` and `recipe.ingredients`, so generating them deterministically
    guarantees they exist, are complete, and can never drift from the entity lists.
    """
    gather: list[Step] = []

    if recipe.tools:
        gather.append(Step(
            order=1,
            kind="gather_tools",
            title="Gather Your Tools",
            instruction="Get these out and within reach before you turn on any heat.",
            detail=(
                "Cooking goes wrong when you're hunting for a spatula with a hot pan going. "
                "Lay everything out first - you'll never be scrambling."
            ),
            tool_ids=[tool.id for tool in recipe.tools],
            provenance="model",
            has_video_clip=False,
        ))

    if recipe.ingredients:
        gather.append(Step(
            order=1,
            kind="gather_ingredients",
            title="Gather Your Ingredients",
            instruction="Measure everything out into small bowls before you start cooking.",
            detail=(
                "Chefs call this mise en place - everything in its place. Once the pan is hot, "
                "things move fast, and you won't have time to measure. Do it now, not later."
            ),
            ingredient_ids=[ing.id for ing in recipe.ingredients],
            provenance="model",
            has_video_clip=False,
        ))

    recipe.steps = gather + recipe.steps
    return relink_steps(recipe)


def relink_steps(recipe: Recipe) -> Recipe:
    """Renumber `order` 1..N and rebuild `depends_on` as the linear chain."""
    previous_id: str | None = None
    for index, step in enumerate(recipe.steps, start=1):
        step.order = index
        step.depends_on = [previous_id] if previous_id else []
        previous_id = step.id
    return recipe


def prune_dangling_references(recipe: Recipe) -> Recipe:
    """Drop ingredient/tool IDs on steps that don't resolve to a real entity.

    The expansion model references entities by ID, and sometimes invents one. A
    dangling ID would break the future mutation path (and any UI that renders a
    step's ingredient list), so we drop them here rather than carry them forward.
    """
    ingredient_ids = {ing.id for ing in recipe.ingredients}
    tool_ids = {tool.id for tool in recipe.tools}

    for step in recipe.steps:
        step.ingredient_ids = [i for i in step.ingredient_ids if i in ingredient_ids]
        step.tool_ids = [t for t in step.tool_ids if t in tool_ids]

    source_ids = {source.id for source in recipe.sources}
    for item in (*recipe.ingredients, *recipe.tools, *recipe.steps):
        if item.source_id and item.source_id not in source_ids:
            item.source_id = None

    return recipe
