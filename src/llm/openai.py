from __future__ import annotations

import os
import json
import re

from openai import OpenAI
from dotenv import load_dotenv

from ..downloaders.base import VideoInfo
from ..schemas import (
    Recipe,
    SceneLog,
    Tool,
    coerce_ingredient_amount,
    compute_frame_indices_from_micro_actions,
    compute_timestamps_from_micro_actions,
)

load_dotenv()


class OpenAIAdapter:
    """Adapter for OpenAI's language models (GPT-4o, GPT-4)."""

    def __init__(self, model: str = "gpt-4o"):
        self._model = model
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model

    def generate_recipe(
        self,
        scene_log: SceneLog,
        video_info: VideoInfo,
        transcript: str | None = None,
        use_timestamps: bool = True
    ) -> Recipe:
        """
        Generate a structured recipe from accumulated scene descriptions.
        
        Args:
            scene_log: Accumulated scene descriptions from VLM analysis
            video_info: Metadata about the video (title, description, etc.)
            transcript: Optional audio transcript from the video
            use_timestamps: If True, use timestamp-based mapping (preferred).
                           If False, use legacy frame-based mapping.
            
        Returns:
            A validated Recipe object
        """
        prompt = self._build_prompt(scene_log, video_info, transcript, use_timestamps)
        
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.7,
        )
        
        content = response.choices[0].message.content
        recipe = self._parse_response(content, video_info)
        recipe = self._seed_tools_from_scene_log(recipe, scene_log)

        # Post-process: compute video clip timing from micro_action_ids
        if use_timestamps:
            # Try timestamp-based first, fall back to frame-based
            recipe = compute_timestamps_from_micro_actions(recipe, scene_log)
            # Also compute frame indices for backward compatibility
            recipe = compute_frame_indices_from_micro_actions(recipe, scene_log)
        else:
            recipe = compute_frame_indices_from_micro_actions(recipe, scene_log)
        
        return recipe

    def _build_prompt(
        self,
        scene_log: SceneLog,
        video_info: VideoInfo,
        transcript: str | None = None,
        use_timestamps: bool = True
    ) -> str:
        """Create the instruction prompt for recipe generation."""
        # Format scene descriptions for the prompt
        scene_text = self._format_scene_log(scene_log)
        
        transcript_section = ""
        if transcript:
            transcript_section = f"""
Audio Transcript: {transcript}

Use the transcript to cross-reference ingredient amounts, clarify steps, and add any missing details.
"""
        else:
            transcript_section = """
(No audio transcript available)
"""

        # Format micro-actions timeline (with timestamps if available)
        micro_actions_text = self._format_micro_actions(scene_log, use_timestamps)

        # Keyframe/transcript amount readings (empty string when there are none)
        observations_section = self._format_observations(scene_log)

        return f"""You are a forensic video analyst building a factual record of what a cooking video showed.

This is the GROUNDING pass. Your ONLY job is fidelity: report what the camera saw, grouped into
coherent chunks, and map each chunk back to the exact micro-actions it came from. A later pass will
turn this into a teachable recipe - so do NOT pad, explain, teach, or invent anything here. If the
video didn't show it, it does not belong in your output.

INPUT DATA:
Video Title: {video_info.title}
Video Description: {video_info.description or "Not provided"}
Video Duration: {video_info.duration_seconds} seconds

TRANSCRIPT:
{transcript_section}

SCENE ANALYSIS:
{scene_text}

MICRO-ACTIONS TIMELINE:
(Format: ID | Timestamp | Duration | Action | Entity | Stated Amount | State Change)
{micro_actions_text}
{observations_section}
YOUR TASK:
1. Identify the dish, and write a short search query for it (`dish_query`) - the next pass uses this
   to look up published recipes for the same dish.
2. List the ingredients the video actually shows, with whatever amounts are visible or stated.
3. List the tools and appliances the video actually shows.
4. Group the micro-actions into logical, CHRONOLOGICAL steps.

GROUPING RULES:
1. **Temporal Contiguity is King:** Only group micro-actions that happen closely together in time.
2. **The "Set Aside" Rule:** If an ingredient is handled (e.g., "sear chicken"), set aside while other
   things happen, then handled again later, THESE MUST BE TWO SEPARATE STEPS. Do not merge them.
3. **Linear Flow:** Step 1 must happen before Step 2.
4. **Clip Tightness:** A step's span is the start of its first micro-action to the end of its last.
5. **Every step must cite micro_action_ids.** A step with no video actions behind it does not belong
   in this pass.

OUTPUT FORMAT:
Return a valid JSON object with this exact structure:

{{
    "reasoning": "Briefly explain how you grouped the timeline.",
    "title": "Recipe name",
    "dish_query": "mushroom and egg quesadilla",
    "description": "Brief description",
    "servings": 4,
    "prep_time_minutes": 15,
    "cook_time_minutes": 30,
    "cuisine": "Italian",
    "tags": ["dinner", "pasta"],
    "ingredients": [
        {{"name": "ingredient", "quantity": 2.0, "unit": "cups", "preparation": "diced"}},
        {{"name": "salt", "quantity": null, "unit": null, "amount_text": "to taste"}}
    ],
    "tools": [
        {{"name": "Nonstick skillet"}}
    ],
    "steps": [
        {{
            "order": 1,
            "title": "Action Verb + Noun (e.g., 'Sear the Chicken')",
            "instruction": "What the video shows happening, plainly stated.",
            "duration_minutes": 5,
            "micro_action_ids": [3, 4, 5]
        }}
    ],
    "nutrition_estimates": {{ "calories": 450, "protein": 25, "carbs": 50, "fats": 15 }}
}}

CONSTRAINTS:
- **Ingredients:** Report amounts exactly as evidenced. If the video, transcript, on-screen text or a
  keyframe reading states a measurable amount, set "quantity" (positive number) + "unit". If the
  amount is honestly vague or never shown, set "quantity": null and put the vague wording in
  "amount_text" (e.g. "to taste", "a splash"). NEVER invent a number for an amount no source stated.
- **Ingredient names:** Be as specific as the evidence allows. Prefer a stated/labeled name
  ("smoked paprika") over a visual guess ("red powder"). If the identity is genuinely unclear,
  keep the honest generic name ("red spice powder") - the next pass resolves it.
- **Tools:** Only what is visible on camera. The next pass adds the ones the video cut away from.
- **Steps:** "micro_action_ids" must be a list of integers from the timeline above.
- **Noise:** Ignore micro-actions labeled "no relevant cooking action".

Return ONLY the JSON object.
"""

    def _seed_tools_from_scene_log(self, recipe: Recipe, scene_log: SceneLog) -> Recipe:
        """Add any tool/appliance the VLM saw that the LLM left off its tools list.

        The VLM already labels entities as ingredient/tool/appliance, and until now that
        classification was only ever printed into a prompt and thrown away.
        """
        known = {tool.name.strip().lower() for tool in recipe.tools}

        for scene in scene_log.scenes:
            for entity in scene.entities:
                if entity.type not in ("tool", "appliance"):
                    continue
                if entity.name.strip().lower() in known:
                    continue
                known.add(entity.name.strip().lower())
                recipe.tools.append(Tool(name=entity.name, provenance="video"))

        return recipe

    def _format_scene_log(self, scene_log: SceneLog) -> str:
        """Format scene log into readable text for the prompt."""
        lines = []
        
        # If we have a summary in metadata (from direct video analysis), show it first
        if scene_log.scenes and scene_log.scenes[0].metadata.get("summary"):
            lines.append(f"Summary: {scene_log.scenes[0].metadata['summary']}")
            
        for i, scene in enumerate(scene_log.scenes):
            if len(scene_log.scenes) > 1:
                lines.append(f"\n--- Scene {i+1} (Frame {scene.frame_index}) ---")
            
            if scene.entities:
                lines.append("Entities identified:")
                for entity in scene.entities:
                    qty = f" ({entity.quantity})" if entity.quantity else ""
                    state = f" [{entity.state}]" if entity.state else ""
                    lines.append(f"  - {entity.name} ({entity.type}){qty}{state}")
            
            if scene.state_changes:
                lines.append("State transformations observed:")
                for change in scene.state_changes:
                    from_state = f"{change.from_state} → " if change.from_state else ""
                    lines.append(f"  - {change.entity}: {from_state}{change.to_state}")
        
        return "\n".join(lines)
    
    def _format_micro_actions(self, scene_log: SceneLog, use_timestamps: bool = True) -> str:
        """Format micro-actions into a timeline for precise clip mapping."""
        all_actions = scene_log.get_all_micro_actions()
        
        if not all_actions:
            return "(No micro-actions extracted - using scene-based mapping)"
        
        # Check if we have timestamp data
        has_timestamps = any(a.timestamp_seconds is not None for a in all_actions)
        
        if has_timestamps and use_timestamps:
            lines = ["ID  | Timestamp | Duration | Action | Entity | Stated Amount | State Change"]
            lines.append("-" * 125)

            for action in all_actions:
                state_change = ""
                if action.state_before and action.state_after:
                    state_change = f"{action.state_before} → {action.state_after}"
                elif action.state_after:
                    state_change = f"→ {action.state_after}"

                # No truncation: cutting entity/action text corrupts spice names
                # ("kashmiri chili powder" -> "kashmiri chi") before the LLM reads them.
                entity = action.entity or ""
                stated = action.stated_amount or "-"
                ts = f"{action.timestamp_seconds:.1f}s" if action.timestamp_seconds is not None else "?"
                duration = f"{action.duration_seconds:.1f}s" if action.duration_seconds else "-"

                lines.append(
                    f"{action.id:3} | {ts:>9} | {duration:>8} | {action.action} | {entity} | {stated} | {state_change}"
                )
        else:
            # Legacy frame-based format
            lines = ["ID    | Frame | Action                        | Entity      | State Change"]
            lines.append("-" * 80)
            
            for action in all_actions:
                state_change = ""
                if action.state_before and action.state_after:
                    state_change = f"{action.state_before} → {action.state_after}"
                elif action.state_after:
                    state_change = f"→ {action.state_after}"
                
                entity = action.entity or ""
                lines.append(
                    f"{action.id:5} | {action.precise_frame_index:5.2f} | {action.action} | {entity} | {state_change}"
                )
        
        return "\n".join(lines)

    def _format_observations(self, scene_log: SceneLog) -> str:
        """Format keyframe/transcript amount readings for the prompt.

        These are the high-res reads the low-res temporal pass can't make:
        on-screen captions, jar labels, spoken amounts. Empty string when none.
        """
        observations = scene_log.get_all_observations()
        if not observations:
            return ""

        lines = [
            "",
            "AMOUNT / LABEL READINGS (from high-res keyframes and the transcript):",
            "Use these to name ingredients precisely and to fill in quantities. They are evidence,",
            "same as the timeline - an amount read here counts as 'shown by the video'.",
        ]
        for obs in observations:
            ts = f"{obs.timestamp_seconds:.1f}s" if obs.timestamp_seconds is not None else "?"
            parts = [f"[{ts}]"]
            if obs.ingredient:
                parts.append(obs.ingredient)
            if obs.stated_amount:
                parts.append(f"amount: {obs.stated_amount}")
            if obs.on_screen_text:
                parts.append(f'on-screen text: "{obs.on_screen_text}"')
            parts.append(f"(source: {obs.source})")
            lines.append("  - " + " | ".join(parts))
        lines.append("")
        return "\n".join(lines)

    def _parse_response(self, content: str, video_info: VideoInfo) -> Recipe:
        """Extract JSON from the LLM response and convert to Recipe."""
        # Try to extract JSON from markdown code blocks first
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If it's raw JSON - find the outermost braces
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                json_str = content[start:end]
            else:
                raise ValueError(f"Could not find JSON in response: {content[:200]}...")
        
        # Parse the JSON
        data = json.loads(json_str)
        
        # Add source URL from video info
        data["source_url"] = video_info.url
        
        # Provide defaults for required fields if missing
        if not data.get("title") or not isinstance(data.get("title"), str):
            data["title"] = video_info.title or "Untitled Recipe"
        
        # Ensure ingredients and steps are lists (even if empty)
        if not isinstance(data.get("ingredients"), list):
            data["ingredients"] = []
        if not isinstance(data.get("steps"), list):
            data["steps"] = []
        if not isinstance(data.get("tools"), list):
            data["tools"] = []
        
        # Handle potential nested nutrition object
        if "nutrition_estimates" in data:
            for k, v in data["nutrition_estimates"].items():
                if k not in data:
                    data[k] = v
        
        if "nutrition" in data and isinstance(data["nutrition"], dict):
            for k, v in data["nutrition"].items():
                if k not in data:
                    data[k] = v
            
        # Normalize ingredient amounts WITHOUT fabricating them. A vague amount the
        # model honestly reported stays vague (quantity=None + amount_text) instead
        # of being silently defaulted to 1.0 with a keyword-guessed unit - that
        # destroyed the vagueness signal and mislabeled a guess as provenance "video".
        data["ingredients"] = [coerce_ingredient_amount(ing) for ing in data.get("ingredients", [])]
        
        # Validate and default servings (required field)
        if data.get("servings") is None or not isinstance(data.get("servings"), int) or data.get("servings") <= 0:
            data["servings"] = 1
        
        # Convert to Recipe (Pydantic handles validation)
        return Recipe(**data)

