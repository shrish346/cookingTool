"""
LLM adapter for timestamp-based recipe generation.

This takes micro-actions with timestamps and generates a recipe with
time-based video clip mappings.
"""
from __future__ import annotations

import os
import json
import re

from openai import OpenAI
from dotenv import load_dotenv

# Add paths to import from main project and local modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from schemas_timestamp import (
    RecipeTimestamp,
    VideoSceneLog,
    compute_timestamps_from_micro_actions,
)
from src.downloaders.base import VideoInfo

load_dotenv()


class TimestampLLMAdapter:
    """Adapter for OpenAI's language models with timestamp-based recipe generation."""

    def __init__(self, model: str = "gpt-4.1"):
        self._model = model
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model

    def generate_recipe(
        self,
        scene_log: VideoSceneLog,
        video_info: VideoInfo,
        transcript: str | None = None
    ) -> RecipeTimestamp:
        """
        Generate a structured recipe from video scene description with timestamps.
        
        Args:
            scene_log: Video scene description with timestamp-based micro-actions
            video_info: Metadata about the video
            transcript: Optional audio transcript
            
        Returns:
            A validated RecipeTimestamp object with time-based video mappings
        """
        prompt = self._build_prompt(scene_log, video_info, transcript)
        
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.7,
        )
        
        content = response.choices[0].message.content
        recipe = self._parse_response(content, video_info)
        
        # Post-process: compute timestamps from micro_action_ids
        recipe = compute_timestamps_from_micro_actions(recipe, scene_log)
        
        return recipe

    def _build_prompt(
        self,
        scene_log: VideoSceneLog,
        video_info: VideoInfo,
        transcript: str | None = None
    ) -> str:
        """Create the instruction prompt for recipe generation."""
        scene_text = self._format_scene_log(scene_log)
        micro_actions_text = self._format_micro_actions(scene_log)
        
        transcript_section = ""
        if transcript:
            transcript_section = f"""
Audio Transcript: {transcript}

Use the transcript to clarify ingredient amounts and add any details not visible in the video.
"""
        else:
            transcript_section = """
(No audio transcript available)
"""

        return f"""You are a professional chef and video editor analyzing a cooking video to create a structured recipe.

INPUT DATA:
Video Title: {video_info.title}
Video Description: {video_info.description or "Not provided"}
Video Duration: {video_info.duration_seconds} seconds

TRANSCRIPT:
{transcript_section}

SCENE ANALYSIS:
{scene_text}

MICRO-ACTIONS TIMELINE:
(Format: ID | Timestamp | Action Description. Note: Some actions may end with a [PREVIEW] tag.)
{micro_actions_text}

YOUR TASK:
1. Build a complete ingredient list.
2. Group micro-actions into logical, CHRONOLOGICAL recipe steps.
3. Ensure steps align with the video flow for accurate clip extraction.

[NEW] GROUPING RULES (CRITICAL):
1. **The Preview Filter (CRITICAL):** Scan the micro-actions for the `[PREVIEW]` tag. These represent the finished dish shown at the start of the video. **Do NOT include these IDs in any recipe step.** The recipe must start at the first micro-action *without* a preview tag.
2. **Temporal Contiguity is King:** Only group micro-actions that happen closely together in time.
3. **The "Set Aside" Rule:** If an ingredient is handled (e.g., "sear chicken"), then set aside while other things happen, and then handled again later, THESE MUST BE TWO SEPARATE STEPS. Do not merge them.
4. **Linear Flow:** Step 1 must happen before Step 2.
5. **Clip Tightness:** A step's duration is defined by the start of its first micro-action and the end of its last.

OUTPUT FORMAT:
Return a valid JSON object with this exact structure:

{{
    "reasoning": "Briefly explain how you handled the timeline, specifically where you determined the preview ended and cooking began.",
    "title": "Recipe name",
    "description": "Brief description",
    "servings": 4,
    "prep_time_minutes": 15,
    "cook_time_minutes": 30,
    "cuisine": "Italian",
    "tags": ["dinner", "pasta"],
    "ingredients": [
        {{"name": "ingredient", "quantity": 2.0, "unit": "cups", "preparation": "diced"}}
    ],
    "steps": [
        {{
            "order": 1,
            "title": "Action Verb + Noun (e.g., 'Sear the Chicken')",
            "instruction": "Detailed instruction for the user.",
            "duration_minutes": 5,
            "tips": ["optional tip"],
            "micro_action_ids": [3, 4, 5], 
            "has_video_clip": true
        }}
    ],
    "nutrition_estimates": {{ "calories": 450, "protein": 25, "carbs": 50, "fats": 15 }}
}}

CONSTRAINTS:
- **Ingredients:** "quantity" must be a number (use 0 if negligible/to taste). "unit" is required (use "count" or "to taste" if unclear).
- **Steps:** "micro_action_ids" must be a list of integers from the timeline. **Ensure NO IDs marked with [PREVIEW] are included.**
- **Video Clips:** If a step is purely instructional (e.g., "Preheat oven to 350") and has no visual action in the timeline, set "has_video_clip": false and "micro_action_ids": [].
- **Noise:** Ignore micro-actions labeled "no relevant cooking action".

Return ONLY the JSON object.
"""

    def _format_scene_log(self, scene_log: VideoSceneLog) -> str:
        """Format scene description for the prompt."""
        scene = scene_log.scene
        lines = []
        
        lines.append(f"Video Duration: {scene.video_duration_seconds}s")
        
        if scene.summary:
            lines.append(f"Summary: {scene.summary}")
        
        if scene.entities:
            lines.append("\nEntities Identified:")
            for entity in scene.entities:
                ts = f" (first seen: {entity.first_seen_timestamp}s)" if entity.first_seen_timestamp else ""
                qty = f" ({entity.quantity})" if entity.quantity else ""
                state = f" [{entity.state}]" if entity.state else ""
                lines.append(f"  - {entity.name} ({entity.type}){qty}{state}{ts}")
        
        if scene.state_changes:
            lines.append("\nState Changes:")
            for change in scene.state_changes:
                from_state = f"{change.from_state} → " if change.from_state else ""
                lines.append(f"  - {change.entity}: {from_state}{change.to_state} (at {change.timestamp_seconds}s)")
        
        return "\n".join(lines)

    def _format_micro_actions(self, scene_log: VideoSceneLog) -> str:
        """Format micro-actions timeline with timestamps."""
        actions = scene_log.get_all_micro_actions()
        
        if not actions:
            return "(No micro-actions extracted)"
        
        lines = ["ID  | Timestamp | Duration | Action                        | Entity      | State Change"]
        lines.append("-" * 95)
        
        for action in actions:
            state_change = ""
            if action.state_before and action.state_after:
                state_change = f"{action.state_before} → {action.state_after}"
            elif action.state_after:
                state_change = f"→ {action.state_after}"
            
            entity = action.entity or ""
            duration = f"{action.duration_seconds:.1f}s" if action.duration_seconds else "?"
            
            lines.append(
                f"{action.id:3} | {action.timestamp_seconds:7.1f}s | {duration:>8} | {action.action[:30]:<30} | {entity[:12]:<12} | {state_change}"
            )
        
        return "\n".join(lines)

    def _parse_response(self, content: str, video_info: VideoInfo) -> RecipeTimestamp:
        """Extract JSON from the LLM response and convert to RecipeTimestamp."""
        # Try to extract JSON from markdown code blocks first
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if json_match:
            json_str = json_match.group(1)
        else:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                json_str = content[start:end]
            else:
                raise ValueError(f"Could not find JSON in response: {content[:200]}...")
        
        data = json.loads(json_str)
        
        # Add source URL
        data["source_url"] = video_info.url
        data["video_duration_seconds"] = video_info.duration_seconds
        
        # Defaults
        if not data.get("title"):
            data["title"] = video_info.title or "Untitled Recipe"
        
        if not isinstance(data.get("ingredients"), list):
            data["ingredients"] = []
        if not isinstance(data.get("steps"), list):
            data["steps"] = []
        
        # Validate ingredients
        validated_ingredients = []
        for ing in data.get("ingredients", []):
            if "quantity" not in ing or ing["quantity"] is None:
                ing["quantity"] = 1.0
            elif not isinstance(ing["quantity"], (int, float)) or ing["quantity"] <= 0:
                ing["quantity"] = 1.0
            
            if "unit" not in ing or not ing.get("unit"):
                name_lower = ing.get("name", "").lower()
                if any(word in name_lower for word in ["salt", "pepper", "spice"]):
                    ing["unit"] = "tsp"
                elif any(word in name_lower for word in ["sauce", "oil", "milk"]):
                    ing["unit"] = "cup"
                else:
                    ing["unit"] = "piece"
            
            validated_ingredients.append(ing)
        
        data["ingredients"] = validated_ingredients
        
        if data.get("servings") is None or not isinstance(data.get("servings"), int) or data.get("servings") <= 0:
            data["servings"] = 1
        
        return RecipeTimestamp(**data)

