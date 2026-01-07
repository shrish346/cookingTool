from __future__ import annotations

import os
import json
import re

from openai import OpenAI
from dotenv import load_dotenv

from ..downloaders.base import VideoInfo
from ..schemas import Recipe, SceneLog, compute_frame_indices_from_micro_actions, compute_timestamps_from_micro_actions

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
            lines = ["ID  | Timestamp | Duration | Action                        | Entity      | State Change"]
            lines.append("-" * 95)
            
            for action in all_actions:
                state_change = ""
                if action.state_before and action.state_after:
                    state_change = f"{action.state_before} → {action.state_after}"
                elif action.state_after:
                    state_change = f"→ {action.state_after}"
                
                entity = action.entity or ""
                ts = f"{action.timestamp_seconds:.1f}s" if action.timestamp_seconds is not None else "?"
                duration = f"{action.duration_seconds:.1f}s" if action.duration_seconds else "-"
                
                lines.append(
                    f"{action.id:3} | {ts:>9} | {duration:>8} | {action.action[:30]:<30} | {entity[:12]:<12} | {state_change}"
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
                    f"{action.id:5} | {action.precise_frame_index:5.2f} | {action.action[:30]:<30} | {entity[:12]:<12} | {state_change}"
                )
        
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
        
        # Handle potential nested nutrition object
        if "nutrition_estimates" in data:
            for k, v in data["nutrition_estimates"].items():
                if k not in data:
                    data[k] = v
        
        if "nutrition" in data and isinstance(data["nutrition"], dict):
            for k, v in data["nutrition"].items():
                if k not in data:
                    data[k] = v
            
        # Validate and fix ingredients - ensure all have quantity and unit
        validated_ingredients = []
        for ing in data.get("ingredients", []):
            # Ensure quantity exists and is valid
            if "quantity" not in ing or ing["quantity"] is None:
                ing["quantity"] = 1.0  # Default to 1 if missing
            elif isinstance(ing["quantity"], str):
                # Try to extract number from string (e.g., "2 cups" -> 2.0)
                match = re.search(r"(\d+\.?\d*)", ing["quantity"])
                if match:
                    ing["quantity"] = float(match.group(1))
                else:
                    ing["quantity"] = 1.0
            elif not isinstance(ing["quantity"], (int, float)) or ing["quantity"] <= 0:
                ing["quantity"] = 1.0  # Fix invalid quantities
            
            # Ensure unit exists
            if "unit" not in ing or not ing.get("unit"):
                # Try to infer unit from ingredient name or use default
                name_lower = ing.get("name", "").lower()
                if any(word in name_lower for word in ["salt", "pepper", "spice", "herb"]):
                    ing["unit"] = "tsp"
                elif any(word in name_lower for word in ["sauce", "oil", "vinegar", "milk", "cream"]):
                    ing["unit"] = "cup"
                elif any(word in name_lower for word in ["cheese", "meat", "chicken", "beef"]):
                    ing["unit"] = "oz"
                else:
                    ing["unit"] = "piece"  # Default fallback
            
            validated_ingredients.append(ing)
        
        data["ingredients"] = validated_ingredients
        
        # Validate and default servings (required field)
        if data.get("servings") is None or not isinstance(data.get("servings"), int) or data.get("servings") <= 0:
            data["servings"] = 1
        
        # Convert to Recipe (Pydantic handles validation)
        return Recipe(**data)

