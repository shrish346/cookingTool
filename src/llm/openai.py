from __future__ import annotations

import os
import json
import re

from openai import OpenAI
from dotenv import load_dotenv

from ..downloaders.base import VideoInfo
from ..schemas import Recipe, SceneLog, compute_frame_indices_from_micro_actions

load_dotenv()


class OpenAIAdapter:
    """Adapter for OpenAI's language models (GPT-4, GPT-3.5)."""

    def __init__(self, model: str = "gpt-4.1"):
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
        transcript: str | None = None
    ) -> Recipe:
        """
        Generate a structured recipe from accumulated scene descriptions.
        
        Args:
            scene_log: Accumulated scene descriptions from VLM analysis
            video_info: Metadata about the video (title, description, etc.)
            transcript: Optional audio transcript from the video
            
        Returns:
            A validated Recipe object
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
        
        # Post-process: compute frame indices from micro_action_ids
        recipe = compute_frame_indices_from_micro_actions(recipe, scene_log)
        
        return recipe

    def _build_prompt(
        self,
        scene_log: SceneLog,
        video_info: VideoInfo,
        transcript: str | None = None
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

        # Format micro-actions timeline
        micro_actions_text = self._format_micro_actions(scene_log)

        return f"""You are a professional chef analyzing scene descriptions from a cooking video to create a complete, well-formatted recipe.

Video Title: {video_info.title}
Video Description: {video_info.description or "Not provided"}
{transcript_section}

Scene Descriptions from Video Analysis:
{scene_text}

MICRO-ACTIONS TIMELINE (Reference for grouping actions into steps):
{micro_actions_text}

Your task:
1. Cross-reference all ingredients across scenes to build a complete ingredient list
2. GROUP related micro-actions into logical recipe steps (e.g., "add butter", "add garlic", "stir" → "Sauté the Aromatics")
3. Infer quantities, units, and preparation methods from the scene descriptions
4. Format everything into a professional recipe

Return your response as a JSON object with this exact structure:

{{
    "reasoning": "Explain how you built the recipe from the scene descriptions",
    "title": "Recipe name",
    "description": "Brief description of the dish",
    "servings": 4,
    "prep_time_minutes": 15,
    "cook_time_minutes": 30,
    "cusine": "Italian",
    "tags": ["dinner", "pasta", "vegetarian"],
    "ingredients": [
        {{"name": "ingredient name", "quantity": 2.0, "unit": "cups", "preparation": "diced"}}
    ],
    "steps": [
        {{
            "order": 1,
            "title": "Short Step Title",
            "instruction": "Detailed step description",
            "duration_minutes": 5,
            "tips": ["optional tip"],
            "micro_action_ids": [500, 501, 502],
            "has_video_clip": true
        }}
    ],
    "calories": 450,
    "protein": 25,
    "carbs": 50,
    "fats": 15
}}

Rules:
- Cross-reference ingredients: if "chopped onions" appears, make sure "onions" is in the ingredients list
- Resolve temporal ordering: use step_number from temporal_actions, or infer from frame_index
- CRITICAL: Every ingredient MUST have "quantity" (a positive number) and "unit" (a string like "cups", "tbsp", "oz", "pieces", etc.)
  - If quantity is not visible, estimate based on typical recipe amounts or use 1.0
  - If unit is unclear, use common units: "cups" for liquids/grains, "tbsp" for small amounts, "oz" for weights, "pieces" for whole items
  - Examples: {{"name": "salt", "quantity": 1.0, "unit": "tsp"}}, {{"name": "onion", "quantity": 1.0, "unit": "piece"}}
- Infer quantities from state_changes and temporal_actions (e.g., "pouring 200ml milk" → quantity: 200, unit: "ml")
- Combine similar steps if they appear in multiple scenes
- For optional fields (prep_time_minutes, cook_time_minutes, calories, etc.), you can omit if unknown

MICRO-ACTION GROUPING (CRITICAL):
- Each step MUST have "title" (short 2-4 word title like "Boil the Pasta")
- Each step MUST group related micro-actions using "micro_action_ids" (list of IDs from the timeline above)
- Group actions that belong together semantically (e.g., adding and stirring ingredients for one component)
- Keep steps focused - don't group unrelated actions just because they're sequential
- "has_video_clip": Set to false if this step has NO matching micro-actions (e.g., "let rest for 10 min", "preheat oven")
  - For steps with no video, use an empty list: "micro_action_ids": []

- Return ONLY the JSON object, no other text"""

    def _format_scene_log(self, scene_log: SceneLog) -> str:
        """Format scene log into readable text for the prompt."""
        lines = []
        for i, scene in enumerate(scene_log.scenes):
            lines.append(f"\n--- Scene {i+1} (Frame {scene.frame_index}) ---")
            
            if scene.entities:
                lines.append("Entities:")
                for entity in scene.entities:
                    qty = f" ({entity.quantity})" if entity.quantity else ""
                    state = f" [{entity.state}]" if entity.state else ""
                    lines.append(f"  - {entity.name} ({entity.type}){qty}{state}")
            
            if scene.state_changes:
                lines.append("State Changes:")
                for change in scene.state_changes:
                    from_state = f"{change.from_state} → " if change.from_state else ""
                    lines.append(f"  - {change.entity}: {from_state}{change.to_state}")
            
            if scene.temporal_actions:
                lines.append("Actions:")
                for action in scene.temporal_actions:
                    step = f"Step {action.step_number}: " if action.step_number else ""
                    lines.append(f"  - {step}{action.action_description}")
                    if action.entities_involved:
                        lines.append(f"    (involves: {', '.join(action.entities_involved)})")
        
        return "\n".join(lines)
    
    def _format_micro_actions(self, scene_log: SceneLog) -> str:
        """Format micro-actions into a timeline for precise clip mapping."""
        all_actions = scene_log.get_all_micro_actions()
        
        if not all_actions:
            return "(No micro-actions extracted - using scene-based mapping)"
        
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
        
        # Validate and fix ingredients - ensure all have quantity and unit
        validated_ingredients = []
        for ing in data.get("ingredients", []):
            # Ensure quantity exists and is valid
            if "quantity" not in ing or ing["quantity"] is None:
                ing["quantity"] = 1.0  # Default to 1 if missing
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

