from __future__ import annotations

import os
import json
import re
import time
import warnings
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv

from ..downloaders.base import VideoInfo
from ..schemas import Recipe, SceneDescription

load_dotenv()


class OpenRouterAdapter:
    """Adapter for OpenRouter's vision-language models."""

    def __init__(self, model: str = "qwen/qwen2.5-vl-72b-instruct"):
        self._model = model
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model

    def analyze_recipe(
        self, 
        video_info: VideoInfo, 
        frames: list[str],
        transcript: str | None = None
    ) -> Recipe:
        """
        Analyze video frames and return a structured Recipe.
        
        .. deprecated:: 2.0
            This method is deprecated. Use analyze_scenes() followed by LLM recipe generation instead.
            Kept for backward compatibility.
        
        Args:
            video_info: Metadata about the video (title, description, etc.)
            frames: List of base64-encoded JPEG images
            transcript: Optional audio transcript from the video
            
        Returns:
            A validated Recipe object
        """
        warnings.warn(
            "analyze_recipe() is deprecated. Use analyze_scenes() followed by LLM recipe generation instead.",
            DeprecationWarning,
            stacklevel=2
        )
        messages = self._build_messages(video_info, frames, transcript)
        
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=4096,
            temperature=0.3,  # Lower = more deterministic
        )
        
        content = response.choices[0].message.content
        return self._parse_response(content, video_info)
    
    def analyze_scenes(
        self,
        video_info: VideoInfo,
        frames: list[str],
        transcript: str | None = None,
        chunk_size: int = 12,
        max_workers: int = 6
    ) -> list[SceneDescription]:
        """
        Analyze video frames and return structured scene descriptions.
        
        Processes frames in chunks in parallel for faster analysis.
        
        Args:
            video_info: Metadata about the video (title, description, etc.)
            frames: List of base64-encoded JPEG images
            transcript: Optional audio transcript from the video
            chunk_size: Number of frames to process in each batch (default 12 for ~3 chunks per 36 frames)
            max_workers: Maximum number of parallel API calls (default 4)
            
        Returns:
            List of SceneDescription objects, one per chunk
        """
        # Build list of chunks to process
        chunks = []
        for start_idx in range(0, len(frames), chunk_size):
            end_idx = min(start_idx + chunk_size, len(frames))
            chunk_indices = list(range(start_idx, end_idx))
            chunk_frames = [frames[i] for i in chunk_indices]
            chunks.append((start_idx, chunk_indices, chunk_frames))
        
        print(f"      Processing {len(frames)} frames in {len(chunks)} chunks (parallel, max {max_workers} workers)...")
        
        scene_descriptions = []
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(
                    self.analyze_frame_batch,
                    video_info,
                    chunk_frames,
                    transcript,
                    start_idx,
                    chunk_indices
                ): (start_idx, chunk_indices)
                for start_idx, chunk_indices, chunk_frames in chunks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                start_idx, chunk_indices = future_to_chunk[future]
                try:
                    scene = future.result()
                    scene_descriptions.append(scene)
                except Exception as e:
                    # Continue with partial data on failure
                    warnings.warn(f"Failed to process chunk starting at frame {start_idx}: {e}", RuntimeWarning)
        
        # Sort by frame_index to maintain temporal order
        scene_descriptions.sort(key=lambda s: s.frame_index)
        
        return scene_descriptions
    
    def analyze_frame_batch(
        self,
        video_info: VideoInfo,
        frames: list[str],
        transcript: str | None = None,
        frame_index: int = 0,
        frame_indices: Optional[list[int]] = None,
        max_retries: int = 3
    ) -> SceneDescription:
        """
        Analyze a batch of frames and return a single scene description.
        
        Args:
            video_info: Metadata about the video
            frames: List of base64-encoded JPEG images (typically a chunk)
            transcript: Optional audio transcript
            frame_index: Starting frame index (for single frame) or chunk start index
            frame_indices: All frame indices if this represents a chunk
            max_retries: Maximum number of retry attempts on failure
            
        Returns:
            A single SceneDescription representing the batch
        """
        messages = self._build_scene_messages(video_info, frames, transcript, frame_index, frame_indices)
        
        last_error = None
        had_retry = False
        
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.2,  # Lower temperature for more consistent scene descriptions
                )
                
                content = response.choices[0].message.content
                result = self._parse_scene_response(content, frame_index, frame_indices)
                
                # If we had to retry, log success
                if had_retry:
                    print(f"[VLM] ✓ Frame {frame_index} processed successfully after {attempt + 1} attempts")
                
                return result
                
            except RateLimitError as e:
                # Rate limited - wait and retry with exponential backoff
                had_retry = True
                wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
                print(f"[VLM] Rate limited on frame {frame_index}, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                last_error = e
                
            except APIError as e:
                # API error (500, 502, etc.) - retry with backoff
                had_retry = True
                wait_time = (2 ** attempt) * 1  # 1s, 2s, 4s
                print(f"[VLM] API error on frame {frame_index}: {e}, retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                last_error = e
                
            except Exception as e:
                # JSON parsing or other error - retry once
                if attempt < max_retries - 1:
                    had_retry = True
                    print(f"[VLM] Failed to parse frame {frame_index}, retrying: {e}")
                    time.sleep(1)
                    last_error = e
                else:
                    raise
        
        # All retries exhausted
        print(f"[VLM] ✗ Frame {frame_index} FAILED after {max_retries} attempts")
        raise last_error or Exception(f"Failed to process frame {frame_index} after {max_retries} attempts")

    def _build_prompt(self, video_info: VideoInfo, transcript: str | None = None) -> str:
        """Create the instruction prompt for the VLM."""
        transcript_section = ""
        if transcript:
            transcript_section = f"""
Audio Transcript: {transcript}

Use BOTH the visual frames AND the audio transcript to extract the recipe. The transcript often contains spoken instructions, ingredient amounts, and tips that may not be visible in the frames.
"""
        else:
            transcript_section = """
(No audio transcript available - extract recipe from visual frames only)
"""

        return f"""You are analyzing frames from a cooking video to extract a recipe.

Video Title: {video_info.title}
Video Description: {video_info.description or "Not provided"}
{transcript_section}
Analyze carefully and extract the complete recipe. Return your response as a JSON object with this exact structure:

{{
    "reasoning": "Explain your thought process here: what you observed in the frames, how you identified ingredients and steps, any uncertainties or assumptions you made",
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
        {{"order": 1, "instruction": "Step description", "duration_minutes": 5, "tips": ["optional tip"]}}
    ],
    "calories": 450,
    "protein": 25,
    "carbs": 50,
    "fats": 15
}}

Rules:
- reasoning should describe what you see in the frames and how you deduced the recipe
- quantity must be a positive number (use decimals like 0.5 for "half")
- order must start at 1 and increment
- Include ALL ingredients and steps you can identify from the video
- If you can't determine a value, omit that field (don't guess wildly)
- Return ONLY the JSON object, no other text"""

    def _build_messages(
        self, 
        video_info: VideoInfo, 
        frames: list[str],
        transcript: str | None = None
    ) -> list[dict]:
        """
        Build the multi-modal message array for the API.
        
        Combines the instruction prompt with all video frames.
        """
        # Start with the text instruction
        content = [
            {"type": "text", "text": self._build_prompt(video_info, transcript)}
        ]
        
        # Add each frame as an image
        for frame_b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}"
                }
            })
        
        return [{"role": "user", "content": content}]

    def _parse_response(self, content: str, video_info: VideoInfo) -> Recipe:
        """
        Extract JSON from the LLM response and convert to Recipe.
        
        LLMs sometimes wrap JSON in markdown code blocks or add extra text.
        This method handles those cases.
        """
        # Try to extract JSON from markdown code blocks first
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if json_match:
            json_str = json_match.group(1)
        else:
            # if it's raw JSON - find the outermost braces
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
        
        # Provide defaults for required fields if missing (handles non-cooking videos or incomplete responses)
        if not data.get("title") or not isinstance(data.get("title"), str):
            data["title"] = video_info.title or "Untitled Recipe"
        
        # Ensure ingredients and steps are lists (even if empty)
        if not isinstance(data.get("ingredients"), list):
            data["ingredients"] = []
        if not isinstance(data.get("steps"), list):
            data["steps"] = []
        
        # Validate and default servings (required field)
        if data.get("servings") is None or not isinstance(data.get("servings"), int) or data.get("servings") <= 0:
            data["servings"] = 1  # Default to 1 serving if missing/invalid
        
        # Convert to Recipe (Pydantic handles validation)
        return Recipe(**data)
    
    def _build_scene_prompt(
        self,
        video_info: VideoInfo,
        transcript: str | None = None,
        frame_index: int = 0,
        frame_indices: Optional[list[int]] = None
    ) -> str:
        """Create the instruction prompt for scene description analysis."""
        # NOTE: transcript is accepted but NOT used - VLM should only analyze visual content
        # The transcript is passed to the LLM later for recipe generation
        
        frame_info = ""
        if frame_indices and len(frame_indices) > 1:
            frame_info = f"Analyzing frames {frame_indices[0]} through {frame_indices[-1]} (chunk of {len(frame_indices)} frames)."
        else:
            frame_info = f"Analyzing frame {frame_index}."

        return f"""You are analyzing cooking video frames to extract structured scene descriptions.

CRITICAL RULES:
- Only describe actions you can VISUALLY SEE happening in the provided frames
- Do NOT infer or assume actions that might have happened but are not visible
- If frames show a static scene (finished dish, ingredients laid out), return empty micro_actions list

Video Title: {video_info.title}
Video Description: {video_info.description or "Not provided"}
{frame_info}

Focus on FOUR key pillars:

1. ENTITY IDENTIFICATION: List all ingredients, tools (e.g., "cast iron skillet", "chef's knife"), and appliances (e.g., "air fryer", "oven") visible in this frame/chunk.
   CRITICAL RULES FOR ENTITY TYPES:
   - Entity type MUST be EXACTLY one of these three strings: "ingredient", "tool", or "appliance"
   - DO NOT use "dish", "container", "plate", "bowl", "pan", "pot", or any other type
   - If you see a finished dish, focus on its ingredients, not the dish itself
   - If you see a plate/bowl/pan, classify it as "tool" (cooking vessel)
   - If you see an oven/stove/microwave, classify it as "appliance"
   - Food items are always "ingredient"
   - Cooking utensils are always "tool"

2. STATE CHANGES: Identify any transformations or state changes. Did onions go from "raw" to "translucent"? Did liquid go from "cold" to "boiling" or "simmering"? Did dough go from "sticky" to "smooth"?
   CRITICAL: Every state_change MUST include "frame_index" set to {frame_index} (the current frame index).

3. TEMPORAL ACTIONS: Describe what action is being performed (e.g., "chopping onions", "pouring 200ml milk", "adding 2 eggs to flour mixture"). If you can determine a step number, include it.
   CRITICAL: Every temporal_action MUST include "frame_index" set to {frame_index} (the current frame index).

4. MICRO-ACTIONS (MOST IMPORTANT): Break down ALL individual atomic cooking actions with PRECISE timing.
   - Each micro-action is a SINGLE, ATOMIC action like "add salt", "stir pan", "flip chicken", "pour oil"
   - ONLY include actions you can VISUALLY SEE happening in the frames - do NOT infer from transcript
   - relative_position MUST reflect WHICH FRAME in the chunk shows the action:
     * If you receive {len(frame_indices) if frame_indices else 12} frames and an action is visible in the 1st frame → 0.0
     * If an action is visible in the middle frames → 0.5
     * If an action is visible in the last frame → 1.0
     * IMPORTANT: If the first few frames show intro/static content and cooking starts at frame 3, the first action should have relative_position ~0.25 (3/12), NOT 0.0
   - If ANY frames show intro content, title cards, or finished dish (no active cooking), add a single micro-action:
     {{"action": "no relevant cooking action", "frame_index": {frame_index}, "relative_position": <position where intro ends>, "entity": null, "state_before": null, "state_after": null}}
   - This enables precise video clip extraction, so BE GRANULAR and PRECISE with timing based on actual frame positions
   - If ALL frames show static content with no cooking, output an empty micro_actions list

Return your response as a JSON object with this exact structure:

{{
    "entities": [
        {{"name": "onion", "type": "ingredient", "quantity": "1 large", "state": "raw", "confidence": 0.95}},
        {{"name": "chef's knife", "type": "tool", "confidence": 0.9}}
    ],
    "state_changes": [
        {{"entity": "onion", "from_state": "raw", "to_state": "chopped", "frame_index": {frame_index}, "confidence": 0.85}}
    ],
    "temporal_actions": [
        {{"step_number": 1, "action_description": "chopping onions", "frame_index": {frame_index}, "entities_involved": ["onion", "chef's knife"]}}
    ],
    "micro_actions": [
        {{"action": "no relevant cooking action", "frame_index": {frame_index}, "relative_position": 0.0, "entity": null, "state_before": null, "state_after": null}},
        {{"action": "add butter to pan", "frame_index": {frame_index}, "relative_position": 0.25, "entity": "butter", "state_before": "solid", "state_after": "melting"}},
        {{"action": "swirl butter around", "frame_index": {frame_index}, "relative_position": 0.4, "entity": "butter", "state_before": "melting", "state_after": "melted"}},
        {{"action": "add diced onions", "frame_index": {frame_index}, "relative_position": 0.6, "entity": "onions", "state_before": null, "state_after": "in pan"}},
        {{"action": "stir onions", "frame_index": {frame_index}, "relative_position": 0.8, "entity": "onions", "state_before": "raw", "state_after": "cooking"}}
    ],
    "metadata": {{"notes": "Additional observations or uncertainties"}}
}}

Rules:
- entities: List ALL visible ingredients, tools, and appliances
  - type MUST be EXACTLY "ingredient", "tool", or "appliance" - NO OTHER VALUES ALLOWED
  - If you see a dish/plate/bowl/pan/pot, classify it as "tool"
  - If you see a finished dish, focus on its ingredients, not the dish itself
- state_changes: Only include if you observe a transformation (can be empty list)
  - REQUIRED: Every state_change MUST have "frame_index" set to {frame_index}
- temporal_actions: Describe actions being performed (can be empty list if no clear action)
  - REQUIRED: Every temporal_action MUST have "frame_index" set to {frame_index}
- micro_actions: CRITICAL - Break down EVERY VISUALLY OBSERVED action into atomic steps
  - ONLY include actions you can SEE happening in the frames - NOT actions inferred from transcript
  - If ALL frames show a static scene with no action, return empty list: "micro_actions": []
  - If SOME frames show intro/outro/static content, add: {{"action": "no relevant cooking action", "relative_position": <where it ends>}}
  - action: Short, specific description (e.g., "add salt", "flip chicken", "pour oil")
  - frame_index: MUST be {frame_index}
  - relative_position: Float 0.0-1.0 based on WHICH FRAME shows the action (NOT just order of actions)
    * If this chunk has 12 frames and an action appears in frame 4, relative_position = 4/12 ≈ 0.33
  - entity: What is being acted on (optional but helpful)
  - state_before/state_after: State changes (optional but helpful)
- step_number: Only include if you can reasonably determine the sequence (can be null)
- confidence: Optional, between 0.0 and 1.0
- quantity: Optional, include if visible (e.g., "2 cups", "200ml", "3 eggs")
- Return ONLY the JSON object, no other text
- DO NOT include any entity with type other than "ingredient", "tool", or "appliance"
- DO NOT omit frame_index from state_changes, temporal_actions, or micro_actions"""
    
    def _build_scene_messages(
        self,
        video_info: VideoInfo,
        frames: list[str],
        transcript: str | None = None,
        frame_index: int = 0,
        frame_indices: Optional[list[int]] = None
    ) -> list[dict]:
        """Build the multi-modal message array for scene description analysis."""
        content = [
            {"type": "text", "text": self._build_scene_prompt(video_info, transcript, frame_index, frame_indices)}
        ]
        
        # Add each frame as an image
        for frame_b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}"
                }
            })
        
        return [{"role": "user", "content": content}]
    
    def _parse_scene_response(
        self,
        content: str,
        frame_index: int,
        frame_indices: Optional[list[int]] = None
    ) -> SceneDescription:
        """Extract JSON from the VLM response and convert to SceneDescription."""
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
                raise ValueError(f"Could not find JSON in scene response: {content[:200]}...")
        
        # Parse the JSON
        data = json.loads(json_str)
        
        # Ensure lists exist
        if not isinstance(data.get("entities"), list):
            data["entities"] = []
        if not isinstance(data.get("state_changes"), list):
            data["state_changes"] = []
        if not isinstance(data.get("temporal_actions"), list):
            data["temporal_actions"] = []
        if not isinstance(data.get("micro_actions"), list):
            data["micro_actions"] = []
        
        # Filter entities - ONLY allow valid types, discard invalid ones
        valid_types = {"ingredient", "tool", "appliance"}
        filtered_entities = []
        for entity in data.get("entities", []):
            entity_type = entity.get("type", "").lower()
            # Only include entities with valid types - discard all others
            if entity_type in valid_types:
                filtered_entities.append(entity)
            # Silently skip invalid types (don't warn, just filter them out)
        
        data["entities"] = filtered_entities
        
        # Ensure frame_index is set in all state_changes
        for state_change in data.get("state_changes", []):
            if state_change.get("frame_index") is None:
                state_change["frame_index"] = frame_index
        
        # Ensure frame_index is set in all temporal_actions
        for temporal_action in data.get("temporal_actions", []):
            if temporal_action.get("frame_index") is None:
                temporal_action["frame_index"] = frame_index
        
        # Ensure frame_index is set in all micro_actions and assign IDs
        for idx, micro_action in enumerate(data.get("micro_actions", [])):
            # Assign unique ID based on frame index and position
            micro_action["id"] = frame_index * 100 + idx  # e.g., frame 5 action 2 = 502
            if micro_action.get("frame_index") is None:
                micro_action["frame_index"] = frame_index
            # Ensure relative_position has a default
            if micro_action.get("relative_position") is None:
                micro_action["relative_position"] = 0.5  # Default to middle of chunk
            
            # Fix VLM outputting lists instead of strings for entity/state fields
            # (happens when multiple entities are involved in an action)
            for field in ["entity", "state_before", "state_after"]:
                if isinstance(micro_action.get(field), list):
                    micro_action[field] = ", ".join(str(v) for v in micro_action[field])
        
        # Add frame_index and frame_indices
        data["frame_index"] = frame_index
        if frame_indices:
            data["frame_indices"] = frame_indices
        
        # Ensure metadata exists
        if "metadata" not in data:
            data["metadata"] = {}
        
        # Convert to SceneDescription (Pydantic handles validation)
        return SceneDescription(**data)