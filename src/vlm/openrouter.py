from __future__ import annotations

import os
import json
import re
import time
import base64
import subprocess
import tempfile
import warnings
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv

from ..downloaders.base import VideoInfo
from ..schemas import Recipe, SceneDescription, MicroAction, Entity, SceneLog
from ..processing.frames import FrameExtractor

load_dotenv()


def compress_video_for_api(
    video_path: str,
    max_size_mb: float = 15.0,
    target_width: int = 720,
    target_fps: int = 15
) -> tuple[str, float]:
    """
    Compress video to stay under API size limits.
    
    Args:
        video_path: Path to original video
        max_size_mb: Target max size in MB
        target_width: Target width (height scaled proportionally)
        target_fps: Target frames per second
        
    Returns:
        Tuple of (path_to_compressed_video, size_in_mb)
    """
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_fd)
    
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vf", f"scale={target_width}:-2,fps={target_fps}",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-an",  # No audio needed for VLM
            "-movflags", "+faststart",
            temp_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg compression failed: {result.stderr}")
        
        size_bytes = os.path.getsize(temp_path)
        size_mb = size_bytes / (1024 * 1024)
        
        # If still too large, compress more aggressively
        if size_mb > max_size_mb:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-vf", "scale=360:-2,fps=6",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "32",
                "-an",
                "-movflags", "+faststart",
                temp_path
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            size_bytes = os.path.getsize(temp_path)
            size_mb = size_bytes / (1024 * 1024)
        
        return temp_path, size_mb
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


class OpenRouterAdapter:
    """Adapter for OpenRouter's vision-language models."""

    def __init__(self, model: str = "google/gemini-2.0-flash-001"):
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

    def analyze_video_with_timestamps(
        self,
        video_info: VideoInfo,
        video_path: str,
        transcript: str | None = None,
        target_fps: float = 2.0,
        max_frames: int = 80,
        chunk_size: int = 16,
        max_workers: int = 4
    ) -> SceneLog:
        """
        Analyze a video using dense frame extraction with timestamps.
        
        This method extracts frames at a high rate (e.g., 2 FPS) and includes
        the exact timestamp for each frame, asking the VLM to map actions
        to specific timestamps.
        
        Args:
            video_info: Metadata about the video
            video_path: Path to the local video file
            transcript: Optional audio transcript
            target_fps: Frames per second to extract (default 2.0)
            max_frames: Maximum frames to extract
            chunk_size: Number of frames per VLM call
            max_workers: Parallel API calls
            
        Returns:
            SceneLog containing all scenes with timestamp-based micro-actions
        """
        # Extract frames with timestamps
        extractor = FrameExtractor(resize_width=512)
        frames_with_timestamps = extractor.extract_with_timestamps(
            video_path, target_fps=target_fps, max_frames=max_frames
        )
        
        print(f"      Extracted {len(frames_with_timestamps)} frames at {target_fps} FPS")
        if frames_with_timestamps:
            print(f"      Time range: 0.0s → {frames_with_timestamps[-1][0]:.1f}s")
        
        # Process frames in chunks
        all_micro_actions = []
        all_entities = []
        all_state_changes = []
        all_scenes = []
        
        # Build chunks
        chunks = []
        for i in range(0, len(frames_with_timestamps), chunk_size):
            chunk = frames_with_timestamps[i:i + chunk_size]
            chunks.append(chunk)
        
        print(f"      Processing {len(chunks)} chunks in parallel (max {max_workers} workers)...")
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(
                    self._analyze_timestamp_chunk,
                    video_info,
                    chunk,
                    transcript
                ): (i, chunk)
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_chunk):
                chunk_idx, chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    
                    # Create a scene description for this chunk
                    start_ts = chunk[0][0]
                    scene = SceneDescription(
                        frame_index=chunk_idx * chunk_size,
                        frame_indices=list(range(chunk_idx * chunk_size, chunk_idx * chunk_size + len(chunk))),
                        entities=[Entity(**e) for e in result.get("entities", [])],
                        micro_actions=[MicroAction(**ma) for ma in result.get("micro_actions", [])],
                        metadata={"start_timestamp": start_ts, "method": "dense_frames_with_timestamps"}
                    )
                    all_scenes.append(scene)
                    
                except Exception as e:
                    warnings.warn(f"Failed to process chunk {chunk_idx}: {e}", RuntimeWarning)
        
        # Sort scenes by frame_index
        all_scenes.sort(key=lambda s: s.frame_index)
        
        # Reassign unique IDs to all micro-actions across all scenes
        action_id = 0
        for scene in all_scenes:
            for ma in scene.micro_actions:
                ma.id = action_id
                action_id += 1
        
        return SceneLog(
            scenes=all_scenes,
            video_info={
                "title": video_info.title,
                "description": video_info.description,
                "url": video_info.url,
                "duration_seconds": video_info.duration_seconds
            }
        )

    def _analyze_timestamp_chunk(
        self,
        video_info: VideoInfo,
        frames_with_timestamps: list[tuple[float, str]],
        transcript: str | None = None,
        max_retries: int = 3
    ) -> dict:
        """
        Analyze a chunk of frames with timestamps.
        
        Returns dict with micro_actions, entities, state_changes lists.
        """
        messages = self._build_timestamp_chunk_messages(video_info, frames_with_timestamps, transcript)
        
        start_ts = frames_with_timestamps[0][0]
        end_ts = frames_with_timestamps[-1][0]
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.3,
                )
                
                content = response.choices[0].message.content
                return self._parse_timestamp_chunk_response(content, frames_with_timestamps)
                
            except RateLimitError as e:
                wait_time = (2 ** attempt) * 2
                print(f"[VLM] Rate limited ({start_ts:.1f}s-{end_ts:.1f}s), waiting {wait_time}s...")
                time.sleep(wait_time)
                last_error = e
                
            except APIError as e:
                wait_time = (2 ** attempt) * 1
                print(f"[VLM] API error ({start_ts:.1f}s-{end_ts:.1f}s): {e}, retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                last_error = e
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[VLM] Error ({start_ts:.1f}s-{end_ts:.1f}s): {e}, retrying...")
                    time.sleep(1)
                    last_error = e
                else:
                    raise
        
        raise last_error or Exception(f"Failed to analyze chunk {start_ts:.1f}s-{end_ts:.1f}s")

    def _build_timestamp_chunk_prompt(
        self,
        video_info: VideoInfo,
        frames_with_timestamps: list[tuple[float, str]],
        transcript: str | None = None
    ) -> str:
        """Create the prompt for analyzing a chunk of timestamped frames."""
        
        start_ts = frames_with_timestamps[0][0]
        end_ts = frames_with_timestamps[-1][0]
        frame_info = ", ".join([f"{ts:.1f}s" for ts, _ in frames_with_timestamps])
        
        transcript_section = ""
        if transcript:
            transcript_section = f"""
Audio Transcript: {transcript}

Use the transcript for context, but base TIMESTAMPS on what you SEE in frames.
"""

        return f"""Watch these cooking video frames carefully and extract ALL cooking actions you observe.

Video Title: {video_info.title}
{transcript_section}

FRAME TIMESTAMPS: {frame_info}
({len(frames_with_timestamps)} frames from {start_ts:.1f}s to {end_ts:.1f}s)

YOUR TASK: Identify cooking actions and map them to EXACT TIMESTAMPS from the frames.
Available timestamps: {frame_info}

This is a cooking video. Look for: cutting, chopping, stirring, adding ingredients, seasoning, flipping, cooking, plating, etc.

Return as JSON:

{{
    "entities": [
        {{"name": "chicken", "type": "ingredient", "quantity": "1 lb", "state": "raw", "confidence": 0.9}}
    ],
    "state_changes": [
        {{"entity": "chicken", "from_state": "raw", "to_state": "cooked", "frame_index": 0, "confidence": 0.8}}
    ],
    "micro_actions": [
        {{"action": "add oil to pan", "timestamp_seconds": {start_ts + 0.5}, "duration_seconds": 2.0, "frame_index": 0, "relative_position": 0.0, "entity": "oil", "state_before": null, "state_after": "in pan", "confidence": 0.9}},
        {{"action": "season chicken", "timestamp_seconds": {start_ts + 2.0}, "duration_seconds": 1.5, "frame_index": 0, "relative_position": 0.3, "entity": "chicken", "state_before": "raw", "state_after": "seasoned", "confidence": 0.9}}
    ]
}}

RULES:
- timestamp_seconds: MUST be one of the frame timestamps: {frame_info}
- duration_seconds: Estimate 1-5 seconds for most actions
- frame_index: Set to 0 (will be adjusted)
- relative_position: Position within this chunk (0.0-1.0)
- Only report VISUALLY OBSERVED actions
- Be GRANULAR: each small action is separate
- entity type: "ingredient", "tool", or "appliance" only
- If no cooking action visible, return empty micro_actions: []

Return ONLY JSON."""

    def _build_timestamp_chunk_messages(
        self,
        video_info: VideoInfo,
        frames_with_timestamps: list[tuple[float, str]],
        transcript: str | None = None
    ) -> list[dict]:
        """Build messages with timestamped frames."""
        content = [
            {"type": "text", "text": self._build_timestamp_chunk_prompt(video_info, frames_with_timestamps, transcript)}
        ]
        
        for timestamp, frame_b64 in frames_with_timestamps:
            content.append({
                "type": "text",
                "text": f"[Frame at {timestamp:.1f}s]"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}"
                }
            })
        
        return [{"role": "user", "content": content}]

    def _parse_timestamp_chunk_response(
        self,
        content: str,
        frames_with_timestamps: list[tuple[float, str]]
    ) -> dict:
        """Extract JSON from chunk response."""
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if json_match:
            json_str = json_match.group(1)
        else:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                json_str = content[start:end]
            else:
                raise ValueError(f"Could not find JSON: {content[:200]}...")
        
        data = json.loads(json_str)
        
        if not isinstance(data.get("entities"), list):
            data["entities"] = []
        if not isinstance(data.get("state_changes"), list):
            data["state_changes"] = []
        if not isinstance(data.get("micro_actions"), list):
            data["micro_actions"] = []
        
        # Filter entities - only valid types
        valid_types = {"ingredient", "tool", "appliance"}
        data["entities"] = [
            e for e in data["entities"]
            if e.get("type", "").lower() in valid_types
        ]
        
        # Process micro-actions
        for idx, ma in enumerate(data.get("micro_actions", [])):
            ma["id"] = idx  # Temporary ID, will be reassigned later
            
            if ma.get("timestamp_seconds") is None:
                ma["timestamp_seconds"] = frames_with_timestamps[0][0]
            
            if ma.get("frame_index") is None:
                ma["frame_index"] = 0
            
            if ma.get("relative_position") is None:
                ma["relative_position"] = 0.5
            
            # Fix lists to strings
            for field in ["entity", "state_before", "state_after"]:
                if isinstance(ma.get(field), list):
                    ma[field] = ", ".join(str(v) for v in ma[field])
        
        return data
    
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

    def analyze_video_direct(
        self,
        video_info: VideoInfo,
        video_path: str,
        max_retries: int = 3
    ) -> SceneLog:
        """
        Analyze video by uploading it directly as base64.
        
        This method compresses the video and uploads it to a video-capable
        VLM (like Gemini) for temporal analysis with M-RoPE grounding.
        
        Args:
            video_info: Metadata about the video
            video_path: Path to the local video file
            max_retries: Maximum retry attempts
            
        Returns:
            SceneLog containing all micro-actions with timestamps
        """
        print(f"      Compressing video for API upload...")
        
        compressed_path, size_mb = compress_video_for_api(
            video_path,
            max_size_mb=10.0,
            target_width=480,
            target_fps=8
        )
        
        try:
            print(f"      Compressed video size: {size_mb:.2f} MB")
            
            with open(compressed_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")
            
            b64_size_mb = len(video_b64) / (1024 * 1024)
            print(f"      Base64 size: {b64_size_mb:.2f} MB")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._build_video_direct_prompt(video_info)},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}
                        }
                    ]
                }
            ]
            
            print(f"      Uploading to VLM (targeting video-capable providers)...")
            
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    start_time = time.perf_counter()
                    
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        max_tokens=8192,
                        temperature=0.4
                    )
                    
                    elapsed = time.perf_counter() - start_time
                    print(f"      VLM response received in {elapsed:.2f}s")
                    
                    content = response.choices[0].message.content
                    scene = self._parse_video_direct_response(content, video_info.duration_seconds)
                    
                    return SceneLog(
                        scenes=[scene],
                        video_info={
                            "title": video_info.title,
                            "description": video_info.description,
                            "url": video_info.url,
                            "duration_seconds": video_info.duration_seconds
                        }
                    )
                    
                except RateLimitError as e:
                    wait_time = (2 ** attempt) * 2
                    print(f"[VLM] Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    last_error = e
                    
                except APIError as e:
                    error_msg = str(e).lower()
                    if "video" in error_msg or "endpoint" in error_msg:
                        raise
                    
                    wait_time = (2 ** attempt) * 1
                    print(f"[VLM] API error: {e}, retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    last_error = e
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"[VLM] Error: {e}, retrying...")
                        time.sleep(1)
                        last_error = e
                    else:
                        raise
            
            raise last_error or Exception("Failed to analyze video")
            
        finally:
            if os.path.exists(compressed_path):
                os.unlink(compressed_path)

    def _build_video_direct_prompt(self, video_info: VideoInfo) -> str:
        """Create the instruction prompt for direct video analysis."""
        return f"""You are a forensic video analyst specializing in culinary processes. Your goal is to create a factual log of cooking actions based ONLY on visual evidence.

VIDEO METADATA:
Title: {video_info.title}
Duration: {video_info.duration_seconds} seconds

*** STEP 1: VISUAL SCRATCHPAD (Mandatory) ***
Watch the video chronologically. Create a raw text log of distinct physical movements.
- Format: [Start MM:SS - End MM:SS] : [Entity] -> [Action]
- Multitasking: If multiple actions occur simultaneously (e.g., stirring while pouring), list them as SEPARATE lines with the same timestamp range.
- Constraint: If nothing significant happens for 5 seconds, do NOT write anything.

*** STEP 2: JSON GENERATION ***
Convert your scratchpad into valid JSON.

CRITICAL FORMATTING RULES:
1. Do not nest actions inside other actions. Every action is a separate object in the main list.
2. Check your brackets. Ensure every '{{' has a closing '}}' before starting a new item.
3. No Duplicates: Do not repeat the same action for the same timestamp.

JSON EXAMPLE (Follow this structure exactly):
{{
  "summary": "Step-by-step preparation of cucumber salad including slicing and seasoning.",
  "entities": [
    {{"name": "Cucumber", "type": "ingredient"}},
    {{"name": "Knife", "type": "tool"}},
    {{"name": "Bowl", "type": "tool"}}
  ],
  "micro_actions": [
    {{
      "action": "Slicing Cucumber into Rounds",
      "start": "00:28",
      "end": "00:30",
      "entity": "Knife",
      "concurrent_with_other_action": false
    }},
    {{
      "action": "Adding sliced cucumber to Bowl",
      "start": "00:31",
      "end": "00:33",
      "entity": "Cucumber",
      "concurrent_with_other_action": false
    }}
  ]
}}

STRICT ACCURACY RULES:
1. Visual Evidence Only: If you do not see a hand touching an object or a tool moving, do not list it.
2. No Filler: Quality over quantity. Do not hallucinate generic actions just to reach a higher count.
3. Valid Actions: Include both STATE CHANGES (e.g., chopping, melting) and ACTIVE PROCESSES (e.g., stirring, whisking, basting). Ignore passive states (e.g., "soup is boiling" with no human interaction).
   - Output format for valid actions: [Action Verb] + [Object] + [Resulting State/Location]
   - Transformation: If the object changes shape/form, describe the new form (e.g., "slicing carrots into rounds").
   - Destination: If the object moves, describe the container (e.g., "transferring onions to the hot skillet").
   - Completion: If the action is a process, describe the goal state (e.g., "whisking eggs until frothy").
   - Accuracy: If the resulting state/location is not clear, do not list it.
4. Concurrency: You MUST list simultaneous actions separately. Do not group them into one generic event.
5. Entity Consistency: Use the exact same name for an entity in the 'micro_actions' list as you defined in the 'entities' list.
6. Time Format: Output timestamps as "MM:SS" strings in the JSON to avoid math errors.

Return ONLY the Scratchpad followed by the JSON object."""

    def _parse_video_direct_response(
        self,
        content: str,
        video_duration: float
    ) -> SceneDescription:
        """Extract JSON from the direct video VLM response."""
        json_content = content
        if "*** STEP 2: JSON GENERATION ***" in content:
            json_content = content.split("*** STEP 2: JSON GENERATION ***")[-1]
        
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_content)
        if json_match:
            json_str = json_match.group(1)
        else:
            start = json_content.find("{")
            end = json_content.rfind("}") + 1
            if start != -1 and end > start:
                json_str = json_content[start:end]
            else:
                raise ValueError(f"Could not find JSON in response: {content[:200]}...")
        
        data = json.loads(json_str)
        
        # Ensure lists exist
        if not isinstance(data.get("entities"), list):
            data["entities"] = []
        if "state_changes" not in data:
            data["state_changes"] = []
        if not isinstance(data.get("micro_actions"), list):
            data["micro_actions"] = []
        
        def time_to_seconds(ts):
            if isinstance(ts, (int, float)):
                return float(ts)
            if isinstance(ts, str):
                if ":" in ts:
                    parts = ts.split(":")
                    if len(parts) == 2:
                        return float(parts[0]) * 60 + float(parts[1])
                    elif len(parts) == 3:
                        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                try:
                    return float(ts)
                except ValueError:
                    pass
            return 0.0

        # Process entities - handle both strings and objects
        processed_entities = []
        valid_types = {"ingredient", "tool", "appliance"}
        for e in data.get("entities", []):
            if isinstance(e, str):
                processed_entities.append({
                    "name": e,
                    "type": "ingredient"
                })
            elif isinstance(e, dict):
                if "name" in e:
                    if "type" not in e or e["type"].lower() not in valid_types:
                        e["type"] = "tool" if "tool" in e.get("name", "").lower() else "ingredient"
                    
                    if "first_seen" in e:
                        pass  # Entity model doesn't have first_seen_timestamp
                    processed_entities.append(e)
        
        # Convert to Entity objects
        entities = []
        for e in processed_entities:
            try:
                entities.append(Entity(**e))
            except Exception:
                pass  # Skip invalid entities
        
        # Process micro-actions
        micro_actions = []
        for idx, ma in enumerate(data.get("micro_actions", [])):
            # Handle start/end to timestamp/duration
            if "start" in ma:
                ma["timestamp_seconds"] = time_to_seconds(ma["start"])
            
            if "end" in ma and "start" in ma:
                ma["duration_seconds"] = max(0.0, time_to_seconds(ma["end"]) - time_to_seconds(ma["start"]))

            if ma.get("timestamp_seconds") is None:
                ma["timestamp_seconds"] = 0.0
            
            # Ensure entity is a string
            for field in ["entity", "state_before", "state_after"]:
                if isinstance(ma.get(field), list):
                    ma[field] = ", ".join(str(v) for v in ma[field])
                elif ma.get(field) is None:
                    ma[field] = ""
            
            # Set required fields
            ma["id"] = idx
            if "frame_index" not in ma:
                ma["frame_index"] = 0
            if "relative_position" not in ma:
                ma["relative_position"] = 0.5
            
            try:
                micro_actions.append(MicroAction(**ma))
            except Exception:
                pass  # Skip invalid micro-actions
        
        return SceneDescription(
            frame_index=0,
            entities=entities,
            state_changes=[],
            temporal_actions=[],
            micro_actions=micro_actions,
            metadata={
                "method": "direct_video",
                "video_duration_seconds": video_duration,
                "summary": data.get("summary", "")
            }
        )