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
from ..schemas import Recipe, SceneDescription, MicroAction, Entity, Observation, SceneLog
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


def _iter_json_objects(text: str):
    """Yield every top-level ``{...}`` substring in ``text``, brace-balanced and
    string-aware (braces inside JSON strings don't count).

    Used to pull the real JSON object out of a VLM response no matter what surrounds it -
    prose, a scratchpad, or one or more code fences - since the object's braces are still
    right there in the raw text.
    """
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                yield text[start : i + 1]
                start = -1


def _extract_scene_json(content: str) -> dict:
    """Robustly extract the scene object from a direct-video VLM response.

    The model is asked to emit a free-text scratchpad *followed by* the JSON object, and
    it may fence either, both, or neither. The old "take the first code fence" approach
    could capture the scratchpad instead - whose lines start with ``[00:28 - ...`` and
    blow up ``json.loads`` at char 2. Instead, scan every brace-balanced object in the
    text and return the first that parses into something shaped like a scene; the
    scratchpad's ``[...]`` lines are ignored because we only look at ``{...}``.
    """
    fallback: Optional[dict] = None
    last_err: Optional[Exception] = None
    for candidate in _iter_json_objects(content):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        if not isinstance(data, dict):
            continue
        if any(k in data for k in ("micro_actions", "entities", "summary")):
            return data
        if fallback is None:
            fallback = data
    if fallback is not None:
        return fallback
    if last_err is not None:
        raise last_err
    raise ValueError(f"Could not find JSON in response: {content[:200]}...")


class OpenRouterAdapter:
    """Adapter for OpenRouter's vision-language models."""

    def __init__(self, model: str = "google/gemini-2.5-flash"):
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

        return f"""You are a forensic video analyst specializing in culinary processes. Your goal is to create a factual log of cooking actions based ONLY on visual evidence.

*** CONTEXT AWARENESS: THE "HERO SHOT" ***
Cooking videos often are non-linear. They usually begin with a "Preview" or "Hero Shot" (showing the finished dish, eating it, or plating it) before cutting back in time to show the raw ingredients.
- **The Reset Point:** Identify the moment the video cuts from a finished/cooked state to a raw/empty state.
- **Tagging:** Any action occurring *before* this reset point involving the finished dish must be tagged.

Return a SINGLE valid JSON object and nothing else. Your entire response must begin with '{{' and end with '}}'.

Work chronologically through the video and fill the object's keys IN ORDER:

1. "scratchpad" (Mandatory, FIRST key): a raw log of distinct physical movements, one string per line.
   - Format per entry: "[Start MM:SS - End MM:SS] : [Entity] -> [Action]"
   - Multitasking: If multiple actions occur simultaneously (e.g., stirring while pouring), list them as SEPARATE entries.
   - Constraint: If nothing significant happens for 5 seconds, do NOT write an entry.
2. "summary", "entities", "micro_actions": derived from the scratchpad you just wrote.

The scratchpad is your reasoning, not a preamble - write it inside the JSON, never before it.

CRITICAL FORMATTING RULES:
1. Do not nest actions. Every action is a separate object.
2. Check your brackets. Ensure every '{{' has a closing '}}'.
3. No Duplicates: Do not repeat the same action for the same timestamp.

JSON EXAMPLE (Follow this structure exactly):
{{
  "scratchpad": [
    "[00:00 - 00:05] : Fork -> Taking a bite of the finished lasagna",
    "[00:28 - 00:30] : Knife -> Slicing cucumber into rounds"
  ],
  "summary": "...",
  "entities": [ ... ],
  "micro_actions": [
    {{
      "action": "Taking a bite of the lasagna [PREVIEW]",
      "start": "00:00",
      "end": "00:05",
      "entity": "Fork",
      "concurrent_with_other_action": false
    }},
    {{
      "action": "Slicing Cucumber into Rounds",
      "start": "00:28",
      "end": "00:30",
      "entity": "Knife",
      "concurrent_with_other_action": false
    }}
  ]
}}

STRICT ACCURACY RULES:
1. Visual Evidence Only: If you do not see a hand touching an object or a tool moving, do not list it.
2. **Preview Detection:** If an action involves the *finished* product (e.g., tasting, cutting a fully cooked slice) but appears at the start of the video *before* raw ingredients are introduced, you MUST append `[PREVIEW]` to the end of the action string.
3. Valid Actions: Include both STATE CHANGES and ACTIVE PROCESSES.
   - Output format: [Action Verb] + [Object] + [Resulting State/Location] + [Optional PREVIEW tag]
   - Transformation: If object changes form, describe it (e.g., "slicing carrots into rounds").
   - Destination: If object moves, describe container.
   - Completion: If action is a process, describe goal state (e.g., "whisking until frothy").
4. Concurrency: List simultaneous actions separately.
5. Entity Consistency: Use exact names from the 'entities' list.
6. Time Format: "MM:SS".

Return ONLY the JSON object, with "scratchpad" as its first key."""

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
        data = _extract_scene_json(content)

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
    "cuisine": "Italian",
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

        return f"""You are a forensic video analyst specializing in culinary processes. Your goal is to create a factual log of cooking actions based ONLY on visual evidence.

*** CONTEXT AWARENESS: THE "HERO SHOT" ***
Cooking videos often are non-linear. They usually begin with a "Preview" or "Hero Shot" (showing the finished dish, eating it, or plating it) before cutting back in time to show the raw ingredients.
- **The Reset Point:** Identify the moment the video cuts from a finished/cooked state to a raw/empty state.
- **Tagging:** Any action occurring *before* this reset point involving the finished dish must be tagged.

Return a SINGLE valid JSON object and nothing else. Your entire response must begin with '{{' and end with '}}'.

Work chronologically through the video and fill the object's keys IN ORDER:

1. "scratchpad" (Mandatory, FIRST key): a raw log of distinct physical movements, one string per line.
   - Format per entry: "[Start MM:SS - End MM:SS] : [Entity] -> [Action]"
   - Multitasking: If multiple actions occur simultaneously (e.g., stirring while pouring), list them as SEPARATE entries.
   - Constraint: If nothing significant happens for 5 seconds, do NOT write an entry.
2. "summary", "entities", "micro_actions": derived from the scratchpad you just wrote.

The scratchpad is your reasoning, not a preamble - write it inside the JSON, never before it.

CRITICAL FORMATTING RULES:
1. Do not nest actions. Every action is a separate object.
2. Check your brackets. Ensure every '{{' has a closing '}}'.
3. No Duplicates: Do not repeat the same action for the same timestamp.

JSON EXAMPLE (Follow this structure exactly):
{{
  "scratchpad": [
    "[00:00 - 00:05] : Fork -> Taking a bite of the finished lasagna",
    "[00:28 - 00:30] : Knife -> Slicing cucumber into rounds"
  ],
  "summary": "...",
  "entities": [ ... ],
  "micro_actions": [
    {{
      "action": "Taking a bite of the lasagna [PREVIEW]",
      "start": "00:00",
      "end": "00:05",
      "entity": "Fork",
      "concurrent_with_other_action": false
    }},
    {{
      "action": "Slicing Cucumber into Rounds",
      "start": "00:28",
      "end": "00:30",
      "entity": "Knife",
      "concurrent_with_other_action": false
    }}
  ]
}}

STRICT ACCURACY RULES:
1. Visual Evidence Only: If you do not see a hand touching an object or a tool moving, do not list it.
2. **Preview Detection:** If an action involves the *finished* product (e.g., tasting, cutting a fully cooked slice) but appears at the start of the video *before* raw ingredients are introduced, you MUST append `[PREVIEW]` to the end of the action string.
3. Valid Actions: Include both STATE CHANGES and ACTIVE PROCESSES.
   - Output format: [Action Verb] + [Object] + [Resulting State/Location] + [Optional PREVIEW tag]
   - Transformation: If object changes form, describe it (e.g., "slicing carrots into rounds").
   - Destination: If object moves, describe container.
   - Completion: If action is a process, describe goal state (e.g., "whisking until frothy").
4. Concurrency: List simultaneous actions separately.
5. Entity Consistency: Use exact names from the 'entities' list.
6. Time Format: "MM:SS".

Return ONLY the JSON object, with "scratchpad" as its first key."""
    
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
        data = _extract_scene_json(content)

        # Reasoning, not a field on the model. SceneDescription(**data) is built from the
        # whole dict, unlike the direct-video path which names its fields explicitly.
        data.pop("scratchpad", None)

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
        transcript: str | None = None,
        max_retries: int = 3,
        debug: bool = False
    ) -> SceneLog:
        """
        Analyze video by uploading it directly as base64.

        This method compresses the video and uploads it to a video-capable
        VLM (like Gemini) for temporal analysis with M-RoPE grounding.

        Args:
            video_info: Metadata about the video
            video_path: Path to the local video file
            transcript: Optional audio transcript. Used only to NAME and QUANTIFY
                what is visually observed (the compressed video is silent) - the
                prompt forbids inventing actions from narration.
            max_retries: Maximum retry attempts
            debug: Whether to print debug information (like micro-actions timeline)

        Returns:
            SceneLog containing all micro-actions with timestamps
        """
        print(f"      Compressing video for API upload...")

        compressed_path, size_mb = compress_video_for_api(
            video_path,
            max_size_mb=10.0,
            target_width=480,
            target_fps=3
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
                        {"type": "text", "text": self._build_video_direct_prompt(video_info, transcript)},
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

                    # "length" means we hit max_tokens and the JSON is cut off; "stop" with an
                    # unparseable body means the model ended its turn early instead. The two look
                    # identical from the parse error alone, so name which one happened.
                    finish_reason = response.choices[0].finish_reason
                    if finish_reason != "stop":
                        print(f"      [VLM] finish_reason={finish_reason}")

                    content = response.choices[0].message.content
                    scene = self._parse_video_direct_response(content, video_info.duration_seconds)
                    
                    # Print micro-actions if debug is True
                    if debug:
                        print("\n" + "="*80)
                        print("EXTRACTED MICRO-ACTIONS TIMELINE:")
                        print("="*80)
                        print(f"Total actions: {len(scene.micro_actions)}")
                        print("-" * 80)
                        for ma in scene.micro_actions:
                            ts = f"{ma.timestamp_seconds:.1f}s"
                            dur = f" (+{ma.duration_seconds:.1f}s)" if ma.duration_seconds else ""
                            print(f"  [{ts}{dur}] {ma.action}")
                        print("="*80 + "\n")

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

    def read_keyframes(
        self,
        video_info: VideoInfo,
        keyframes: list[tuple[float, str]],
        max_retries: int = 3,
    ) -> list[Observation]:
        """Read high-res keyframes for text/labels/amounts the low-res pass can't see.

        Args:
            video_info: Metadata about the video
            keyframes: (timestamp_seconds, base64 JPEG) pairs from
                keyframes.extract_reading_keyframes
            max_retries: Maximum retry attempts

        Returns:
            Observations (on-screen text, labels, stated amounts) to attach to the
            SceneLog before the grounding pass.
        """
        if not keyframes:
            return []

        content: list[dict] = [{"type": "text", "text": self._build_keyframe_reading_prompt(video_info)}]
        for ts, b64 in keyframes:
            content.append({"type": "text", "text": f"Frame at {ts:.1f}s:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        messages = [{"role": "user", "content": content}]

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.2,  # reading, not writing - stay literal
                )
                return self._parse_keyframe_response(response.choices[0].message.content)

            except RateLimitError as e:
                wait_time = (2 ** attempt) * 2
                print(f"[VLM] Keyframe reading rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                last_error = e

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[VLM] Keyframe reading error: {e}, retrying...")
                    time.sleep(1)
                    last_error = e
                else:
                    raise

        raise last_error or Exception("Failed to read keyframes")

    def _build_keyframe_reading_prompt(self, video_info: VideoInfo) -> str:
        return f"""You are reading high-resolution frames from a cooking video ("{video_info.title}").
These are the moments where an ingredient is added or measured. Your ONLY job is to read: extract
exact text and visually-evidenced amounts. You are not describing the cooking.

For EACH frame, report:
1. on_screen_text - ALL overlay captions, recipe-card text, jar/package label text, VERBATIM.
   This is the highest-value field; creators put amounts here ("1 tsp cumin").
2. ingredient - what ingredient is being handled, named as specifically as the evidence allows.
   A readable label or caption beats appearance ("smoked paprika", not "red powder"). If the
   identity is unreadable, give an honest visual description ("red spice powder").
3. stated_amount - ONLY if the frame evidences it: a caption/label amount, or a clearly visible
   measuring spoon/cup ("1 tsp measuring spoon of turmeric"). Never estimate an unmeasured pile.
4. confidence - 0.0-1.0 for the ingredient identification.

Skip a frame entirely if it shows nothing readable and no identifiable ingredient.

Return ONLY a JSON object:
{{
  "observations": [
    {{"timestamp": 12.5, "ingredient": "smoked paprika", "stated_amount": "1 tsp",
      "on_screen_text": "1 tsp smoked paprika", "confidence": 0.95}},
    {{"timestamp": 31.0, "ingredient": "soy sauce", "stated_amount": null,
      "on_screen_text": "Kikkoman soy sauce (label)", "confidence": 0.9}}
  ]
}}

Use each frame's stated timestamp for "timestamp". Omit nothing you can read; invent nothing you can't."""

    def _parse_keyframe_response(self, content: str) -> list[Observation]:
        """Parse the reading response into Observations, skipping invalid entries."""
        data = None
        for candidate in _iter_json_objects(content):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "observations" in parsed:
                data = parsed
                break
        if data is None:
            return []

        observations = []
        for obs in data.get("observations", []):
            if not isinstance(obs, dict):
                continue
            # An observation with nothing read carries no signal - drop it.
            if not any(obs.get(k) for k in ("ingredient", "stated_amount", "on_screen_text")):
                continue
            try:
                observations.append(Observation(
                    timestamp_seconds=obs.get("timestamp"),
                    ingredient=obs.get("ingredient"),
                    stated_amount=obs.get("stated_amount"),
                    on_screen_text=obs.get("on_screen_text"),
                    source="keyframe",
                    confidence=obs.get("confidence"),
                ))
            except Exception:
                continue  # one bad reading is not worth losing the rest
        return observations

    def _build_video_direct_prompt(self, video_info: VideoInfo, transcript: str | None = None) -> str:
        """Create the instruction prompt for direct video analysis."""
        description_section = ""
        if video_info.description:
            # Creator captions often carry the ingredient list; also often hashtag walls.
            description_section = f"""
CREATOR'S DESCRIPTION (may name ingredients/amounts; ignore hashtags and promo):
{video_info.description[:1500]}
"""

        transcript_section = ""
        if transcript:
            transcript_section = f"""
AUDIO TRANSCRIPT (the video you receive is silent - this is what was said):
{transcript}

TRANSCRIPT RULES:
- Use the transcript ONLY to NAME and QUANTIFY what you visually observe. If you see a red powder
  added while the narrator says "add the paprika", log "Adding paprika" - not "adding red powder".
- If the narrator states an amount ("a teaspoon of cumin") for something you SEE happen, record it
  in that action's "stated_amount".
- NEVER log an action you did not see, no matter what the narration says. Narration describes;
  only the camera evidences.
"""

        return f"""You are a forensic video analyst specializing in culinary processes. Your goal is to create a factual log of cooking actions based ONLY on visual evidence.

VIDEO METADATA:
Title: {video_info.title}
Duration: {video_info.duration_seconds} seconds
{description_section}{transcript_section}
Return a SINGLE valid JSON object and nothing else. Your entire response must begin with '{{' and end with '}}'.

Work chronologically through the video and fill the object's keys IN ORDER:

1. "scratchpad" (Mandatory, FIRST key): a raw log of distinct physical movements, one string per line.
   - Format per entry: "[Start MM:SS - End MM:SS] : [Entity] -> [Action]"
   - Multitasking: If multiple actions occur simultaneously (e.g., stirring while pouring), list them as SEPARATE entries with the same timestamp range.
   - Constraint: If nothing significant happens for 5 seconds, do NOT write an entry.
2. "summary", "entities", "micro_actions": derived from the scratchpad you just wrote.

The scratchpad is your reasoning, not a preamble - write it inside the JSON, never before it.

CRITICAL FORMATTING RULES:
1. Do not nest actions inside other actions. Every action is a separate object in the main list.
2. Check your brackets. Ensure every '{{' has a closing '}}' before starting a new item.
3. No Duplicates: Do not repeat the same action for the same timestamp.

JSON EXAMPLE (Follow this structure exactly):
{{
  "scratchpad": [
    "[00:28 - 00:30] : Knife -> Slicing cucumber into rounds",
    "[00:31 - 00:33] : Smoked paprika -> Sprinkling over cucumber in bowl"
  ],
  "summary": "Step-by-step preparation of cucumber salad including slicing and seasoning.",
  "entities": [
    {{"name": "Cucumber", "type": "ingredient", "quantity": "2 whole"}},
    {{"name": "Smoked paprika", "type": "ingredient", "quantity": "1 tsp"}},
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
      "action": "Sprinkling smoked paprika over cucumber in Bowl",
      "start": "00:31",
      "end": "00:33",
      "entity": "Smoked paprika",
      "stated_amount": "1 tsp",
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
7. Read On-Screen Text: Short-form videos often overlay captions ("1 tsp cumin") or show labeled
   jars and packages. READ them. A caption or label naming an ingredient or amount is visual
   evidence - use it to name the entity precisely and fill "stated_amount" on the matching action.
8. Ingredient Specificity: Name every ingredient as specifically as the evidence allows - a label,
   caption, or narration beats a visual guess ("smoked paprika" not "red powder"). When the identity
   is genuinely unreadable, log an honest visual description ("red spice powder", "dark sauce") -
   do NOT guess a specific name without evidence, and do NOT collapse distinct additions into one
   generic "spices".
9. Stated Amounts: "stated_amount" is ONLY for amounts the video evidences (caption, label, visible
   measuring spoon/cup, or narration). If no source states one, omit the field - never estimate.

Return ONLY the JSON object, with "scratchpad" as its first key."""

    def _parse_video_direct_response(
        self,
        content: str,
        video_duration: float
    ) -> SceneDescription:
        """Extract JSON from the direct video VLM response."""
        data = _extract_scene_json(content)
        
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
            for field in ["entity", "state_before", "state_after", "stated_amount"]:
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