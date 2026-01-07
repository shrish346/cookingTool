"""
VLM adapter for timestamp-based micro-action extraction.

Uses direct video upload - Uploads compressed video as base64.

This leverages Qwen2.5-VL's M-RoPE for temporal grounding.
"""
from __future__ import annotations

import os
import json
import re
import time
import base64
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv

# Add paths to import from main project and local modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from schemas_timestamp import (
    MicroActionTimestamp,
    EntityTimestamp,
    StateChangeTimestamp,
    VideoSceneDescription,
    VideoSceneLog,
)
from src.downloaders.base import VideoInfo

load_dotenv()


def compress_video_for_api(
    video_path: str,
    max_size_mb: float = 15.0,  # Allow slightly larger for better quality
    target_width: int = 360,  # Higher resolution for better action detection
    target_fps: int = 2  # Higher FPS for smoother temporal analysis
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
    # Create temp file for compressed video
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_fd)
    
    try:
        # Use two-pass encoding for better quality at target size
        # First, try with reasonable quality settings
        cmd = [
            "ffmpeg","-y","-i", video_path,
            "-vf", f"scale={target_width}:-2,fps={target_fps}",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",  # Better quality for action detection
            "-an",  # No audio needed for VLM
            "-movflags", "+faststart",
            temp_path
        ]
        
        result = subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg compression failed: {result.stderr}")
        
        # Check file size
        size_bytes = os.path.getsize(temp_path)
        size_mb = size_bytes / (1024 * 1024)
        
        # If still too large, compress more aggressively
        if size_mb > max_size_mb:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-vf", f"scale=360:-2,fps=6",  # Even smaller
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "32",  # More compression
                "-an",
                "-movflags", "+faststart",
                temp_path
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            size_bytes = os.path.getsize(temp_path)
            size_mb = size_bytes / (1024 * 1024)
        
        return temp_path, size_mb
        
    except Exception as e:
        # Cleanup on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


class VideoVLMAdapter:
    """
    Adapter for any OpenRouter models
    
    Uploads video as base64 to the model.
    """

    def __init__(self, model: str = "google/gemini-2.0-flash-001"):
        self._model = model
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    @property
    def model_name(self) -> str:
        return self._model

    def analyze_video(
        self,
        video_info: VideoInfo,
        video_path: str,
    ) -> VideoSceneLog:
        """
        Analyze a video using direct video upload.
        
        Args:
            video_info: Metadata about the video
            video_path: Path to the local video file
            
        Returns:
            VideoSceneLog containing all micro-actions with timestamps
        """
        return self._analyze_video_direct(video_info, video_path)

    def _analyze_video_direct(
        self,
        video_info: VideoInfo,
        video_path: str,
        max_retries: int = 3,
        debug: bool = False
    ) -> VideoSceneLog:
        """
        Analyze video by uploading it directly as base64.
        """
        print(f"      Compressing video for API upload...")
        
        # Compress video to reduce size
        compressed_path, size_mb = compress_video_for_api(
            video_path,
            max_size_mb=10.0,
            target_width=480,
            target_fps=8
        )
        
        try:
            print(f"      Compressed video size: {size_mb:.2f} MB")
            
            # Encode as base64
            with open(compressed_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")
            
            b64_size_mb = len(video_b64) / (1024 * 1024)
            print(f"      Base64 size: {b64_size_mb:.2f} MB")
            
            # Build messages with video
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._build_video_prompt(video_info)},
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
                    scene = self._parse_video_response(content, video_info.duration_seconds)
                    
                    # Print full list of microactions as requested
                    if(debug):
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

                    return VideoSceneLog(
                        scene=scene,
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
                    # Check if it's a "video not supported" error
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
            # Cleanup compressed video
            if os.path.exists(compressed_path):
                os.unlink(compressed_path)

    def _build_video_prompt(
        self,
        video_info: VideoInfo,
        transcript: str | None = None
    ) -> str:
        """Create the instruction prompt for video analysis."""

        return f"""You are a forensic video analyst specializing in culinary processes. Your goal is to create a factual log of cooking actions based ONLY on visual evidence.

*** CONTEXT AWARENESS: THE "HERO SHOT" ***
Cooking videos often are non-linear. They usually begin with a "Preview" or "Hero Shot" (showing the finished dish, eating it, or plating it) before cutting back in time to show the raw ingredients.
- **The Reset Point:** Identify the moment the video cuts from a finished/cooked state to a raw/empty state.
- **Tagging:** Any action occurring *before* this reset point involving the finished dish must be tagged.

*** STEP 1: VISUAL SCRATCHPAD (Mandatory) ***
Watch the video chronologically. Create a raw text log of distinct physical movements.
- Format: [Start MM:SS - End MM:SS] : [Entity] -> [Action]
- Multitasking: If multiple actions occur simultaneously (e.g., stirring while pouring), list them as SEPARATE lines.
- Constraint: If nothing significant happens for 5 seconds, do NOT write anything.

*** STEP 2: JSON GENERATION ***
Convert your scratchpad into valid JSON.

CRITICAL FORMATTING RULES:
1. Do not nest actions. Every action is a separate object.
2. Check your brackets. Ensure every '{{' has a closing '}}'.
3. No Duplicates: Do not repeat the same action for the same timestamp.

JSON EXAMPLE (Follow this structure exactly):
{{
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

Return ONLY the Scratchpad followed by the JSON object."""

    def _parse_video_response(
        self,
        content: str,
        video_duration: float
    ) -> VideoSceneDescription:
        """Extract JSON from the VLM response."""
        # Clean up any "Scratchpad" text before parsing JSON
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
        
        # Map timestamp formats
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
                # If just a string, assume it's a tool/ingredient and give it a default type
                processed_entities.append({
                    "name": e,
                    "type": "ingredient"  # Default to ingredient if unknown
                })
            elif isinstance(e, dict):
                if "name" in e:
                    if "type" not in e or e["type"].lower() not in valid_types:
                        e["type"] = "tool" if "tool" in e.get("name", "").lower() else "ingredient"
                    
                    if "first_seen" in e:
                        e["first_seen_timestamp"] = time_to_seconds(e["first_seen"])
                    processed_entities.append(e)
        data["entities"] = processed_entities
        
        # Process micro-actions
        for idx, ma in enumerate(data.get("micro_actions", [])):
            ma["id"] = idx
            
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
        
        data["video_duration_seconds"] = video_duration
        
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["method"] = "direct_video"
        
        return VideoSceneDescription(**data)
