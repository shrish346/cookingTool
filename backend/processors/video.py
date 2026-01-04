"""
Video clip extraction using FFmpeg.
Extracts short loops for each recipe step and uploads to S3.
"""
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional

from backend.api.config import get_settings


class VideoClipExtractor:
    """
    Extracts video clips for each recipe step using FFmpeg.
    Uploads clips to S3 for streaming.
    """
    
    def __init__(self, s3_client, bucket_name: str):
        self.s3 = s3_client
        self.bucket_name = bucket_name
        self.settings = get_settings()
    
    def extract_and_upload_clips(
        self,
        video_path: str,
        video_id: str,
        steps: list[dict],
        fps: float = 30.0,
        frame_interval: int = 1
    ) -> list[dict]:
        """
        Extract video clips for each step and upload to S3.
        
        Supports both V1 (scene-based) and V2 (micro-action based) frame indices:
        - V1: Integer frame indices (0-34) - multiply by frame_interval
        - V2: Float frame indices (5.3 = frame 5, position 0.3) - precise timing
        
        Args:
            video_path: Path to the source video file
            video_id: Unique video identifier for S3 folder
            steps: List of step dictionaries with frame indices
            fps: Video frames per second for timestamp calculation
            frame_interval: Multiplier to convert scene indices to actual frame numbers
            
        Returns:
            Updated steps list with video_clip_url populated
        """
        # Ensure fps is valid
        if fps is None or fps <= 0:
            fps = 30.0
        if frame_interval is None or frame_interval < 1:
            frame_interval = 1
        
        updated_steps = []
        
        print(f"\n{'='*70}")
        print(f"[ClipExtractor] Processing {len(steps)} steps")
        print(f"  Video: fps={fps}, frame_interval={frame_interval}")
        print(f"{'='*70}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for step in steps:
                step_order = step.get("order", 1)
                step_title = step.get("title", "Untitled")
                
                print(f"\n{'─'*70}")
                print(f"  STEP {step_order}: {step_title}")
                print(f"{'─'*70}")
                
                # Show micro-action descriptions if available
                micro_action_descriptions = step.get("micro_action_descriptions", [])
                if micro_action_descriptions:
                    print(f"  Micro-actions ({len(micro_action_descriptions)}):")
                    for i, desc in enumerate(micro_action_descriptions):
                        print(f"    {i+1}. {desc}")
                
                # Check if this step should have no video clip
                has_video_clip = step.get("has_video_clip", True)
                if not has_video_clip:
                    print(f"  ⏭  Skipping - marked as no video clip")
                    updated_steps.append(step)
                    continue
                
                # Get frame indices (can be int or float in V2)
                scene_start = step.get("start_frame_index")
                scene_end = step.get("end_frame_index")
                
                # Skip if no frame indices
                if scene_start is None or scene_end is None:
                    print(f"  ⏭  Skipping - no frame indices")
                    updated_steps.append(step)
                    continue
                
                # Convert to float for consistent handling
                scene_start = float(scene_start)
                scene_end = float(scene_end)
                
                # micro-action based: use precise frame mapping
                # scene_start=5.3 means frame index 5, position 0.3 within that chunk
                # Actual frame = (frame_index + relative_position) * frame_interval
                start_frame = scene_start * frame_interval
                end_frame = scene_end * frame_interval
                
                print(f"  Frame mapping: scene {scene_start:.2f}-{scene_end:.2f} → frames {start_frame:.0f}-{end_frame:.0f}")
                
                # Calculate timestamps from actual frame numbers
                start_time = start_frame / fps
                end_time = end_frame / fps
                duration = end_time - start_time
                
                # Minimum clip duration - scale based on number of micro-actions
                # Single micro-action steps need at least 4s for context
                # Multi-action steps need at least 3s
                num_micro_actions = len(micro_action_descriptions) if micro_action_descriptions else 1
                min_duration = 4.0 if num_micro_actions == 1 else 3.0
                
                if duration < min_duration:
                    print(f"  Duration {duration:.1f}s too short, extending to {min_duration}s")
                    duration = min_duration
                
                # Maximum clip duration of 14 seconds - cooking steps should be short loops
                # If the LLM gives a wider range, we cap it to prevent overly long clips
                if duration > 14:
                    print(f"  ⚠  WARNING: Capping duration from {duration:.1f}s to 14s")
                    duration = 14
                
                print(f"  Extracting: {start_time:.2f}s → {start_time + duration:.2f}s (duration: {duration:.1f}s)")
                
                # Extract clip (always use mp4 for compatibility)
                clip_filename = f"step_{step_order}.mp4"
                clip_path = os.path.join(temp_dir, clip_filename)
                
                success = self._extract_clip(
                    video_path=video_path,
                    output_path=clip_path,
                    start_time=start_time,
                    duration=duration
                )
                
                if success and os.path.exists(clip_path):
                    # Upload to S3
                    s3_key = f"{video_id}/{clip_filename}"
                    try:
                        clip_url = self._upload_to_s3(clip_path, s3_key)
                        print(f"  ✓ Uploaded: {s3_key}")
                        
                        # Update step with clip URL
                        step_copy = step.copy()
                        step_copy["video_clip_url"] = clip_url
                        updated_steps.append(step_copy)
                    except Exception as e:
                        print(f"  ✗ S3 upload failed: {e}")
                        updated_steps.append(step)
                else:
                    print(f"  ✗ FFmpeg extraction failed")
                    # Keep step without clip URL
                    updated_steps.append(step)
        
        print(f"\n{'='*70}")
        print(f"[ClipExtractor] Completed: {len([s for s in updated_steps if s.get('video_clip_url')])} clips uploaded")
        print(f"{'='*70}\n")
        
        return updated_steps
    
    def _extract_clip(
        self,
        video_path: str,
        output_path: str,
        start_time: float,
        duration: float
    ) -> bool:
        """
        Extract a clip from the video using FFmpeg.
        
        Uses accurate seeking for precise clip boundaries. This is slower than
        keyframe-only seeking but ensures the clip starts/ends at the exact times.
        
        Returns True if successful, False otherwise.
        """
        try:
            # FFmpeg command for low-bitrate clip extraction without audio
            # Use MP4 with libx264 for maximum compatibility
            # Note: -ss after -i is slower but more accurate for clip boundaries
            cmd = [
                self.settings.ffmpeg_path,
                "-y",  # Overwrite output
                "-accurate_seek",  # Enable accurate seeking (default in newer ffmpeg but explicit is safer)
                "-ss", str(start_time),  # Seek to start time (before -i for fast input seeking)
                "-i", video_path,
                "-ss", "0",  # Output-level seek to ensure we start from the exact frame
                "-t", str(duration),  # Duration
                "-an",  # No audio
                "-vf", "scale=640:-2",  # Scale to 640px width, maintain aspect ratio
                "-c:v", "libx264",  # Use H.264 codec (widely supported)
                "-preset", "fast",  # Encoding speed
                "-crf", "28",  # Quality (lower = better quality, 18-28 is good range)
                "-movflags", "+faststart",  # Enable streaming
                "-avoid_negative_ts", "make_zero",  # Ensure timestamps start at 0
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout per clip
            )
            
            if result.returncode != 0:
                print(f"[FFmpeg] STDERR: {result.stderr[:500]}")
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
    
    def _upload_to_s3(self, file_path: str, s3_key: str) -> str:
        """
        Upload a file to S3 and return the public URL.
        """
        content_type = "video/mp4"  # Always use mp4
        
        self.s3.upload_file(
            file_path,
            self.bucket_name,
            s3_key,
            ExtraArgs={
                "ContentType": content_type,
                "CacheControl": "max-age=31536000",  # 1 year cache
            }
        )
        
        # Return a region-aware virtual-hosted–style S3 URL.
        # Using the region hostname avoids 301 redirects for buckets outside us-east-1.
        return f"https://{self.bucket_name}.s3.{self.settings.aws_region}.amazonaws.com/{s3_key}"
    
    def get_total_frames(self, video_path: str) -> int:
        """
        Get the total number of frames in a video using FFprobe.
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0",
                video_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip())
        except Exception:
            pass
        
        # Fallback: estimate from duration and fps
        fps = self.get_video_fps(video_path)
        duration = self._get_video_duration(video_path)
        return int(fps * duration) if duration > 0 else 1000
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                video_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass
        
        return 0.0
    
    def get_video_fps(self, video_path: str) -> float:
        """
        Get the FPS of a video using FFprobe.
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "csv=p=0",
                video_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse fraction like "30/1" or "30000/1001"
                fps_str = result.stdout.strip()
                if "/" in fps_str:
                    num, den = map(float, fps_str.split("/"))
                    return num / den
                return float(fps_str)
        except Exception:
            pass
        
        # Default to 30 FPS
        return 30.0

