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
        
        print(f"[ClipExtractor] Processing {len(steps)} steps with fps={fps}, frame_interval={frame_interval}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for step in steps:
                # Scene indices from LLM (0-34 range typically)
                scene_start = step.get("start_frame_index")
                scene_end = step.get("end_frame_index")
                step_order = step.get("order", 1)
                
                # Skip if no frame indices
                if scene_start is None or scene_end is None:
                    print(f"[ClipExtractor] Skipping step {step_order} - no frame indices")
                    updated_steps.append(step)
                    continue
                
                # Convert scene indices to actual frame numbers
                start_frame = scene_start * frame_interval
                end_frame = scene_end * frame_interval
                
                print(f"[ClipExtractor] Step {step_order}: scene={scene_start}-{scene_end} -> frames={start_frame}-{end_frame}")
                
                # Calculate timestamps from actual frame numbers
                start_time = start_frame / fps
                end_time = end_frame / fps
                duration = end_time - start_time
                
                # Minimum clip duration of 2 seconds
                if duration < 2:
                    duration = 2
                
                # Extract clip (always use mp4 for compatibility)
                clip_filename = f"step_{step_order}.mp4"
                clip_path = os.path.join(temp_dir, clip_filename)
                
                print(f"[ClipExtractor] Extracting clip: start={start_time:.2f}s, duration={duration:.2f}s")
                
                success = self._extract_clip(
                    video_path=video_path,
                    output_path=clip_path,
                    start_time=start_time,
                    duration=duration
                )
                
                if success and os.path.exists(clip_path):
                    # Upload to S3
                    s3_key = f"{video_id}/{clip_filename}"
                    print(f"[ClipExtractor] Uploading to S3: {s3_key}")
                    try:
                        clip_url = self._upload_to_s3(clip_path, s3_key)
                        print(f"[ClipExtractor] Success! URL: {clip_url}")
                        
                        # Update step with clip URL
                        step_copy = step.copy()
                        step_copy["video_clip_url"] = clip_url
                        updated_steps.append(step_copy)
                    except Exception as e:
                        print(f"[ClipExtractor] S3 upload failed: {e}")
                        updated_steps.append(step)
                else:
                    print(f"[ClipExtractor] FFmpeg extraction failed for step {step_order}")
                    # Keep step without clip URL
                    updated_steps.append(step)
        
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
        
        Returns True if successful, False otherwise.
        """
        try:
            # FFmpeg command for low-bitrate clip extraction without audio
            # Use MP4 with libx264 for maximum compatibility
            cmd = [
                self.settings.ffmpeg_path,
                "-y",  # Overwrite output
                "-ss", str(start_time),  # Start time (before -i for faster seeking)
                "-i", video_path,
                "-t", str(duration),  # Duration
                "-an",  # No audio
                "-vf", "scale=640:-2",  # Scale to 640px width, maintain aspect ratio
                "-c:v", "libx264",  # Use H.264 codec (widely supported)
                "-preset", "fast",  # Encoding speed
                "-crf", "28",  # Quality (lower = better quality, 18-28 is good range)
                "-movflags", "+faststart",  # Enable streaming
                output_path
            ]
            
            print(f"[FFmpeg] Running: {' '.join(cmd)}")
            
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

