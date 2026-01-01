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
        fps: float = 30.0
    ) -> list[dict]:
        """
        Extract video clips for each step and upload to S3.
        
        Args:
            video_path: Path to the source video file
            video_id: Unique video identifier for S3 folder
            steps: List of step dictionaries with frame indices
            fps: Video frames per second for timestamp calculation
            
        Returns:
            Updated steps list with video_clip_url populated
        """
        updated_steps = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for step in steps:
                start_frame = step.get("start_frame_index")
                end_frame = step.get("end_frame_index")
                step_order = step.get("order", 1)
                
                # Skip if no frame indices
                if start_frame is None or end_frame is None:
                    updated_steps.append(step)
                    continue
                
                # Calculate timestamps from frame indices
                start_time = start_frame / fps
                end_time = end_frame / fps
                duration = end_time - start_time
                
                # Minimum clip duration of 2 seconds
                if duration < 2:
                    duration = 2
                
                # Extract clip
                clip_filename = f"step_{step_order}.{self.settings.clip_format}"
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
                    clip_url = self._upload_to_s3(clip_path, s3_key)
                    
                    # Update step with clip URL
                    step_copy = step.copy()
                    step_copy["video_clip_url"] = clip_url
                    updated_steps.append(step_copy)
                else:
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
            cmd = [
                self.settings.ffmpeg_path,
                "-y",  # Overwrite output
                "-ss", str(start_time),  # Start time (before -i for faster seeking)
                "-i", video_path,
                "-t", str(duration),  # Duration
                "-an",  # No audio
                "-vf", "scale=640:-2",  # Scale to 640px width, maintain aspect ratio
                "-b:v", self.settings.clip_bitrate,  # Low bitrate
                "-c:v", "libvpx-vp9" if self.settings.clip_format == "webm" else "libx264",
                "-crf", "30",  # Quality (higher = lower quality/smaller file)
                "-preset", "fast" if self.settings.clip_format == "mp4" else "",
                output_path
            ]
            
            # Remove empty strings from command
            cmd = [c for c in cmd if c]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout per clip
            )
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
    
    def _upload_to_s3(self, file_path: str, s3_key: str) -> str:
        """
        Upload a file to S3 and return the public URL.
        """
        content_type = "video/webm" if s3_key.endswith(".webm") else "video/mp4"
        
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

