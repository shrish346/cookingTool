"""
Video clip extraction using FFmpeg.
Extracts short loops for each recipe step and uploads to S3.
"""
import subprocess
import tempfile
import os
from pathlib import Path

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
        steps: list[dict]
    ) -> list[dict]:
        """
        Extract video clips for each step and upload to S3.
        
        Requires 'start_timestamp_seconds' and 'end_timestamp_seconds' to be present 
        in each step dictionary.
        """
        updated_steps = []
        
        print(f"\n{'='*70}")
        print(f"[ClipExtractor] Processing {len(steps)} steps")
        print("="*70 + "\n")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for step in steps:
                step_order = step.get("order", 1)
                step_title = step.get("title", "Untitled")
                instruction = step.get("instruction", "")
                
                print(f"─── STEP {step_order}: {step_title} ".ljust(70, "─"))
                if instruction:
                    preview = (instruction[:65] + '...') if len(instruction) > 65 else instruction
                    print(f"  Instruction: {preview}")
                
                # Check if this step should have no video clip
                if not step.get("has_video_clip", True):
                    print(f"  ⏭  Skipping - marked as no video clip\n")
                    updated_steps.append(step)
                    continue
                
                # Timing Logic (Timestamp only)
                start_time = step.get("start_timestamp_seconds") or step.get("start_timestamp")
                end_time = step.get("end_timestamp_seconds") or step.get("end_timestamp")
                
                if start_time is not None and end_time is not None:
                    start_time = float(start_time)
                    end_time = float(end_time)
                    print(f"  Timing: {start_time:.2f}s → {end_time:.2f}s")
                else:
                    print(f"  ⏭  Skipping - no timestamp timing data found\n")
                    updated_steps.append(step)
                    continue

                # Micro-actions for context
                micro_actions = step.get("micro_action_descriptions", [])
                if micro_actions:
                    print(f"  Actions ({len(micro_actions)}):")
                    for action in micro_actions[:3]:
                        print(f"    • {action}")
                    if len(micro_actions) > 3:
                        print(f"    • ... and {len(micro_actions)-3} more")

                # Duration Checks
                duration = end_time - start_time
                micro_count = len(micro_actions)
                min_duration = 4.0 if micro_count <= 1 else 3.0
                
                if duration < min_duration:
                    print(f"  ⚠️  Duration {duration:.2f}s too short, extending to {min_duration}s")
                    duration = min_duration
                
                if duration > 14:
                    print(f"  ⚠️  WARNING: Capping duration from {duration:.2f}s to 14s")
                    duration = 14
                
                print(f"  Extracting: {start_time:.2f}s → {start_time + duration:.2f}s (duration: {duration:.1f}s)")
                
                # Extraction & Upload
                clip_filename = f"step_{step_order}.mp4"
                clip_path = os.path.join(temp_dir, clip_filename)
                
                success = self._extract_clip(
                    video_path=video_path,
                    output_path=clip_path,
                    start_time=start_time,
                    duration=duration
                )
                
                if success and os.path.exists(clip_path):
                    s3_key = f"{video_id}/{clip_filename}"
                    try:
                        clip_url = self._upload_to_s3(clip_path, s3_key)
                        print(f"  ✓ Uploaded: {s3_key}\n")
                        
                        step_copy = step.copy()
                        step_copy["video_clip_url"] = clip_url
                        updated_steps.append(step_copy)
                    except Exception as e:
                        print(f"  ✗ S3 upload failed: {e}\n")
                        updated_steps.append(step)
                else:
                    print(f"  ✗ FFmpeg extraction failed\n")
                    updated_steps.append(step)
        
        print("="*70)
        print(f"[ClipExtractor] Completed: {len([s for s in updated_steps if s.get('video_clip_url')])} clips uploaded")
        print("="*70 + "\n")
        
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
            # Note: Putting -ss after -i is slower but ensures frame-accurate seeking
            # and correct duration for the extracted clip.
            cmd = [
                self.settings.ffmpeg_path,
                "-y",  # Overwrite output
                "-i", video_path,
                "-ss", str(start_time),  # Accurate seek after input
                "-t", str(duration),     # Duration from the seek point
                "-an",  # No audio
                "-vf", "scale=640:-2",  # Scale to 640px width
                "-c:v", "libx264",      # Use H.264
                "-preset", "fast",      # Speed
                "-crf", "28",           # Quality
                "-movflags", "+faststart",
                "-avoid_negative_ts", "make_zero",
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

