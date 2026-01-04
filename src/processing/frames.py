import cv2
import base64
import os
from typing import Optional


class FrameExtractor:
    def __init__(self, resize_width: int = 640):
        self.resize_width = resize_width
    
    def extract(self, video_path: str) -> list[str]:
        """
        Extract frames from a video at regular intervals.
        
        Legacy method - returns frames without timestamps.
        For timestamp-based extraction, use extract_with_timestamps().
        """
        frames = []
        frame_limit = 40

        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise FileNotFoundError(f"Video file is not found at {video_path}")

        try:
            fps = vid.get(cv2.CAP_PROP_FPS)
            frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) # has to be an int for range() to work
            duration = frame_count / fps
            frame_interval = duration / frame_limit
            interval = int(frame_interval * fps)

            for i in range(0, frame_count, interval):
                vid.set(cv2.CAP_PROP_POS_FRAMES, i)
                success, frame = vid.read()
                if not success:
                    continue  # Skip corrupted frames instead of crashing
                frame = cv2.resize(frame, (self.resize_width, int(frame.shape[0] * self.resize_width / frame.shape[1])))
                _, buffer = cv2.imencode('.jpg', frame)
                frames.append(base64.b64encode(buffer).decode('utf-8'))
        finally:
            vid.release()

        return frames
    
    def extract_with_timestamps(
        self,
        video_path: str,
        target_fps: float = 2.0,
        max_frames: int = 80
    ) -> list[tuple[float, str]]:
        """
        Extract frames from a video with their timestamps.
        
        This method extracts frames at a specified FPS rate and returns
        each frame paired with its exact timestamp in seconds.
        
        Args:
            video_path: Path to the video file
            target_fps: Frames per second to extract (default 2.0 = one frame every 0.5s)
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of (timestamp_seconds, base64_image) tuples
        """
        frames_with_timestamps = []

        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise FileNotFoundError(f"Video file is not found at {video_path}")

        try:
            video_fps = vid.get(cv2.CAP_PROP_FPS)
            frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / video_fps if video_fps > 0 else 0
            
            # Calculate frame interval based on target FPS
            frame_interval = int(video_fps / target_fps) if target_fps > 0 else 1
            frame_interval = max(1, frame_interval)  # At least 1 frame
            
            extracted_count = 0
            for frame_num in range(0, frame_count, frame_interval):
                if extracted_count >= max_frames:
                    break
                    
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                success, frame = vid.read()
                if not success:
                    continue
                
                # Calculate timestamp for this frame
                timestamp = frame_num / video_fps if video_fps > 0 else 0
                
                # Resize frame
                frame = cv2.resize(
                    frame, 
                    (self.resize_width, int(frame.shape[0] * self.resize_width / frame.shape[1]))
                )
                
                # Encode as base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frames_with_timestamps.append((timestamp, frame_b64))
                extracted_count += 1
                
        finally:
            vid.release()

        return frames_with_timestamps
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata.
        
        Returns:
            Dict with fps, frame_count, duration_seconds
        """
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise FileNotFoundError(f"Video file is not found at {video_path}")
        
        try:
            fps = vid.get(cv2.CAP_PROP_FPS)
            frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            return {
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": duration
            }
        finally:
            vid.release()