from __future__ import annotations

from typing import Protocol, runtime_checkable
import warnings

from ..downloaders.base import VideoInfo
from ..schemas import Observation, Recipe, SceneDescription, SceneLog

@runtime_checkable
class VLMAdapter(Protocol):
    def analyze_recipe(
        self, 
        video_info: VideoInfo, 
        frames: list[str],
        transcript: str | None = None
    ) -> Recipe:
        """
        Analyze video frames and return a structured Recipe.
        
        .. deprecated:: 2.0
            Use :meth:`analyze_scenes` followed by LLM recipe generation instead.
            This method is kept for backward compatibility.
        """
        ...

    def analyze_scenes(
        self,
        video_info: VideoInfo,
        frames: list[str],
        transcript: str | None = None,
        chunk_size: int = 5
    ) -> list[SceneDescription]:
        """
        Analyze video frames and return structured scene descriptions.
        
        Args:
            video_info: Metadata about the video (title, description, etc.)
            frames: List of base64-encoded JPEG images
            transcript: Optional audio transcript from the video
            chunk_size: Number of frames to process in each batch (for adaptive processing)
            
        Returns:
            List of SceneDescription objects, one per frame or chunk
        """
        ...

    def analyze_frame_batch(
        self,
        video_info: VideoInfo,
        frames: list[str],
        transcript: str | None = None
    ) -> SceneDescription:
        """
        Analyze a batch of frames and return a single scene description.
        
        Args:
            video_info: Metadata about the video
            frames: List of base64-encoded JPEG images (typically a chunk)
            transcript: Optional audio transcript
            
        Returns:
            A single SceneDescription representing the batch
        """
        ...

    def analyze_video_direct(
        self,
        video_info: VideoInfo,
        video_path: str,
        transcript: str | None = None,
        max_retries: int = 3,
        debug: bool = False,
    ) -> SceneLog:
        """
        Analyze a video by uploading it directly (no frame extraction).

        This method compresses and uploads the video as base64 to a
        video-capable VLM (like Gemini) for temporal analysis.

        Args:
            video_info: Metadata about the video
            video_path: Path to the local video file
            transcript: Optional audio transcript, used only to name/quantify
                what is visually observed - never to invent actions
            max_retries: Maximum retry attempts
            debug: Whether to print debug information

        Returns:
            SceneLog containing all micro-actions with timestamps
        """
        ...

    def read_keyframes(
        self,
        video_info: VideoInfo,
        keyframes: list[tuple[float, str]],
        max_retries: int = 3,
    ) -> list[Observation]:
        """
        Read high-res keyframes for on-screen text, labels and amounts that the
        compressed temporal pass can't resolve.

        Args:
            video_info: Metadata about the video
            keyframes: (timestamp_seconds, base64 JPEG) pairs
            max_retries: Maximum retry attempts

        Returns:
            Observations to attach to the SceneLog before the grounding pass
        """
        ...

    @property
    def model_name(self) -> str:
        ...



