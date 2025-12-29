from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..downloaders.base import VideoInfo
from ..schemas import Recipe, SceneLog

@runtime_checkable
class LLMAdapter(Protocol):
    """Protocol for LLM adapters that generate recipes from scene descriptions."""
    
    def generate_recipe(
        self,
        scene_log: SceneLog,
        video_info: VideoInfo,
        transcript: str | None = None
    ) -> Recipe:
        """
        Generate a structured recipe from accumulated scene descriptions.
        
        Args:
            scene_log: Accumulated scene descriptions from VLM analysis
            video_info: Metadata about the video (title, description, etc.)
            transcript: Optional audio transcript from the video
            
        Returns:
            A validated Recipe object
        """
        ...
    
    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

