from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .downloaders.base import VideoInfo
from .schemas import Recipe, SceneLog, SceneDescription
from .vlm.base import VLMAdapter
from .llm.base import LLMAdapter


class RecipeChef:
    """
    Orchestrates the two-stage pipeline: VLM scene analysis → LLM recipe generation.
    
    This class coordinates the flow:
    1. VLM analyzes frames and generates scene descriptions
    2. Scene descriptions are accumulated into a SceneLog
    3. LLM takes the SceneLog and generates a structured Recipe
    """

    def __init__(
        self,
        vlm_adapter: VLMAdapter,
        llm_adapter: LLMAdapter,
        save_scenes_path: Optional[str | Path] = None
    ):
        """
        Initialize the RecipeChef with VLM and LLM adapters.
        
        Args:
            vlm_adapter: Adapter for vision-language model (scene analysis)
            llm_adapter: Adapter for language model (recipe generation)
            save_scenes_path: Optional path to save SceneLog for debugging/caching
        """
        self._vlm = vlm_adapter
        self._llm = llm_adapter
        self._save_scenes_path = Path(save_scenes_path) if save_scenes_path else None

    def generate_recipe(
        self,
        video_info: VideoInfo,
        frames: list[str],
        transcript: str | None = None,
        chunk_size: int = 5
    ) -> Recipe:
        """
        Generate a recipe from video frames using the two-stage pipeline.
        
        Args:
            video_info: Metadata about the video (title, description, etc.)
            frames: List of base64-encoded JPEG images
            transcript: Optional audio transcript from the video
            chunk_size: Number of frames to process in each VLM batch
            
        Returns:
            A validated Recipe object
        """
        # Stage 1: VLM analyzes frames and generates scene descriptions
        scene_descriptions = self._vlm.analyze_scenes(
            video_info,
            frames,
            transcript,
            chunk_size
        )
        
        # Create SceneLog from descriptions
        scene_log = SceneLog(
            scenes=scene_descriptions,
            video_info={
                "title": video_info.title,
                "description": video_info.description,
                "url": video_info.url,
                "duration_seconds": video_info.duration_seconds
            }
        )
        
        # Optionally save SceneLog for debugging/caching
        if self._save_scenes_path:
            self._save_scene_log(scene_log)
        
        # Stage 2: LLM generates recipe from scene descriptions
        recipe = self._llm.generate_recipe(scene_log, video_info, transcript)
        
        return recipe

    def _save_scene_log(self, scene_log: SceneLog) -> None:
        """Save SceneLog to JSON file for debugging or caching."""
        if self._save_scenes_path:
            self._save_scenes_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._save_scenes_path, 'w') as f:
                json.dump(scene_log.to_dict(), f, indent=2)

    @classmethod
    def load_scene_log(cls, path: str | Path) -> SceneLog:
        """
        Load a SceneLog from a JSON file.
        
        Useful for debugging or re-running recipe generation with cached scene descriptions.
        """
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return SceneLog.from_dict(data)

