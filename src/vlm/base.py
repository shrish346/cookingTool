from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..downloaders.base import VideoInfo
from ..schemas import Recipe

@runtime_checkable
class VLMAdapter(Protocol):
    def analyze_recipe(
        self, 
        video_info: VideoInfo, 
        frames: list[str],
        transcript: str | None = None
    ) -> Recipe:
        ...

    @property
    def model_name(self) -> str:
        ...



