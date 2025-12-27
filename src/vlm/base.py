from ..downloaders.base import VideoInfo
from ..schemas import Recipe
from typing import Protocol, runtime_checkable

@runtime_checkable
class VLMAdapter(Protocol):
    def analyze_recipe(self, video_info: VideoInfo, frames: list[str]) -> Recipe:
        ...

    @property
    def model_name(self) -> str:
        ...



