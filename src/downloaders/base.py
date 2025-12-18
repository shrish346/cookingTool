from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Optional, runtime_checkable

@dataclass
class VideoInfo():
    title: str
    file_path: Path
    url: str
    duration_seconds: int
    description: Optional[str] = None

@runtime_checkable
class VideoDownloader(Protocol):
    def download(self, url:str) -> VideoInfo:
        ...
    def supports(self, url:str) -> bool:
        ...
    def cleanup(self, video_info: VideoInfo) -> None:
        ...

    