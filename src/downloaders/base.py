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

@dataclass
class VideoMetadata:
    """Lightweight metadata for validation (without downloading)."""
    title: str
    video_id: str
    duration_seconds: int
    thumbnail_url: Optional[str] = None
    description: Optional[str] = None


@runtime_checkable
class VideoDownloader(Protocol):
    def download(self, url: str) -> VideoInfo:
        ...
    
    def supports(self, url: str) -> bool:
        ...
    
    def cleanup(self, video_info: VideoInfo) -> None:
        ...
    
    def get_info(self, url: str) -> Optional[VideoMetadata]:
        """Get video metadata without downloading (for validation)."""
        ...
