"""Video downloaders for various platforms."""

from .base import VideoInfo, VideoDownloader
from .youtube import YouTubeDownloader
from .factory import get_downloader

__all__ = [
    'VideoInfo',
    'VideoDownloader',
    'YouTubeDownloader',
    'get_downloader',
    'get_supported_platforms',
]

