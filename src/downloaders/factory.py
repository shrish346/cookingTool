from typing import Optional
from .base import VideoDownloader
from .youtube import YouTubeDownloader


def get_downloader(url: str, cookies: Optional[str] = None) -> VideoDownloader | None:
    # Initialize downloaders on demand
    downloaders = [
        YouTubeDownloader(cookies=cookies),
        # TikTokDownloader(),
    ]
    
    for downloader in downloaders:
        if downloader.supports(url):
            return downloader
    return None
