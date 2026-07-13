from typing import Optional
from .base import VideoDownloader
from .youtube import YouTubeDownloader
from .tiktok import TikTokDownloader
from .instagram import InstagramDownloader


def get_downloader(url: str, cookies: Optional[str] = None) -> VideoDownloader | None:
    # Initialize downloaders on demand. Each reads its own cookie env var
    # (YOUTUBE_COOKIES / TIKTOK_COOKIES / INSTAGRAM_COOKIES); `cookies`
    # overrides whichever one matches.
    downloaders = [
        YouTubeDownloader(cookies=cookies),
        TikTokDownloader(cookies=cookies),
        InstagramDownloader(cookies=cookies),
    ]

    for downloader in downloaders:
        if downloader.supports(url):
            return downloader
    return None
