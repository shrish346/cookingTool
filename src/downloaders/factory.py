from .base import VideoDownloader
from .youtube import YouTubeDownloader

DOWNLOADERS = {YouTubeDownloader() 
#TikTokDownloader(), 
# #InstagramDownloader(), 
# #X(Twitter)Downloader()
}


def get_downloader(url: str) -> VideoDownloader:
    for downloader in DOWNLOADERS:
        if downloader.supports(url):
            return downloader
    raise ValueError(f"No downloader found for URL: {url}")
