from .base import VideoDownloader
from .youtube import YouTubeDownloader

DOWNLOADERS = {YouTubeDownloader()}
#TikTokDownloader(), 
# #InstagramDownloader(), 
# #X(Twitter)Downloader()


def get_downloader(url: str) -> VideoDownloader | None:
    for downloader in DOWNLOADERS:
        if downloader.supports(url):
            return downloader
    return None
