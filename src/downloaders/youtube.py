import re

from .ytdlp import YtDlpDownloader


class YouTubeDownloader(YtDlpDownloader):
    # Cheap host/path check — does NOT hit the network. Whether the video actually
    # loads is decided later by get_info()/download(), which surface the real error.
    URL_RE = re.compile(
        r'(?:https?://)?(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|shorts/|embed/|live/)|youtu\.be/)',
        re.IGNORECASE,
    )
    FORMAT = 'best[height<=480][ext=mp4]/best[height<=480]/worst'
    # YouTube blocks datacenter IPs; a cookie file gets a real session through.
    COOKIES_ENV = 'YOUTUBE_COOKIES'
