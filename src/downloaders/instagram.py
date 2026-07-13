import re
from typing import Optional

from .ytdlp import YtDlpDownloader

# Instagram serves anonymous requests a login wall rather than an error page, and
# yt-dlp surfaces that as one of these. It is the single most likely reason a
# Reel fails, and the fix is always the same: give the worker a session cookie.
_LOGIN_WALL_SIGNS = (
    'login required',
    'rate-limit reached',
    'You need to log in',
    'requested content is not available',
    'Requested content is not available',
    'empty media response',
)


class InstagramDownloader(YtDlpDownloader):
    # Reels appear as /reel/, /reels/, and (for older or cross-posted clips) /p/
    # and /tv/. Share-sheet links (/share/...) 302 to one of those, and yt-dlp
    # follows the redirect, so match them too. Host/path check only — no network.
    URL_RE = re.compile(
        r'(?:https?://)?(?:www\.|m\.)?instagram\.com/'
        r'(?:[\w.-]+/)?(?:reels?|p|tv|share)/',
        re.IGNORECASE,
    )
    # Instagram has no 480p rendition to ask for — take the best muxed mp4 and
    # let compress_video_for_api() shrink it before the VLM call.
    FORMAT = 'best[ext=mp4]/best'
    # Unlike YouTube's, these cookies are not a datacenter-IP workaround —
    # Instagram walls most Reels behind a login for *anyone* not signed in, so
    # the worker needs a real session to fetch anything but the most public clips.
    COOKIES_ENV = 'INSTAGRAM_COOKIES'
    PROXY_ENV = 'INSTAGRAM_PROXY'

    def explain_error(self, exc: Exception) -> Optional[str]:
        msg = str(exc)
        if any(sign in msg for sign in _LOGIN_WALL_SIGNS):
            return (
                "Instagram wouldn't serve this Reel without a login (it walls "
                "most Reels behind one, and rate-limits logged-out requests). "
                "Set INSTAGRAM_COOKIES on the download worker to a Netscape "
                "cookie file exported from a signed-in Instagram session."
            )
        return None
