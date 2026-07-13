import re
from typing import Optional

from .ytdlp import YtDlpDownloader

# A blocked TikTok fails at the TLS handshake, so yt-dlp never gets a page and
# reports a bare SSL error plus "report this issue on GitHub" — useless to the
# person who pasted the link, and not a yt-dlp bug at all.
_BLOCKED_SIGNS = ('[SSL]', 'SSLError', 'ssl.c', 'Connection reset', 'Failed to resolve')


class TikTokDownloader(YtDlpDownloader):
    # Matches full video URLs (tiktok.com/@user/video/123) and the short-link
    # hosts (vm./vt.tiktok.com/CODE, tiktok.com/t/CODE) that the share sheet
    # produces. Host check only — no network; yt-dlp follows the redirect.
    URL_RE = re.compile(
        r'(?:https?://)?(?:www\.|m\.|vm\.|vt\.)?tiktok\.com/',
        re.IGNORECASE,
    )
    # TikTok has no 480p rendition to ask for — take the best muxed mp4 and let
    # compress_video_for_api() shrink it before the VLM call.
    FORMAT = 'best[ext=mp4]/best'
    # Usually unnecessary, unlike YouTube; set it if TikTok starts age- or
    # region-gating downloads from the worker's IP.
    COOKIES_ENV = 'TIKTOK_COOKIES'
    # Some ISPs (notably in India, where TikTok is banned) DNS-hijack
    # tiktok.com, so the TLS handshake dies before yt-dlp sees a page. The
    # download worker runs on a residential line, so it inherits that block —
    # point TIKTOK_PROXY at a proxy on an unblocked network to get around it.
    PROXY_ENV = 'TIKTOK_PROXY'

    def explain_error(self, exc: Exception) -> Optional[str]:
        if any(sign in str(exc) for sign in _BLOCKED_SIGNS):
            return (
                "Can't reach TikTok from the download worker's network — the "
                "connection is being blocked before it starts (some ISPs block "
                "TikTok outright). Set TIKTOK_PROXY to a proxy on a network that "
                "can reach TikTok, or run the worker somewhere that can."
            )
        return None
