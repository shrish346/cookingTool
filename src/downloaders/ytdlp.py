import yt_dlp
import re
import tempfile
import os
from pathlib import Path
from typing import Optional
from .base import VideoInfo, VideoMetadata


class YtDlpDownloader:
    """Shared yt-dlp download/probe logic. Subclasses declare the platform.

    A platform subclass sets:
      URL_RE       — cheap host/path check for supports(); no network.
      FORMAT       — yt-dlp format selector.
      COOKIES_ENV  — env var holding a Netscape cookie file, if the platform
                     needs one to serve datacenter IPs.
      PROXY_ENV    — env var holding a proxy URL, for a platform the worker's
                     own network can't reach (see TikTokDownloader).
    """

    URL_RE: re.Pattern
    FORMAT: str = "best[ext=mp4]/best"
    COOKIES_ENV: Optional[str] = None
    PROXY_ENV: Optional[str] = None

    def __init__(self, cookies: Optional[str] = None):
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        env_cookies = os.getenv(self.COOKIES_ENV) if self.COOKIES_ENV else None
        self._cookies = cookies or env_cookies
        self._proxy = os.getenv(self.PROXY_ENV) if self.PROXY_ENV else None

    def _get_ydl_opts(self, temp_dir_path: Optional[str] = None) -> dict:
        opts = {
            'format': self.FORMAT,
            'quiet': True,
            'no_warnings': True,
        }

        if self._proxy:
            opts['proxy'] = self._proxy

        if temp_dir_path:
            opts['outtmpl'] = str(Path(temp_dir_path) / '%(title)s.%(ext)s')

        if self._cookies:
            # Create a temporary cookie file
            cookie_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            cookie_file.write(self._cookies)
            cookie_file.close()
            opts['cookiefile'] = cookie_file.name

        return opts

    def supports(self, url: str) -> bool:
        """Whether this looks like a URL for this platform (pattern match only).

        Deliberately does not hit the network: a network call here both made
        validation slow (two fetches) and swallowed the real error, so callers
        saw a misleading "unsupported platform". Reachability is checked by
        get_info()/download(), which let the actual yt-dlp error propagate.
        """
        return bool(self.URL_RE.search(url or ""))

    def explain_error(self, exc: Exception) -> Optional[str]:
        """Hook: rewrite a known yt-dlp failure into an actionable message.

        Returning None keeps the raw yt-dlp error, which is the right default —
        it names the real cause. Override only for failures the operator can
        actually fix, since this string is what the user sees on a failed job.
        """
        return None

    def download(self, url: str) -> VideoInfo:
        """Download a video and return VideoInfo."""
        # Create temp directory (not using 'with' so it persists)
        self._temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = self._temp_dir.name

        ydl_opts = self._get_ydl_opts(temp_dir_path)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first to get metadata
                info = ydl.extract_info(url, download=False)

                # Download the video
                ydl.download([url])

                # Find the downloaded file
                temp_path = Path(temp_dir_path)
                files = [f for f in temp_path.iterdir() if f.is_file()]

                if not files:
                    raise FileNotFoundError(f"No video file found in {temp_dir_path}")

                video_path = files[0]

                return VideoInfo(
                    title=info.get('title', 'Unknown'),
                    file_path=video_path,
                    url=url,
                    duration_seconds=int(info.get('duration', 0)),
                    description=info.get('description'),
                )
        except yt_dlp.utils.DownloadError as e:
            friendly = self.explain_error(e)
            if friendly:
                raise RuntimeError(friendly) from e
            raise
        finally:
            if 'cookiefile' in ydl_opts:
                Path(ydl_opts['cookiefile']).unlink(missing_ok=True)

    def cleanup(self, video_info: VideoInfo) -> None:
        """Delete the downloaded video file and temp directory."""
        # Delete the video file if it exists
        if video_info.file_path.exists():
            video_info.file_path.unlink()

        # Clean up the temp directory
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def get_info(self, url: str) -> Optional[VideoMetadata]:
        """
        Get video metadata without downloading.
        Used for deep validation to check if video exists and is accessible.
        """
        ydl_opts = self._get_ydl_opts()
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                if info is None:
                    return None

                return VideoMetadata(
                    title=info.get('title', 'Unknown'),
                    video_id=info.get('id', ''),
                    duration_seconds=int(info.get('duration', 0)),
                    thumbnail_url=info.get('thumbnail'),
                    description=info.get('description'),
                )
        except yt_dlp.utils.DownloadError as e:
            friendly = self.explain_error(e)
            if friendly:
                raise RuntimeError(friendly) from e
            raise
        finally:
            if 'cookiefile' in ydl_opts:
                Path(ydl_opts['cookiefile']).unlink(missing_ok=True)
