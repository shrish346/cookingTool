import yt_dlp
import re
import tempfile
import os
import time
from pathlib import Path
from typing import Optional
from .base import VideoInfo, VideoMetadata

# yt-dlp colours its errors for a terminal. That escape sequence travels all the way
# into the job's error field and is rendered as mojibake in the browser, so strip it
# from anything a user might read.
_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


def clean_error_message(exc: object) -> str:
    """The text of an error, without terminal colour codes."""
    return _ANSI_ESCAPE.sub('', str(exc)).strip()


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

    # Attempts for a download that fails for no reason we can name. Linear backoff:
    # a flake clears in seconds, and the user is watching a progress bar meanwhile.
    DOWNLOAD_ATTEMPTS: int = 3
    RETRY_BACKOFF_SECONDS: float = 2.0

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
        """Download a video and return VideoInfo, retrying a transient failure.

        YouTube intermittently hands back an empty body for a media URL it just
        served ("ERROR: The downloaded file is empty"). Nothing about the request is
        wrong and the next attempt usually works, so a single flake must not cost the
        user their whole job.

        A failure `explain_error` recognises is *not* retried: those are settled
        conditions (a login wall, an unreachable network) that a second attempt would
        only fail slower against.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.DOWNLOAD_ATTEMPTS + 1):
            try:
                return self._download_once(url)
            except (yt_dlp.utils.DownloadError, FileNotFoundError) as e:
                friendly = self.explain_error(e)
                if friendly:
                    raise RuntimeError(friendly) from e

                last_error = e
                self._discard_temp_dir()

                if attempt < self.DOWNLOAD_ATTEMPTS:
                    time.sleep(self.RETRY_BACKOFF_SECONDS * attempt)

        raise RuntimeError(
            f"The video failed to download after {self.DOWNLOAD_ATTEMPTS} attempts. "
            "This is usually temporary - please try again. "
            f"(Last error: {clean_error_message(last_error)})"
        ) from last_error

    def _download_once(self, url: str) -> VideoInfo:
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
        finally:
            if 'cookiefile' in ydl_opts:
                Path(ydl_opts['cookiefile']).unlink(missing_ok=True)

    def _discard_temp_dir(self) -> None:
        """Drop the temp dir from a failed attempt, so a retry starts on empty ground
        and a half-written file can't be mistaken for the download."""
        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
            self._temp_dir = None

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
            # Raw yt-dlp text still names the real cause, which is what we want here -
            # but not with the terminal colour codes still in it.
            raise RuntimeError(clean_error_message(e)) from e
        finally:
            if 'cookiefile' in ydl_opts:
                Path(ydl_opts['cookiefile']).unlink(missing_ok=True)
