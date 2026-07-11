import yt_dlp
import re
import tempfile
import os
from pathlib import Path
from typing import Optional
from .base import VideoInfo, VideoDownloader, VideoMetadata


# Cheap host/path check — does NOT hit the network. Whether the video actually
# loads is decided later by get_info()/download(), which surface the real error.
_YOUTUBE_URL_RE = re.compile(
    r'(?:https?://)?(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|shorts/|embed/|live/)|youtu\.be/)',
    re.IGNORECASE,
)


class YouTubeDownloader:
    def __init__(self, cookies: Optional[str] = None):
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._cookies = cookies or os.getenv("YOUTUBE_COOKIES")

    def _get_ydl_opts(self, temp_dir_path: Optional[str] = None) -> dict:
        opts = {
            'format': 'best[height<=480][ext=mp4]/best[height<=480]/worst',
            'quiet': True,
            'no_warnings': True,
        }
        
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
        """Whether this looks like a YouTube URL (pattern match only).

        Deliberately does not hit the network: a network call here both made
        validation slow (two fetches) and swallowed the real error, so callers
        saw a misleading "unsupported platform". Reachability is checked by
        get_info()/download(), which let the actual yt-dlp error propagate.
        """
        return bool(_YOUTUBE_URL_RE.search(url or ""))

    def download(self, url: str) -> VideoInfo:
        """Download a YouTube video and return VideoInfo."""
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
        finally:
            if 'cookiefile' in ydl_opts:
                Path(ydl_opts['cookiefile']).unlink(missing_ok=True)