import yt_dlp
import tempfile
import os
from pathlib import Path
from typing import Optional
from .base import VideoInfo, VideoDownloader


class YouTubeDownloader:
    def __init__(self):
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def supports(self, url: str) -> bool:
        """Check if the URL is a supported YouTube URL."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                ydl.extract_info(url, download=False)
            return True
        except Exception:
            return False

    def download(self, url: str) -> VideoInfo:
        """Download a YouTube video and return VideoInfo."""
        # Create temp directory (not using 'with' so it persists)
        self._temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = self._temp_dir.name
        
        ydl_opts = {
            'outtmpl': os.path.join(temp_dir_path, '%(title)s.%(ext)s'),
            'format': 'best[height<=480][ext=mp4]/best[height<=480]/worst',  # Lower quality for smaller files
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to get metadata
            info = ydl.extract_info(url, download=False)
            
            # Download the video
            ydl.download([url])
            
            # Find the downloaded file - yt-dlp saves it based on outtmpl pattern
            # Since we used '%(title)s.%(ext)s', we need to find the actual file
            # The title gets sanitized, so we search for the file
            temp_path = Path(temp_dir_path)
            files = [f for f in temp_path.iterdir() if f.is_file()]
            
            if not files:
                raise FileNotFoundError(f"No video file found in {temp_dir_path}")
            
            # Take the first (and should be only) file
            video_path = files[0]
            
            return VideoInfo(
                title=info.get('title', 'Unknown'),
                file_path=video_path,
                url=url,
                duration_seconds=int(info.get('duration', 0)),
                description=info.get('description'),
            )

    def cleanup(self, video_info: VideoInfo) -> None:
        """Delete the downloaded video file and temp directory."""
        # Delete the video file if it exists
        if video_info.file_path.exists():
            video_info.file_path.unlink()
        
        # Clean up the temp directory
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None