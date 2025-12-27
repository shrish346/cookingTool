"""
Audio extraction and transcription for cooking videos.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class AudioTranscriber:
    """Extract and transcribe audio from video files."""
    
    # Minimum words to consider transcript meaningful
    MIN_WORDS_THRESHOLD = 10
    
    def __init__(self):
        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self._temp_audio: Path | None = None
    
    def extract_audio(self, video_path: str | Path) -> Path:
        """
        Extract audio track from video file using ffmpeg.
        
        Returns path to temporary MP3 file.
        """
        video_path = Path(video_path)
        
        # Create temp file for audio
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(temp_fd)
        self._temp_audio = Path(temp_path)
        
        # Extract audio with ffmpeg
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",                    # No video
            "-acodec", "libmp3lame",  # MP3 codec
            "-q:a", "4",              # Quality (0-9, lower is better)
            "-y",                     # Overwrite output
            str(self._temp_audio)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        
        return self._temp_audio
    
    def transcribe(self, audio_path: str | Path) -> str | None:
        """
        Transcribe audio using OpenAI Whisper API.
        
        Returns transcript text, or None if no meaningful speech detected.
        """
        audio_path = Path(audio_path)
        
        with open(audio_path, "rb") as audio_file:
            response = self._client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        transcript = response.strip() if isinstance(response, str) else response.text.strip()
        
        # Check if transcript has meaningful content
        if not self._has_meaningful_speech(transcript):
            return None
        
        return transcript
    
    def _has_meaningful_speech(self, transcript: str) -> bool:
        """
        Check if transcript contains meaningful spoken instructions.
        
        Filters out:
        - Empty transcripts
        - Very short transcripts (likely just music or noise)
        - Music-only indicators like "[Music]" or "♪"
        """
        if not transcript:
            return False
        
        # Remove common non-speech markers
        cleaned = transcript
        noise_markers = ["[Music]", "[Applause]", "[Laughter]", "♪", "♫"]
        for marker in noise_markers:
            cleaned = cleaned.replace(marker, "")
        
        cleaned = cleaned.strip()
        
        # Check word count
        words = cleaned.split()
        if len(words) < self.MIN_WORDS_THRESHOLD:
            return False
        
        return True
    
    def process_video(self, video_path: str | Path) -> str | None:
        """
        Extract audio and transcribe in one step.
        
        Returns transcript if meaningful speech found, None otherwise.
        """
        try:
            audio_path = self.extract_audio(video_path)
            return self.transcribe(audio_path)
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Remove temporary audio file."""
        if self._temp_audio and self._temp_audio.exists():
            self._temp_audio.unlink()
            self._temp_audio = None

