# cookingTool

## Requirements

### System Dependencies

- **ffmpeg** - Required for audio extraction from video files
  - **macOS**: `brew install ffmpeg`
  - **Linux (Debian/Ubuntu)**: `sudo apt-get install ffmpeg`
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Python Dependencies

Install Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Python packages included:**

- `yt-dlp` - YouTube video downloading
- `openai` - OpenRouter API client
- `opencv-python` - Video frame extraction
- `pydantic` - Data validation and schemas
- `python-dotenv` - Environment variable management
