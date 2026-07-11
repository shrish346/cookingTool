"""
Application configuration loaded from environment variables.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    app_name: str = "Chef's Loop API"
    debug: bool = False

    # Comma-separated list of allowed browser origins.
    cors_origins: str = "http://localhost:5173"

    # Redis (Upstash)
    redis_url: str = "redis://localhost:6379"

    # Object storage (S3-compatible; Cloudflare R2 in production)
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_region: str = "auto"  # R2 requires "auto"; AWS wants a real region
    s3_bucket_name: str = "chefs-loop-clips"
    # Empty means "talk to real AWS S3". For R2:
    # https://<account_id>.r2.cloudflarestorage.com
    s3_endpoint_url: str = ""
    # Domain clips are served from. R2's r2.dev is rate-limited and unfit for
    # production, so this should be a custom domain. Empty falls back to AWS.
    s3_public_base_url: str = ""

    # Rate Limiting
    rate_limit_per_hour: int = 20
    
    # Cache Settings
    cache_refresh_threshold: int = 5  # Re-run pipeline every N hits
    
    # AI Provider Keys
    openrouter_api_key: str = ""
    openai_api_key: str = ""
    
    # Video Processing
    ffmpeg_path: str = "ffmpeg"
    clip_format: str = "webm"  # webm or mp4
    clip_bitrate: str = "500k"
    
    # YouTube Cookies
    youtube_cookies: str = ""

    # Residential download worker
    # Shared secret authenticating the home worker's /internal/* calls.
    downloader_secret: str = ""
    # How long the pipeline waits for the worker to upload the raw video
    # before failing the job with "Downloader offline".
    download_wait_timeout: int = 300

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()



