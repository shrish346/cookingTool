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
    
    # Redis (Upstash)
    redis_url: str = "redis://localhost:6379"
    
    # AWS S3
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "chefs-loop-clips"
    
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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

