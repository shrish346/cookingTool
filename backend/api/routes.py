"""
API routes for the Chef's Loop backend.
"""
import re
import asyncio
from typing import Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from pydantic import BaseModel, Field, HttpUrl

from .deps import get_cache_manager, CacheManager
from .config import get_settings


router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class VideoSource(str, Enum):
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    UNKNOWN = "unknown"


class ValidateRequest(BaseModel):
    url: str = Field(..., description="YouTube or TikTok URL to validate")


class ValidateResponse(BaseModel):
    valid: bool
    source: VideoSource
    video_id: Optional[str] = None
    error: Optional[str] = None


class ProcessRequest(BaseModel):
    url: str = Field(..., description="YouTube or TikTok URL to process")


class ProcessResponse(BaseModel):
    job_id: str
    video_id: str
    status: str = "processing"
    message: str = "Link found! Generating recipe..."


class JobStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    EXTRACTING_CLIPS = "extracting_clips"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100, description="Progress percentage")
    message: str
    recipe: Optional[dict] = None
    clips_ready: bool = False
    error: Optional[str] = None


# ============================================================================
# URL Validation Helpers
# ============================================================================

# YouTube patterns
YOUTUBE_PATTERNS = [
    r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
]

# TikTok patterns
TIKTOK_PATTERNS = [
    r'(?:https?://)?(?:www\.)?tiktok\.com/@[\w.-]+/video/(\d+)',
    r'(?:https?://)?(?:vm\.)?tiktok\.com/(\w+)',
    r'(?:https?://)?(?:www\.)?tiktok\.com/t/(\w+)',
]


def extract_video_id(url: str) -> tuple[VideoSource, Optional[str]]:
    """
    Extract video ID and source from a URL.
    Returns (source, video_id) tuple.
    """
    # Check YouTube patterns
    for pattern in YOUTUBE_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return VideoSource.YOUTUBE, match.group(1)
    
    # Check TikTok patterns
    for pattern in TIKTOK_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return VideoSource.TIKTOK, match.group(1)
    
    return VideoSource.UNKNOWN, None


async def deep_validate_url(url: str, source: VideoSource, video_id: str) -> tuple[bool, Optional[str]]:
    """
    Perform deep validation by checking if the video actually exists.
    Returns (is_valid, error_message).
    """
    from src.downloaders.factory import get_downloader
    
    try:
        downloader = get_downloader(url)
        if downloader is None:
            return False, "Unsupported video platform"
        
        # Try to get video info without downloading
        # This validates the video exists and is accessible
        info = await asyncio.to_thread(downloader.get_info, url)
        if info is None:
            return False, "Video not found or is private"
        
        return True, None
    except Exception as e:
        return False, str(e)


# ============================================================================
# Routes
# ============================================================================

@router.post("/validate", response_model=ValidateResponse)
async def validate_url(request: ValidateRequest):
    """
    Deep validation of a YouTube or TikTok URL.
    Checks if the URL format is valid AND if the video exists.
    """
    url = request.url.strip()
    
    # Step 1: Pattern matching (fast)
    source, video_id = extract_video_id(url)
    
    if source == VideoSource.UNKNOWN or video_id is None:
        return ValidateResponse(
            valid=False,
            source=VideoSource.UNKNOWN,
            error="Invalid URL format. Please enter a valid YouTube or TikTok link."
        )
    
    # Step 2: Deep validation (network call)
    is_valid, error = await deep_validate_url(url, source, video_id)
    
    if not is_valid:
        return ValidateResponse(
            valid=False,
            source=source,
            video_id=video_id,
            error=error or "Could not access video"
        )
    
    return ValidateResponse(
        valid=True,
        source=source,
        video_id=video_id
    )


@router.post("/process", response_model=ProcessResponse)
async def process_video(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    cache: CacheManager = Depends(get_cache_manager)
):
    """
    Start processing a video URL.
    Returns immediately with a job ID for status polling.
    """
    url = request.url.strip()
    
    # Validate URL first
    source, video_id = extract_video_id(url)
    if source == VideoSource.UNKNOWN or video_id is None:
        raise HTTPException(status_code=400, detail="Invalid URL")
    
    # Increment hit counter
    hit_count = await cache.increment_hit_count(video_id)
    
    # Check if we should use cache or refresh
    should_refresh = await cache.should_refresh_cache(video_id)
    
    if not should_refresh:
        # Try to get cached recipe
        cached_recipe = await cache.get_cached_recipe(video_id)
        if cached_recipe:
            # Store the cached result for immediate retrieval
            from .deps import get_redis
            redis_client = await get_redis()
            await redis_client.set(
                f"job:{video_id}:status",
                "completed",
                ex=3600
            )
            await redis_client.set(
                f"job:{video_id}:progress",
                "100",
                ex=3600
            )
            
            return ProcessResponse(
                job_id=video_id,
                video_id=video_id,
                status="completed",
                message="Recipe ready!"
            )
    
    # Start background processing
    background_tasks.add_task(
        run_pipeline,
        url=url,
        video_id=video_id,
        source=source
    )
    
    return ProcessResponse(
        job_id=video_id,
        video_id=video_id,
        status="processing",
        message="Link found! Generating recipe..."
    )


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(
    job_id: str,
    cache: CacheManager = Depends(get_cache_manager)
):
    """
    Get the status of a processing job.
    Poll this endpoint to track progress.
    """
    from .deps import get_redis
    redis_client = await get_redis()
    
    # Get job status from Redis
    status = await redis_client.get(f"job:{job_id}:status")
    progress = await redis_client.get(f"job:{job_id}:progress")
    message = await redis_client.get(f"job:{job_id}:message")
    error = await redis_client.get(f"job:{job_id}:error")
    
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get recipe if completed
    recipe = None
    clips_ready = False
    
    if status == "completed":
        cached = await cache.get_cached_recipe(job_id)
        if cached:
            recipe = cached
            clips_ready = cached.get("clips_ready", False)
    
    return StatusResponse(
        job_id=job_id,
        status=JobStatus(status),
        progress=int(progress or 0),
        message=message or "Processing...",
        recipe=recipe,
        clips_ready=clips_ready,
        error=error
    )


# ============================================================================
# Background Pipeline Task
# ============================================================================

async def run_pipeline(url: str, video_id: str, source: VideoSource):
    """
    Background task that runs the full VLM → LLM → FFmpeg pipeline.
    Updates Redis with progress for status polling.
    """
    from .deps import get_redis, get_cache_manager
    from src.downloaders.factory import get_downloader
    from src.processing.frames import FrameExtractor
    from src.processing.audio import AudioTranscriber
    from src.vlm.openrouter import OpenRouterAdapter
    from src.llm.openai import OpenAIAdapter
    from src.chef import RecipeChef
    from backend.processors.video import VideoClipExtractor
    
    redis_client = await get_redis()
    cache = await get_cache_manager()
    settings = get_settings()
    
    async def update_status(status: str, progress: int, message: str):
        await redis_client.set(f"job:{video_id}:status", status, ex=3600)
        await redis_client.set(f"job:{video_id}:progress", str(progress), ex=3600)
        await redis_client.set(f"job:{video_id}:message", message, ex=3600)
    
    try:
        # Step 1: Download video
        await update_status("downloading", 10, "Downloading video...")
        
        downloader = get_downloader(url)
        video_info = await asyncio.to_thread(downloader.download, url)
        video_path = str(video_info.file_path)
        
        try:
            # Step 2: Extract frames
            await update_status("analyzing", 25, "Extracting frames...")
            
            extractor = FrameExtractor(resize_width=512)
            frames = await asyncio.to_thread(extractor.extract, video_path)
            
            # Step 3: Transcribe audio
            await update_status("analyzing", 40, "Transcribing audio...")
            
            transcriber = AudioTranscriber()
            transcript = await asyncio.to_thread(transcriber.process_video, video_path)
            
            # Step 4: VLM + LLM pipeline
            await update_status("generating", 55, "Analyzing cooking steps...")
            
            vlm_adapter = OpenRouterAdapter()
            llm_adapter = OpenAIAdapter()
            
            chef = RecipeChef(vlm_adapter=vlm_adapter, llm_adapter=llm_adapter)
            recipe = await asyncio.to_thread(
                chef.generate_recipe,
                video_info,
                frames,
                transcript
            )
            
            # Add video metadata to recipe
            recipe_dict = recipe.model_dump()
            recipe_dict["video_id"] = video_id
            recipe_dict["clips_ready"] = False
            
            # Save recipe (without clips)
            await update_status("generating", 70, "Recipe generated! Preparing video clips...")
            await cache.save_recipe(video_id, recipe_dict)
            
            # Step 5: Extract video clips
            await update_status("extracting_clips", 80, "Extracting video segments...")
            
            clip_extractor = VideoClipExtractor(
                s3_client=cache.s3,
                bucket_name=settings.s3_bucket_name
            )
            
            # Extract and upload clips for each step
            updated_steps = await asyncio.to_thread(
                clip_extractor.extract_and_upload_clips,
                video_path=video_path,
                video_id=video_id,
                steps=recipe_dict["steps"],
                fps=recipe_dict.get("video_fps", 30.0)
            )
            
            # Update recipe with clip URLs
            recipe_dict["steps"] = updated_steps
            recipe_dict["clips_ready"] = True
            
            await update_status("uploading", 95, "Uploading clips...")
            await cache.save_recipe(video_id, recipe_dict)
            
            # Done!
            await update_status("completed", 100, "Recipe ready!")
            
        finally:
            # Cleanup downloaded video
            await asyncio.to_thread(downloader.cleanup, video_info)
    
    except Exception as e:
        # Rollback on failure
        await cache.delete_recipe(video_id)
        await redis_client.set(f"job:{video_id}:status", "failed", ex=3600)
        await redis_client.set(f"job:{video_id}:error", str(e), ex=3600)
        await redis_client.set(f"job:{video_id}:message", "Failed to generate recipe", ex=3600)

