"""
API routes for the Chef's Loop backend.
"""
import re
import os
import json
import asyncio
import tempfile
from pathlib import Path
from typing import Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, Header
from pydantic import BaseModel, Field, HttpUrl

from backend.api.deps import get_cache_manager, CacheManager
from backend.api.config import get_settings


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


# NOTE: deep (network) validation was removed. The cloud can't reach YouTube
# from its datacenter IP; the residential download worker is what actually
# fetches the video. So /validate is pattern-only and a video's real
# reachability surfaces at process time (as a failed job with the yt-dlp error).


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
    
    # Reachability is confirmed later by the residential download worker (the
    # cloud can't reach YouTube from its datacenter IP), so this is pattern-only.
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
            from backend.api.deps import get_redis
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
    from backend.api.deps import get_redis
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
# Internal endpoints — used only by the residential download worker
# ============================================================================

DOWNLOAD_QUEUE_KEY = "dl:queue"


class JobCompleteRequest(BaseModel):
    video_id: str
    ext: Optional[str] = "mp4"
    title: Optional[str] = None
    duration: Optional[int] = 0
    description: Optional[str] = None
    error: Optional[str] = None


def _check_worker_auth(secret: Optional[str]):
    settings = get_settings()
    if not settings.downloader_secret or secret != settings.downloader_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/internal/next-job")
async def internal_next_job(x_downloader_secret: Optional[str] = Header(None)):
    """The download worker polls this to claim the next pending job (or {})."""
    _check_worker_auth(x_downloader_secret)
    from backend.api.deps import get_redis
    redis_client = await get_redis()
    item = await redis_client.lpop(DOWNLOAD_QUEUE_KEY)
    return json.loads(item) if item else {}


@router.post("/internal/job-complete")
async def internal_job_complete(
    req: JobCompleteRequest,
    x_downloader_secret: Optional[str] = Header(None),
):
    """The worker reports the raw video is in R2 (or that the download failed)."""
    _check_worker_auth(x_downloader_secret)
    from backend.api.deps import get_redis
    redis_client = await get_redis()
    if req.error:
        await redis_client.set(f"job:{req.video_id}:source_error", req.error, ex=600)
    else:
        meta = {
            "ext": req.ext or "mp4",
            "title": req.title or "Unknown",
            "duration": req.duration or 0,
            "description": req.description,
        }
        await redis_client.set(f"job:{req.video_id}:source", json.dumps(meta), ex=600)
    return {"ok": True}


async def _await_source(redis_client, video_id: str, timeout: int) -> dict:
    """Wait for the worker to upload the raw video and report back."""
    waited = 0
    interval = 3
    while waited < timeout:
        err = await redis_client.get(f"job:{video_id}:source_error")
        if err:
            raise RuntimeError(err)
        meta = await redis_client.get(f"job:{video_id}:source")
        if meta:
            return json.loads(meta)
        await asyncio.sleep(interval)
        waited += interval
    raise RuntimeError(
        "Downloader offline — start your local download worker and try again."
    )


# ============================================================================
# Background Pipeline Task
# ============================================================================

async def run_pipeline(url: str, video_id: str, source: VideoSource):
    """
    Background task that runs the full VLM → LLM → FFmpeg pipeline.

    The download step does NOT happen here — the cloud can't reach YouTube from
    its datacenter IP. Instead we enqueue a download job, wait for the residential
    worker to upload the raw video to R2, pull it back, and run the rest of the
    pipeline as before. Updates Redis with progress for status polling.
    """
    from backend.api.deps import get_redis, get_cache_manager
    from src.processing.audio import AudioTranscriber
    from src.vlm.openrouter import OpenRouterAdapter
    from src.llm.openai import OpenAIAdapter
    from src.chef import RecipeChef
    from src.downloaders.base import VideoInfo
    from backend.processors.video import VideoClipExtractor

    redis_client = await get_redis()
    cache = await get_cache_manager()
    settings = get_settings()

    async def update_status(status: str, progress: int, message: str):
        await redis_client.set(f"job:{video_id}:status", status, ex=3600)
        await redis_client.set(f"job:{video_id}:progress", str(progress), ex=3600)
        await redis_client.set(f"job:{video_id}:message", message, ex=3600)

    tmp_path: Optional[str] = None
    source_key: Optional[str] = None
    try:
        # Step 1: Hand the download off to the residential worker and wait for it.
        await update_status("downloading", 10, "Downloading video...")

        # Clear any stale source signals from a previous run of this video.
        await redis_client.delete(
            f"job:{video_id}:source", f"job:{video_id}:source_error"
        )
        # Also clear a lingering error from a prior failed run (it isn't cleared
        # elsewhere, so a re-run could surface a stale error).
        await redis_client.delete(f"job:{video_id}:error")

        await redis_client.rpush(
            DOWNLOAD_QUEUE_KEY, json.dumps({"video_id": video_id, "url": url})
        )
        meta = await _await_source(redis_client, video_id, settings.download_wait_timeout)

        # Pull the raw video the worker uploaded to R2 down to a local tempfile.
        ext = meta.get("ext") or "mp4"
        source_key = f"{video_id}/source.{ext}"
        fd, tmp_path = tempfile.mkstemp(suffix=f".{ext}")
        os.close(fd)
        await asyncio.to_thread(
            cache.s3.download_file,
            settings.s3_bucket_name,
            source_key,
            tmp_path,
        )

        video_info = VideoInfo(
            title=meta.get("title") or "Unknown",
            file_path=Path(tmp_path),
            url=url,
            duration_seconds=meta.get("duration") or 0,
            description=meta.get("description"),
        )
        video_path = tmp_path

        # Step 2: Transcribe audio
        await update_status("analyzing", 25, "Transcribing audio...")

        transcriber = AudioTranscriber()
        transcript = await asyncio.to_thread(transcriber.process_video, video_path)

        # Step 3: VLM + LLM pipeline (using direct video upload to Gemini)
        await update_status("generating", 50, "Analyzing steps and generating recipe...")

        vlm_adapter = OpenRouterAdapter()  # Defaults to google/gemini-2.5-flash
        llm_adapter = OpenAIAdapter()

        chef = RecipeChef(vlm_adapter=vlm_adapter, llm_adapter=llm_adapter)
        # Use direct video upload (no frame extraction)
        recipe = await asyncio.to_thread(
            chef.generate_recipe_direct,
            video_info,
            video_path,
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
            steps=recipe_dict["steps"]
        )

        # Update recipe with clip URLs
        recipe_dict["steps"] = updated_steps
        recipe_dict["clips_ready"] = True

        await update_status("uploading", 95, "Uploading clips...")
        await cache.save_recipe(video_id, recipe_dict)

        # Done!
        await update_status("completed", 100, "Recipe ready!")

    except Exception as e:
        # Rollback on failure
        await cache.delete_recipe(video_id)
        await redis_client.set(f"job:{video_id}:status", "failed", ex=3600)
        await redis_client.set(f"job:{video_id}:error", str(e), ex=3600)
        await redis_client.set(f"job:{video_id}:message", "Failed to generate recipe", ex=3600)

    finally:
        # Clean up the local tempfile and the raw source object in R2 — the
        # source video is only needed for the duration of this pipeline run.
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if source_key:
            try:
                await asyncio.to_thread(
                    cache.s3.delete_object,
                    Bucket=settings.s3_bucket_name,
                    Key=source_key,
                )
            except Exception:
                pass

