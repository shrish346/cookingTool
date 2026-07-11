"""
Residential download worker for MakerAI.

Why this exists: YouTube bot-blocks the cloud backend's datacenter IP (Railway
runs on GCP; YouTube is Google), so the cloud cannot download videos itself.
This worker runs on a residential connection (your laptop, a Raspberry Pi, ...)
and does ONLY the download step. Everything else — transcription, VLM/LLM,
clip extraction — stays on the cloud.

Flow (pull-based, because the home machine is behind NAT and can only make
outbound connections):

    1. Poll  GET  {CLOUD_API_URL}/api/internal/next-job   for a queued job.
    2. Download the video with yt-dlp over the residential IP.
    3. Upload the raw file to R2 at  {video_id}/source.{ext}.
    4. Report back POST {CLOUD_API_URL}/api/internal/job-complete
       (or the same endpoint with an `error` on failure).

All calls are authenticated with the shared DOWNLOADER_SECRET, so Redis never
has to be exposed to the internet.

Run it with:  python downloader_worker.py
Requires a populated .env (S3_*, YOUTUBE_COOKIES, CLOUD_API_URL, DOWNLOADER_SECRET).
"""
import os
import sys
import time
from urllib.parse import urlparse

import boto3
import httpx
from dotenv import load_dotenv

from src.downloaders.factory import get_downloader

load_dotenv()

CLOUD_API_URL = os.getenv("CLOUD_API_URL", "").rstrip("/")
DOWNLOADER_SECRET = os.getenv("DOWNLOADER_SECRET", "")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "chefs-loop-clips")
YOUTUBE_COOKIES = os.getenv("YOUTUBE_COOKIES", "")

POLL_INTERVAL = 5  # seconds between polls when the queue is empty


def _make_s3_client():
    """S3/R2 client. Mirrors backend/api/deps.py: strip any path from the
    endpoint so boto3 doesn't double the bucket into every key."""
    endpoint = (os.getenv("S3_ENDPOINT_URL") or "").strip()
    if endpoint:
        parsed = urlparse(endpoint)
        endpoint = f"{parsed.scheme}://{parsed.netloc}"
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY", ""),
        region_name=os.getenv("S3_REGION", "auto"),
        endpoint_url=endpoint or None,
    )


def _auth_headers() -> dict:
    return {"X-Downloader-Secret": DOWNLOADER_SECRET}


def _report(client: httpx.Client, payload: dict):
    """POST job-complete (success meta or {error}) back to the cloud."""
    resp = client.post(
        f"{CLOUD_API_URL}/api/internal/job-complete",
        headers=_auth_headers(),
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()


def handle_job(client: httpx.Client, s3, video_id: str, url: str):
    """Download one video and hand the raw file off to the cloud via R2."""
    print(f"[worker] job {video_id}: downloading {url}")
    downloader = get_downloader(url, cookies=YOUTUBE_COOKIES or None)
    if downloader is None:
        _report(client, {"video_id": video_id, "error": "Unsupported URL"})
        return

    video_info = None
    try:
        video_info = downloader.download(url)
        ext = video_info.file_path.suffix.lstrip(".") or "mp4"
        key = f"{video_id}/source.{ext}"

        print(f"[worker] job {video_id}: uploading {key} to R2")
        s3.upload_file(str(video_info.file_path), S3_BUCKET_NAME, key)

        _report(client, {
            "video_id": video_id,
            "ext": ext,
            "title": video_info.title,
            "duration": video_info.duration_seconds,
            "description": video_info.description,
        })
        print(f"[worker] job {video_id}: done")
    except Exception as e:
        # A failed download must not wedge the cloud job — tell it to fail fast.
        print(f"[worker] job {video_id}: FAILED — {e}")
        _report(client, {"video_id": video_id, "error": str(e)})
    finally:
        if video_info is not None:
            try:
                downloader.cleanup(video_info)
            except Exception:
                pass


def main():
    if not CLOUD_API_URL or not DOWNLOADER_SECRET:
        sys.exit("Set CLOUD_API_URL and DOWNLOADER_SECRET in .env before running.")

    s3 = _make_s3_client()
    print(f"[worker] polling {CLOUD_API_URL} every {POLL_INTERVAL}s — Ctrl-C to stop")

    with httpx.Client() as client:
        while True:
            try:
                resp = client.get(
                    f"{CLOUD_API_URL}/api/internal/next-job",
                    headers=_auth_headers(),
                    timeout=30,
                )
                resp.raise_for_status()
                job = resp.json()
            except Exception as e:
                print(f"[worker] poll error: {e}")
                time.sleep(POLL_INTERVAL)
                continue

            if not job:
                time.sleep(POLL_INTERVAL)
                continue

            handle_job(client, s3, job["video_id"], job["url"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[worker] stopped")
