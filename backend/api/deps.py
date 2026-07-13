"""
Dependency injection for FastAPI routes.
Provides Redis and S3 clients.
"""
import json
from typing import Optional
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import redis.asyncio as redis
import boto3
from botocore.exceptions import ClientError

from backend.api.config import get_settings
from src.schemas import SCHEMA_VERSION


# Global client instances
_redis_client: Optional[redis.Redis] = None
_s3_client = None


async def get_redis() -> redis.Redis:
    """Get Redis client instance."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
    return _redis_client


def get_s3_client():
    """Get S3 client instance (AWS S3 or any S3-compatible provider)."""
    global _s3_client
    if _s3_client is None:
        settings = get_settings()
        # The endpoint must be host-only. A trailing /<bucket> (an easy paste
        # mistake) makes boto3 double the bucket into every key, so strip any path.
        endpoint = settings.s3_endpoint_url.strip()
        if endpoint:
            parsed = urlparse(endpoint)
            endpoint = f"{parsed.scheme}://{parsed.netloc}"
        _s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name=settings.s3_region,
            # None routes to real AWS; set it to target R2 instead.
            endpoint_url=endpoint or None,
        )
    return _s3_client


class CacheManager:
    """Manages recipe caching with Redis and S3."""
    
    def __init__(self, redis_client: redis.Redis, s3_client):
        self.redis = redis_client
        self.s3 = s3_client
        self.settings = get_settings()
    
    async def get_hit_count(self, video_id: str) -> int:
        """Get the hit count for a video."""
        count = await self.redis.get(f"hits:{video_id}")
        return int(count) if count else 0
    
    async def increment_hit_count(self, video_id: str) -> int:
        """Increment and return the hit count for a video."""
        return await self.redis.incr(f"hits:{video_id}")
    
    async def should_refresh_cache(self, video_id: str) -> bool:
        """Check if we should re-run the pipeline based on hit count."""
        count = await self.get_hit_count(video_id)
        threshold = self.settings.cache_refresh_threshold
        return count > 0 and count % threshold == 0
    
    async def get_cached_recipe(self, video_id: str) -> Optional[dict]:
        """
        Try to get cached recipe from Redis first, then S3.
        Returns None if not found, or if the cached recipe predates the current schema.
        """
        # Try Redis first (fastest)
        cached = await self.redis.get(f"recipe:{video_id}")
        if cached:
            return self._if_current(video_id, json.loads(cached))

        # Try S3 (persistent storage)
        try:
            response = self.s3.get_object(
                Bucket=self.settings.s3_bucket_name,
                Key=f"{video_id}/recipe.json"
            )
            recipe_data = json.loads(response['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

        if self._if_current(video_id, recipe_data) is None:
            return None

        # Cache in Redis for future requests
        await self.redis.set(
            f"recipe:{video_id}",
            json.dumps(recipe_data),
            ex=3600  # 1 hour TTL
        )
        return recipe_data

    def _if_current(self, video_id: str, recipe_data: dict) -> Optional[dict]:
        """Treat a recipe from an older schema as a miss, so it gets regenerated.

        S3 recipes have no TTL, so without this an old-shape recipe.json would be
        served verbatim forever - missing every field the frontend now expects.
        """
        version = recipe_data.get("schema_version", 1)
        if version != SCHEMA_VERSION:
            print(f"[Cache] {video_id}: schema v{version} != v{SCHEMA_VERSION}, regenerating")
            return None
        return recipe_data

    async def save_recipe(self, video_id: str, recipe_data: dict):
        """Save recipe to both Redis and S3."""
        recipe_data = {**recipe_data, "schema_version": SCHEMA_VERSION}
        recipe_json = json.dumps(recipe_data, indent=2)

        # Save to Redis
        await self.redis.set(
            f"recipe:{video_id}",
            recipe_json,
            ex=3600  # 1 hour TTL
        )

        # Save to S3
        self.s3.put_object(
            Bucket=self.settings.s3_bucket_name,
            Key=f"{video_id}/recipe.json",
            Body=recipe_json.encode('utf-8'),
            ContentType='application/json'
        )
    
    async def delete_recipe(self, video_id: str):
        """Delete recipe from Redis and S3 (for rollback on failure)."""
        # Delete from Redis
        await self.redis.delete(f"recipe:{video_id}")
        
        # Delete from S3 - recipe.json and all clips
        try:
            # List all objects with the video_id prefix
            response = self.s3.list_objects_v2(
                Bucket=self.settings.s3_bucket_name,
                Prefix=f"{video_id}/"
            )
            if 'Contents' in response:
                objects = [{'Key': obj['Key']} for obj in response['Contents']]
                self.s3.delete_objects(
                    Bucket=self.settings.s3_bucket_name,
                    Delete={'Objects': objects}
                )
        except ClientError:
            pass  # Ignore deletion errors


async def get_cache_manager() -> CacheManager:
    """Get CacheManager instance."""
    redis_client = await get_redis()
    s3_client = get_s3_client()
    return CacheManager(redis_client, s3_client)

