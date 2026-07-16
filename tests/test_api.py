"""API layer: video-ID extraction, CacheManager semantics, and endpoints.

Endpoints run against a TestClient with fake Redis/S3 injected - no network.
"""
import json

import pytest

from backend.api.deps import CacheManager
from backend.api.routes import VideoSource, extract_video_id
from src.schemas import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# extract_video_id - the ID is the only cache key; collisions serve the wrong
# recipe, so non-YouTube IDs must be namespaced.
# ---------------------------------------------------------------------------

class TestExtractVideoId:
    @pytest.mark.parametrize("url, source, video_id", [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", VideoSource.YOUTUBE, "dQw4w9WgXcQ"),
        ("https://youtube.com/shorts/K_L-FVXOR0o", VideoSource.YOUTUBE, "K_L-FVXOR0o"),
        ("https://youtu.be/dQw4w9WgXcQ", VideoSource.YOUTUBE, "dQw4w9WgXcQ"),
        ("https://www.tiktok.com/@cook/video/7301234567890123456", VideoSource.TIKTOK,
         "tt-7301234567890123456"),
        ("https://vm.tiktok.com/ZMhabcdef/", VideoSource.TIKTOK, "tt-ZMhabcdef"),
        ("https://www.instagram.com/reel/C1a2B3c4D5e/", VideoSource.INSTAGRAM,
         "ig-C1a2B3c4D5e"),
        ("https://www.instagram.com/p/C1a2B3c4D5e/", VideoSource.INSTAGRAM,
         "ig-C1a2B3c4D5e"),
    ])
    def test_extraction(self, url, source, video_id):
        assert extract_video_id(url) == (source, video_id)

    def test_junk_url(self):
        assert extract_video_id("https://example.com/nope") == (VideoSource.UNKNOWN, None)

    def test_instagram_and_youtube_ids_cannot_collide(self):
        """An IG shortcode uses the same 11-char alphabet as a YouTube ID -
        the ig- prefix is what keeps their cache entries apart."""
        _, yt = extract_video_id("https://youtu.be/C1a2B3c4D5e")
        _, ig = extract_video_id("https://www.instagram.com/reel/C1a2B3c4D5e/")
        assert yt != ig


# ---------------------------------------------------------------------------
# CacheManager against the fakes
# ---------------------------------------------------------------------------

def _recipe_dict(version=SCHEMA_VERSION) -> dict:
    return {"schema_version": version, "title": "t", "clips_ready": True}


class TestCacheManager:
    pytestmark = pytest.mark.anyio

    async def test_save_stamps_current_version(self, fake_redis, fake_s3):
        cache = CacheManager(fake_redis, fake_s3)
        await cache.save_recipe("vid1", {"title": "t"})
        stored = json.loads(fake_redis.store["recipe:vid1"])
        assert stored["schema_version"] == SCHEMA_VERSION
        assert "vid1/recipe.json" in fake_s3.store

    async def test_redis_hit(self, fake_redis, fake_s3):
        cache = CacheManager(fake_redis, fake_s3)
        fake_redis.store["recipe:vid1"] = json.dumps(_recipe_dict())
        assert (await cache.get_cached_recipe("vid1"))["title"] == "t"

    async def test_old_schema_in_redis_is_a_miss(self, fake_redis, fake_s3):
        cache = CacheManager(fake_redis, fake_s3)
        fake_redis.store["recipe:vid1"] = json.dumps(_recipe_dict(version=SCHEMA_VERSION - 1))
        assert await cache.get_cached_recipe("vid1") is None

    async def test_s3_hit_backfills_redis(self, fake_redis, fake_s3):
        cache = CacheManager(fake_redis, fake_s3)
        fake_s3.store["vid1/recipe.json"] = json.dumps(_recipe_dict()).encode()
        assert (await cache.get_cached_recipe("vid1"))["title"] == "t"
        assert "recipe:vid1" in fake_redis.store

    async def test_old_schema_in_s3_is_a_miss(self, fake_redis, fake_s3):
        """S3 recipes have no TTL - without the version check an old-shape
        recipe.json would be served forever."""
        cache = CacheManager(fake_redis, fake_s3)
        fake_s3.store["vid1/recipe.json"] = json.dumps(_recipe_dict(version=1)).encode()
        assert await cache.get_cached_recipe("vid1") is None

    async def test_total_miss(self, fake_redis, fake_s3):
        cache = CacheManager(fake_redis, fake_s3)
        assert await cache.get_cached_recipe("vid1") is None

    async def test_delete_clears_whole_prefix(self, fake_redis, fake_s3):
        cache = CacheManager(fake_redis, fake_s3)
        fake_redis.store["recipe:vid1"] = "{}"
        fake_s3.store["vid1/recipe.json"] = b"{}"
        fake_s3.store["vid1/clips/step1.mp4"] = b"..."
        await cache.delete_recipe("vid1")
        assert "recipe:vid1" not in fake_redis.store
        assert not any(k.startswith("vid1/") for k in fake_s3.store)

    async def test_refresh_threshold(self, fake_redis, fake_s3):
        cache = CacheManager(fake_redis, fake_s3)
        threshold = cache.settings.cache_refresh_threshold
        for _ in range(threshold - 1):
            await cache.increment_hit_count("vid1")
        assert await cache.should_refresh_cache("vid1") is False
        await cache.increment_hit_count("vid1")
        assert await cache.should_refresh_cache("vid1") is True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class TestEndpoints:
    def test_health(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_validate_good_url(self, api_client):
        response = api_client.post("/api/validate",
                                   json={"url": "https://youtu.be/dQw4w9WgXcQ"})
        body = response.json()
        assert body["valid"] is True
        assert body["source"] == "youtube"
        assert body["video_id"] == "dQw4w9WgXcQ"

    def test_validate_bad_url(self, api_client):
        response = api_client.post("/api/validate", json={"url": "https://example.com"})
        body = response.json()
        assert body["valid"] is False
        assert body["error"]

    def test_status_unknown_job_404(self, api_client):
        assert api_client.get("/api/status/nope").status_code == 404

    def test_status_completed_returns_recipe(self, api_client, fake_redis, fake_s3):
        fake_redis.store["job:vid1:status"] = "completed"
        fake_redis.store["job:vid1:progress"] = "100"
        fake_redis.store["recipe:vid1"] = json.dumps(_recipe_dict())
        body = api_client.get("/api/status/vid1").json()
        assert body["status"] == "completed"
        assert body["recipe"]["title"] == "t"
        assert body["clips_ready"] is True

    def test_status_completed_with_stale_recipe_omits_it(self, api_client, fake_redis):
        fake_redis.store["job:vid1:status"] = "completed"
        fake_redis.store["recipe:vid1"] = json.dumps(_recipe_dict(version=1))
        body = api_client.get("/api/status/vid1").json()
        assert body["recipe"] is None

    def test_process_invalid_url_400(self, api_client):
        response = api_client.post("/api/process", json={"url": "https://example.com"})
        assert response.status_code == 400

    def test_process_cache_hit_completes_immediately(self, api_client, fake_redis,
                                                     monkeypatch):
        import backend.api.routes as routes

        async def fail_if_called(**kwargs):
            raise AssertionError("run_pipeline must not run on a cache hit")

        monkeypatch.setattr(routes, "run_pipeline", fail_if_called)

        video_id = "dQw4w9WgXcQ"
        fake_redis.store[f"recipe:{video_id}"] = json.dumps(_recipe_dict())
        response = api_client.post("/api/process",
                                   json={"url": f"https://youtu.be/{video_id}"})
        body = response.json()
        assert body["status"] == "completed"
        assert body["job_id"] == video_id
        assert fake_redis.store[f"job:{video_id}:status"] == "completed"

    def test_process_cache_miss_starts_pipeline(self, api_client, monkeypatch):
        import backend.api.routes as routes

        calls = {}

        async def fake_pipeline(url, video_id, source):
            calls["video_id"] = video_id

        monkeypatch.setattr(routes, "run_pipeline", fake_pipeline)

        response = api_client.post(
            "/api/process", json={"url": "https://www.instagram.com/reel/C1a2B3c4D5e/"})
        body = response.json()
        assert body["status"] == "processing"
        assert body["job_id"] == "ig-C1a2B3c4D5e"  # namespaced - the collision guard
        assert calls["video_id"] == "ig-C1a2B3c4D5e"  # BackgroundTasks ran on response
