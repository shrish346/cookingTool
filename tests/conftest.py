"""Shared fixtures and fakes for the pytest suite.

The whole suite runs with no API keys and no network. Model adapters are
constructed via ``__new__`` (never hitting their __init__ clients) or replaced
with fakes; Redis/S3 are dict-backed stubs injected through FastAPI's
dependency overrides.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from src.downloaders.base import VideoInfo
from src.schemas import Entity, MicroAction, Observation, SceneDescription, SceneLog


# ---------------------------------------------------------------------------
# Domain object factories
# ---------------------------------------------------------------------------

@pytest.fixture
def video_info(tmp_path) -> VideoInfo:
    return VideoInfo(
        title="Rice Cooker Fried Rice",
        file_path=tmp_path / "video.mp4",
        url="https://www.youtube.com/shorts/abc123def45",
        duration_seconds=35,
        description="Full recipe: 1 tsp cumin, 2 tomatoes. #food #easyrecipe #yum",
    )


@pytest.fixture
def make_scene_log():
    """Factory for a SceneLog with sensible defaults, overridable per test."""

    def _make(
        micro_actions: list[MicroAction] | None = None,
        entities: list[Entity] | None = None,
        observations: list[Observation] | None = None,
    ) -> SceneLog:
        if micro_actions is None:
            micro_actions = [
                MicroAction(id=0, action="Slicing chicken into strips",
                            timestamp_seconds=2.0, duration_seconds=2.0, entity="Knife"),
                MicroAction(id=1, action="Adding cumin to pan",
                            timestamp_seconds=10.0, duration_seconds=1.0, entity="Cumin",
                            stated_amount="1 tsp"),
                MicroAction(id=2, action="Stirring pan",
                            timestamp_seconds=20.0, duration_seconds=3.0, entity="Pan"),
                MicroAction(id=3, action="Sprinkling smoked paprika over chicken",
                            timestamp_seconds=30.0, duration_seconds=1.0, entity="Smoked paprika"),
            ]
        if entities is None:
            entities = [
                Entity(name="Chicken", type="ingredient"),
                Entity(name="Cumin", type="ingredient", quantity="1 tsp"),
                Entity(name="Smoked paprika", type="ingredient"),
                Entity(name="Knife", type="tool"),
                Entity(name="Pan", type="tool"),
            ]
        scene = SceneDescription(
            frame_index=0,
            entities=entities,
            micro_actions=micro_actions,
            observations=observations or [],
            metadata={"method": "direct_video"},
        )
        return SceneLog(scenes=[scene], video_info={"title": "test"})

    return _make


@pytest.fixture
def scene_log(make_scene_log) -> SceneLog:
    return make_scene_log()


# ---------------------------------------------------------------------------
# Synthetic fixture video (session-scoped; skips when ffmpeg is missing)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> str:
    """A 30s test-pattern video with a burned-in caption, like a creator overlay."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")

    path = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc2=duration=30:size=1280x720:rate=24",
            "-vf", "drawtext=text='1 tsp cumin':fontsize=48:fontcolor=white:x=50:y=50",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg could not build fixture video: {result.stderr[-200:]}")
    return str(path)


# ---------------------------------------------------------------------------
# Fake Redis / S3
# ---------------------------------------------------------------------------

class FakeRedis:
    """Dict-backed async stand-in for the redis client (string values only)."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = str(value)

    async def incr(self, key):
        value = int(self.store.get(key, 0)) + 1
        self.store[key] = str(value)
        return value

    async def delete(self, key):
        self.store.pop(key, None)


class FakeS3:
    """Dict-backed stand-in for boto3's S3 client."""

    def __init__(self):
        self.store: dict[str, bytes] = {}

    def get_object(self, Bucket, Key):
        if Key not in self.store:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        class _Body:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        return {"Body": _Body(self.store[Key])}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.store[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def list_objects_v2(self, Bucket, Prefix):
        keys = [k for k in self.store if k.startswith(Prefix)]
        return {"Contents": [{"Key": k} for k in keys]} if keys else {}

    def delete_objects(self, Bucket, Delete):
        for obj in Delete["Objects"]:
            self.store.pop(obj["Key"], None)


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def fake_s3() -> FakeS3:
    return FakeS3()


# ---------------------------------------------------------------------------
# FastAPI test client with fake cache wiring
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client(fake_redis, fake_s3, monkeypatch):
    """TestClient whose CacheManager and get_redis both hit the fakes.

    No `with` block: that skips the lifespan handler, which would otherwise
    ping a real Redis on startup.
    """
    from fastapi.testclient import TestClient

    import backend.api.deps as deps
    from backend.api.deps import CacheManager, get_cache_manager
    from backend.api.main import app

    cache = CacheManager(fake_redis, fake_s3)

    async def fake_get_cache_manager():
        return cache

    async def fake_get_redis():
        return fake_redis

    # Routes fetch get_redis lazily via `from backend.api.deps import get_redis`
    # inside the handler bodies, so patching the module attribute covers them.
    monkeypatch.setattr(deps, "get_redis", fake_get_redis)
    app.dependency_overrides[get_cache_manager] = fake_get_cache_manager

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def anyio_backend():
    return "asyncio"
