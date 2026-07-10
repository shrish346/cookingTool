# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MakerAI (internally "Chef's Loop") turns cooking short-form videos (YouTube Shorts, TikTok, Reels) into interactive step-by-step recipes, where each step is backed by a looping video clip cut from the source video. The core problem the code solves is *temporal grounding*: mapping each written recipe step back to the exact seconds of video it came from.

## Commands

### Backend (Python, run from repo root)

```bash
pip install -r backend/requirements.txt   # API server deps
pip install -r requirements.txt           # superset, includes CLI-only deps (gemini, whisper, etc.)
uvicorn backend.api.main:app --reload     # dev server on :8000
python main.py <video_url_or_path> -v     # CLI pipeline, no Redis/S3 needed
```

### Frontend

```bash
cd frontend
npm install
npm run dev      # :5173
npm run build    # tsc -b && vite build
npm run lint     # eslint
```

### Full stack

```bash
docker-compose up --build   # backend :8000, frontend :5173, redis :6379
```

### Tests

There is no pytest setup and no unit tests. Everything in `tests/` is a standalone script that hits real APIs and downloads real videos:

```bash
python tests/test_pipeline.py [youtube_url]   # end-to-end VLM→LLM→clips, writes ./test_output/clips/
python tests/test_frames.py                   # FrameExtractor only
```

Requires a populated `.env` (copy from `env.example`) and `ffmpeg` on PATH. These cost API credits — don't run them casually.

## Architecture

### Two-stage pipeline

The central abstraction is `RecipeChef` (`src/chef.py`), which orchestrates:

1. **VLM stage** — a vision model watches the video and emits a `SceneLog`: a list of `MicroAction`s, each an atomic cooking action ("add salt", "flip chicken") stamped with `timestamp_seconds` and an integer `id`.
2. **LLM stage** — a text model reads the `SceneLog` and emits a `Recipe`, where each `Step` cites the `micro_action_ids` it was derived from.
3. **Deterministic mapping** — `compute_timestamps_from_micro_actions()` in `src/schemas.py` resolves those IDs back into `start_timestamp_seconds` / `end_timestamp_seconds` on each step. The LLM never invents timestamps directly; it only cites IDs, and the code derives the timing. This is the key design decision — preserve it.
4. **Clip extraction** — `backend/processors/video.py` runs FFmpeg per step and uploads clips to S3, writing `video_clip_url` back onto each step.

`RecipeChef` exposes three entry points, in order of preference:

- `generate_recipe_direct()` — **the one in production use.** Compresses the video (`compress_video_for_api` in `src/vlm/openrouter.py`) and uploads it whole to a video-capable VLM (Gemini via OpenRouter). Most accurate temporal grounding.
- `generate_recipe_with_timestamps()` — dense frame extraction with timestamps.
- `generate_recipe()` — legacy frame-chunk path. **Stale: it calls `analyze_video_direct()` with a frame list, but that method's signature takes a video path.** It will break if called. `main.py` still routes to it.

### Adapters

`src/vlm/` and `src/llm/` define `Protocol` interfaces (`VLMAdapter`, `LLMAdapter`) with concrete adapters alongside. The protocols have drifted from the implementations — `LLMAdapter.generate_recipe` doesn't declare the `use_timestamps` kwarg every caller passes, and `analyze_video_with_timestamps` isn't in `VLMAdapter` at all. Treat the concrete adapters as the source of truth, and prefer fixing the protocol over working around it.

Providers: OpenRouter (VLM + LLM), OpenAI, Gemini. `src/downloaders/factory.py` picks a downloader by URL; only YouTube is wired up (TikTok is commented out, despite the API validating TikTok URLs).

### API surface

`backend/api/routes.py` — three endpoints under `/api`:

- `POST /validate` — regex match, then a real `yt-dlp` info fetch to confirm the video exists.
- `POST /process` — returns immediately with a `job_id`, runs `run_pipeline()` as a FastAPI `BackgroundTasks` job.
- `GET /status/{job_id}` — poll for progress; returns the recipe once `status == "completed"`.

**`job_id` is the video ID**, not a per-request UUID. Two users processing the same video share a job and a cache entry. Progress lives in Redis under `job:{video_id}:{status,progress,message,error}` with a 1h TTL.

### Caching

`CacheManager` (`backend/api/deps.py`) is two-tier: Redis (1h TTL) in front of S3 (`{video_id}/recipe.json`, `{video_id}/clips/*`). A hit counter re-runs the whole pipeline every `CACHE_REFRESH_THRESHOLD` (default 5) hits. On pipeline failure the recipe is deleted from both tiers to avoid serving a half-built recipe.

The recipe is saved to cache *twice*: once after the LLM stage with `clips_ready: false`, again after clips upload with `clips_ready: true`. The frontend renders the recipe text while clips are still rendering, so don't collapse these into one write.

### Frontend

Vite + React + TypeScript + Tailwind, PWA via `vite-plugin-pwa`. Single-file state machine in `App.tsx`: `landing → loading → recipe → cooking`, polling `/api/status` on an interval. `frontend/src/api/client.ts` reads `VITE_API_URL`. Recipes persist to localStorage via `useSavedRecipe`.

## Gotchas

- `docker-compose.yml` hardcodes `VITE_API_URL` to a LAN IP (`http://10.71.118.71:8000`) so phones on the same network can reach the PWA. Change it for your machine; don't assume `localhost` works.
- CORS is `allow_origins=["*"]` — dev-only, needs tightening before any real deploy.
- YouTube blocks datacenter IPs. `YOUTUBE_COOKIES` (a Netscape cookie file, passed through `get_downloader(url, cookies=...)`) exists to work around this.
- Frames are passed between stages as base64 JPEG strings, not file paths.
- `frames/` and `monitoring/` are scratch/disabled artifacts, not live code. Prometheus/Grafana are commented out in `docker-compose.yml` and would need `prometheus-fastapi-instrumentator` wired into `backend/api/main.py` to have anything to scrape.
- `BETA_LAUNCH.md` is stale and does not describe current behavior.
