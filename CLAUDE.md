# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MakerAI turns cooking short-form videos (YouTube Shorts, TikTok, Reels) into interactive step-by-step recipes, where each step is backed by a looping video clip cut from the source video. The core problem the code solves is _temporal grounding_: mapping each written recipe step back to the exact seconds of video it came from.

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

### Phone testing

Two helper scripts start Redis + backend (bound to `0.0.0.0`) + frontend with `VITE_API_URL` wired to the right host. `VITE_API_URL` is baked in when the Vite dev server starts, so it must point at a host the phone can reach — this is what the scripts handle.

```bash
./dev-lan.sh      # phone on same WiFi; auto-detects LAN IP, serves http://<ip>:5173
./dev-tunnel.sh   # public https via cloudflared; needed for real PWA install/offline testing
```

- `dev-lan.sh` is plain http on a LAN IP, which is **not** a secure context, so the PWA service worker won't register on the phone. Fine for UI/recipe flow; not for PWA install. Override the detected IP with `LAN_IP_OVERRIDE=192.168.x.x ./dev-lan.sh`.
- `dev-tunnel.sh` opens **two** cloudflared tunnels (frontend + backend) — an https page can't call an http backend, so the backend tunnel URL is captured first and fed into `VITE_API_URL`. Requires `cloudflared` (`brew install cloudflared`).
- Both scripts clean up child processes and the Redis container on Ctrl-C.
- If the phone can't connect over LAN, it's usually the macOS firewall blocking incoming connections for python/node.

### Tests

There is no pytest setup and no unit tests. Everything in `tests/` is a standalone script that hits real APIs and downloads real videos:

```bash
python tests/test_pipeline.py [youtube_url]   # end-to-end VLM→LLM→clips, writes ./test_output/clips/
python tests/test_frames.py                   # FrameExtractor only
```

Requires a populated `.env` (copy from `env.example`) and `ffmpeg` on PATH. These cost API credits — don't run them casually.

## Architecture

### The pipeline

The central abstraction is `RecipeChef` (`src/chef.py`), which orchestrates:

1. **VLM stage** — a vision model watches the video and emits a `SceneLog`: a list of `MicroAction`s, each an atomic cooking action ("add salt", "flip chicken") stamped with `timestamp_seconds` and an integer `id`.
2. **Grounding pass (pass 1)** — `src/llm/openai.py` (gpt-4o) reads the `SceneLog` and emits a `Recipe`, where each `Step` cites the `micro_action_ids` it was derived from. **This pass is about fidelity only** — it reports what the camera saw and invents nothing.
3. **Deterministic mapping** — `compute_timestamps_from_micro_actions()` in `src/schemas.py` resolves those IDs back into `start_timestamp_seconds` / `end_timestamp_seconds` on each step. The LLM never invents timestamps directly; it only cites IDs, and the code derives the timing. This is the key design decision — preserve it.
4. **Expansion pass (pass 2)** — `src/llm/expander.py` takes the grounded skeleton and rewrites it for a cook who has never cooked before: filling in quantities the video never stated, adding a `doneness_cue` to every step, and **inserting new steps the video never showed** (prep it skipped, technique it never taught). Runs on OpenRouter with the `:online` suffix, so it can pull real quantities from published recipes and cite them in `Recipe.sources`.
5. **Assembly** — `src/recipe_assembly.py`. `merge_expansion()` re-attaches the frozen timing, `build_gather_steps()` synthesizes the two setup steps, `relink_steps()` renumbers.
6. **Clip extraction** — `backend/processors/video.py` runs FFmpeg per step and uploads clips to S3, writing `video_clip_url` back onto each step. Clips are keyed by `step.id`, **not** `order`, so inserting or reordering steps can't collide with or orphan a clip.

**How grounding survives pass 2 — do not break this.** Pass 2 is never shown a timestamp and never writes one. It echoes back the `grounded_step_id` of the step it derived from (or `null` for a step it invented), and `merge_expansion()` copies the frozen `micro_action_ids` / timestamps back on by ID, stripping anything the model tried to set. The model *cannot* drift the timing because it never gets the opportunity to write it. `has_video_clip` is likewise derived in code, never trusted from the model.

If pass 2 throws, returns bad JSON, or drops >30% of the grounded steps, `RecipeChef._expand_for_beginner` logs it and returns the pass-1 recipe with `expansion_failed: true`. Worst case is the old, un-expanded output — never a broken request.

**The expanded recipe is the only recipe ever cached or served.** Pass 2 runs inside `generate_recipe_direct()`, which returns before `run_pipeline`'s first `cache.save_recipe()`. The grounded skeleton is an intermediate value; don't move pass 2 out of the chef or you'll start caching it.

`RecipeChef` exposes three entry points, in order of preference:

- `generate_recipe_direct()` — **the one in production use.** Compresses the video (`compress_video_for_api` in `src/vlm/openrouter.py`) and uploads it whole to a video-capable VLM (Gemini via OpenRouter). Most accurate temporal grounding. Pass `expand=False` to get the raw grounded recipe, which is useful for diffing what expansion actually changed.
- `generate_recipe_with_timestamps()` — dense frame extraction with timestamps. **Does not run pass 2.**
- `generate_recipe()` — legacy frame-chunk path. **Stale: it calls `analyze_video_direct()` with a frame list, but that method's signature takes a video path.** It will break if called. `main.py` still routes to it.

### Recipe schema and provenance

`Recipe` (`src/schemas.py`) carries a `schema_version`; bump it when the shape changes incompatibly, and bump `RECIPE_SCHEMA_VERSION` in `frontend/src/hooks/useLocalStorage.ts` to match. `CacheManager.get_cached_recipe` treats a version mismatch as a miss — without that, S3 recipes (which have no TTL) would be served forever in an old shape.

Ingredients, tools and steps each carry a `provenance`: `video` (the camera showed it), `reference` (from a published recipe, cites `source_id` into `Recipe.sources`), or `model` (the model's estimate). The frontend surfaces this on the ingredients list — it's the trust surface for everything pass 2 invents, so don't drop it when touching that view.

Steps reference ingredients and tools by ID (`ingredient_ids`, `tool_ids`) rather than only naming them in prose. That's deliberate groundwork for recipe mutation: "swap chicken for tofu" should become a patch to the ingredient entity plus a regeneration of only the steps referencing it. `Step.depends_on` is a linear chain today and read by nothing — it's the seam a real DAG grows from.

### Adapters

`src/vlm/` and `src/llm/` define `Protocol` interfaces (`VLMAdapter`, `LLMAdapter`) with concrete adapters alongside. The protocols have drifted from the implementations — `LLMAdapter.generate_recipe` doesn't declare the `use_timestamps` kwarg every caller passes, and `analyze_video_with_timestamps` isn't in `VLMAdapter` at all. Treat the concrete adapters as the source of truth, and prefer fixing the protocol over working around it.

Providers: OpenRouter (VLM + LLM), OpenAI, Gemini.

`src/downloaders/factory.py` picks a downloader by URL. YouTube, TikTok and Instagram are all wired up and share `YtDlpDownloader` (`src/downloaders/ytdlp.py`) — a subclass is just a URL regex, a yt-dlp format selector, the names of its cookie/proxy env vars, and an optional `explain_error` hook that rewrites a known yt-dlp failure into something the operator can act on. Add a platform by subclassing it, not by copying the download loop. Each downloader reads its own cookie env var, so callers should call `get_downloader(url)` and let it; the `cookies=` kwarg is an explicit override for whichever downloader matches.

`extract_video_id()` (`backend/api/routes.py`) namespaces every non-YouTube ID (`tt-`, `ig-`). This is load-bearing, not cosmetic: the ID is the *only* cache key — Redis job keys, the S3 `{id}/` prefix, and the public `job_id` — and an Instagram shortcode is 11 chars of the same alphabet as a YouTube ID, so bare IDs could collide across platforms and serve one video's recipe for another's. YouTube stays un-prefixed so recipes already cached under a plain ID keep hitting.

### API surface

`backend/api/routes.py` — three endpoints under `/api`:

- `POST /validate` — regex match, then a real `yt-dlp` info fetch to confirm the video exists.
- `POST /process` — returns immediately with a `job_id`, runs `run_pipeline()` as a FastAPI `BackgroundTasks` job.
- `GET /status/{job_id}` — poll for progress; returns the recipe once `status == "completed"`.

**`job_id` is the video ID**, not a per-request UUID. Two users processing the same video share a job and a cache entry. Progress lives in Redis under `job:{video_id}:{status,progress,message,error}` with a 1h TTL.

### Caching

`CacheManager` (`backend/api/deps.py`) is two-tier: Redis (1h TTL) in front of S3 (`{video_id}/recipe.json`, `{video_id}/clips/*`). A hit counter re-runs the whole pipeline every `CACHE_REFRESH_THRESHOLD` (default 5) hits. On pipeline failure the recipe is deleted from both tiers to avoid serving a half-built recipe.

The recipe is saved to cache _twice_: once after the LLM stage with `clips_ready: false`, again after clips upload with `clips_ready: true`. The frontend renders the recipe text while clips are still rendering, so don't collapse these into one write.

### Frontend

Vite + React + TypeScript + Tailwind, PWA via `vite-plugin-pwa`. Single-file state machine in `App.tsx`: `landing → loading → recipe → cooking`, polling `/api/status` on an interval. `frontend/src/api/client.ts` reads `VITE_API_URL`. Recipes persist to localStorage via `useSavedRecipe`.

## Gotchas

- `docker-compose.yml` hardcodes `VITE_API_URL` to a LAN IP (`http://10.71.118.71:8000`) so phones on the same network can reach the PWA. Change it for your machine; don't assume `localhost` works.
- CORS is `allow_origins=["*"]` — dev-only, needs tightening before any real deploy.
- YouTube blocks datacenter IPs. `YOUTUBE_COOKIES` (a Netscape cookie file, read by `YouTubeDownloader`) exists to work around this.
- **Instagram Reels need `INSTAGRAM_COOKIES` to work at all.** Instagram answers a logged-out request with an "empty media response" for most Reels, so yt-dlp gets nothing without a signed-in session cookie file on the worker. Use a throwaway account — sustained automated fetching can get a session rate-limited. `InstagramDownloader.explain_error` turns that empty response into a message naming the cookie fix.
- **TikTok is unreachable from an Indian ISP** — it's banned there, `www.tiktok.com` DNS-resolves to a block address and the TLS handshake is refused, so yt-dlp fails before it loads a page. The download worker runs on a residential line, so it inherits whatever its ISP blocks. `TIKTOK_PROXY` (a proxy URL, TikTok-only) is the way through; without it, `TikTokDownloader.explain_error` turns the raw SSL error into a message that says so. TikTok's *code path* is verified; an actual TikTok download has never been run from this machine's network.
- Frames are passed between stages as base64 JPEG strings, not file paths.
- `frames/` and `monitoring/` are scratch/disabled artifacts, not live code. Prometheus/Grafana are commented out in `docker-compose.yml` and would need `prometheus-fastapi-instrumentator` wired into `backend/api/main.py` to have anything to scrape.
- `BETA_LAUNCH.md` is stale and does not describe current behavior.
