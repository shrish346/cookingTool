"""Run the production model pipeline against one video, locally.

Mirrors the VLM -> grounding -> expansion sequence of backend/api/routes.py's
run_pipeline(), minus Redis/R2/the residential download worker: this machine can
reach YouTube directly, so it downloads in-process instead.

Unlike main.py (which still calls the stale frame-chunk generate_recipe path),
this goes through generate_recipe_direct, so it reproduces what production does.

    python3 scripts/repro_pipeline.py <url> [--no-transcript] [--no-expand]

Exits non-zero and prints a full traceback on failure.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.chef import RecipeChef
from src.downloaders.factory import get_downloader
from src.llm.openai import OpenAIAdapter
from src.processing.audio import AudioTranscriber
from src.vlm.openrouter import OpenRouterAdapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--no-transcript", action="store_true",
                        help="Skip transcription (saves a Whisper API call)")
    parser.add_argument("--no-expand", action="store_true",
                        help="Stop after the grounded skeleton (skips pass 2)")
    args = parser.parse_args()

    print(f"[1/4] Downloading {args.url} ...")
    downloader = get_downloader(args.url)
    if not downloader:
        print(f"Unsupported URL: {args.url}", file=sys.stderr)
        return 1
    video_info = downloader.download(args.url)
    video_path = str(video_info.file_path)
    size_mb = Path(video_path).stat().st_size / (1024 * 1024)
    print(f"      Title:    {video_info.title}")
    print(f"      Duration: {video_info.duration_seconds}s")
    print(f"      Source:   {size_mb:.2f} MB -> {video_path}")

    transcript = None
    if args.no_transcript:
        print("[2/4] Skipping transcription (--no-transcript)")
    else:
        print("[2/4] Transcribing audio ...")
        transcript = AudioTranscriber().process_video(video_path)
        words = len(transcript.split()) if transcript else 0
        print(f"      {words} words")

    print("[3/4] Running VLM + grounding" + ("" if args.no_expand else " + expansion") + " ...")
    chef = RecipeChef(vlm_adapter=OpenRouterAdapter(), llm_adapter=OpenAIAdapter())
    recipe = chef.generate_recipe_direct(
        video_info,
        video_path,
        transcript,
        True,                  # debug
        not args.no_expand,    # expand
        lambda stage: print(f"      -> stage: {stage}"),
    )

    print("[4/4] Done.")
    print(f"      Title:       {recipe.title}")
    print(f"      Steps:       {len(recipe.steps)}")
    print(f"      Ingredients: {len(recipe.ingredients)}")
    if getattr(recipe, "expansion_failed", False):
        print("      NOTE: expansion_failed=True (serving grounded recipe)")
    for step in recipe.steps:
        span = ""
        if step.start_timestamp_seconds is not None:
            span = f" [{step.start_timestamp_seconds:.1f}s-{step.end_timestamp_seconds:.1f}s]"
        print(f"      {step.order}. {step.instruction[:70]}{span}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n--- PIPELINE FAILED ---", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
