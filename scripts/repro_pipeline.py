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

    print("[4/4] Done.\n")
    _print_recipe(recipe)
    return 0


def _prov(item) -> str:
    """Compact provenance tag, e.g. 'video' or 'reference->s1'."""
    tag = getattr(item, "provenance", None) or "?"
    src = getattr(item, "source_id", None)
    return f"{tag}->{src}" if src else tag


def _amount(ing) -> str:
    if ing.quantity is not None:
        return f"{ing.quantity} {ing.unit or ''}".strip()
    return ing.amount_text or "amount not shown"


def _print_recipe(recipe) -> None:
    print("=" * 72)
    print(recipe.title)
    if recipe.description:
        print(recipe.description)
    meta = [f"serves {recipe.servings}"]
    if recipe.difficulty:
        meta.append(recipe.difficulty)
    if recipe.prep_time_minutes:
        meta.append(f"prep {recipe.prep_time_minutes}m")
    if recipe.cook_time_minutes:
        meta.append(f"cook {recipe.cook_time_minutes}m")
    if recipe.cuisine:
        meta.append(recipe.cuisine)
    print("  ".join(meta))
    if getattr(recipe, "expansion_failed", False):
        print("\n!! expansion_failed=True — this is the raw grounded (pass-1) recipe")
    print("=" * 72)

    print(f"\nINGREDIENTS ({len(recipe.ingredients)}):")
    for ing in recipe.ingredients:
        prep = f", {ing.preparation}" if ing.preparation else ""
        opt = " (optional)" if ing.optional else ""
        print(f"  - {ing.name}: {_amount(ing)}{prep}{opt}  [{_prov(ing)}]")
        if ing.note:
            print(f"      note: {ing.note}")

    if recipe.tools:
        print(f"\nTOOLS ({len(recipe.tools)}):")
        for tool in recipe.tools:
            ess = "essential" if tool.essential else "optional"
            sub = f", sub: {tool.substitute}" if tool.substitute else ""
            print(f"  - {tool.name} ({ess}){sub}  [{_prov(tool)}]")

    if recipe.sources:
        print(f"\nSOURCES ({len(recipe.sources)}):")
        for src in recipe.sources:
            site = f" — {src.site}" if src.site else ""
            url = f" ({src.url})" if src.url else ""
            print(f"  - [{src.id}] {src.title}{site}{url}")

    print(f"\nSTEPS ({len(recipe.steps)}):")
    for step in recipe.steps:
        span = ""
        if step.start_timestamp_seconds is not None:
            span = f" [{step.start_timestamp_seconds:.1f}s-{step.end_timestamp_seconds:.1f}s]"
        elif not step.has_video_clip:
            span = " [no clip]"
        dur = f" (~{step.duration_minutes}m)" if step.duration_minutes else ""
        print(f"\n  {step.order}. [{step.kind}] {step.title or ''}{dur}{span}  [{_prov(step)}]")
        print(f"     {step.instruction}")
        if step.detail:
            print(f"     detail: {step.detail}")
        if step.doneness_cue:
            print(f"     done when: {step.doneness_cue}")
        for tip in step.tips or []:
            print(f"     tip: {tip}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n--- PIPELINE FAILED ---", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
