#!/usr/bin/env python3
"""
CLI entry point for the cooking video recipe extractor.

Usage:
    python main.py <video_url_or_path> [options]
"""

import sys
import argparse
import json
from pathlib import Path

from src.downloaders.factory import get_downloader
from src.downloaders.youtube import YouTubeDownloader
from src.processing.frames import FrameExtractor
from src.processing.audio import AudioTranscriber
from src.vlm.openrouter import OpenRouterAdapter
from src.llm.openai import OpenAIAdapter
from src.llm.gemini import GeminiAdapter
from src.llm.openrouter import OpenRouterLLMAdapter
from src.chef import RecipeChef


def create_llm_adapter(provider: str, model: str | None = None):
    """Create an LLM adapter based on provider name."""
    if provider.lower() == "openai":
        return OpenAIAdapter(model=model or "gpt-4")
    elif provider.lower() == "gemini":
        return GeminiAdapter(model=model or "gemini-pro")
    elif provider.lower() == "openrouter":
        return OpenRouterLLMAdapter(model=model or "openai/gpt-4")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Choose from: openai, gemini, openrouter")


def format_recipe_markdown(recipe) -> str:
    """Format a Recipe object as Markdown."""
    lines = []
    lines.append(f"# {recipe.title}\n")
    
    if recipe.description:
        lines.append(f"{recipe.description}\n")
    
    lines.append("---\n")
    
    if recipe.prep_time_minutes or recipe.cook_time_minutes:
        times = []
        if recipe.prep_time_minutes:
            times.append(f"Prep: {recipe.prep_time_minutes} min")
        if recipe.cook_time_minutes:
            times.append(f"Cook: {recipe.cook_time_minutes} min")
        if times:
            lines.append(f"{' | '.join(times)}\n")
    
    if recipe.servings:
        lines.append(f"Serves: {recipe.servings}\n")
    
    if recipe.cusine:
        lines.append(f"Cuisine: {recipe.cusine}\n")
    
    if recipe.tags:
        lines.append(f"Tags: {', '.join(recipe.tags)}\n")
    
    lines.append("\n## Ingredients\n")
    for ing in recipe.ingredients:
        prep = f" ({ing.preparation})" if ing.preparation else ""
        lines.append(f"- {ing.quantity} {ing.unit} {ing.name}{prep}\n")
    
    lines.append("\n## Instructions\n")
    for step in recipe.steps:
        duration = f" [{step.duration_minutes} min]" if step.duration_minutes else ""
        lines.append(f"{step.order}. {step.instruction}{duration}\n")
        if step.tips:
            for tip in step.tips:
                lines.append(f"   💡 {tip}\n")
    
    if recipe.calories or recipe.protein or recipe.carbs or recipe.fats:
        lines.append("\n## Nutrition\n")
        if recipe.calories:
            lines.append(f"- Calories: {recipe.calories}\n")
        if recipe.protein:
            lines.append(f"- Protein: {recipe.protein}g\n")
        if recipe.carbs:
            lines.append(f"- Carbs: {recipe.carbs}g\n")
        if recipe.fats:
            lines.append(f"- Fats: {recipe.fats}g\n")
    
    if recipe.source_url:
        lines.append(f"\nSource: {recipe.source_url}\n")
    
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract recipes from cooking videos using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py https://www.youtube.com/watch?v=...
  python main.py video.mp4 --llm-provider openai --llm-model gpt-4
  python main.py video.mp4 --save-scenes scenes.json --output both
        """
    )
    
    parser.add_argument(
        "input",
        help="Video URL (YouTube, etc.) or path to local video file"
    )
    
    parser.add_argument(
        "--vlm-model",
        default="qwen/qwen2.5-vl-72b-instruct",
        help="VLM model to use (default: qwen/qwen2.5-vl-72b-instruct)"
    )
    
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "gemini", "openrouter"],
        default="openai",
        help="LLM provider to use (default: openai)"
    )
    
    parser.add_argument(
        "--llm-model",
        help="LLM model to use (default depends on provider)"
    )
    
    parser.add_argument(
        "--save-scenes",
        metavar="PATH",
        help="Save intermediate scene descriptions to JSON file"
    )
    
    parser.add_argument(
        "--output",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format (default: both)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show progress and intermediate results"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Number of frames to process in each VLM batch (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        # Step 1: Download or validate video
        if args.verbose:
            print("[1/5] Processing video input...")
        
        input_path = Path(args.input)
        if input_path.exists() and input_path.is_file():
            # Local file
            from src.downloaders.base import VideoInfo
            video_info = VideoInfo(
                title=input_path.stem,
                file_path=input_path,
                url=str(input_path),
                duration_seconds=0  # Could extract with ffprobe if needed
            )
            video_path = str(input_path)
        else:
            # URL - download it
            downloader = get_downloader(args.input)
            if not downloader:
                print(f"Error: Unsupported video URL: {args.input}", file=sys.stderr)
                sys.exit(1)
            
            video_info = downloader.download(args.input)
            video_path = str(video_info.file_path)
            
            if args.verbose:
                print(f"      Title: {video_info.title}")
                print(f"      Duration: {video_info.duration_seconds}s")
        
        try:
            # Step 2: Extract frames
            if args.verbose:
                print("[2/5] Extracting frames...")
            
            extractor = FrameExtractor(resize_width=512)
            frames = extractor.extract(video_path)
            
            if args.verbose:
                print(f"      Extracted {len(frames)} frames")
            
            # Step 3: Transcribe audio
            if args.verbose:
                print("[3/5] Transcribing audio...")
            
            transcriber = AudioTranscriber()
            transcript = transcriber.process_video(video_path)
            
            if args.verbose:
                if transcript:
                    print(f"      Found speech ({len(transcript.split())} words)")
                else:
                    print("      No meaningful speech detected")
            
            # Step 4: Initialize adapters
            if args.verbose:
                print("[4/5] Initializing AI models...")
            
            vlm_adapter = OpenRouterAdapter(model=args.vlm_model)
            llm_adapter = create_llm_adapter(args.llm_provider, args.llm_model)
            
            if args.verbose:
                print(f"      VLM: {vlm_adapter.model_name}")
                print(f"      LLM: {llm_adapter.model_name}")
            
            # Step 5: Generate recipe
            if args.verbose:
                print("[5/5] Generating recipe (this may take a while)...")
            
            chef = RecipeChef(
                vlm_adapter=vlm_adapter,
                llm_adapter=llm_adapter,
                save_scenes_path=args.save_scenes
            )
            
            recipe = chef.generate_recipe(
                video_info,
                frames,
                transcript,
                chunk_size=args.chunk_size
            )
            
            # Output results
            if args.output in ["json", "both"]:
                print("\n" + "="*50)
                print("RECIPE (JSON)")
                print("="*50)
                print(json.dumps(recipe.model_dump(), indent=2))
            
            if args.output in ["markdown", "both"]:
                if args.output == "both":
                    print("\n" + "="*50)
                    print("RECIPE (MARKDOWN)")
                    print("="*50)
                print(format_recipe_markdown(recipe))
            
            if args.verbose and args.save_scenes:
                print(f"\nScene descriptions saved to: {args.save_scenes}")
        
        finally:
            # Cleanup downloaded video if it was downloaded
            if not (input_path.exists() and input_path.is_file()):
                downloader.cleanup(video_info)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

