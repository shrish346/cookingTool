"""
Test the full pipeline: Download -> Extract Frames -> Analyze with VLM
"""
import base64
import shutil
from pathlib import Path

from src.downloaders.youtube import YouTubeDownloader
from src.processing.frames import FrameExtractor
from src.processing.audio import AudioTranscriber
from src.vlm.openrouter import OpenRouterAdapter

FRAMES_DIR = Path("frames")

# Use a short cooking video for testing (regular video, not Shorts - more reliable)
TEST_URL = "https://www.youtube.com/shorts/-2t0WNgtsZI"  # 3-min eggs recipe

def main():
    downloader = YouTubeDownloader()
    extractor = FrameExtractor(resize_width=512)  # Smaller for faster uploads
    transcriber = AudioTranscriber()
    adapter = OpenRouterAdapter()
    
    print(f"Using model: {adapter.model_name}")
    print(f"Testing with: {TEST_URL}\n")
    
    # Step 1: Download
    print("[1/4] Downloading video...")
    video_info = downloader.download(TEST_URL)
    print(f"      Title: {video_info.title}")
    print(f"      Duration: {video_info.duration_seconds}s")
    print(f"      File: {video_info.file_path}\n")
    
    try:
        # Step 2: Extract frames
        print("[2/4] Extracting frames...")
        frames = extractor.extract(str(video_info.file_path))
        print(f"      Extracted {len(frames)} frames")
        
        # Save frames to folder
        if FRAMES_DIR.exists():
            shutil.rmtree(FRAMES_DIR)  # Clear old frames
        FRAMES_DIR.mkdir()
        
        for i, frame_b64 in enumerate(frames, start=1):
            frame_bytes = base64.b64decode(frame_b64)
            (FRAMES_DIR / f"{i}.jpg").write_bytes(frame_bytes)
        print(f"      Saved to ./{FRAMES_DIR}/\n")
        
        # Step 3: Transcribe audio (if meaningful speech exists)
        print("[3/4] Transcribing audio...")
        transcript = transcriber.process_video(video_info.file_path)
        if transcript:
            print(f"      Found speech ({len(transcript.split())} words)")
            print(f"      Preview: {transcript[:100]}...")
        else:
            print("      No meaningful speech detected, skipping transcript")
        print()
        
        # Step 4: Analyze with VLM
        print("[4/4] Analyzing with VLM (this may take 30-60 seconds)...")
        recipe = adapter.analyze_recipe(video_info, frames, transcript)
        
        # Print results
        print("\n" + "="*50)
        print("RECIPE EXTRACTED")
        print("="*50)
        
        if recipe.reasoning:
            print("\n[Model's Reasoning]")
            print(recipe.reasoning)
            print()
        
        print(f"{recipe.title}")
        print(f"   {recipe.description}\n")
        
        print(f"Prep: {recipe.prep_time_minutes} min | Cook: {recipe.cook_time_minutes} min")
        print(f"Servings: {recipe.servings}")
        
        print(f"\nIngredients ({len(recipe.ingredients)}):")
        for ing in recipe.ingredients:
            prep = f" ({ing.preparation})" if ing.preparation else ""
            print(f"   - {ing.quantity} {ing.unit} {ing.name}{prep}")
        
        print(f"\nSteps ({len(recipe.steps)}):")
        for step in recipe.steps:
            duration = f" [{step.duration_minutes} min]" if step.duration_minutes else ""
            print(f"   {step.order}. {step.instruction}{duration}")
        
        if recipe.calories:
            print(f"\nNutrition: {recipe.calories} cal | {recipe.protein}g protein | {recipe.carbs}g carbs | {recipe.fats}g fat")
        
    finally:
        # Always cleanup
        print("\nCleaning up temp files...")
        downloader.cleanup(video_info)
        print("Done!")

if __name__ == "__main__":
    main()

