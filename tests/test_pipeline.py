"""
Test the full pipeline: Download -> Extract Frames -> Analyze with VLM -> Generate Recipe with LLM
"""
import sys
from pathlib import Path
import time

# Add project root to Python path for cross-platform imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import base64
import shutil

from src.downloaders.youtube import YouTubeDownloader
from src.processing.frames import FrameExtractor
from src.processing.audio import AudioTranscriber
from src.vlm.openrouter import OpenRouterAdapter
from src.llm.openrouter import OpenRouterLLMAdapter
from src.chef import RecipeChef
from src.schemas import SceneLog, SceneDescription

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
    print("[1/5] Downloading video...")
    video_info = downloader.download(TEST_URL)
    print(f"      Title: {video_info.title}")
    print(f"      Duration: {video_info.duration_seconds}s")
    print(f"      File: {video_info.file_path}\n")
    
    try:
        # Step 2: Extract frames
        print("[2/5] Extracting frames...")
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
        print("[3/5] Transcribing audio...")
        transcript = transcriber.process_video(video_info.file_path)
        if transcript:
            print(f"      Found speech ({len(transcript.split())} words)")
            print(f"      Preview: {transcript[:100]}...")
        else:
            print("      No meaningful speech detected, skipping transcript")
        print()
        
        # Step 4: Test two-stage pipeline (VLM → LLM)

        print("[4/5] Analyzing scenes with VLM (parallel processing)...")
        start = time.perf_counter()
        vlm_adapter = OpenRouterAdapter()
        scene_descriptions = vlm_adapter.analyze_scenes(video_info, frames, transcript, chunk_size=12)
        end = time.perf_counter()
        print(f"      Time taken: {end - start:.2f} seconds")
        print(f"      Generated {len(scene_descriptions)} scene descriptions")
        
        # Show sample scene description
        if scene_descriptions:
            sample = scene_descriptions[0]
            print(f"      Sample scene (frame {sample.frame_index}):")
            print(f"        - Entities: {len(sample.entities)}")
            print(f"        - State changes: {len(sample.state_changes)}")
            print(f"        - Actions: {len(sample.temporal_actions)}")
        
        # Step 5: Generate recipe with LLM
        print("[5/5] Generating recipe with LLM (this may take 30-60 seconds)...")
        llm_adapter = OpenRouterLLMAdapter()
        scene_log = SceneLog(
            scenes=scene_descriptions,
            video_info={
                "title": video_info.title,
                "description": video_info.description,
                "url": video_info.url,
                "duration_seconds": video_info.duration_seconds
            }
        )
        recipe = llm_adapter.generate_recipe(scene_log, video_info, transcript)
        
        # Print results
        print("\n" + "="*50)
        print("RECIPE EXTRACTED (Two-Stage Pipeline)")
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


def test_recipe_chef():
    """Test the RecipeChef orchestrator."""
    print("\n" + "="*50)
    print("TESTING RECIPE CHEF ORCHESTRATOR")
    print("="*50 + "\n")
    
    downloader = YouTubeDownloader()
    extractor = FrameExtractor(resize_width=512)
    transcriber = AudioTranscriber()
    
    print(f"Testing with: {TEST_URL}\n")
    
    # Download and process
    print("[1/4] Downloading video...")
    video_info = downloader.download(TEST_URL)
    print(f"      Title: {video_info.title}\n")
    
    try:
        print("[2/4] Extracting frames...")
        frames = extractor.extract(str(video_info.file_path))
        print(f"      Extracted {len(frames)} frames\n")
        
        print("[3/4] Transcribing audio...")
        transcript = transcriber.process_video(video_info.file_path)
        if transcript:
            print(f"      Found speech ({len(transcript.split())} words)\n")
        else:
            print("      No speech detected\n")
        
        print("[4/4] Using RecipeChef (VLM → LLM pipeline)...")
        vlm_adapter = OpenRouterAdapter()
        llm_adapter = OpenRouterLLMAdapter()
        
        chef = RecipeChef(
            vlm_adapter=vlm_adapter,
            llm_adapter=llm_adapter,
            save_scenes_path="test_scenes.json"
        )
        
        recipe = chef.generate_recipe(video_info, frames, transcript, chunk_size=12)
        
        print("\n" + "="*50)
        print("RECIPE FROM CHEF")
        print("="*50)
        print(f"\n{recipe.title}")
        print(f"Ingredients: {len(recipe.ingredients)}")
        print(f"Steps: {len(recipe.steps)}")
        
        # Test loading scene log
        if Path("test_scenes.json").exists():
            loaded_log = RecipeChef.load_scene_log("test_scenes.json")
            print(f"\nLoaded scene log: {len(loaded_log.scenes)} scenes")
            Path("test_scenes.json").unlink()  # Cleanup
        
    finally:
        downloader.cleanup(video_info)
        print("\nDone!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "chef":
        test_recipe_chef()
    else:
        main()