import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be found
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

import os
import json
import subprocess
import shutil
from typing import Optional

from src.downloaders.base import VideoInfo
from src.schemas import Recipe, SceneLog, SceneDescription
from src.vlm.base import VLMAdapter
from src.vlm.openrouter import OpenRouterAdapter
from src.llm.openrouter import OpenRouterLLMAdapter
from src.llm.openai import OpenAIAdapter
from src.chef import RecipeChef
from src.downloaders.factory import get_downloader
from src.processing.audio import AudioTranscriber
from backend.processors.video import VideoClipExtractor

url = "https://www.youtube.com/shorts/Ll307dapW64"

if __name__ == "__main__":
    # Step 1: Download video
    print(f"Starting Step 1: Downloading video from {url}...")
    downloader = get_downloader(url)
    video_info = downloader.download(url)
    video_path = str(video_info.file_path)
    print(f"Finished Step 1: Video downloaded to {video_path}")
    
    # Step 2: Transcribe audio
    print("Starting Step 2: Transcribing audio...")
    transcriber = AudioTranscriber()
    transcript = transcriber.process_video(video_path)
    print(f"Finished Step 2: Audio transcribed ({len(transcript) if transcript else 0} characters)")
    
    # Step 3: VLM + LLM pipeline (using direct video upload to Gemini)
    print("Starting Step 3: VLM + LLM Pipeline (direct video upload)...")
    vlm_adapter = OpenRouterAdapter("google/gemini-2.0-flash-001")
    llm_adapter = OpenAIAdapter("gpt-4o")
    
    scene_log_path = Path("tests/video_timestamp_test/scene_log.json")
    chef = RecipeChef(vlm_adapter=vlm_adapter, llm_adapter=llm_adapter, save_scenes_path=scene_log_path)
    
    # Use direct video upload (no frame extraction)
    recipe = chef.generate_recipe_direct(
        video_info,
        video_path,
        transcript,
        True
    )
    print("Finished Step 3: Pipeline complete.")
    
    print("\n" + "="*50)
    print("FINAL RECIPE RESULT:")
    print("="*50)      
    print(chef.load_scene_log(chef._save_scenes_path))
    print(recipe)

    # Output steps and timestamps
    print("\n" + "="*50)
    print("RECIPE STEPS AND CLIP TIMESTAMPS:")
    print("="*50)
    for step in recipe.steps:
        start = step.start_timestamp_seconds
        end = step.end_timestamp_seconds
        timing = f"{start:.2f}s -> {end:.2f}s" if start is not None and end is not None else "No timing"
        print(f"Step {step.order}: {step.title} ({timing})")

    # Extract clips to tests/official_test_output
    output_dir = Path("tests/official_test_output")
    
    # Clear directory before extracting new videos
    if output_dir.exists():
        print(f"\nClearing previous results in {output_dir}...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full video
    full_video_dest = output_dir / "full_video.mp4"
    shutil.copy(video_path, full_video_dest)
    print(f"\nSaved full video to: {full_video_dest}")

    # Define a mock S3 client that saves files locally instead of cloud
    class LocalS3Mock:
        def __init__(self, output_dir):
            self.output_dir = output_dir
        def upload_file(self, local_path, bucket, key, ExtraArgs=None):
            # The key is usually f"{video_id}/{filename}"
            filename = os.path.basename(key)
            dest = self.output_dir / filename
            shutil.copy(local_path, dest)
            return str(dest)

    print("\nExtracting step clips using VideoClipExtractor...")
    extractor = VideoClipExtractor(
        s3_client=LocalS3Mock(output_dir),
        bucket_name="local-test-output"
    )
    
    # Convert recipe model to dict list for the extractor
    recipe_dict = recipe.model_dump()
    
    # This will run the logic from VideoClipExtractor but "upload" to our local folder
    extractor.extract_and_upload_clips(
        video_path=video_path,
        video_id="official_test",
        steps=recipe_dict["steps"]
    )

    print(f"\nAll assets saved to: {output_dir}")
