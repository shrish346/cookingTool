"""
Test the timestamp-based VLM-LLM pipeline with direct video upload.

This test approach:
1. Downloads a cooking video
2. Uploads video DIRECTLY to VLM (no frame extraction)
3. VLM outputs micro-actions mapped to TIMESTAMPS (not frames)
4. LLM groups micro-actions into recipe steps with time ranges
5. Clips are extracted based on timestamp ranges

This leverages Qwen2.5-VL's M-RoPE for accurate temporal grounding.

Usage:
    python tests/video_timestamp_test/test_timestamp_pipeline.py [youtube_url]
"""
import sys
from pathlib import Path
import time
import subprocess
import shutil
import json

# Add project root and local test directory to Python path
project_root = Path(__file__).parent.parent.parent
test_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(test_dir))

# Import from main project
from src.downloaders.youtube import YouTubeDownloader
from src.processing.audio import AudioTranscriber

# Import from local test modules
from vlm_video import VideoVLMAdapter
from llm_timestamp import TimestampLLMAdapter
from schemas_timestamp import VideoSceneLog, RecipeTimestamp

# Output directories
OUTPUT_DIR = Path(__file__).parent / "output"
CLIPS_DIR = OUTPUT_DIR / "clips"

# Default test video (short cooking video)
DEFAULT_TEST_URL = "https://www.youtube.com/shorts/Ll307dapW64"


def extract_clip_by_timestamp(
    video_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    ffmpeg_path: str = "ffmpeg"
) -> bool:
    """Extract a clip from the video based on timestamps."""
    try:
        duration = end_time - start_time
        
        # Minimum 2 seconds, maximum 10 seconds
        if duration < 2:
            duration = 2
        if duration > 10:
            duration = 10
        
        cmd = [
            ffmpeg_path,
            "-y",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-an",  # No audio
            "-vf", "scale=640:-2",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "28",
            "-movflags", "+faststart",
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"      Error extracting clip: {e}")
        return False


def get_video_duration(video_path: str) -> float:
    """Get video duration using FFprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    
    return 0.0


def main(test_url: str = DEFAULT_TEST_URL):
    """Run the timestamp-based pipeline test."""
    
    print("\n" + "=" * 80)
    print("  TIMESTAMP-BASED VLM-LLM PIPELINE TEST")
    print("  (Direct video upload - no frame extraction)")
    print("=" * 80)
    print(f"\nTest URL: {test_url}\n")
    
    # Setup output directories
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    CLIPS_DIR.mkdir()
    
    # Initialize components
    downloader = YouTubeDownloader()
    transcriber = AudioTranscriber()
    vlm_adapter = VideoVLMAdapter()
    llm_adapter = TimestampLLMAdapter()
    
    print(f"VLM Model: {vlm_adapter.model_name}")
    print(f"LLM Model: {llm_adapter.model_name}\n")
    
    # =========================================================================
    # STEP 1: Download video
    # =========================================================================
    print("─" * 80)
    print("[1/5] DOWNLOADING VIDEO")
    print("─" * 80)
    
    video_info = downloader.download(test_url)
    video_path = str(video_info.file_path)
    
    print(f"  Title: {video_info.title}")
    print(f"  Duration: {video_info.duration_seconds}s")
    print(f"  File: {video_path}\n")
    
    try:
        # =====================================================================
        # STEP 2: Transcribe audio (optional, still useful for context)
        # =====================================================================
        print("─" * 80)
        print("[2/5] TRANSCRIBING AUDIO")
        print("─" * 80)
        
        transcript = transcriber.process_video(video_path)
        
        if transcript:
            print(f"  Found speech ({len(transcript.split())} words)")
            print(f"  Preview: {transcript[:150]}...\n")
        else:
            print("  No meaningful speech detected\n")
        
        # =====================================================================
        # STEP 3: VLM Video Analysis (Direct Video Upload)
        # =====================================================================
        print("─" * 80)
        print("[3/5] VLM VIDEO ANALYSIS")
        print("─" * 80)
        print("\n  Using direct video upload (leverages M-RoPE)...\n")
        
        start_time = time.perf_counter()
        scene_log = vlm_adapter.analyze_video(video_info, video_path)
        vlm_time = time.perf_counter() - start_time
        
        print(f"\n  VLM Time: {vlm_time:.2f}s")
        
        # =====================================================================
        # OUTPUT: All Micro-Actions with Timestamps
        # =====================================================================
        print("\n" + "=" * 80)
        print("  ALL MICRO-ACTIONS WITH TIMESTAMPS")
        print("=" * 80)
        
        all_micro_actions = scene_log.get_all_micro_actions()
        print(f"\n  Total: {len(all_micro_actions)} micro-actions")
        print(f"  Video Duration: {scene_log.scene.video_duration_seconds}s\n")
        
        print(f"  {'ID':<4} {'Timestamp':<12} {'Duration':<10} {'Action':<35} {'Entity':<12} {'State Change'}")
        print("  " + "─" * 100)
        
        for ma in all_micro_actions:
            state_change = ""
            if ma.state_before and ma.state_after:
                state_change = f"{ma.state_before} → {ma.state_after}"
            elif ma.state_after:
                state_change = f"→ {ma.state_after}"
            
            entity = ma.entity or ""
            duration_str = f"{ma.duration_seconds:.1f}s" if ma.duration_seconds else "-"
            
            print(f"  {ma.id:<4} {ma.timestamp_seconds:>7.1f}s     {duration_str:<10} {ma.action[:35]:<35} {entity[:12]:<12} {state_change}")
        
        print()
        
        # Save scene log for debugging
        scene_log_path = OUTPUT_DIR / "scene_log.json"
        with open(scene_log_path, "w") as f:
            json.dump(scene_log.to_dict(), f, indent=2)
        print(f"  Scene log saved to: {scene_log_path}\n")
        
        # =====================================================================
        # STEP 4: LLM Recipe Generation
        # =====================================================================
        print("─" * 80)
        print("[4/5] LLM RECIPE GENERATION")
        print("─" * 80)
        
        start_time = time.perf_counter()
        recipe = llm_adapter.generate_recipe(scene_log, video_info, transcript)
        llm_time = time.perf_counter() - start_time
        
        print(f"  Time: {llm_time:.2f}s")
        print(f"  Steps generated: {len(recipe.steps)}\n")
        
        # =====================================================================
        # OUTPUT: Recipe with Time Ranges
        # =====================================================================
        print("=" * 80)
        print("  RECIPE WITH TIME-BASED VIDEO CLIPS")
        print("=" * 80)
        
        print(f"\n  Title: {recipe.title}")
        print(f"  Description: {recipe.description}")
        print(f"  Servings: {recipe.servings}")
        
        if recipe.prep_time_minutes or recipe.cook_time_minutes:
            print(f"  Time: {recipe.prep_time_minutes or 0} min prep, {recipe.cook_time_minutes or 0} min cook")
        
        if recipe.cuisine:
            print(f"  Cuisine: {recipe.cuisine}")
        
        if recipe.tags:
            print(f"  Tags: {', '.join(recipe.tags)}")
        
        # Ingredients
        print(f"\n  INGREDIENTS ({len(recipe.ingredients)}):")
        print("  " + "─" * 50)
        for ing in recipe.ingredients:
            prep = f" ({ing.preparation})" if ing.preparation else ""
            print(f"    • {ing.quantity} {ing.unit} {ing.name}{prep}")
        
        # Steps with Time Ranges
        print("\n" + "=" * 80)
        print("  STEPS WITH TIME RANGES")
        print("=" * 80)
        
        for step in recipe.steps:
            print(f"\n  ┌{'─' * 76}┐")
            print(f"  │ STEP {step.order}: {step.title or 'Untitled':<66} │")
            print(f"  └{'─' * 76}┘")
            
            print(f"\n  Instruction: {step.instruction}")
            
            if step.duration_minutes:
                print(f"  Duration: {step.duration_minutes} minutes")
            
            if step.tips:
                print(f"  Tips: {', '.join(step.tips)}")
            
            # Time range info
            print(f"\n  Has Video Clip: {'Yes' if step.has_video_clip else 'No'}")
            
            if step.has_video_clip and step.start_timestamp_seconds is not None:
                print(f"  Time Range: {step.start_timestamp_seconds:.1f}s → {step.end_timestamp_seconds:.1f}s")
                print(f"  Clip Duration: {step.clip_duration_seconds:.1f}s")
            
            # Micro-actions used
            if step.micro_action_ids:
                print(f"\n  Micro-Actions ({len(step.micro_action_ids)}):")
                
                for ma_id in step.micro_action_ids:
                    ma = next((m for m in all_micro_actions if m.id == ma_id), None)
                    if ma:
                        print(f"    [{ma_id}] {ma.timestamp_seconds:.1f}s: {ma.action}")
                    else:
                        print(f"    [{ma_id}] (not found)")
            else:
                print(f"\n  Micro-Actions: None")
        
        # Save recipe for debugging
        recipe_path = OUTPUT_DIR / "recipe.json"
        with open(recipe_path, "w") as f:
            json.dump(recipe.model_dump(), f, indent=2, default=str)
        print(f"\n  Recipe saved to: {recipe_path}")
        
        # =====================================================================
        # STEP 5: Extract Video Clips
        # =====================================================================
        print("\n" + "─" * 80)
        print("[5/5] EXTRACTING VIDEO CLIPS BY TIMESTAMP")
        print("─" * 80 + "\n")
        
        clips_extracted = 0
        
        for step in recipe.steps:
            step_order = step.order
            step_title = step.title or "Untitled"
            
            if not step.has_video_clip or step.start_timestamp_seconds is None:
                print(f"  Step {step_order}: Skipped (no video clip)")
                continue
            
            start_ts = step.start_timestamp_seconds
            end_ts = step.end_timestamp_seconds
            original_duration = end_ts - start_ts
            
            # Apply duration limits
            duration = original_duration
            if duration < 2:
                duration = 2
            if duration > 10:
                duration = 10
            
            # Adjust end time based on clamped duration
            actual_end_ts = start_ts + duration
            
            # Extract clip
            clip_filename = f"step_{step_order:02d}_{step_title.replace(' ', '_')[:30]}.mp4"
            clip_path = CLIPS_DIR / clip_filename
            
            print(f"  Step {step_order}: {step_title}")
            print(f"    Time Range: {start_ts:.1f}s → {end_ts:.1f}s (raw: {original_duration:.1f}s)")
            if original_duration != duration:
                print(f"    Clamped to: {duration:.1f}s")
            print(f"    Extracting: {start_ts:.1f}s → {actual_end_ts:.1f}s")
            
            success = extract_clip_by_timestamp(
                video_path=video_path,
                output_path=str(clip_path),
                start_time=start_ts,
                end_time=actual_end_ts
            )
            
            if success:
                print(f"    ✓ Saved: {clip_path.name}")
                step.video_clip_path = str(clip_path)
                clips_extracted += 1
            else:
                print(f"    ✗ Failed to extract")
            
            print()
        
        # =====================================================================
        # Summary
        # =====================================================================
        print("=" * 80)
        print("  SUMMARY")
        print("=" * 80)
        print(f"\n  Video: {video_info.title}")
        print(f"  Duration: {video_info.duration_seconds}s")
        print(f"  Approach: Direct video upload (M-RoPE temporal grounding)")
        print(f"  Micro-actions extracted: {len(all_micro_actions)}")
        print(f"  Recipe steps: {len(recipe.steps)}")
        print(f"  Video clips extracted: {clips_extracted}")
        print(f"\n  Output directory: {OUTPUT_DIR.absolute()}")
        print(f"    - scene_log.json (VLM output)")
        print(f"    - recipe.json (LLM output)")
        print(f"    - clips/ ({clips_extracted} videos)")
        print(f"\n  Total time: VLM {vlm_time:.1f}s + LLM {llm_time:.1f}s = {vlm_time + llm_time:.1f}s")
        print()
        
    finally:
        # Cleanup downloaded video
        print("─" * 80)
        print("Cleaning up downloaded video...")
        downloader.cleanup(video_info)
        print("Done!\n")


if __name__ == "__main__":
    # Get URL from command line or use default
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEST_URL
    main(url)

