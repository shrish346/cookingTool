"""
Test the VLM-LLM pipeline and local clip extraction.

Outputs:
- Frame extraction details (which frames go into which chunks)
- All micro-actions from VLM with frame mappings
- All recipe steps from LLM with associated micro-actions
- Local video clips for each step (saved to ./test_output/clips/)

Usage:
    python tests/test_pipeline.py [youtube_url]
    
    If no URL provided, uses a default test video.
"""
import sys
from pathlib import Path
import time
import subprocess
import base64
import shutil
import hashlib

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.downloaders.youtube import YouTubeDownloader
from src.processing.frames import FrameExtractor
from src.processing.audio import AudioTranscriber
from src.vlm.openrouter import OpenRouterAdapter
from src.llm.openai import OpenAIAdapter
from src.schemas import SceneLog, compute_frame_indices_from_micro_actions

# Output directories
OUTPUT_DIR = Path("test_output")
FRAMES_DIR = OUTPUT_DIR / "frames"
CLIPS_DIR = OUTPUT_DIR / "clips"

# Default test video (short cooking video)
DEFAULT_TEST_URL = "https://www.youtube.com/shorts/K_L-FVXOR0o"


def extract_local_clip(
    video_path: str,
    output_path: str,
    start_time: float,
    duration: float,
    ffmpeg_path: str = "ffmpeg"
) -> bool:
    """Extract a clip from the video using FFmpeg (local only, no S3)."""
    try:
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


def get_video_fps(video_path: str) -> float:
    """Get video FPS using FFprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            fps_str = result.stdout.strip()
            if "/" in fps_str:
                num, den = map(float, fps_str.split("/"))
                return num / den
            return float(fps_str)
    except Exception:
        pass
    
    return 30.0


def get_total_frames(video_path: str) -> int:
    """Get total frame count using FFprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception:
        pass
    
    # Fallback: estimate
    return 1000


def compute_frame_hash(frame_b64: str) -> str:
    """Compute a short hash of a frame to detect duplicates."""
    return hashlib.md5(frame_b64.encode()).hexdigest()[:8]


def main(test_url: str = DEFAULT_TEST_URL):
    """Run the full pipeline test."""
    
    print("\n" + "=" * 80)
    print("  VLM-LLM PIPELINE TEST (with frame extraction debugging)")
    print("=" * 80)
    print(f"\nTest URL: {test_url}\n")
    
    # Setup output directories
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()
    FRAMES_DIR.mkdir()
    CLIPS_DIR.mkdir()
    
    # Initialize components
    downloader = YouTubeDownloader()
    extractor = FrameExtractor(resize_width=512)
    transcriber = AudioTranscriber()
    vlm_adapter = OpenRouterAdapter()
    llm_adapter = OpenAIAdapter()
    
    print(f"VLM Model: {vlm_adapter.model_name}")
    print(f"LLM Model: {llm_adapter.model_name}\n")
    
    # =========================================================================
    # STEP 1: Download video
    # =========================================================================
    print("─" * 80)
    print("[1/6] DOWNLOADING VIDEO")
    print("─" * 80)
    
    video_info = downloader.download(test_url)
    video_path = str(video_info.file_path)
    
    print(f"  Title: {video_info.title}")
    print(f"  Duration: {video_info.duration_seconds}s")
    print(f"  File: {video_path}\n")
    
    try:
        # =====================================================================
        # STEP 2: Extract frames
        # =====================================================================
        print("─" * 80)
        print("[2/6] EXTRACTING FRAMES")
        print("─" * 80)
        
        frames = extractor.extract(video_path)
        num_frames = len(frames)
        
        # Save frames for inspection and compute hashes
        frame_hashes = []
        for i, frame_b64 in enumerate(frames):
            frame_bytes = base64.b64decode(frame_b64)
            (FRAMES_DIR / f"frame_{i:02d}.jpg").write_bytes(frame_bytes)
            frame_hashes.append(compute_frame_hash(frame_b64))
        
        print(f"  Extracted {num_frames} frames")
        print(f"  Saved to: {FRAMES_DIR}/\n")
        
        # Get video metadata for clip extraction
        fps = get_video_fps(video_path)
        total_frames = get_total_frames(video_path)
        frame_interval = max(1, total_frames // num_frames) if num_frames > 0 else 1
        
        print(f"  Video FPS: {fps:.2f}")
        print(f"  Total video frames: {total_frames}")
        print(f"  Sample interval: every {frame_interval} frames")
        print(f"  Duration per sample: {frame_interval / fps:.2f}s\n")
        
        # Show frame hashes to detect duplicates
        print("  Frame Hashes (to detect duplicate frames):")
        unique_hashes = set()
        duplicates = []
        for i, h in enumerate(frame_hashes):
            if h in unique_hashes:
                duplicates.append((i, h))
            unique_hashes.add(h)
            # Show first 10 and last 5
            if i < 10 or i >= num_frames - 5:
                timestamp = (i * frame_interval) / fps
                print(f"    Frame {i:2d}: {h} (at {timestamp:.1f}s)")
            elif i == 10:
                print(f"    ... ({num_frames - 15} more frames) ...")
        
        if duplicates:
            print(f"\n  ⚠ WARNING: {len(duplicates)} duplicate frames detected!")
            for idx, h in duplicates:
                print(f"    Frame {idx} has same hash as earlier frame")
        else:
            print(f"\n  ✓ All {num_frames} frames are unique")
        
        # =====================================================================
        # Show VLM chunk breakdown
        # =====================================================================
        chunk_size = 12  # Default chunk size
        print(f"\n  VLM Chunk Breakdown (chunk_size={chunk_size}):")
        num_chunks = (num_frames + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_frames)
            chunk_frames = list(range(start_idx, end_idx))
            
            start_time = (start_idx * frame_interval) / fps
            end_time = (end_idx * frame_interval) / fps
            
            print(f"    Chunk {chunk_idx}: frames {start_idx}-{end_idx-1} ({len(chunk_frames)} frames)")
            print(f"              video time: {start_time:.1f}s → {end_time:.1f}s")
        
        print()
        
        # =====================================================================
        # STEP 3: Transcribe audio
        # =====================================================================
        print("─" * 80)
        print("[3/6] TRANSCRIBING AUDIO")
        print("─" * 80)
        
        transcript = transcriber.process_video(video_path)
        
        if transcript:
            print(f"  Found speech ({len(transcript.split())} words)")
            print(f"  Preview: {transcript[:150]}...\n")
        else:
            print("  No meaningful speech detected\n")
        
        # =====================================================================
        # STEP 4: VLM Scene Analysis
        # =====================================================================
        print("─" * 80)
        print("[4/6] VLM SCENE ANALYSIS")
        print("─" * 80)
        
        print(f"\n  Processing {num_frames} frames in {num_chunks} chunks...")
        print("  (VLM retry messages will appear below if any chunks fail)\n")
        
        start_time = time.perf_counter()
        scene_descriptions = vlm_adapter.analyze_scenes(
            video_info, frames, transcript, chunk_size=chunk_size
        )
        vlm_time = time.perf_counter() - start_time
        
        # Create scene log
        scene_log = SceneLog(
            scenes=scene_descriptions,
            video_info={
                "title": video_info.title,
                "description": video_info.description,
                "url": video_info.url,
                "duration_seconds": video_info.duration_seconds
            }
        )
        
        print(f"\n  VLM Time: {vlm_time:.2f}s")
        print(f"  Scenes returned: {len(scene_descriptions)}")
        
        # Show scene breakdown
        print(f"\n  Scene Details:")
        for scene in scene_descriptions:
            frame_range = f"{scene.frame_indices[0]}-{scene.frame_indices[-1]}" if scene.frame_indices else str(scene.frame_index)
            print(f"    Scene at frame {scene.frame_index}: {len(scene.micro_actions)} micro-actions, frames {frame_range}")
        
        # Get all micro-actions
        all_micro_actions = scene_log.get_all_micro_actions()
        id_to_frame = scene_log.build_micro_action_lookup(chunk_size)
        
        # =====================================================================
        # OUTPUT: All Micro-Actions
        # =====================================================================
        print("\n" + "=" * 80)
        print("  ALL MICRO-ACTIONS FROM VLM")
        print("=" * 80)
        print(f"\n  Total: {len(all_micro_actions)} micro-actions\n")
        
        # Group by chunk for clarity
        # Calculate actual frame numbers from chunk start + relative_position * chunk_size
        print(f"  {'ID':<6} {'Frame':<6} {'Time':<8} {'File':<16} {'Action':<28} {'Entity':<10} {'State Change'}")
        print("  " + "─" * 105)
        
        current_chunk = -1
        for ma in all_micro_actions:
            chunk = ma.frame_index // chunk_size
            chunk_start = chunk * chunk_size
            chunk_end = min((chunk + 1) * chunk_size - 1, num_frames - 1)
            
            if chunk != current_chunk:
                if current_chunk != -1:
                    print("  " + "─" * 105)
                chunk_start_time = (chunk_start * frame_interval) / fps
                chunk_end_time = ((chunk_end + 1) * frame_interval) / fps
                print(f"  CHUNK {chunk}: frame_{chunk_start:02d}.jpg - frame_{chunk_end:02d}.jpg ({chunk_start_time:.1f}s - {chunk_end_time:.1f}s)")
                print("  " + "─" * 105)
                current_chunk = chunk
            
            state_change = ""
            if ma.state_before and ma.state_after:
                state_change = f"{ma.state_before} → {ma.state_after}"
            elif ma.state_after:
                state_change = f"→ {ma.state_after}"
            
            entity = ma.entity or ""
            
            # Use the new get_actual_frame method
            actual_frame = ma.get_actual_frame(chunk_size)
            actual_frame_int = min(int(actual_frame), num_frames - 1)
            frame_file = f"frame_{actual_frame_int:02d}.jpg"
            
            # Calculate video timestamp
            video_time = (actual_frame * frame_interval) / fps
            
            print(f"  {ma.id:<6} {actual_frame:<6.1f} ({video_time:>5.1f}s) {frame_file:<16} {ma.action[:28]:<28} {entity[:10]:<10} {state_change}")
        
        print()
        
        # =====================================================================
        # STEP 5: LLM Recipe Generation
        # =====================================================================
        print("─" * 80)
        print("[5/6] LLM RECIPE GENERATION")
        print("─" * 80)
        
        start_time = time.perf_counter()
        recipe = llm_adapter.generate_recipe(scene_log, video_info, transcript)
        llm_time = time.perf_counter() - start_time
        
        print(f"  Time: {llm_time:.2f}s")
        print(f"  Steps generated: {len(recipe.steps)}\n")
        
        # =====================================================================
        # OUTPUT: Recipe Details
        # =====================================================================
        print("=" * 80)
        print("  RECIPE GENERATED BY LLM")
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
        
        # Nutrition
        if recipe.calories:
            print(f"\n  NUTRITION:")
            print("  " + "─" * 50)
            print(f"    Calories: {recipe.calories} | Protein: {recipe.protein}g | Carbs: {recipe.carbs}g | Fat: {recipe.fats}g")
        
        # =====================================================================
        # OUTPUT: Steps with Micro-Action Mapping
        # =====================================================================
        print("\n" + "=" * 80)
        print("  STEPS WITH MICRO-ACTION MAPPING")
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
            
            # Video clip info
            print(f"\n  Video Clip: {'Yes' if step.has_video_clip else 'No'}")
            
            if step.has_video_clip and step.start_frame_index is not None:
                # Frame indices are now actual extracted frame numbers
                start_frame = step.start_frame_index
                end_frame = step.end_frame_index
                
                print(f"  Frame Range: frame_{int(start_frame):02d} → frame_{int(end_frame):02d} ({start_frame:.1f} → {end_frame:.1f})")
                
                # Convert to actual video timestamps
                start_time_sec = (start_frame * frame_interval) / fps
                end_time_sec = (end_frame * frame_interval) / fps
                
                print(f"  Video Time: {start_time_sec:.1f}s → {end_time_sec:.1f}s (duration: {end_time_sec - start_time_sec:.1f}s)")
                
                # Check if this spans multiple chunks
                if step.micro_action_ids:
                    chunks_used = set()
                    for ma_id in step.micro_action_ids:
                        ma = next((m for m in all_micro_actions if m.id == ma_id), None)
                        if ma:
                            chunks_used.add(ma.frame_index // chunk_size)
                    
                    if len(chunks_used) > 1:
                        print(f"  ⚠ WARNING: This step spans {len(chunks_used)} chunks: {sorted(chunks_used)}")
            
            # Micro-actions
            if step.micro_action_ids:
                print(f"\n  Micro-Actions ({len(step.micro_action_ids)}):")
                
                for ma_id in step.micro_action_ids:
                    if ma_id in id_to_frame:
                        # Find the actual micro-action
                        ma = next((m for m in all_micro_actions if m.id == ma_id), None)
                        if ma:
                            # Use the new get_actual_frame method
                            actual_frame = ma.get_actual_frame(chunk_size)
                            actual_frame_int = min(int(actual_frame), num_frames - 1)
                            
                            # Calculate video timestamp
                            video_time = (actual_frame * frame_interval) / fps
                            
                            print(f"    [{ma_id}] frame_{actual_frame_int:02d}.jpg ({video_time:.1f}s): {ma.action}")
                        else:
                            print(f"    [{ma_id}] Unknown")
                    else:
                        print(f"    [{ma_id}] (invalid ID)")
            else:
                print(f"\n  Micro-Actions: None (no video for this step)")
        
        # =====================================================================
        # STEP 6: Extract Local Clips
        # =====================================================================
        print("\n" + "─" * 80)
        print("[6/6] EXTRACTING LOCAL VIDEO CLIPS")
        print("─" * 80 + "\n")
        
        clips_extracted = 0
        
        for step in recipe.steps:
            step_order = step.order
            step_title = step.title or "Untitled"
            
            if not step.has_video_clip or step.start_frame_index is None:
                print(f"  Step {step_order}: Skipped (no video clip)")
                continue
            
            # Calculate timestamps - frame indices are now actual extracted frame numbers
            start_time_sec = (step.start_frame_index * frame_interval) / fps
            end_time_sec = (step.end_frame_index * frame_interval) / fps
            duration = end_time_sec - start_time_sec
            
            # Minimum 2 seconds, maximum 10 seconds
            original_duration = duration
            if duration < 2:
                duration = 2
            if duration > 10:
                duration = 10
            
            # Extract clip
            clip_filename = f"step_{step_order:02d}_{step_title.replace(' ', '_')[:30]}.mp4"
            clip_path = CLIPS_DIR / clip_filename
            
            print(f"  Step {step_order}: {step_title}")
            print(f"    Frames: frame_{int(step.start_frame_index):02d}.jpg → frame_{int(step.end_frame_index):02d}.jpg")
            print(f"    Video time: {start_time_sec:.1f}s → {end_time_sec:.1f}s (raw: {original_duration:.1f}s)")
            if original_duration != duration:
                print(f"    Clamped to: {duration:.1f}s")
            print(f"    Extracting: {start_time_sec:.1f}s → {start_time_sec + duration:.1f}s")
            
            success = extract_local_clip(
                video_path=video_path,
                output_path=str(clip_path),
                start_time=start_time_sec,
                duration=duration
            )
            
            if success:
                print(f"    ✓ Saved: {clip_path.name}")
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
        print(f"  Frames extracted: {num_frames} (one every {frame_interval} frames / {frame_interval/fps:.2f}s)")
        print(f"  VLM chunks: {num_chunks} (chunk_size={chunk_size})")
        print(f"  Micro-actions extracted: {len(all_micro_actions)}")
        print(f"  Recipe steps: {len(recipe.steps)}")
        print(f"  Video clips extracted: {clips_extracted}")
        print(f"\n  Output directory: {OUTPUT_DIR.absolute()}")
        print(f"    - Frames: {FRAMES_DIR.name}/ ({num_frames} images)")
        print(f"    - Clips: {CLIPS_DIR.name}/ ({clips_extracted} videos)")
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
