from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from .downloaders.base import VideoInfo
from .schemas import Recipe, SceneLog, SceneDescription
from .vlm.base import VLMAdapter
from .llm.base import LLMAdapter

if TYPE_CHECKING:
    from .llm.expander import RecipeExpander


class RecipeChef:
    """
    Orchestrates the two-stage pipeline: VLM scene analysis → LLM recipe generation.
    
    This class coordinates the flow:
    1. VLM analyzes frames and generates scene descriptions
    2. Scene descriptions are accumulated into a SceneLog
    3. LLM takes the SceneLog and generates a structured Recipe
    """

    def __init__(
        self,
        vlm_adapter: VLMAdapter,
        llm_adapter: LLMAdapter,
        save_scenes_path: Optional[str | Path] = None,
        expander: Optional["RecipeExpander"] = None,
    ):
        """
        Initialize the RecipeChef with VLM and LLM adapters.

        Args:
            vlm_adapter: Adapter for vision-language model (scene analysis)
            llm_adapter: Adapter for language model (recipe generation)
            save_scenes_path: Optional path to save SceneLog for debugging/caching
            expander: Beginner-expansion pass. Defaults to RecipeExpander() when the
                      expansion stage runs.
        """
        self._vlm = vlm_adapter
        self._llm = llm_adapter
        self._save_scenes_path = Path(save_scenes_path) if save_scenes_path else None
        self._expander = expander

    def generate_recipe(
        self,
        video_info: VideoInfo,
        frames: list[str],
        transcript: str | None = None,
        chunk_size: int = 5
    ) -> Recipe:
        """
        Generate a recipe from video frames using the two-stage pipeline.
        
        Legacy method using frame-based analysis.
        For timestamp-based analysis, use generate_recipe_with_timestamps().
        
        Args:
            video_info: Metadata about the video (title, description, etc.)
            frames: List of base64-encoded JPEG images
            transcript: Optional audio transcript from the video
            chunk_size: Number of frames to process in each VLM batch
            
        Returns:
            A validated Recipe object
        """
        # Stage 1: VLM analyzes frames and generates scene descriptions
        scene_descriptions = self._vlm.analyze_video_direct(
            video_info,
            frames,
            transcript,
            chunk_size
        )
        
        # Create SceneLog from descriptions
        scene_log = SceneLog(
            scenes=scene_descriptions,
            video_info={
                "title": video_info.title,
                "description": video_info.description,
                "url": video_info.url,
                "duration_seconds": video_info.duration_seconds
            }
        )
        
        # Optionally save SceneLog for debugging/caching
        if self._save_scenes_path:
            self._save_scene_log(scene_log)
        
        # Stage 2: LLM generates recipe from scene descriptions
        recipe = self._llm.generate_recipe(scene_log, video_info, transcript, use_timestamps=False)
        
        return recipe

    def generate_recipe_with_timestamps(
        self,
        video_info: VideoInfo,
        video_path: str,
        transcript: str | None = None,
        target_fps: float = 2.0,
        max_frames: int = 80,
        chunk_size: int = 16
    ) -> Recipe:
        """
        Generate a recipe using dense frame extraction with timestamps.
        
        This method extracts frames at a high rate and uses timestamp-based
        mapping for more accurate video clip extraction.
        
        Args:
            video_info: Metadata about the video (title, description, etc.)
            video_path: Path to the local video file
            transcript: Optional audio transcript from the video
            target_fps: Frames per second to extract (default 2.0)
            max_frames: Maximum frames to extract
            chunk_size: Number of frames per VLM batch
            
        Returns:
            A validated Recipe object with timestamp-based video clips
        """
        # Stage 1: VLM analyzes video with timestamps
        scene_log = self._vlm.analyze_video_with_timestamps(
            video_info,
            video_path,
            transcript,
            target_fps=target_fps,
            max_frames=max_frames,
            chunk_size=chunk_size
        )
        
        # Optionally save SceneLog for debugging/caching
        if self._save_scenes_path:
            self._save_scene_log(scene_log)
        
        # Stage 2: LLM generates recipe with timestamp-based mapping
        recipe = self._llm.generate_recipe(scene_log, video_info, transcript, use_timestamps=True)
        
        return recipe

    def generate_recipe_direct(
        self,
        video_info: VideoInfo,
        video_path: str,
        transcript: str | None = None,
        debug: bool = True,
        expand: bool = True,
        on_stage: Optional[Callable[[str], None]] = None,
    ) -> Recipe:
        """
        Generate a recipe by uploading video directly to a video-capable VLM.

        This method compresses and uploads the video directly (no frame extraction)
        to a video-capable VLM like Gemini, leveraging M-RoPE for temporal grounding.
        This is the most accurate method for timestamp-based clip extraction.

        Three stages:
          1. VLM watches the video and emits a SceneLog of micro-actions.
          2. LLM groups those into a video-grounded skeleton, and the code (not the
             model) resolves micro_action_ids into timestamps.
          3. A second LLM expands that skeleton for a total beginner, filling in
             everything the video left out. The expanded recipe is what callers get,
             and therefore what gets cached and served.

        Args:
            video_info: Metadata about the video (title, description, etc.)
            video_path: Path to the local video file
            transcript: Optional audio transcript from the video
            debug: Print VLM diagnostics
            expand: Run the beginner-expansion pass. Set False to get the raw grounded
                    recipe - useful for diffing what expansion actually changed.
            on_stage: Called with "grounding" / "expanding" as each stage begins, so a
                      caller can report progress across what is now three model calls.

        Returns:
            A validated Recipe object with timestamp-based video clips
        """
        def stage(name: str) -> None:
            if on_stage:
                on_stage(name)

        # Stage 1: VLM analyzes video directly (no frame extraction). The transcript
        # goes along so the VLM can name/quantify what it sees (the compressed upload
        # is silent) - the prompt forbids inventing actions from narration.
        scene_log = self._vlm.analyze_video_direct(video_info, video_path, transcript=transcript, debug=debug)

        # Stage 1.5: high-res keyframe reading. The temporal pass runs on a hard-compressed
        # upload that can't read captions/labels, so we pull sharp frames at the moments
        # ingredients are introduced and read them with a cheap multi-image call. Additive
        # evidence only - a failure here just means no observations, never a failed recipe.
        stage("reading")
        try:
            self._read_keyframes_into(scene_log, video_info, video_path, debug=debug)
        except Exception as e:
            print(f"[Chef] Keyframe reading failed ({e}); continuing without observations")

        # Optionally save SceneLog for debugging/caching
        if self._save_scenes_path:
            self._save_scene_log(scene_log)

        # Stage 2: LLM groups micro-actions into a grounded skeleton, and
        # compute_timestamps_from_micro_actions freezes the video timing onto it.
        stage("grounding")
        grounded = self._llm.generate_recipe(scene_log, video_info, transcript, use_timestamps=True)

        if not expand:
            return grounded

        stage("expanding")

        # Stage 3: expand for a beginner. The grounded recipe is the fallback, so a
        # failure here degrades to today's output rather than breaking the request.
        return self._expand_for_beginner(grounded, video_info)

    def _read_keyframes_into(
        self,
        scene_log: SceneLog,
        video_info: VideoInfo,
        video_path: str,
        debug: bool = False,
    ) -> None:
        """Attach high-res keyframe observations to the SceneLog, in place.

        VLM-guided selection: the micro-action timeline says when ingredients are
        introduced; those moments get a sharp frame and a reading pass. Scene-cut
        detection fills in when the timeline yields too few events (fast montages).
        """
        from .vlm.keyframes import (
            _dedupe_and_cap,
            detect_scene_cut_timestamps,
            extract_reading_keyframes,
            select_reading_timestamps,
        )

        if not scene_log.scenes or not hasattr(self._vlm, "read_keyframes"):
            return

        timestamps = select_reading_timestamps(scene_log)
        if len(timestamps) < 3:
            merged = timestamps + detect_scene_cut_timestamps(video_path)
            timestamps = _dedupe_and_cap(merged, 14)
        if not timestamps:
            return

        start = time.perf_counter()
        keyframes = extract_reading_keyframes(video_path, timestamps)
        observations = self._vlm.read_keyframes(video_info, keyframes)
        scene_log.scenes[0].observations = observations
        elapsed = time.perf_counter() - start

        if debug:
            print(f"[Chef] Keyframe reading: {len(keyframes)} frames -> "
                  f"{len(observations)} observations in {elapsed:.1f}s")
            for obs in observations:
                ts = f"{obs.timestamp_seconds:.1f}s" if obs.timestamp_seconds is not None else "?"
                print(f"  [{ts}] {obs.ingredient or '?'} | amount: {obs.stated_amount or '-'} | "
                      f"text: {obs.on_screen_text or '-'}")

    def _expand_for_beginner(self, grounded: Recipe, video_info: VideoInfo) -> Recipe:
        from .llm.expander import RecipeExpander
        from .recipe_assembly import (
            merge_expansion,
            build_gather_steps,
            prune_dangling_references,
        )

        try:
            expander = self._expander or RecipeExpander()
            expanded = expander.expand(grounded, video_info)
            recipe = merge_expansion(grounded, expanded)
            recipe = prune_dangling_references(recipe)
            recipe = build_gather_steps(recipe)
            return recipe
        except Exception as e:
            print(f"[Expansion] Failed ({type(e).__name__}: {e}) - serving the grounded recipe instead")
            grounded.expansion_failed = True
            return grounded

    def _save_scene_log(self, scene_log: SceneLog) -> None:
        """Save SceneLog to JSON file for debugging or caching."""
        if self._save_scenes_path:
            self._save_scenes_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._save_scenes_path, 'w') as f:
                json.dump(scene_log.to_dict(), f, indent=2)

    @classmethod
    def load_scene_log(cls, path: str | Path) -> SceneLog:
        """
        Load a SceneLog from a JSON file.
        
        Useful for debugging or re-running recipe generation with cached scene descriptions.
        """
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return SceneLog.from_dict(data)

