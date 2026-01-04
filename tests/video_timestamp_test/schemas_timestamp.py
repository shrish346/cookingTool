"""
Timestamp-based schemas for direct video analysis.

Instead of mapping actions to frame indices, we map them directly to timestamps.
This leverages the VLM's native video understanding with M-RoPE for temporal grounding.
"""
from pydantic import BaseModel, Field, computed_field
from typing import Optional, Literal


class MicroActionTimestamp(BaseModel):
    """
    A single atomic cooking action with precise timestamp.
    
    Unlike the frame-based MicroAction, this uses timestamps (seconds)
    for precise temporal positioning within the video.
    """
    id: int = Field(ge=0, description="Unique ID for this micro-action within the video")
    action: str = Field(description="Brief action description (e.g., 'add salt', 'stir pan', 'flip chicken')")
    timestamp_seconds: float = Field(ge=0.0, description="Timestamp in seconds where this action occurs")
    duration_seconds: Optional[float] = Field(default=None, ge=0.0, description="Duration of this action in seconds")
    entity: Optional[str] = Field(default=None, description="What is being acted upon (e.g., 'onions', 'pan')")
    state_before: Optional[str] = Field(default=None, description="State before action (e.g., 'raw', 'whole')")
    state_after: Optional[str] = Field(default=None, description="State after action (e.g., 'browning', 'diced')")
    concurrent_with_other_action: Optional[bool] = Field(default=None, description="Whether this action happens at the same time as another")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score for this action")

    @computed_field
    @property
    def end_timestamp(self) -> float:
        """Calculate the end timestamp based on duration."""
        if self.duration_seconds:
            return self.timestamp_seconds + self.duration_seconds
        # Default: assume 2 second action if no duration specified
        return self.timestamp_seconds + 2.0


class EntityTimestamp(BaseModel):
    """Represents an ingredient, tool, or appliance identified in the video."""
    name: str
    type: Literal["ingredient", "tool", "appliance"]
    first_seen_timestamp: Optional[float] = Field(default=None, ge=0.0, description="First timestamp where entity is visible")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    quantity: Optional[str] = Field(default=None, description="Observed quantity if visible")
    state: Optional[str] = Field(default=None, description="Current state if applicable")


class StateChangeTimestamp(BaseModel):
    """Tracks a transformation observed in the video with timestamp."""
    entity: str = Field(description="Name of the entity that changed state")
    from_state: Optional[str] = Field(default=None, description="Previous state")
    to_state: str = Field(description="New state")
    timestamp_seconds: float = Field(ge=0.0, description="Timestamp where change was observed")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class VideoSceneDescription(BaseModel):
    """Container for full video analysis from the VLM."""
    video_duration_seconds: float = Field(ge=0.0, description="Total video duration")
    entities: list[EntityTimestamp] = Field(default_factory=list, description="All entities visible in video")
    state_changes: list[StateChangeTimestamp] = Field(default_factory=list, description="State transformations observed")
    micro_actions: list[MicroActionTimestamp] = Field(default_factory=list, description="All atomic actions with timestamps")
    summary: Optional[str] = Field(default=None, description="Brief summary of what's happening in the video")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")
    
    def get_sorted_micro_actions(self) -> list[MicroActionTimestamp]:
        """Return micro-actions sorted by timestamp."""
        return sorted(self.micro_actions, key=lambda a: a.timestamp_seconds)
    
    def build_timestamp_lookup(self) -> dict[int, float]:
        """Build a lookup table mapping micro-action IDs to their timestamps."""
        return {ma.id: ma.timestamp_seconds for ma in self.micro_actions}
    
    def build_action_description_lookup(self) -> dict[int, str]:
        """Build a lookup table mapping micro-action IDs to their descriptions."""
        return {ma.id: ma.action for ma in self.micro_actions}


class VideoSceneLog(BaseModel):
    """Wrapper for video scene description with video metadata."""
    scene: VideoSceneDescription
    video_info: Optional[dict] = Field(default=None, description="Video metadata for context")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: dict) -> "VideoSceneLog":
        """Create from dictionary (for loading from JSON)."""
        return cls(**data)
    
    def get_all_micro_actions(self) -> list[MicroActionTimestamp]:
        """Get all micro-actions sorted by timestamp."""
        return self.scene.get_sorted_micro_actions()


class StepTimestamp(BaseModel):
    """Recipe step with timestamp-based video mapping."""
    order: int = Field(ge=1)
    title: Optional[str] = Field(default=None, description="Short title for this step")
    instruction: str
    duration_minutes: Optional[int] = None
    tips: Optional[list[str]] = None
    # Timestamp-based video mapping
    micro_action_ids: Optional[list[int]] = Field(default=None, description="IDs of micro-actions that comprise this step")
    micro_action_descriptions: Optional[list[str]] = Field(default=None, description="Descriptions of micro-actions")
    # Time range for video clip
    start_timestamp_seconds: Optional[float] = Field(default=None, ge=0, description="Start time in seconds")
    end_timestamp_seconds: Optional[float] = Field(default=None, ge=0, description="End time in seconds")
    has_video_clip: bool = Field(default=True, description="Whether this step has a matching video clip")
    video_clip_path: Optional[str] = Field(default=None, description="Local path to the extracted video clip")
    
    @computed_field
    @property
    def clip_duration_seconds(self) -> Optional[float]:
        """Calculate clip duration from timestamps."""
        if self.start_timestamp_seconds is not None and self.end_timestamp_seconds is not None:
            return self.end_timestamp_seconds - self.start_timestamp_seconds
        return None
    
    @computed_field
    @property
    def display_title(self) -> str:
        """Auto-generate a title from instruction if not provided."""
        if self.title:
            return self.title
        words = self.instruction.split()[:5]
        return ' '.join(words) + ('...' if len(self.instruction.split()) > 5 else '')


class IngredientTimestamp(BaseModel):
    """Ingredient with optional timestamp for when it first appears."""
    name: str
    quantity: float = Field(gt=0)
    unit: str
    preparation: Optional[str] = None
    first_seen_timestamp: Optional[float] = Field(default=None, ge=0, description="When ingredient first appears in video")


class RecipeTimestamp(BaseModel):
    """Recipe with timestamp-based video mapping."""
    title: str
    description: Optional[str] = None
    reasoning: Optional[str] = Field(default=None, description="Model's thought process")
    ingredients: list[IngredientTimestamp]
    steps: list[StepTimestamp]
    servings: int = Field(gt=0)
    prep_time_minutes: Optional[int] = None
    cook_time_minutes: Optional[int] = None
    source_url: Optional[str] = None
    cuisine: Optional[str] = None
    tags: Optional[list[str]] = None
    # Nutrition (optional)
    calories: Optional[int] = None
    protein: Optional[int] = None
    carbs: Optional[int] = None
    fats: Optional[int] = None
    # Video metadata
    video_duration_seconds: Optional[float] = Field(default=None, description="Source video duration")
    
    @computed_field
    @property
    def total_time_minutes(self) -> Optional[int]:
        if self.prep_time_minutes is not None and self.cook_time_minutes is not None:
            return self.prep_time_minutes + self.cook_time_minutes
        elif self.prep_time_minutes is not None:
            return self.prep_time_minutes
        elif self.cook_time_minutes is not None:
            return self.cook_time_minutes
        return None


def compute_timestamps_from_micro_actions(
    recipe: RecipeTimestamp,
    scene_log: VideoSceneLog
) -> RecipeTimestamp:
    """
    Post-process a recipe to compute start/end timestamps from micro_action_ids.
    
    This function derives video clip timing from the VLM's micro-action
    timestamp data, ensuring consistency between micro_action_ids and timestamps.
    
    Args:
        recipe: Recipe with steps containing micro_action_ids
        scene_log: VideoSceneLog containing all micro-actions with timestamps
        
    Returns:
        Recipe with start_timestamp_seconds and end_timestamp_seconds populated for each step
    """
    # Build lookup tables
    id_to_timestamp = scene_log.scene.build_timestamp_lookup()
    id_to_description = scene_log.scene.build_action_description_lookup()
    
    # Get all micro-actions for duration lookup
    all_actions = {ma.id: ma for ma in scene_log.get_all_micro_actions()}
    
    # Process each step
    for step in recipe.steps:
        if not step.has_video_clip or not step.micro_action_ids:
            step.start_timestamp_seconds = None
            step.end_timestamp_seconds = None
            continue
        
        timestamps = []
        descriptions = []
        invalid_ids = []
        
        for ma_id in step.micro_action_ids:
            if ma_id in id_to_timestamp:
                action_desc = id_to_description.get(ma_id, f"ID {ma_id}")
                descriptions.append(action_desc)
                
                # Skip "no relevant cooking action" when calculating time range
                if "no relevant cooking" not in action_desc.lower():
                    start_ts = id_to_timestamp[ma_id]
                    timestamps.append(start_ts)
                    
                    # Also consider the action's end timestamp
                    if ma_id in all_actions:
                        end_ts = all_actions[ma_id].end_timestamp
                        timestamps.append(end_ts)
            else:
                invalid_ids.append(ma_id)
        
        step.micro_action_descriptions = descriptions
        
        if invalid_ids:
            print(f"[TimestampMapping] Warning: Step {step.order} references invalid IDs: {invalid_ids}")
        
        if timestamps:
            step.start_timestamp_seconds = min(timestamps)
            step.end_timestamp_seconds = max(timestamps)
        else:
            print(f"[TimestampMapping] Warning: Step {step.order} has no valid timestamps, disabling video clip")
            step.has_video_clip = False
            step.start_timestamp_seconds = None
            step.end_timestamp_seconds = None
    
    return recipe

