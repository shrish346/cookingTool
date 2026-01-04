from pydantic import BaseModel, Field, computed_field
from typing import Optional, Literal

class Ingredient(BaseModel):
    name: str
    quantity: float = Field(gt=0)
    unit: str
    preparation: Optional[str] = None

class Step(BaseModel):
    order: int = Field(ge=1)
    title: Optional[str] = Field(default=None, description="Short title for this step (e.g., 'Boil the Pasta')")
    instruction: str
    duration_minutes: Optional[int] = None
    tips: Optional[list[str]] = None
    # Micro-action based video mapping (V2 approach)
    micro_action_ids: Optional[list[int]] = Field(default=None, description="IDs of micro-actions that comprise this step")
    micro_action_descriptions: Optional[list[str]] = Field(default=None, description="Descriptions of micro-actions (for debugging)")
    # Video loop fields for the Cooking Mode
    start_frame_index: Optional[float] = Field(default=None, ge=0, description="Precise frame index where this step starts (from micro-actions)")
    end_frame_index: Optional[float] = Field(default=None, ge=0, description="Precise frame index where this step ends (from micro-actions)")
    has_video_clip: bool = Field(default=True, description="Whether this step has a matching video clip")
    video_clip_url: Optional[str] = Field(default=None, description="S3 URL for the extracted video clip")
    
    @computed_field
    @property
    def display_title(self) -> str:
        """Auto-generate a title from instruction if not provided."""
        if self.title:
            return self.title
        # Take first 5 words of instruction as title
        words = self.instruction.split()[:5]
        return ' '.join(words) + ('...' if len(self.instruction.split()) > 5 else '')

class Recipe(BaseModel):
    title: str
    description: Optional[str] = None
    reasoning: Optional[str] = Field(default=None, description="Model's thought process for extracting this recipe")
    ingredients: list[Ingredient]
    steps: list[Step]
    servings: int = Field(gt=0)
    prep_time_minutes: Optional[int] = None
    cook_time_minutes: Optional[int] = None
    source_url: Optional[str] = None
    cusine: Optional[str] = None
    tags: Optional[list[str]] = None
    nutrition: Optional[dict] = None
    calories: Optional[int] = None
    protein: Optional[int] = None
    carbs: Optional[int] = None
    fats: Optional[int] = None
    cholesterol: Optional[int] = None
    sodium: Optional[int] = None
    sugar: Optional[int] = None
    vitamin_a: Optional[int] = None
    vitamin_c: Optional[int] = None
    calcium: Optional[int] = None
    # Video metadata for caching and clip extraction
    video_id: Optional[str] = Field(default=None, description="YouTube/TikTok video ID for caching")
    video_fps: Optional[float] = Field(default=None, gt=0, description="Source video FPS for timestamp calculations")
    clips_ready: bool = Field(default=False, description="Whether all video clips have been uploaded to S3")

    @computed_field
    @property
    def total_time_minutes(self) -> Optional[int]:
        if self.prep_time_minutes is not None and self.cook_time_minutes is not None:
            return self.prep_time_minutes + self.cook_time_minutes
        elif self.prep_time_minutes is not None:
            return self.prep_time_minutes
        elif self.cook_time_minutes is not None:
            return self.cook_time_minutes
        else:
            return None


# Scene Description Models for VLM → LLM Pipeline

class MicroAction(BaseModel):
    """
    A single atomic cooking action with precise timing.
    
    This enables accurate video clip extraction by tracking exactly when
    each small action occurs within a scene/chunk.
    """
    id: int = Field(ge=0, description="Unique ID for this micro-action within the video")
    action: str = Field(description="Brief action description (e.g., 'add salt', 'stir pan', 'flip chicken')")
    frame_index: int = Field(ge=0, description="Chunk start frame index (0, 12, 24, etc.)")
    relative_position: float = Field(ge=0.0, le=1.0, description="Position within the chunk (0.0=start, 1.0=end)")
    entity: Optional[str] = Field(default=None, description="What is being acted upon (e.g., 'onions', 'pan')")
    state_before: Optional[str] = Field(default=None, description="State before action (e.g., 'raw', 'whole')")
    state_after: Optional[str] = Field(default=None, description="State after action (e.g., 'browning', 'diced')")
    
    @computed_field
    @property
    def precise_frame_index(self) -> float:
        """
        Legacy: Returns frame_index + relative_position for backward compatibility.
        For accurate frame calculation, use get_actual_frame(chunk_size) instead.
        """
        return self.frame_index + self.relative_position
    
    def get_actual_frame(self, chunk_size: int = 12) -> float:
        """
        Compute the actual extracted frame number where this action occurs.
        
        Args:
            chunk_size: Number of frames per chunk (default 12)
            
        Returns:
            Actual frame number (e.g., chunk 0 + 0.20 relative = frame 2.4)
        """
        return self.frame_index + (self.relative_position * chunk_size)


class Entity(BaseModel):
    """Represents an ingredient, tool, or appliance identified in a frame."""
    name: str
    type: Literal["ingredient", "tool", "appliance"]
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score if available")
    quantity: Optional[str] = Field(default=None, description="Observed quantity if visible (e.g., '2 cups', '200ml')")
    state: Optional[str] = Field(default=None, description="Current state if applicable (e.g., 'raw', 'chopped', 'boiling')")


class StateChange(BaseModel):
    """Tracks a transformation or state change observed in the video."""
    entity: str = Field(description="Name of the entity that changed state")
    from_state: Optional[str] = Field(default=None, description="Previous state (e.g., 'raw', 'solid')")
    to_state: str = Field(description="New state (e.g., 'translucent', 'boiling', 'chopped')")
    frame_index: int = Field(ge=0, description="Frame index where change was observed")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class TemporalAction(BaseModel):
    """Records an action with temporal ordering information."""
    step_number: Optional[int] = Field(default=None, ge=1, description="Sequential step number if determinable")
    action_description: str = Field(description="Description of the action (e.g., 'chopping onions', 'pouring 200ml milk')")
    frame_index: int = Field(ge=0, description="Frame index where action was observed")
    timestamp_seconds: Optional[float] = Field(default=None, ge=0, description="Approximate timestamp in video")
    entities_involved: Optional[list[str]] = Field(default=None, description="List of entity names involved in this action")


class SceneDescription(BaseModel):
    """Container for one frame or chunk analysis from the VLM."""
    frame_index: int = Field(ge=0, description="Frame index (or starting index for chunk)")
    frame_indices: Optional[list[int]] = Field(default=None, description="All frame indices if this represents a chunk")
    entities: list[Entity] = Field(default_factory=list, description="Ingredients, tools, and appliances visible")
    state_changes: list[StateChange] = Field(default_factory=list, description="State transformations observed")
    temporal_actions: list[TemporalAction] = Field(default_factory=list, description="Actions performed")
    micro_actions: list["MicroAction"] = Field(default_factory=list, description="Granular atomic actions with precise timing")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata (confidence, notes, etc.)")


class SceneLog(BaseModel):
    """Accumulated list of scene descriptions from VLM analysis."""
    scenes: list[SceneDescription] = Field(default_factory=list)
    video_info: Optional[dict] = Field(default=None, description="Video metadata for context")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: dict) -> "SceneLog":
        """Create from dictionary (for loading from JSON)."""
        return cls(**data)
    
    def get_all_micro_actions(self) -> list[MicroAction]:
        """
        Collect all micro-actions from all scenes, sorted by precise frame index.
        Useful for the LLM to see a complete timeline of actions.
        """
        all_actions = []
        for scene in self.scenes:
            all_actions.extend(scene.micro_actions)
        return sorted(all_actions, key=lambda a: a.precise_frame_index)
    
    def build_micro_action_lookup(self, chunk_size: int = 12) -> dict[int, float]:
        """
        Build a lookup table mapping micro-action IDs to their actual frame numbers.
        
        Args:
            chunk_size: Number of frames per chunk (default 12)
            
        Returns:
            Dict mapping micro_action_id -> actual_frame_number
        """
        return {ma.id: ma.get_actual_frame(chunk_size) for ma in self.get_all_micro_actions()}
    
    def build_micro_action_description_lookup(self) -> dict[int, str]:
        """
        Build a lookup table mapping micro-action IDs to their action descriptions.
        
        Returns:
            Dict mapping micro_action_id -> action description
        """
        return {ma.id: ma.action for ma in self.get_all_micro_actions()}


def compute_frame_indices_from_micro_actions(
    recipe: "Recipe",
    scene_log: "SceneLog",
    chunk_size: int = 12
) -> "Recipe":
    """
    Post-process a recipe to compute start/end frame indices from micro_action_ids.
    
    This function deterministically derives frame timing from the VLM's micro-action
    data, ensuring consistency between micro_action_ids and frame indices.
    
    Also adds micro_action_descriptions to each step for debugging purposes.
    
    Args:
        recipe: Recipe with steps containing micro_action_ids
        scene_log: SceneLog containing all micro-actions with frame indices
        chunk_size: Number of frames per VLM chunk (default 12)
        
    Returns:
        Recipe with start_frame_index and end_frame_index populated for each step
    """
    # Build lookup tables using actual frame numbers
    id_to_frame = scene_log.build_micro_action_lookup(chunk_size)
    id_to_description = scene_log.build_micro_action_description_lookup()
    
    # Process each step
    for step in recipe.steps:
        # Skip steps marked as no video clip or with no micro-action IDs
        if not step.has_video_clip or not step.micro_action_ids:
            step.start_frame_index = None
            step.end_frame_index = None
            continue
        
        # Look up frame indices and descriptions for all referenced micro-action IDs
        frame_indices = []
        descriptions = []
        invalid_ids = []
        
        for ma_id in step.micro_action_ids:
            if ma_id in id_to_frame:
                action_desc = id_to_description.get(ma_id, f"ID {ma_id}")
                descriptions.append(action_desc)
                
                # Skip "no relevant cooking action" when calculating frame range
                # (but still include in descriptions for transparency)
                if "no relevant cooking" not in action_desc.lower():
                    frame_indices.append(id_to_frame[ma_id])
            else:
                invalid_ids.append(ma_id)
        
        # Store descriptions for debugging
        step.micro_action_descriptions = descriptions
        
        # Warn about invalid IDs (optional - could be logged)
        if invalid_ids:
            print(f"[FrameMapping] Warning: Step {step.order} references invalid micro-action IDs: {invalid_ids}")
        
        # Compute frame range from valid IDs
        if frame_indices:
            step.start_frame_index = min(frame_indices)
            step.end_frame_index = max(frame_indices)
        else:
            # All IDs were invalid - mark as no video clip
            print(f"[FrameMapping] Warning: Step {step.order} has no valid micro-action IDs, disabling video clip")
            step.has_video_clip = False
            step.start_frame_index = None
            step.end_frame_index = None
    
    return recipe

