from pydantic import BaseModel, Field, computed_field
from typing import Optional, Literal
from uuid import uuid4

# Bump whenever the Recipe shape changes incompatibly. Cached recipes written
# under an older version are treated as a cache miss and regenerated.
SCHEMA_VERSION = 3


def _new_id() -> str:
    return uuid4().hex[:8]


Provenance = Literal["video", "reference", "model"]
"""Where a piece of the recipe came from.

video     - observed in the source video (grounded in micro_action_ids)
reference - taken from a web recipe; cites source_id
model     - the model's own cooking knowledge; an estimate
"""

StepKind = Literal[
    "gather_tools",
    "gather_ingredients",
    "prep_component",
    "prep",
    "cook",
    "assemble",
    "rest",
    "serve",
    "technique",
    "safety",
]
"""prep_component: making a pre-required component the source video assumed was
already done (cooked rice, steamed potatoes). Clipless, grouped as "Make ahead"
right after the gather steps."""


class Source(BaseModel):
    """A web recipe the expansion pass leaned on to fill gaps the video left."""
    id: str = Field(default_factory=_new_id)
    title: str
    url: Optional[str] = None
    site: Optional[str] = Field(default=None, description="Human-readable site name, e.g. 'Serious Eats'")


class Artifact(BaseModel):
    """Side-panel content attached to a step. Schema only - nothing populates these yet."""
    type: str
    title: Optional[str] = None
    payload: dict = Field(default_factory=dict)


class Tool(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    essential: bool = True
    substitute: Optional[str] = Field(default=None, description="What to use instead, e.g. 'no whisk? a fork works'")
    provenance: Provenance = "video"
    source_id: Optional[str] = None


class Ingredient(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    quantity: float = Field(gt=0)
    unit: str
    preparation: Optional[str] = None
    optional: bool = False
    provenance: Provenance = "video"
    source_id: Optional[str] = None
    note: Optional[str] = Field(default=None, description="Why this value, when the video didn't state it")

class Step(BaseModel):
    id: str = Field(default_factory=_new_id, description="Stable step ID. Video clips are keyed off this, not order.")
    order: int = Field(ge=1)
    kind: StepKind = "cook"
    title: Optional[str] = Field(default=None, description="Short title for this step (e.g., 'Boil the Pasta')")
    instruction: str
    detail: Optional[str] = Field(default=None, description="Expanded hand-holding explanation for a total beginner")
    doneness_cue: Optional[str] = Field(default=None, description="How the cook knows this step worked")
    duration_minutes: Optional[int] = None
    tips: Optional[list[str]] = None
    # Entity references into Recipe.ingredients / Recipe.tools. Groundwork for
    # recipe mutation: changing an ingredient means regenerating only the steps
    # whose ingredient_ids contain it.
    ingredient_ids: list[str] = Field(default_factory=list)
    tool_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list, description="Step IDs this step follows. Linear chain for now; the seam a DAG grows from.")
    provenance: Provenance = "video"
    source_id: Optional[str] = None
    artifacts: list[Artifact] = Field(default_factory=list)
    # Micro-action based video mapping (V2 approach)
    micro_action_ids: Optional[list[int]] = Field(default=None, description="IDs of micro-actions that comprise this step")
    micro_action_descriptions: Optional[list[str]] = Field(default=None, description="Descriptions of micro-actions (for debugging)")
    # Timestamp-based video clip fields (preferred)
    start_timestamp_seconds: Optional[float] = Field(default=None, ge=0, description="Start time in seconds for this step's video clip")
    end_timestamp_seconds: Optional[float] = Field(default=None, ge=0, description="End time in seconds for this step's video clip")
    # Frame-based video clip fields (legacy, for backward compatibility)
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
    schema_version: int = SCHEMA_VERSION
    title: str
    description: Optional[str] = None
    reasoning: Optional[str] = Field(default=None, description="Model's thought process for extracting this recipe")
    ingredients: list[Ingredient]
    tools: list[Tool] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list, description="Web recipes the expansion pass used to fill gaps")
    steps: list[Step]
    servings: int = Field(gt=0)
    difficulty: Optional[str] = None
    prep_time_minutes: Optional[int] = None
    cook_time_minutes: Optional[int] = None
    source_url: Optional[str] = None
    cuisine: Optional[str] = None
    tags: Optional[list[str]] = None
    dish_query: Optional[str] = Field(default=None, description="Short search string for this dish, used for web grounding")
    expansion_failed: bool = Field(default=False, description="True if the beginner-expansion pass failed and this is the raw grounded recipe")
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
    
    Supports both frame-based (legacy) and timestamp-based (new) positioning.
    Timestamp-based is preferred for accurate video clip extraction.
    """
    id: int = Field(ge=0, description="Unique ID for this micro-action within the video")
    action: str = Field(description="Brief action description (e.g., 'add salt', 'stir pan', 'flip chicken')")
    # Timestamp-based fields (preferred)
    timestamp_seconds: Optional[float] = Field(default=None, ge=0.0, description="Timestamp in seconds where this action occurs")
    duration_seconds: Optional[float] = Field(default=None, ge=0.0, description="Duration of this action in seconds")
    # Frame-based fields (legacy, for backward compatibility)
    frame_index: int = Field(default=0, ge=0, description="Chunk start frame index (0, 12, 24, etc.)")
    relative_position: float = Field(default=0.5, ge=0.0, le=1.0, description="Position within the chunk (0.0=start, 1.0=end)")
    # Common fields
    entity: Optional[str] = Field(default=None, description="What is being acted upon (e.g., 'onions', 'pan')")
    state_before: Optional[str] = Field(default=None, description="State before action (e.g., 'raw', 'whole')")
    state_after: Optional[str] = Field(default=None, description="State after action (e.g., 'browning', 'diced')")
    concurrent_with_other_action: Optional[bool] = Field(default=None, description="Whether this action happens at the same time as another")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score")
    
    @computed_field
    @property
    def precise_frame_index(self) -> float:
        """
        Legacy: Returns frame_index + relative_position for backward compatibility.
        For accurate frame calculation, use get_actual_frame(chunk_size) instead.
        """
        return self.frame_index + self.relative_position
    
    @computed_field
    @property
    def end_timestamp(self) -> Optional[float]:
        """Calculate the end timestamp based on duration."""
        if self.timestamp_seconds is not None:
            if self.duration_seconds:
                return self.timestamp_seconds + self.duration_seconds
            return self.timestamp_seconds + 2.0  # Default 2 second action
        return None
    
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
    
    def build_timestamp_lookup(self) -> dict[int, float]:
        """
        Build a lookup table mapping micro-action IDs to their timestamps.
        
        Returns:
            Dict mapping micro_action_id -> timestamp_seconds
        """
        return {ma.id: ma.timestamp_seconds for ma in self.get_all_micro_actions() if ma.timestamp_seconds is not None}
    
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


def compute_timestamps_from_micro_actions(
    recipe: "Recipe",
    scene_log: "SceneLog"
) -> "Recipe":
    """
    Post-process a recipe to compute start/end timestamps from micro_action_ids.
    
    This function derives video clip timing from timestamp-based micro-actions,
    which is more accurate than frame-based mapping.
    
    Args:
        recipe: Recipe with steps containing micro_action_ids
        scene_log: SceneLog containing all micro-actions with timestamps
        
    Returns:
        Recipe with start_timestamp_seconds and end_timestamp_seconds populated
    """
    # Build lookup tables
    id_to_timestamp = scene_log.build_timestamp_lookup()
    id_to_description = scene_log.build_micro_action_description_lookup()
    
    # Get all micro-actions for end timestamp lookup
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
                    if ma_id in all_actions and all_actions[ma_id].end_timestamp:
                        timestamps.append(all_actions[ma_id].end_timestamp)
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

