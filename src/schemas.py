from pydantic import BaseModel, Field, computed_field
from typing import Optional, Literal

class Ingredient(BaseModel):
    name: str
    quantity: float = Field(gt=0)
    unit: str
    preparation: Optional[str] = None

class Step(BaseModel):
    order: int = Field(ge=1)
    title: str = Field(description="Short title for this step (e.g., 'Boil the Pasta')")
    instruction: str
    duration_minutes: Optional[int] = None
    tips: Optional[list[str]] = None
    # Video loop fields for the Cooking Mode
    start_frame_index: Optional[int] = Field(default=None, ge=0, description="Frame index where this step starts in the video")
    end_frame_index: Optional[int] = Field(default=None, ge=0, description="Frame index where this step ends in the video")
    video_clip_url: Optional[str] = Field(default=None, description="S3 URL for the extracted video clip")

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

