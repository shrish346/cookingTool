from pydantic import BaseModel, Field, computed_field
from typing import Optional

class Ingredient(BaseModel):
    name: str
    quantity: float = Field(gt=0)
    unit: str
    preparation: Optional[str] = None

class Step(BaseModel):
    order: int = Field(ge=1)
    instruction: str
    duration_minutes: Optional[int] = None
    tips: Optional[list[str]] = None

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

