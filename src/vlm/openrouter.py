import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

from ..downloaders.base import VideoInfo
from ..schemas import Recipe, Ingredient, Step

load_dotenv()


class OpenRouterAdapter:
    """Adapter for OpenRouter's vision-language models."""

    def __init__(self, model: str = "qwen/qwen2.5-vl-72b-instruct"):
        self._model = model
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model

    def analyze_recipe(self, video_info: VideoInfo, frames: list[str]) -> Recipe:
        """
        Analyze video frames and return a structured Recipe.
        
        Args:
            video_info: Metadata about the video (title, description, etc.)
            frames: List of base64-encoded JPEG images
            
        Returns:
            A validated Recipe object
        """
        messages = self._build_messages(video_info, frames)
        
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=4096,
            temperature=0.3,  # Lower = more deterministic
        )
        
        content = response.choices[0].message.content
        return self._parse_response(content, video_info)

    def _build_prompt(self, video_info: VideoInfo) -> str:
        """Create the instruction prompt for the VLM."""
        return f"""You are analyzing frames from a cooking video to extract a recipe.

Video Title: {video_info.title}
Video Description: {video_info.description or "Not provided"}

Analyze these frames carefully and extract the complete recipe. Return your response as a JSON object with this exact structure:

{{
    "title": "Recipe name",
    "description": "Brief description of the dish",
    "servings": 4,
    "prep_time_minutes": 15,
    "cook_time_minutes": 30,
    "cusine": "Italian",
    "tags": ["dinner", "pasta", "vegetarian"],
    "ingredients": [
        {{"name": "ingredient name", "quantity": 2.0, "unit": "cups", "preparation": "diced"}}
    ],
    "steps": [
        {{"order": 1, "instruction": "Step description", "duration_minutes": 5, "tips": ["optional tip"]}}
    ],
    "calories": 450,
    "protein": 25,
    "carbs": 50,
    "fats": 15
}}

Rules:
- quantity must be a positive number (use decimals like 0.5 for "half")
- order must start at 1 and increment
- Include ALL ingredients and steps you can identify from the video
- If you can't determine a value, omit that field (don't guess wildly)
- Return ONLY the JSON object, no other text"""

    def _build_messages(self, video_info: VideoInfo, frames: list[str]) -> list[dict]:
        """
        Build the multi-modal message array for the API.
        
        Combines the instruction prompt with all video frames.
        """
        # Start with the text instruction
        content = [
            {"type": "text", "text": self._build_prompt(video_info)}
        ]
        
        # Add each frame as an image
        for frame_b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}"
                }
            })
        
        return [{"role": "user", "content": content}]

    def _parse_response(self, content: str, video_info: VideoInfo) -> Recipe:
        """
        Extract JSON from the LLM response and convert to Recipe.
        
        LLMs sometimes wrap JSON in markdown code blocks or add extra text.
        This method handles those cases.
        """
        # Try to extract JSON from markdown code blocks first
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Maybe it's raw JSON - find the outermost braces
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                json_str = content[start:end]
            else:
                raise ValueError(f"Could not find JSON in response: {content[:200]}...")
        
        # Parse the JSON
        data = json.loads(json_str)
        
        # Add source URL from video info
        data["source_url"] = video_info.url
        
        # Convert to Recipe (Pydantic handles validation)
        return Recipe(**data)

