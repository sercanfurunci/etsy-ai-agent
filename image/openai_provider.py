import base64
import os
from pathlib import Path
from openai import OpenAI
from agent.config import OPENAI_API_KEY
from image.base import ImageProvider

MODEL = "gpt-image-2"  # change here to switch models
OUTPUT_DIR = Path("output")


class OpenAIImageProvider(ImageProvider):
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        self._client = OpenAI(api_key=OPENAI_API_KEY)

    def generate(self, prompt: str) -> str:
        OUTPUT_DIR.mkdir(exist_ok=True)

        response = self._client.images.generate(
            model=MODEL,
            prompt=prompt,
            n=1,
            size="1088x1920",
        )

        image_data = base64.b64decode(response.data[0].b64_json)
        # unique filename from first 40 chars of prompt
        slug = "".join(c if c.isalnum() else "_" for c in prompt[:40]).strip("_")
        file_path = OUTPUT_DIR / f"{slug}.png"
        file_path.write_bytes(image_data)
        return str(file_path)
