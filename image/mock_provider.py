import hashlib
from image.base import ImageProvider


class MockImageProvider(ImageProvider):
    def generate(self, prompt: str) -> str:
        slug = hashlib.md5(prompt.encode()).hexdigest()[:10]
        return f"output/mock_image_{slug}.png"
