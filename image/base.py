from abc import ABC, abstractmethod


class ImageProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, negative_prompt: str = "", on_usage=None) -> str:
        """Return a URL or local file path for the generated image."""
        ...
