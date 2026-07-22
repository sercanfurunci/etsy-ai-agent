from abc import ABC, abstractmethod


class ImageProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return a URL or local file path for the generated image."""
        ...
