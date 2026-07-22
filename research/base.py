from abc import ABC, abstractmethod
from research.models import Product


class ResearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, limit: int = 20) -> list[Product]:
        ...
