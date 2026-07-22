from urllib.parse import urlparse
from ddgs import DDGS
from research.base import ResearchProvider
from research.models import Product


class WebResearchProvider(ResearchProvider):
    def search(self, query: str, limit: int = 20) -> list[Product]:
        results = DDGS().text(query, max_results=limit)
        products = []
        for r in results:
            url = r.get("href", "")
            domain = urlparse(url).netloc.removeprefix("www.")
            products.append(Product(
                source=domain,
                title=r.get("title", ""),
                description=r.get("body", ""),
                product_url=url,
            ))
        return products
