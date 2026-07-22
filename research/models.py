from dataclasses import dataclass, field


@dataclass
class Product:
    title: str
    source: str = ""
    description: str = ""
    price: float = 0.0
    currency: str = "USD"
    review_count: int = 0
    rating: float = 0.0
    tags: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    shop_name: str = ""
    product_url: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "title": self.title,
            "description": self.description,
            "price": self.price,
            "currency": self.currency,
            "review_count": self.review_count,
            "rating": self.rating,
            "tags": self.tags,
            "shop_name": self.shop_name,
            "product_url": self.product_url,
        }
