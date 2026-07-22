# ponytail: temporary test — remove once web provider is wired into main pipeline
from research.web_provider import WebResearchProvider
from research.models import Product

if __name__ == "__main__":
    provider = WebResearchProvider()
    query = "handmade ceramic mug Etsy"
    print(f"Searching: '{query}'\n")

    products = provider.search(query, limit=5)

    assert len(products) > 0, "no results returned"
    assert all(isinstance(p, Product) for p in products), "non-Product in results"
    assert all(p.title for p in products), "product missing title"
    assert all(p.product_url for p in products), "product missing url"

    for i, p in enumerate(products, 1):
        print(f"[{i}] {p.title}")
        print(f"     source:  {p.source}")
        print(f"     url:     {p.product_url}")
        print(f"     snippet: {p.description[:120]}...")
        print()

    print(f"[OK] {len(products)} normalized Product objects returned.")
