# ponytail: temporary test script — remove once pipeline is wired end-to-end
import textwrap
from research.mock_provider import MockResearchProvider
from research.models import Product
from agent.analyzer import analyze

def _wrap(label: str, text: str, indent: int = 6) -> str:
    prefix = " " * indent
    wrapped = textwrap.fill(text, width=100, initial_indent=prefix, subsequent_indent=prefix)
    return f"{' ' * (indent - 2)}{label}:\n{wrapped}"

if __name__ == "__main__":
    provider = MockResearchProvider()

    all_products = provider.search("wall art printable poster etsy")
    assert len(all_products) > 0, "search returned no products"
    assert all(isinstance(p, Product) for p in all_products), "non-Product returned"
    limited = provider.search("wall art", limit=2)
    assert len(limited) == 2, f"limit not respected: got {len(limited)}"
    print(f"[OK] Provider: {len(all_products)} products, limit=2 → {len(limited)}\n")

    products = provider.search("wall art printable poster etsy", limit=5)
    print(f"Sending {len(products)} products to Claude...\n")

    try:
        result = analyze([p.to_dict() for p in products])
    except Exception as e:
        print(f"[Error] {e}")
        raise SystemExit(1)

    print(f"Niche: {result['niche']}\n")

    print("Market observations:")
    for obs in result.get("market_observations", []):
        print(f"  • {obs}")

    print("\nRecurring patterns:")
    for p in result.get("recurring_patterns", []):
        print(f"  • {p}")

    print("\nOpportunities:")
    for opp in result.get("potential_opportunities", []):
        print(f"  • {opp}")

    concepts = result.get("poster_concepts", [])
    print(f"\nPoster concepts ({len(concepts)}):")
    for i, c in enumerate(concepts, 1):
        print(f"\n  [{i}] {c.get('name', '—')}  [{c.get('single_or_set', '—')}]")
        print(f"      Niche:      {c.get('niche', '—')}")
        print(f"      Style:      {c.get('art_style', '—')}")
        print(f"      Subject:    {c.get('subject', '—')}")
        print(f"      Palette:    {c.get('color_palette', '—')}")
        print(f"      Ratio:      {c.get('aspect_ratio', '—')}")
        print(f"      Etsy title: {c.get('suggested_etsy_title', '—')}")
        print(f"      Tags: {', '.join(c.get('suggested_etsy_tags', []))}")
        print(_wrap("Image prompt", c.get("image_generation_prompt", "—")))
        print(_wrap("Negative prompt", c.get("negative_prompt", "—")))
        print(_wrap("Set notes", c.get("set_consistency_notes", "—")))

    print("\n[OK] Validation passed — all required fields present.")
