# ponytail: temporary e2e test — superseded once main.py drives the full pipeline
import textwrap
from agent.pipeline import run_pipeline

def _wrap(label: str, text: str, indent: int = 6) -> str:
    prefix = " " * indent
    wrapped = textwrap.fill(text, width=100, initial_indent=prefix, subsequent_indent=prefix)
    return f"{' ' * (indent - 2)}{label}:\n{wrapped}"

if __name__ == "__main__":
    query = input("Research query: ").strip()
    if not query:
        raise SystemExit("No query provided.")

    print(f"\nFetching web results for '{query}'...")
    try:
        result = run_pipeline(query, limit=10)
    except ValueError as e:
        raise SystemExit(f"[Error] {e}")
    except Exception as e:
        raise SystemExit(f"[API error] {e}")

    print(f"\nNiche: {result['niche']}\n")

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
        print(f"      Ratio:      {c.get('aspect_ratio', '—')}")
        print(f"      Etsy title: {c.get('suggested_etsy_title', '—')}")
        print(f"      Tags: {', '.join(c.get('suggested_etsy_tags', []))}")
        print(_wrap("Image prompt", c.get("image_generation_prompt", "—")))
        print(_wrap("Negative prompt", c.get("negative_prompt", "—")))
        print(_wrap("Set notes", c.get("set_consistency_notes", "—")))
