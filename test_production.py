import json
from pathlib import Path

from agent.production_orchestrator import ProductionRequest, run_production

if __name__ == "__main__":
    print("=== Etsy AI Agent — Production Run ===\n")

    query = input("Research query: ").strip()
    if not query:
        raise SystemExit("Query cannot be empty.")

    size_raw = input("Collection size [3–8, Enter for 3]: ").strip()
    size = int(size_raw) if size_raw.isdigit() and 3 <= int(size_raw) <= 8 else 3

    idx_raw = input("Concept index to select [Enter for auto]: ").strip()
    concept_idx = int(idx_raw) if idx_raw.isdigit() and int(idx_raw) >= 1 else None

    retries_raw = input("Max image retries [0–3, Enter for 1]: ").strip()
    max_retries = int(retries_raw) if retries_raw.isdigit() and 0 <= int(retries_raw) <= 3 else 1

    skip_mockups_raw = input("Skip mockup generation? [y/N]: ").strip().lower()
    skip_mockups = skip_mockups_raw == "y"

    skip_listing_raw = input("Skip listing generation? [y/N]: ").strip().lower()
    skip_listing = skip_listing_raw == "y"

    request = ProductionRequest(
        query=query,
        collection_size=size,
        output_root="outputs",
        selected_concept_index=concept_idx,
        max_image_retries=max_retries,
        skip_mockups=skip_mockups,
        skip_listing=skip_listing,
    )

    print()
    result = run_production(request)
    manifest = result.manifest
    out = Path(manifest.output_directory)

    # Count total image attempts across all posters
    total_attempts = 0
    if result.collection_plan:
        for poster in result.collection_plan.poster_items:
            attempts_file = out / "images" / f"poster_{poster.index:02d}" / "attempts.json"
            if attempts_file.exists():
                total_attempts += len(json.loads(attempts_file.read_text()))

    print()
    print("=" * 60)
    print("PRODUCTION SUMMARY")
    print("=" * 60)
    print(f"  Production ID:    {manifest.production_id}")
    print(f"  Status:           {manifest.status}")
    print(f"  Output directory: {manifest.output_directory}")
    if result.selected_concept:
        print(f"  Selected concept: {result.selected_concept.get('name', '—')}")
    if result.collection_plan:
        print(f"  Posters:          {result.collection_plan.collection_size}")
    print(f"  Image attempts:   {total_attempts}")
    if result.listing_plan and hasattr(result.listing_plan, "listing_title"):
        print(f"  Listing title:    {result.listing_plan.listing_title}")
    print("=" * 60)
