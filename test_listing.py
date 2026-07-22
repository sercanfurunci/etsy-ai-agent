import textwrap
from research.mock_provider import MockResearchProvider
from agent.analyzer import analyze
from agent.prompt_optimizer import optimize
from agent.collection_generator import generate_collection
from agent.mockup_generator import generate_mockup_plan
from agent.listing_generator import generate_listing_plan

def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(str(text), width=100, initial_indent=prefix, subsequent_indent=prefix)

def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def _sub(title: str) -> None:
    print(f"\n  {'─' * 56}")
    print(f"  {title}")
    print(f"  {'─' * 56}")

if __name__ == "__main__":
    # ── Research & Analysis ────────────────────────────────────────
    products = MockResearchProvider().search("wall art printable poster etsy", limit=5)
    print(f"Analyzing {len(products)} mock products with Claude...\n")
    result = analyze([p.to_dict() for p in products])
    concepts = result.get("poster_concepts", [])

    print(f"Found {len(concepts)} poster concepts:\n")
    for i, c in enumerate(concepts, 1):
        print(f"  [{i}] {c.get('name', '—')}  [{c.get('single_or_set', '—')}]")
        print(f"      {c.get('niche', '—')} · {c.get('art_style', '—')}")

    print()
    raw = input(f"Select concept [1–{len(concepts)}]: ").strip()
    try:
        concept = concepts[int(raw) - 1]
    except (ValueError, IndexError):
        raise SystemExit("Invalid selection.")

    # ── Prompt Optimizer ───────────────────────────────────────────
    print(f"\nOptimizing prompt for: {concept.get('name', '—')}\n")
    try:
        opt = optimize(
            concept,
            concept.get("image_generation_prompt", ""),
            concept.get("negative_prompt", ""),
        )
    except Exception as e:
        raise SystemExit(f"[Optimizer error] {e}")
    print("[OK] Prompt optimized.\n")

    # ── Collection Generator ───────────────────────────────────────
    size_raw = input("Collection size [3–8, Enter for 3]: ").strip()
    size = int(size_raw) if size_raw.isdigit() and 3 <= int(size_raw) <= 8 else 3

    print(f"\nGenerating collection (size={size})...\n")
    try:
        collection_plan = generate_collection(
            concept,
            opt["optimized_image_prompt"],
            opt["optimized_negative_prompt"],
            collection_size=size,
        )
    except RuntimeError as e:
        raise SystemExit(f"[JSON error] {e}")
    except ValueError as e:
        raise SystemExit(f"[Validation error] {e}")
    except Exception as e:
        raise SystemExit(f"[Collection generator error] {e}")

    bible = collection_plan.collection_bible
    print(f"[OK] Collection '{bible.collection_name}' — {collection_plan.collection_size} posters.\n")

    # ── Mockup Generator ───────────────────────────────────────────
    print("Generating mockup plan...\n")
    try:
        mockup_plan = generate_mockup_plan(collection_plan)
    except RuntimeError as e:
        raise SystemExit(f"[JSON error] {e}")
    except ValueError as e:
        raise SystemExit(f"[Validation error] {e}")
    except Exception as e:
        raise SystemExit(f"[Mockup generator error] {e}")

    print(f"[OK] Mockup plan ready.\n")

    # ── Listing Generator ──────────────────────────────────────────
    print("Generating listing package...\n")
    try:
        listing = generate_listing_plan(collection_plan, mockup_plan)
    except RuntimeError as e:
        raise SystemExit(f"[JSON error] {e}")
    except ValueError as e:
        raise SystemExit(f"[Validation error] {e}")
    except Exception as e:
        raise SystemExit(f"[Listing generator error] {e}")

    # ── LISTING OVERVIEW ───────────────────────────────────────────
    _section("LISTING OVERVIEW")
    print(f"\n  Shop section:      {listing.shop_section}")
    print(f"  Listing type:      {listing.listing_type}")
    print(f"  Primary category:  {listing.primary_category}")
    print(f"  Secondary category:{listing.secondary_category}")
    print(f"  Target customer:\n{_wrap(listing.target_customer)}")

    # ── TITLES ────────────────────────────────────────────────────
    _section("TITLES")
    print(f"\n  Listing title ({len(listing.listing_title)} chars):")
    print(f"    {listing.listing_title}")
    print(f"\n  Short title ({len(listing.short_title)} chars):")
    print(f"    {listing.short_title}")

    # ── DESCRIPTION ───────────────────────────────────────────────
    _section("DESCRIPTION")
    # Print description preserving section headings with line breaks
    for line in listing.listing_description.splitlines():
        stripped = line.strip()
        if not stripped:
            print()
        elif stripped.startswith("✦"):
            print(f"\n  {stripped}")
        else:
            print(_wrap(stripped, indent=4))

    # ── BULLET FEATURES ───────────────────────────────────────────
    _section(f"BULLET FEATURES  ({len(listing.bullet_features)})")
    for b in listing.bullet_features:
        print(f"    • {b}")

    # ── ETSY TAGS ─────────────────────────────────────────────────
    _section(f"ETSY TAGS  ({len(listing.etsy_tags)}/13)")
    for i, tag in enumerate(listing.etsy_tags, 1):
        print(f"    [{i:02d}] {tag}  ({len(tag)} chars)")

    # ── SEO KEYWORDS ──────────────────────────────────────────────
    _section(f"SEO KEYWORDS  ({len(listing.seo_keywords)})")
    # Print in 2-column layout
    kws = listing.seo_keywords
    mid = (len(kws) + 1) // 2
    left, right = kws[:mid], kws[mid:]
    for l, r in zip(left, right + [""]):
        print(f"    {l:<38}  {r}")
    if len(left) > len(right):
        pass  # already handled by zip padding

    # ── MATERIALS ─────────────────────────────────────────────────
    _section("MATERIALS")
    for m in listing.materials:
        print(f"    • {m}")

    # ── IMAGE ORDER ───────────────────────────────────────────────
    _section(f"IMAGE ORDER  ({len(listing.image_order)} images)")
    for img in sorted(listing.image_order, key=lambda x: x.position):
        print(f"    [{img.position:02d}] {img.description}")
        print(f"         source: {img.source}")

    # ── DOWNLOAD PACKAGE ──────────────────────────────────────────
    _section("DOWNLOAD PACKAGE")
    dp = listing.download_package
    print(f"\n  ZIP name:          {dp.zip_name}")
    print(f"  Total files:       {dp.total_file_count}")
    print(f"  Formats:           {', '.join(dp.formats_included)}")
    print(f"  Includes guide:    {'yes' if dp.includes_print_guide else 'no'}")
    print(f"  Includes license:  {'yes' if dp.includes_license else 'no'}")
    print(f"  Includes readme:   {'yes' if dp.includes_readme else 'no'}")
    print(f"\n  Size variants:")
    for sv in dp.size_variants:
        print(f"    • {sv}")
    print(f"\n  File list:")
    for f in dp.file_list:
        print(f"    {f}")

    # ── CUSTOMER NOTES ────────────────────────────────────────────
    _section("CUSTOMER NOTES")
    for n in listing.customer_notes:
        print(f"    • {n}")

    # ── FAQ ───────────────────────────────────────────────────────
    _section(f"FAQ  ({len(listing.faq)} questions)")
    for i, faq in enumerate(listing.faq, 1):
        _sub(f"Q{i}: {faq.question}")
        print(_wrap(faq.answer, indent=4))

    # ── EVALUATION ────────────────────────────────────────────────
    _section("LISTING EVALUATION")

    def _score(label: str, s) -> None:
        print(f"\n  {label}: {s.score}/10")
        print(_wrap(s.reason, indent=4))

    _score("SEO",                  listing.evaluation.seo_score)
    _score("Commercial appeal",    listing.evaluation.commercial_appeal_score)
    _score("Customer clarity",     listing.evaluation.customer_clarity_score)
    _score("Conversion potential", listing.evaluation.conversion_potential_score)
    _score("Brand consistency",    listing.evaluation.brand_consistency_score)
    _score("Professionalism",      listing.evaluation.professionalism_score)
    _score("Overall (computed)",   listing.evaluation.overall_score)
    print(f"\n  Reasoning:\n{_wrap(listing.evaluation.reasoning)}")
    print(f"\n  Confidence: {listing.confidence_score}/10")

    # ── SUMMARY ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  [OK] Listing package validated.")
    print(f"       Title: {listing.listing_title[:70]}{'...' if len(listing.listing_title) > 70 else ''}")
    print(f"       Tags: {len(listing.etsy_tags)}/13  |  SEO keywords: {len(listing.seo_keywords)}"
          f"  |  Images: {len(listing.image_order)}  |  FAQs: {len(listing.faq)}")
    print(f"       Overall score: {listing.evaluation.overall_score.score}/10")
    print(f"{'=' * 60}\n")
