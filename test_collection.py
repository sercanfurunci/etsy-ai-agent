import textwrap
from research.mock_provider import MockResearchProvider
from agent.analyzer import analyze
from agent.prompt_optimizer import optimize
from agent.collection_generator import generate_collection

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

    final_prompt = opt["optimized_image_prompt"]
    final_negative = opt["optimized_negative_prompt"]
    print(f"[OK] Prompt optimized.\n")

    # ── Collection Generator ───────────────────────────────────────
    size_raw = input("Collection size [3–8, Enter for auto]: ").strip()
    size = int(size_raw) if size_raw.isdigit() else None

    print(f"\nGenerating collection (size={size or 'auto'})...\n")
    try:
        plan = generate_collection(concept, final_prompt, final_negative, collection_size=size)
    except RuntimeError as e:
        raise SystemExit(f"[JSON error] {e}")
    except ValueError as e:
        raise SystemExit(f"[Validation error] {e}")
    except Exception as e:
        raise SystemExit(f"[Collection generator error] {e}")

    bible = plan.collection_bible
    ev = plan.evaluation

    # ── Print Collection Bible ─────────────────────────────────────
    _section("COLLECTION BIBLE")
    print(f"\n  Name:            {bible.collection_name}")
    print(f"  Target customer: {bible.target_customer}")
    print(f"  Room style:      {bible.recommended_room_style}")
    print(f"\n  Story:\n{_wrap(bible.collection_story)}")
    print(f"\n  Brand identity:\n{_wrap(bible.brand_identity)}")
    print(f"\n  Visual identity:\n{_wrap(bible.visual_identity)}")

    print(f"\n  Rendering medium: {bible.shared_rendering_medium}")
    print(f"  Linework:         {bible.shared_linework}")
    print(f"  Lighting:         {bible.shared_lighting}")
    print(f"  Camera angle:     {bible.shared_camera_angle}")
    print(f"  Perspective:      {bible.shared_perspective}")
    print(f"  Atmosphere:       {bible.shared_atmosphere}")
    print(f"  Detail level:     {bible.shared_detail_level}")
    print(f"  Print treatment:  {bible.shared_print_treatment}")

    print(f"\n  Shared palette:  {', '.join(bible.shared_palette)}")
    if bible.shared_accent_colour_rules:
        print(f"  Accent rules:")
        for r in bible.shared_accent_colour_rules:
            print(f"    • {r}")

    print(f"\n  Style DNA:")
    for s in bible.style_dna:
        print(f"    • {s}")

    print(f"\n  Composition rules:")
    for r in bible.shared_composition_rules:
        print(f"    • {r}")

    print(f"\n  Storytelling rules:")
    for r in bible.shared_storytelling_rules:
        print(f"    • {r}")

    print(f"\n  Style rules:")
    for r in bible.shared_style_rules:
        print(f"    • {r}")

    print(f"\n  Consistency rules:")
    for r in bible.consistency_rules:
        print(f"    • {r}")

    print(f"\n  Full-bleed rules:")
    for r in bible.full_bleed_rules:
        print(f"    • {r}")

    print(f"\n  Forbidden elements:")
    for r in bible.forbidden_elements:
        print(f"    ✗ {r}")

    print(f"\n  Shared negative prompt:\n{_wrap(bible.shared_negative_prompt)}")

    # ── Print Collection Evaluation ────────────────────────────────
    _section("COLLECTION EVALUATION")

    def _score(label: str, s) -> None:
        print(f"\n  {label}: {s.score}/10")
        print(_wrap(s.reason, indent=4))

    _score("Consistency",       ev.consistency_score)
    _score("Commercial appeal", ev.commercial_score)
    _score("Variation",         ev.variation_score)
    _score("Brand identity",    ev.brand_identity_score)
    _score("Print collection",  ev.print_collection_score)
    _score("Market uniqueness", ev.market_uniqueness_score)
    _score("Overall (computed)",ev.overall_score)
    print(f"\n  Reasoning:\n{_wrap(ev.reasoning)}")
    print(f"\n  Confidence: {plan.confidence_score}/10")

    if plan.collection_consistency_notes:
        print(f"\n  Consistency notes:")
        for n in plan.collection_consistency_notes:
            print(f"    • {n}")

    # ── Print Posters ──────────────────────────────────────────────
    _section(f"POSTER COLLECTION — {bible.collection_name}  ({plan.collection_size} posters)")

    for p in plan.poster_items:
        _sub(f"[{p.index}/{plan.collection_size}] {p.title}")

        print(f"\n  Subject:              {p.subject}")
        print(f"  Aspect ratio:         {p.aspect_ratio}")
        print(f"  Unique hook:          {p.unique_hook}")
        print(f"  Focal point:          {p.focal_point}")
        print(f"  Lighting:             {p.lighting_variation}")
        print(f"  Weather / time:       {p.weather_or_time_variation}")
        print(f"  Mockup room:          {p.mockup_room_style}")
        print(f"  Palette variation:    {', '.join(p.palette_variation)}")

        print(f"\n  Scene concept:\n{_wrap(p.scene_concept)}")
        print(f"\n  Storytelling focus:\n{_wrap(p.storytelling_focus)}")

        print(f"\n  Foreground: {', '.join(p.foreground_elements)}")
        print(f"  Midground:  {', '.join(p.midground_elements)}")
        print(f"  Background: {', '.join(p.background_elements)}")

        print(f"\n  Image prompt:\n{_wrap(p.image_prompt, indent=4)}")
        print(f"\n  Negative prompt:\n{_wrap(p.negative_prompt, indent=4)}")

        print(f"\n  Etsy title: {p.suggested_etsy_title}")
        print(f"  Etsy tags ({len(p.suggested_etsy_tags)}): {', '.join(p.suggested_etsy_tags)}")

        print(f"\n  Consistency notes:")
        for n in p.consistency_notes:
            print(f"    • {n}")

    print(f"\n{'=' * 60}")
    print(f"  [OK] Collection validated — {plan.collection_size} posters, all checks passed.")
    print(f"{'=' * 60}\n")
