import textwrap
from research.mock_provider import MockResearchProvider
from agent.analyzer import analyze
from agent.prompt_optimizer import optimize
from agent.collection_generator import generate_collection
from agent.mockup_generator import generate_mockup_plan

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
    print("[OK] Prompt optimized.\n")

    # ── Collection Generator ───────────────────────────────────────
    size_raw = input("Collection size [3–8, Enter for 3]: ").strip()
    size = int(size_raw) if size_raw.isdigit() and 3 <= int(size_raw) <= 8 else 3

    print(f"\nGenerating collection (size={size})...\n")
    try:
        collection_plan = generate_collection(concept, final_prompt, final_negative, collection_size=size)
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

    # ── MOCKUP PLAN ────────────────────────────────────────────────
    _section(f"MOCKUP PLAN — {mockup_plan.collection_name}")

    print(f"\n  Rendering mode:  {mockup_plan.rendering_mode}")
    print(f"\n  Shared mockup rules:")
    for r in mockup_plan.shared_mockup_rules:
        print(f"    • {r}")

    print(f"\n  Forbidden mockup elements:")
    for r in mockup_plan.forbidden_mockup_elements:
        print(f"    ✗ {r}")

    # ── INDIVIDUAL MOCKUPS ─────────────────────────────────────────
    _section(f"INDIVIDUAL MOCKUPS  ({len(mockup_plan.individual_mockups)} total)")

    for m in mockup_plan.individual_mockups:
        _sub(f"[{m.poster_index}/{collection_plan.collection_size}] {m.mockup_name}")
        print(f"\n  Poster:           {m.poster_title}")
        print(f"  Room type:        {m.room_type}")
        print(f"  Room style:       {m.room_style}")
        print(f"  Wall colour:      {m.wall_colour}")
        print(f"  Frame style:      {m.frame_style}")
        print(f"  Frame colour:     {m.frame_colour}")
        print(f"  Frame orient.:    {m.frame_orientation}")
        print(f"  Artwork ratio:    {m.artwork_aspect_ratio}")
        print(f"  Artwork scale:    {m.artwork_scale}")
        print(f"  Camera angle:     {m.camera_angle}")
        print(f"  Camera distance:  {m.camera_distance}")
        print(f"  Lighting:         {m.lighting}")
        print(f"  Furniture:        {', '.join(m.furniture_elements)}")
        print(f"  Décor:            {', '.join(m.decor_elements)}")
        print(f"\n  Placement rules:")
        for r in m.placement_rules:
            print(f"    • {r}")
        print(f"\n  Image prompt:\n{_wrap(m.image_prompt)}")
        print(f"\n  Negative prompt:\n{_wrap(m.negative_prompt)}")
        print(f"\n  Commercial purpose:\n{_wrap(m.commercial_purpose)}")

    # ── COLLECTION GALLERY MOCKUP ──────────────────────────────────
    _section("COLLECTION GALLERY MOCKUP")

    cm = mockup_plan.collection_mockup
    if cm is None:
        print("\n  [skipped]")
    else:
        print(f"\n  Name:           {cm.mockup_name}")
        print(f"  Room type:      {cm.room_type}")
        print(f"  Room style:     {cm.room_style}")
        print(f"  Wall colour:    {cm.wall_colour}")
        print(f"  Gallery layout: {cm.gallery_layout}")
        print(f"  Poster order:   {cm.poster_order}")
        print(f"  Frame style:    {cm.frame_style}")
        print(f"  Frame colour:   {cm.frame_colour}")
        print(f"  Artwork scale:  {cm.artwork_scale}")
        print(f"  Camera angle:   {cm.camera_angle}")
        print(f"  Lighting:       {cm.lighting}")
        print(f"  Furniture:      {', '.join(cm.furniture_elements)}")
        print(f"  Décor:          {', '.join(cm.decor_elements)}")
        print(f"\n  Spacing rules:")
        for r in cm.spacing_rules:
            print(f"    • {r}")
        print(f"\n  Image prompt:\n{_wrap(cm.image_prompt)}")
        print(f"\n  Negative prompt:\n{_wrap(cm.negative_prompt)}")
        print(f"\n  Commercial purpose:\n{_wrap(cm.commercial_purpose)}")

    # ── ETSY HERO MOCKUP ──────────────────────────────────────────
    _section("ETSY HERO MOCKUP")

    hm = mockup_plan.hero_mockup
    if hm is None:
        print("\n  [skipped]")
    else:
        print(f"\n  Name:             {hm.mockup_name}")
        print(f"  Hero strategy:\n{_wrap(hm.hero_strategy)}")
        print(f"\n  Primary poster:   #{hm.primary_poster_index}")
        print(f"  Secondary:        {hm.secondary_poster_indexes}")
        print(f"  Layout:\n{_wrap(hm.layout_description)}")
        print(f"\n  Background style: {hm.background_style}")
        print(f"  Wall colour:      {hm.wall_colour}")
        print(f"  Frame style:      {hm.frame_style}")
        print(f"  Camera angle:     {hm.camera_angle}")
        print(f"  Lighting:         {hm.lighting}")
        print(f"\n  Thumbnail readability rules:")
        for r in hm.thumbnail_readability_rules:
            print(f"    • {r}")
        print(f"\n  Image prompt:\n{_wrap(hm.image_prompt)}")
        print(f"\n  Negative prompt:\n{_wrap(hm.negative_prompt)}")
        print(f"\n  Commercial purpose:\n{_wrap(hm.commercial_purpose)}")

    # ── MOCKUP EVALUATION ─────────────────────────────────────────
    _section("MOCKUP EVALUATION")

    ev = mockup_plan.evaluation

    def _score(label: str, s) -> None:
        print(f"\n  {label}: {s.score}/10")
        print(_wrap(s.reason, indent=4))

    _score("Artwork visibility",      ev.artwork_visibility_score)
    _score("Commercial clarity",      ev.commercial_clarity_score)
    _score("Room fit",                ev.room_fit_score)
    _score("Collection presentation", ev.collection_presentation_score)
    _score("Thumbnail readability",   ev.thumbnail_readability_score)
    _score("Brand alignment",         ev.brand_alignment_score)
    _score("Overall (computed)",      ev.overall_score)
    print(f"\n  Reasoning:\n{_wrap(ev.reasoning)}")
    print(f"\n  Confidence: {mockup_plan.confidence_score}/10")

    print(f"\n{'=' * 60}")
    print(f"  [OK] Mockup plan validated — {len(mockup_plan.individual_mockups)} individual mockups, all checks passed.")
    print(f"{'=' * 60}\n")
