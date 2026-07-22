import json
import re
from dataclasses import dataclass, field
import anthropic
from agent.claude_client import ANTHROPIC_API_KEY, MODEL
from agent.vision_critic import ScoreWithReason
from agent.collection_generator import CollectionPlan

_MAX_TOKENS = 8192
_DEFAULT_RENDERING_MODE = "compositing"

# ── Validation term sets ───────────────────────────────────────────────────────
# At least 2 of these must appear in every image prompt (immutable-asset intent)
_IMMUTABLE_TERMS = ["immutable", "source asset", "original pixel", "unchanged"]
# At least 1 must appear (no-crop intent)
_NO_CROP_TERMS = ["no crop", "no cropping", "contain-fit", "contain fit"]
# At least 1 must appear (no-stretch intent)
_NO_STRETCH_TERMS = ["no stretch", "no stretching", "aspect ratio"]
# At least 1 must appear in hero prompt (equal physical frame size intent)
_HERO_EQUAL_SIZE_TERMS = ["equal size", "same size", "uniform size", "identical size",
                           "equal-size", "same-size", "uniform frame"]

# ── Shared prompt blocks ───────────────────────────────────────────────────────

_IMMUTABLE_BLOCK = """\
IMMUTABLE SOURCE ASSET RULES — every image_prompt you write must include all of these concepts
(use natural wording, do not copy mechanically):
  • Treat each supplied poster as an immutable source asset
  • Insert it as a flat rectangular image layer inside the assigned frame
  • Preserve every original pixel — do not redraw, repaint, regenerate, reinterpret, extend,
    inpaint, recolour, sharpen, stylise, relight, or add elements inside the artwork area
  • Use contain-fit inside the frame — no cropping, no stretching
  • Preserve the exact aspect ratio of the supplied artwork
  • Perspective transformation may affect only the rectangular frame plane, not the internal artwork
  • Do not apply generative fill inside the poster area
"""

_ARTWORK_REFERENCE_RULES = """\
ARTWORK REFERENCE RULES — mandatory, no exceptions:
  • Refer to each artwork ONLY by its poster index and title: "Poster 1 — [Title]"
  • Do NOT describe characters, scenes, colour palette, rendering style, or any visual content
    inside the supplied artwork
  • Do NOT ask the image model to recreate or imitate the poster style inside the frames
  • Use wording such as: "Insert supplied artwork asset Poster 2 — Platform Waiting into the
    frame as an immutable source asset, unchanged, contain-fit, no cropping, no stretching."
"""

# ── Prompts ────────────────────────────────────────────────────────────────────

_INDIVIDUAL_PROMPT = """\
You are a commercial mockup art director for an Etsy printable wall art shop.
Rendering mode: compositing. The final production system receives real poster image assets.
Your job: create individual mockup scene plans. The actual poster artworks will be composited
into the frames by a separate system — you describe only the room scene.

Collection Bible:
{bible}

Posters in this collection ({collection_size} total):
{poster_list}

{immutable_block}

{artwork_reference_rules}

ROOM SCENE CONSTRAINTS (individual mockups):
  • Maximum 1 major furniture element (e.g. a low sideboard OR a reading chair — not both)
  • Maximum 1 lighting element (e.g. a floor lamp OR wall sconce)
  • Maximum 2 small décor elements (e.g. one plant and one vase — no more)
  • No string lights unless the collection's brand identity strongly requires them
  • No combinations such as desk + chair + shelf + record player + plant + blanket
  • The artwork must occupy at least 45% of the final image area
  • Generous visual breathing room — room context is secondary to the artwork
  • Each room tailored to the poster's mockup_room_style while matching collection brand identity

NEGATIVE PROMPT — every negative_prompt must include:
altered artwork, regenerated artwork, changed colours, cropped poster, stretched poster,
distorted frame, incorrect aspect ratio, obscured artwork, plant covering artwork, glare,
glass reflection, watermark, logos, readable wall text, extra artwork, branded products,
people blocking artwork, extreme wide-angle distortion, fisheye lens, cluttered room,
dark underexposure, overexposure, inpainted artwork area, generative fill inside frame.

Return ONLY a valid JSON object — no markdown, no commentary:
{{
  "shared_mockup_rules": ["rule applying to every mockup in this collection", "..."],
  "forbidden_mockup_elements": ["element that must never appear in any mockup", "..."],
  "individual_mockups": [
    {{
      "poster_index": integer,
      "poster_title": "string",
      "mockup_name": "string",
      "room_type": "string",
      "room_style": "string",
      "wall_colour": "string",
      "frame_style": "string",
      "frame_colour": "string",
      "frame_orientation": "portrait | landscape | square",
      "artwork_aspect_ratio": "string — from poster metadata, e.g. 2:3",
      "artwork_scale": "string — e.g. large, fills upper two-thirds of wall",
      "camera_angle": "string",
      "camera_distance": "string",
      "lighting": "string — room and frame lighting only, no artwork relighting",
      "furniture_elements": ["maximum 1 item"],
      "decor_elements": ["maximum 2 items"],
      "placement_rules": [
        "artwork centred on wall, occupies at least 45% of image area",
        "..."
      ],
      "image_prompt": "string — compositing scene: describe only the room and frame. Reference artwork only as: Insert supplied artwork asset Poster N — [Title] into the frame as an immutable source asset, original pixels unchanged, contain-fit, no cropping, no stretching.",
      "negative_prompt": "string",
      "commercial_purpose": "string"
    }}
  ]
}}
"""

_COLLECTION_HERO_PROMPT = """\
You are a commercial mockup art director for an Etsy printable wall art shop.
Rendering mode: compositing. The final production system receives real poster image assets.
Your job: create a Collection Gallery Mockup and an Etsy Hero Mockup plan.
The actual poster artworks will be composited into the frames — you describe only scene and layout.

Collection Bible:
{bible}

Posters in this collection ({collection_size} total):
{poster_list}

Shared mockup rules already established:
{shared_rules}

{immutable_block}

{artwork_reference_rules}

ASSET-SLOT MAPPING — mandatory for both collection and hero image_prompt:
  Map every included poster to a named frame position explicitly, e.g.:
    "left frame: Poster 1 — [Title]"
    "centre frame: Poster 2 — [Title]"
    "right frame: Poster 3 — [Title]"
  Do not describe the artwork content — only name the slot and the asset reference.

COLLECTION GALLERY MOCKUP RULES:
  • Every poster must be visible — no obscured or cropped frames
  • Consistent frame style
  • Artwork remains readable — no small or distant frames
  • Spacing visually balanced
  • No unrelated artwork, no readable wall text, no branded décor
  • Preferred layout for portrait posters: horizontal triptych or balanced gallery arrangement

HERO MOCKUP RULES — PHYSICAL REALISM:
  • All frames must use the EQUAL physical size — no one dominant large frame with smaller ones
  • All posters fully visible and recognisable at thumbnail size
  • Central or lead poster may receive visual emphasis through placement or lighting only,
    not through an unrealistically larger physical frame
  • Frames occupy most of the image — minimal room context
  • Must look like a believable photographed wall arrangement, not a floating digital collage
  • No text overlay
  • No Etsy badges or fake interface elements
  • No misleading physical product claims

ROOM LIGHTING RULE (both mockups):
  Room lighting may illuminate only the physical frame and wall.
  It must not relight, recolour, or modify the supplied artwork pixels.
  Do not write phrases such as "poster glows", "internal neon illuminates", or
  "lighting highlights brushwork inside artwork".

NEGATIVE PROMPT (both mockups):
altered artwork, regenerated artwork, changed colours, cropped poster, stretched poster,
distorted frame, incorrect aspect ratio, duplicate frames, missing poster, obscured artwork,
plant covering artwork, glare, glass reflection, watermark, logos, readable wall text,
extra artwork, branded products, people blocking artwork, extreme wide-angle distortion,
fisheye lens, cluttered room, dark underexposure, overexposure,
inpainted artwork area, generative fill inside frame, size-mismatched frames.

Return ONLY a valid JSON object — no markdown, no commentary:
{{
  "collection_mockup": {{
    "mockup_name": "string",
    "room_type": "string",
    "room_style": "string",
    "wall_colour": "string",
    "gallery_layout": "string",
    "poster_order": [integer, "..."],
    "spacing_rules": ["string", "..."],
    "frame_style": "string",
    "frame_colour": "string",
    "artwork_scale": "string",
    "camera_angle": "string",
    "lighting": "string — room and frame only",
    "furniture_elements": ["string", "..."],
    "decor_elements": ["string", "..."],
    "image_prompt": "string — compositing scene: describe room and gallery layout, include explicit asset-slot mapping for every poster, include immutable source asset language for each slot",
    "negative_prompt": "string",
    "commercial_purpose": "string"
  }},
  "hero_mockup": {{
    "mockup_name": "string",
    "hero_strategy": "string — conversion rationale for this hero design",
    "primary_poster_index": integer,
    "secondary_poster_indexes": [integer, "..."],
    "layout_description": "string — must state that all frames use equal physical sizes",
    "thumbnail_readability_rules": ["string", "..."],
    "background_style": "string",
    "wall_colour": "string",
    "frame_style": "string",
    "camera_angle": "string",
    "lighting": "string — room and frame only",
    "image_prompt": "string — compositing scene: describe close wall arrangement, all equal-size frames, include explicit asset-slot mapping for every poster, include immutable source asset language for each slot",
    "negative_prompt": "string",
    "commercial_purpose": "string"
  }},
  "evaluation": {{
    "artwork_visibility_score":        {{"score": integer 1-10, "reason": "string"}},
    "commercial_clarity_score":        {{"score": integer 1-10, "reason": "string"}},
    "room_fit_score":                  {{"score": integer 1-10, "reason": "string"}},
    "collection_presentation_score":   {{"score": integer 1-10, "reason": "string"}},
    "thumbnail_readability_score":     {{"score": integer 1-10, "reason": "string"}},
    "brand_alignment_score":           {{"score": integer 1-10, "reason": "string"}},
    "reasoning": "string — one paragraph overall assessment"
  }},
  "confidence_score": integer 1-10
}}
"""


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class IndividualMockup:
    poster_index: int
    poster_title: str
    mockup_name: str
    room_type: str
    room_style: str
    wall_colour: str
    frame_style: str
    frame_colour: str
    frame_orientation: str
    artwork_aspect_ratio: str
    artwork_scale: str
    camera_angle: str
    camera_distance: str
    lighting: str
    furniture_elements: list[str]
    decor_elements: list[str]
    placement_rules: list[str]
    image_prompt: str
    negative_prompt: str
    commercial_purpose: str


@dataclass
class CollectionMockup:
    mockup_name: str
    room_type: str
    room_style: str
    wall_colour: str
    gallery_layout: str
    poster_order: list[int]
    spacing_rules: list[str]
    frame_style: str
    frame_colour: str
    artwork_scale: str
    camera_angle: str
    lighting: str
    furniture_elements: list[str]
    decor_elements: list[str]
    image_prompt: str
    negative_prompt: str
    commercial_purpose: str


@dataclass
class HeroMockup:
    mockup_name: str
    hero_strategy: str
    primary_poster_index: int
    secondary_poster_indexes: list[int]
    layout_description: str
    thumbnail_readability_rules: list[str]
    background_style: str
    wall_colour: str
    frame_style: str
    camera_angle: str
    lighting: str
    image_prompt: str
    negative_prompt: str
    commercial_purpose: str


@dataclass
class MockupEvaluation:
    artwork_visibility_score:       ScoreWithReason
    commercial_clarity_score:       ScoreWithReason
    room_fit_score:                 ScoreWithReason
    collection_presentation_score:  ScoreWithReason
    thumbnail_readability_score:    ScoreWithReason
    brand_alignment_score:          ScoreWithReason
    overall_score:                  ScoreWithReason  # computed in Python
    reasoning: str


@dataclass
class MockupPlan:
    collection_name: str
    individual_mockups: list[IndividualMockup]
    collection_mockup: CollectionMockup | None
    hero_mockup: HeroMockup | None
    shared_mockup_rules: list[str]
    forbidden_mockup_elements: list[str]
    evaluation: MockupEvaluation
    confidence_score: int
    rendering_mode: str = _DEFAULT_RENDERING_MODE


# ── Helpers ────────────────────────────────────────────────────────────────────

def _validate_score(label: str, value: int) -> None:
    if not isinstance(value, int) or not (1 <= value <= 10):
        raise ValueError(f"Score '{label}' is {value!r} — must be an integer from 1 to 10.")


def _swr(d: dict, key: str) -> ScoreWithReason:
    v = d[key]
    _validate_score(key, v["score"])
    return ScoreWithReason(score=v["score"], reason=v["reason"])


def _compute_overall(scores: list[int]) -> int:
    return round(sum(scores) / len(scores))


def _claude(client: anthropic.Anthropic, prompt: str) -> dict:
    message = client.messages.create(
        model=MODEL,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw[:300].replace("\n", " ")
        raise RuntimeError(
            f"Claude returned invalid JSON: {exc}\n"
            f"Response preview: {preview!r}"
        ) from exc


def _poster_list_text(collection_plan: CollectionPlan) -> str:
    lines = []
    for p in collection_plan.poster_items:
        lines.append(
            f"  Poster {p.index} — {p.title} | subject: {p.subject} "
            f"| aspect_ratio: {p.aspect_ratio} | mockup_room_style: {p.mockup_room_style}"
        )
    return "\n".join(lines)


# ── Prompt validation helpers ──────────────────────────────────────────────────

def _has_immutable_language(prompt: str) -> bool:
    lower = prompt.lower()
    # ponytail: require at least 2 of the key terms to avoid false positives on single-word matches
    return sum(1 for t in _IMMUTABLE_TERMS if t in lower) >= 2


def _has_no_crop_language(prompt: str) -> bool:
    lower = prompt.lower()
    return any(t in lower for t in _NO_CROP_TERMS)


def _has_no_stretch_language(prompt: str) -> bool:
    lower = prompt.lower()
    return any(t in lower for t in _NO_STRETCH_TERMS)


def _has_asset_reference(prompt: str, poster_index: int) -> bool:
    return f"poster {poster_index}" in prompt.lower()


def _all_indexes_mapped(prompt: str, indexes: set[int]) -> bool:
    return all(_has_asset_reference(prompt, i) for i in indexes)


def _has_equal_size_language(prompt: str) -> bool:
    lower = prompt.lower()
    return any(t in lower for t in _HERO_EQUAL_SIZE_TERMS)


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate(
    plan: MockupPlan,
    collection_plan: CollectionPlan,
    include_individual: bool,
    include_collection: bool,
    include_hero: bool,
) -> None:
    valid_indexes = {p.index for p in collection_plan.poster_items}
    size = collection_plan.collection_size

    # ── Individual mockups ────────────────────────────────────────────────────
    if include_individual:
        if len(plan.individual_mockups) != size:
            raise ValueError(
                f"Expected {size} individual mockups, got {len(plan.individual_mockups)}."
            )
        seen: set[int] = set()
        for m in plan.individual_mockups:
            label = f"IndividualMockup[{m.poster_index}]"

            if m.poster_index not in valid_indexes:
                raise ValueError(f"{label}: unknown poster_index {m.poster_index}.")
            if m.poster_index in seen:
                raise ValueError(f"Duplicate poster_index {m.poster_index} in individual_mockups.")
            seen.add(m.poster_index)

            if not m.artwork_aspect_ratio:
                raise ValueError(f"{label}: missing artwork_aspect_ratio.")
            if not m.image_prompt:
                raise ValueError(f"{label}: image_prompt is empty.")
            if not m.placement_rules:
                raise ValueError(f"{label}: has no placement_rules.")

            # Furniture / décor limits
            if len(m.furniture_elements) > 1:
                raise ValueError(
                    f"{label}: furniture_elements has {len(m.furniture_elements)} items "
                    f"(maximum 1). Got: {m.furniture_elements}"
                )
            if len(m.decor_elements) > 2:
                raise ValueError(
                    f"{label}: decor_elements has {len(m.decor_elements)} items "
                    f"(maximum 2). Got: {m.decor_elements}"
                )

            # Immutable asset language
            if not _has_immutable_language(m.image_prompt):
                raise ValueError(
                    f"{label}: image_prompt lacks immutable source asset language "
                    f"(need ≥2 of: {_IMMUTABLE_TERMS})."
                )
            if not _has_no_crop_language(m.image_prompt):
                raise ValueError(
                    f"{label}: image_prompt lacks no-crop intent "
                    f"(need one of: {_NO_CROP_TERMS})."
                )
            if not _has_no_stretch_language(m.image_prompt):
                raise ValueError(
                    f"{label}: image_prompt lacks no-stretch/aspect-ratio intent "
                    f"(need one of: {_NO_STRETCH_TERMS})."
                )

            # Asset reference
            if not _has_asset_reference(m.image_prompt, m.poster_index):
                raise ValueError(
                    f"{label}: image_prompt does not reference 'Poster {m.poster_index}' — "
                    f"artwork must be referenced by index, not described."
                )

    # ── Collection mockup ─────────────────────────────────────────────────────
    if include_collection and plan.collection_mockup is not None:
        cm = plan.collection_mockup

        if not cm.image_prompt:
            raise ValueError("CollectionMockup: image_prompt is empty.")

        order_set = set(cm.poster_order)
        if order_set != valid_indexes:
            raise ValueError(
                f"CollectionMockup: poster_order {cm.poster_order} does not match "
                f"collection indexes {sorted(valid_indexes)}."
            )
        if len(cm.poster_order) != size:
            raise ValueError(
                f"CollectionMockup: poster_order has {len(cm.poster_order)} entries, "
                f"expected {size}."
            )

        if not _has_immutable_language(cm.image_prompt):
            raise ValueError(
                "CollectionMockup: image_prompt lacks immutable source asset language."
            )
        if not _has_no_crop_language(cm.image_prompt):
            raise ValueError("CollectionMockup: image_prompt lacks no-crop intent.")
        if not _has_no_stretch_language(cm.image_prompt):
            raise ValueError("CollectionMockup: image_prompt lacks no-stretch intent.")
        if not _all_indexes_mapped(cm.image_prompt, valid_indexes):
            missing = [i for i in sorted(valid_indexes)
                       if not _has_asset_reference(cm.image_prompt, i)]
            raise ValueError(
                f"CollectionMockup: image_prompt does not map all poster indexes to frame slots. "
                f"Missing: Poster {missing}."
            )

    # ── Hero mockup ───────────────────────────────────────────────────────────
    if include_hero and plan.hero_mockup is not None:
        hm = plan.hero_mockup

        if not hm.image_prompt:
            raise ValueError("HeroMockup: image_prompt is empty.")
        if hm.primary_poster_index not in valid_indexes:
            raise ValueError(
                f"HeroMockup: primary_poster_index {hm.primary_poster_index} not in collection."
            )
        for idx in hm.secondary_poster_indexes:
            if idx not in valid_indexes:
                raise ValueError(
                    f"HeroMockup: secondary_poster_index {idx} not in collection."
                )
        all_hero_idxs = [hm.primary_poster_index] + hm.secondary_poster_indexes
        if len(all_hero_idxs) != len(set(all_hero_idxs)):
            raise ValueError("HeroMockup: duplicate poster indexes.")

        if not _has_immutable_language(hm.image_prompt):
            raise ValueError("HeroMockup: image_prompt lacks immutable source asset language.")
        if not _has_no_crop_language(hm.image_prompt):
            raise ValueError("HeroMockup: image_prompt lacks no-crop intent.")
        if not _has_no_stretch_language(hm.image_prompt):
            raise ValueError("HeroMockup: image_prompt lacks no-stretch intent.")

        hero_indexes = set(all_hero_idxs)
        if not _all_indexes_mapped(hm.image_prompt, hero_indexes):
            missing = [i for i in sorted(hero_indexes)
                       if not _has_asset_reference(hm.image_prompt, i)]
            raise ValueError(
                f"HeroMockup: image_prompt does not map all included poster indexes to frame slots. "
                f"Missing: Poster {missing}."
            )
        if not _has_equal_size_language(hm.image_prompt):
            raise ValueError(
                "HeroMockup: image_prompt must confirm equal physical frame sizes "
                f"(need one of: {_HERO_EQUAL_SIZE_TERMS})."
            )

    # ── Unique image prompts ──────────────────────────────────────────────────
    all_prompts = [m.image_prompt for m in plan.individual_mockups]
    if include_collection and plan.collection_mockup:
        all_prompts.append(plan.collection_mockup.image_prompt)
    if include_hero and plan.hero_mockup:
        all_prompts.append(plan.hero_mockup.image_prompt)
    if len(all_prompts) != len(set(all_prompts)):
        raise ValueError("Duplicate image prompts detected across mockups.")

    # ── Scores ────────────────────────────────────────────────────────────────
    ev = plan.evaluation
    for label, swr in [
        ("artwork_visibility",       ev.artwork_visibility_score),
        ("commercial_clarity",       ev.commercial_clarity_score),
        ("room_fit",                 ev.room_fit_score),
        ("collection_presentation",  ev.collection_presentation_score),
        ("thumbnail_readability",    ev.thumbnail_readability_score),
        ("brand_alignment",          ev.brand_alignment_score),
        ("overall",                  ev.overall_score),
    ]:
        _validate_score(label, swr.score)
    _validate_score("confidence_score", plan.confidence_score)


# ── Public interface ───────────────────────────────────────────────────────────

def generate_mockup_plan(
    collection_plan: CollectionPlan,
    include_individual_mockups: bool = True,
    include_collection_mockup: bool = True,
    include_hero_mockup: bool = True,
    rendering_mode: str = _DEFAULT_RENDERING_MODE,
) -> MockupPlan:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set in .env")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    bible_json = json.dumps({
        "collection_name":        collection_plan.collection_bible.collection_name,
        "collection_story":       collection_plan.collection_bible.collection_story,
        "brand_identity":         collection_plan.collection_bible.brand_identity,
        "target_customer":        collection_plan.collection_bible.target_customer,
        "recommended_room_style": collection_plan.collection_bible.recommended_room_style,
        "visual_identity":        collection_plan.collection_bible.visual_identity,
        "shared_palette":         collection_plan.collection_bible.shared_palette,
        "shared_lighting":        collection_plan.collection_bible.shared_lighting,
        "shared_atmosphere":      collection_plan.collection_bible.shared_atmosphere,
        "style_dna":              collection_plan.collection_bible.style_dna,
    }, indent=2)
    poster_list = _poster_list_text(collection_plan)
    size = collection_plan.collection_size

    # ── Call 1: Shared rules + Individual mockups ──────────────────────────────
    print("  [1/2] Generating individual mockups...")
    d1 = _claude(client, _INDIVIDUAL_PROMPT.format(
        bible=bible_json,
        poster_list=poster_list,
        collection_size=size,
        immutable_block=_IMMUTABLE_BLOCK,
        artwork_reference_rules=_ARTWORK_REFERENCE_RULES,
    ))

    shared_rules = d1.get("shared_mockup_rules", [])
    forbidden = d1.get("forbidden_mockup_elements", [])

    individual_mockups: list[IndividualMockup] = []
    if include_individual_mockups:
        for item in d1.get("individual_mockups", []):
            individual_mockups.append(IndividualMockup(
                poster_index=item["poster_index"],
                poster_title=item["poster_title"],
                mockup_name=item["mockup_name"],
                room_type=item["room_type"],
                room_style=item["room_style"],
                wall_colour=item["wall_colour"],
                frame_style=item["frame_style"],
                frame_colour=item["frame_colour"],
                frame_orientation=item.get("frame_orientation", "portrait"),
                artwork_aspect_ratio=item["artwork_aspect_ratio"],
                artwork_scale=item["artwork_scale"],
                camera_angle=item["camera_angle"],
                camera_distance=item["camera_distance"],
                lighting=item["lighting"],
                furniture_elements=item.get("furniture_elements", []),
                decor_elements=item.get("decor_elements", []),
                placement_rules=item.get("placement_rules", []),
                image_prompt=item["image_prompt"],
                negative_prompt=item["negative_prompt"],
                commercial_purpose=item["commercial_purpose"],
            ))

    # ── Call 2: Collection + Hero + Evaluation ─────────────────────────────────
    print("  [2/2] Generating collection mockup, hero mockup, and evaluation...")
    d2 = _claude(client, _COLLECTION_HERO_PROMPT.format(
        bible=bible_json,
        poster_list=poster_list,
        collection_size=size,
        shared_rules="\n".join(f"- {r}" for r in shared_rules),
        immutable_block=_IMMUTABLE_BLOCK,
        artwork_reference_rules=_ARTWORK_REFERENCE_RULES,
    ))

    collection_mockup: CollectionMockup | None = None
    if include_collection_mockup and "collection_mockup" in d2:
        cm = d2["collection_mockup"]
        collection_mockup = CollectionMockup(
            mockup_name=cm["mockup_name"],
            room_type=cm["room_type"],
            room_style=cm["room_style"],
            wall_colour=cm["wall_colour"],
            gallery_layout=cm["gallery_layout"],
            poster_order=cm["poster_order"],
            spacing_rules=cm.get("spacing_rules", []),
            frame_style=cm["frame_style"],
            frame_colour=cm["frame_colour"],
            artwork_scale=cm["artwork_scale"],
            camera_angle=cm["camera_angle"],
            lighting=cm["lighting"],
            furniture_elements=cm.get("furniture_elements", []),
            decor_elements=cm.get("decor_elements", []),
            image_prompt=cm["image_prompt"],
            negative_prompt=cm["negative_prompt"],
            commercial_purpose=cm["commercial_purpose"],
        )

    hero_mockup: HeroMockup | None = None
    if include_hero_mockup and "hero_mockup" in d2:
        hm = d2["hero_mockup"]
        hero_mockup = HeroMockup(
            mockup_name=hm["mockup_name"],
            hero_strategy=hm["hero_strategy"],
            primary_poster_index=hm["primary_poster_index"],
            secondary_poster_indexes=hm.get("secondary_poster_indexes", []),
            layout_description=hm["layout_description"],
            thumbnail_readability_rules=hm.get("thumbnail_readability_rules", []),
            background_style=hm["background_style"],
            wall_colour=hm["wall_colour"],
            frame_style=hm["frame_style"],
            camera_angle=hm["camera_angle"],
            lighting=hm["lighting"],
            image_prompt=hm["image_prompt"],
            negative_prompt=hm["negative_prompt"],
            commercial_purpose=hm["commercial_purpose"],
        )

    e = d2["evaluation"]
    sub_scores = [
        _swr(e, "artwork_visibility_score").score,
        _swr(e, "commercial_clarity_score").score,
        _swr(e, "room_fit_score").score,
        _swr(e, "collection_presentation_score").score,
        _swr(e, "thumbnail_readability_score").score,
        _swr(e, "brand_alignment_score").score,
    ]
    evaluation = MockupEvaluation(
        artwork_visibility_score=      _swr(e, "artwork_visibility_score"),
        commercial_clarity_score=      _swr(e, "commercial_clarity_score"),
        room_fit_score=                _swr(e, "room_fit_score"),
        collection_presentation_score= _swr(e, "collection_presentation_score"),
        thumbnail_readability_score=   _swr(e, "thumbnail_readability_score"),
        brand_alignment_score=         _swr(e, "brand_alignment_score"),
        overall_score=ScoreWithReason(
            score=_compute_overall(sub_scores),
            reason="Equal-weighted mean of six mockup evaluation scores.",
        ),
        reasoning=e.get("reasoning", ""),
    )

    plan = MockupPlan(
        collection_name=collection_plan.collection_bible.collection_name,
        individual_mockups=individual_mockups,
        collection_mockup=collection_mockup,
        hero_mockup=hero_mockup,
        shared_mockup_rules=shared_rules,
        forbidden_mockup_elements=forbidden,
        evaluation=evaluation,
        confidence_score=int(d2.get("confidence_score", 0)),
        rendering_mode=rendering_mode,
    )

    _validate(plan, collection_plan, include_individual_mockups,
              include_collection_mockup, include_hero_mockup)
    return plan
