import json
import re
from dataclasses import dataclass
import anthropic
from agent.claude_client import ANTHROPIC_API_KEY, MODEL
from agent.vision_critic import ScoreWithReason
from agent.collection_generator import CollectionPlan
from agent.mockup_generator import MockupPlan

_MAX_TOKENS = 8192

_LISTING_PROMPT = """\
You are a senior Etsy SEO copywriter specialising in premium printable wall art.

You will receive a collection overview and mockup plan summary.
Your job: generate a complete Etsy listing package for this printable art collection.

Collection overview:
{collection_summary}

Mockup plan summary:
{mockup_summary}

TITLE RULES:
  • Maximum 140 characters
  • Natural English — readable by a human buyer
  • Front-load strongest keywords
  • Include collection size naturally (e.g. "Set of 3")
  • No keyword stuffing, no ALL CAPS spam, no repeated words (PRINT PRINT DIGITAL DIGITAL)
  • One clear value proposition

SHORT TITLE RULES:
  • 40–60 characters
  • For internal shop organisation — concise and descriptive

DESCRIPTION RULES:
  • Use these sections with these exact headings:
    ✦ Overview
    ✦ The Collection
    ✦ What's Included
    ✦ Printing Tips
    ✦ Perfect For
    ✦ Digital Download
    ✦ Notes
  • Sound premium, warm, and human — not AI-generated
  • No fake scarcity, no exaggerated claims, no em-dash overuse
  • Total 350–550 words

BULLET FEATURES RULES:
  • 6–10 bullets
  • Short, punchy, benefit-focused
  • Examples: "Instant download — no waiting", "High-res files for crisp large prints"

SEO KEYWORDS:
  • 20–40 keywords and short phrases
  • Natural search queries a buyer would type
  • Mix of broad and specific
  • No duplicates, no single-character entries

ETSY TAGS:
  • Exactly 13 tags
  • Each tag ≤ 20 characters (including spaces)
  • No duplicates
  • No singular/plural duplicates (e.g. not both "wall art" and "wall arts")
  • Each tag must be a useful, searchable term
  • Mix of style, niche, product type, and room type tags

MATERIALS:
  • List realistic digital product materials
  • Examples: digital download, JPEG file, PNG file, PDF guide, ZIP archive
  • 3–6 items

IMAGE ORDER:
  • Plan up to 10 listing images in order
  • Every entry needs: position (1–10), description (what this image shows), source (where it comes from — e.g. hero_mockup, gallery_mockup, poster_1_individual_mockup, poster_2_close_crop, size_guide, download_instructions, thank_you_card)
  • Image 1 must be the hero mockup — highest conversion priority
  • Maximise variety and buyer information

DOWNLOAD PACKAGE:
  • zip_name: descriptive filename with no spaces (use underscores)
  • file_list: list of files/folders as they appear inside the ZIP (human-readable)
  • size_variants: list of print size/ratio variants included (e.g. "2:3 (40x60cm)", "A3", "5x7 inch")
  • formats_included: list of file formats (e.g. "JPG", "PNG", "PDF")
  • includes_print_guide: true/false
  • includes_license: true/false
  • includes_readme: true/false
  • total_file_count: integer

CUSTOMER NOTES:
  • 4–6 concise factual notes
  • Examples: "Frame not included.", "Colours may vary by monitor and printer.", "For personal use only."

FAQ:
  • 8–12 FAQs
  • Cover: how to download, printing at home, print lab recommendations, paper type,
    frame size, commercial use, refunds, colour accuracy, file formats, resizing
  • Answers: 2–4 sentences, warm and helpful

EVALUATION:
  • seo_score: how well this listing will rank in Etsy search
  • commercial_appeal_score: how likely a buyer who sees it will want it
  • customer_clarity_score: how clearly a first-time buyer understands what they get
  • conversion_potential_score: overall likelihood of turning a view into a sale
  • brand_consistency_score: how well the listing reflects the collection's brand identity
  • professionalism_score: listing polish and quality compared to top Etsy sellers
  • reasoning: one paragraph

Return ONLY a valid JSON object — no markdown, no commentary:
{{
  "shop_section": "string — e.g. Wall Art Prints, Botanical Prints, Gallery Wall Sets",
  "listing_type": "digital",
  "listing_title": "string ≤140 chars",
  "short_title": "string 40–60 chars",
  "listing_description": "string — full multi-section description with ✦ headings",
  "bullet_features": ["string", "..."],
  "seo_keywords": ["string", "..."],
  "etsy_tags": ["tag1", "tag2", "..."],
  "materials": ["string", "..."],
  "primary_category": "string",
  "secondary_category": "string",
  "room_styles": ["string", "..."],
  "target_customer": "string",
  "colour_palette": ["string", "..."],
  "image_order": [
    {{
      "position": integer,
      "description": "string",
      "source": "string"
    }}
  ],
  "download_package": {{
    "zip_name": "string",
    "file_list": ["string", "..."],
    "size_variants": ["string", "..."],
    "formats_included": ["string", "..."],
    "includes_print_guide": boolean,
    "includes_license": boolean,
    "includes_readme": boolean,
    "total_file_count": integer
  }},
  "customer_notes": ["string", "..."],
  "faq": [
    {{
      "question": "string",
      "answer": "string"
    }}
  ],
  "evaluation": {{
    "seo_score":                  {{"score": integer 1-10, "reason": "string"}},
    "commercial_appeal_score":    {{"score": integer 1-10, "reason": "string"}},
    "customer_clarity_score":     {{"score": integer 1-10, "reason": "string"}},
    "conversion_potential_score": {{"score": integer 1-10, "reason": "string"}},
    "brand_consistency_score":    {{"score": integer 1-10, "reason": "string"}},
    "professionalism_score":      {{"score": integer 1-10, "reason": "string"}},
    "reasoning": "string"
  }},
  "confidence_score": integer 1-10
}}
"""


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class FAQ:
    question: str
    answer: str


@dataclass
class ImageOrderItem:
    position: int
    description: str
    source: str


@dataclass
class DownloadPackage:
    zip_name: str
    file_list: list[str]
    size_variants: list[str]
    formats_included: list[str]
    includes_print_guide: bool
    includes_license: bool
    includes_readme: bool
    total_file_count: int


@dataclass
class ListingEvaluation:
    seo_score:                  ScoreWithReason
    commercial_appeal_score:    ScoreWithReason
    customer_clarity_score:     ScoreWithReason
    conversion_potential_score: ScoreWithReason
    brand_consistency_score:    ScoreWithReason
    professionalism_score:      ScoreWithReason
    overall_score:              ScoreWithReason  # computed in Python
    reasoning: str


@dataclass
class ListingPlan:
    shop_section: str
    listing_type: str
    listing_title: str
    short_title: str
    listing_description: str
    bullet_features: list[str]
    seo_keywords: list[str]
    etsy_tags: list[str]
    materials: list[str]
    primary_category: str
    secondary_category: str
    room_styles: list[str]
    target_customer: str
    colour_palette: list[str]
    image_order: list[ImageOrderItem]
    download_package: DownloadPackage
    customer_notes: list[str]
    faq: list[FAQ]
    evaluation: ListingEvaluation
    confidence_score: int


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


def _claude(client: anthropic.Anthropic, prompt: str, on_usage=None) -> dict:
    message = client.messages.create(
        model=MODEL,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    if on_usage is not None:
        on_usage({
            "provider": "anthropic",
            "model": MODEL,
            "call_type": "text",
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        })
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


def _build_collection_summary(plan: CollectionPlan) -> dict:
    bible = plan.collection_bible
    return {
        "collection_name":    bible.collection_name,
        "collection_story":   bible.collection_story,
        "brand_identity":     bible.brand_identity,
        "target_customer":    bible.target_customer,
        "recommended_room_style": bible.recommended_room_style,
        "visual_identity":    bible.visual_identity,
        "shared_palette":     bible.shared_palette,
        "style_dna":          bible.style_dna,
        "collection_size":    plan.collection_size,
        "posters": [
            {
                "index":               p.index,
                "title":               p.title,
                "subject":             p.subject,
                "aspect_ratio":        p.aspect_ratio,
                "suggested_etsy_title": p.suggested_etsy_title,
                "suggested_etsy_tags": p.suggested_etsy_tags,
            }
            for p in plan.poster_items
        ],
    }


def _build_mockup_summary(plan: MockupPlan) -> dict:
    summary: dict = {
        "collection_name": plan.collection_name,
        "rendering_mode":  plan.rendering_mode,
        "shared_rules":    plan.shared_mockup_rules[:5],  # top 5 to keep context compact
    }
    if plan.hero_mockup:
        summary["hero_mockup"] = {
            "name":           plan.hero_mockup.mockup_name,
            "hero_strategy":  plan.hero_mockup.hero_strategy,
            "camera_angle":   plan.hero_mockup.camera_angle,
            "wall_colour":    plan.hero_mockup.wall_colour,
        }
    if plan.collection_mockup:
        summary["gallery_mockup"] = {
            "name":           plan.collection_mockup.mockup_name,
            "gallery_layout": plan.collection_mockup.gallery_layout,
            "room_style":     plan.collection_mockup.room_style,
        }
    summary["individual_mockups"] = [
        {
            "poster_index": m.poster_index,
            "room_type":    m.room_type,
            "room_style":   m.room_style,
        }
        for m in plan.individual_mockups
    ]
    return summary


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate(plan: ListingPlan) -> None:
    # Title
    if len(plan.listing_title) > 140:
        raise ValueError(
            f"listing_title is {len(plan.listing_title)} chars — maximum 140. "
            f"Title: {plan.listing_title!r}"
        )

    # Short title
    if not (40 <= len(plan.short_title) <= 60):
        raise ValueError(
            f"short_title is {len(plan.short_title)} chars — must be 40–60. "
            f"Title: {plan.short_title!r}"
        )

    # Tags
    if len(plan.etsy_tags) != 13:
        raise ValueError(
            f"etsy_tags has {len(plan.etsy_tags)} entries — exactly 13 required."
        )
    for tag in plan.etsy_tags:
        if len(tag) > 20:
            raise ValueError(
                f"Etsy tag {tag!r} is {len(tag)} chars — maximum 20."
            )
    if len(plan.etsy_tags) != len(set(t.lower() for t in plan.etsy_tags)):
        seen: set[str] = set()
        for tag in plan.etsy_tags:
            low = tag.lower()
            if low in seen:
                raise ValueError(f"Duplicate Etsy tag: {tag!r}")
            seen.add(low)

    # SEO keywords
    if not (20 <= len(plan.seo_keywords) <= 40):
        raise ValueError(
            f"seo_keywords has {len(plan.seo_keywords)} entries — must be 20–40."
        )

    # Bullet features
    if not (6 <= len(plan.bullet_features) <= 10):
        raise ValueError(
            f"bullet_features has {len(plan.bullet_features)} entries — must be 6–10."
        )

    # Image order
    if len(plan.image_order) > 10:
        raise ValueError(
            f"image_order has {len(plan.image_order)} entries — maximum 10."
        )
    positions = [img.position for img in plan.image_order]
    if len(positions) != len(set(positions)):
        raise ValueError("image_order has duplicate position values.")

    # FAQ
    if not (8 <= len(plan.faq) <= 12):
        raise ValueError(
            f"faq has {len(plan.faq)} entries — must be 8–12."
        )

    # Description non-empty
    if not plan.listing_description.strip():
        raise ValueError("listing_description is empty.")

    # Scores
    ev = plan.evaluation
    sub_scores = []
    for label, swr in [
        ("seo",                  ev.seo_score),
        ("commercial_appeal",    ev.commercial_appeal_score),
        ("customer_clarity",     ev.customer_clarity_score),
        ("conversion_potential", ev.conversion_potential_score),
        ("brand_consistency",    ev.brand_consistency_score),
        ("professionalism",      ev.professionalism_score),
        ("overall",              ev.overall_score),
    ]:
        _validate_score(label, swr.score)
        if label != "overall":
            sub_scores.append(swr.score)
    _validate_score("confidence_score", plan.confidence_score)


# ── Public interface ───────────────────────────────────────────────────────────

def generate_listing_plan(
    collection_plan: CollectionPlan,
    mockup_plan: MockupPlan,
    on_usage=None,
) -> ListingPlan:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set in .env")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = _LISTING_PROMPT.format(
        collection_summary=json.dumps(_build_collection_summary(collection_plan), indent=2),
        mockup_summary=json.dumps(_build_mockup_summary(mockup_plan), indent=2),
    )

    print("  Generating listing package...")
    d = _claude(client, prompt, on_usage=on_usage)

    # ── Parse download package ─────────────────────────────────────────────────
    dp = d["download_package"]
    download_package = DownloadPackage(
        zip_name=dp["zip_name"],
        file_list=dp["file_list"],
        size_variants=dp["size_variants"],
        formats_included=dp["formats_included"],
        includes_print_guide=bool(dp.get("includes_print_guide", False)),
        includes_license=bool(dp.get("includes_license", False)),
        includes_readme=bool(dp.get("includes_readme", False)),
        total_file_count=int(dp.get("total_file_count", 0)),
    )

    # ── Parse image order ──────────────────────────────────────────────────────
    image_order = [
        ImageOrderItem(
            position=item["position"],
            description=item["description"],
            source=item["source"],
        )
        for item in d["image_order"]
    ]

    # ── Parse FAQ ──────────────────────────────────────────────────────────────
    faq = [FAQ(question=f["question"], answer=f["answer"]) for f in d["faq"]]

    # ── Parse evaluation ───────────────────────────────────────────────────────
    e = d["evaluation"]
    sub_scores = [
        _swr(e, "seo_score").score,
        _swr(e, "commercial_appeal_score").score,
        _swr(e, "customer_clarity_score").score,
        _swr(e, "conversion_potential_score").score,
        _swr(e, "brand_consistency_score").score,
        _swr(e, "professionalism_score").score,
    ]
    evaluation = ListingEvaluation(
        seo_score=                  _swr(e, "seo_score"),
        commercial_appeal_score=    _swr(e, "commercial_appeal_score"),
        customer_clarity_score=     _swr(e, "customer_clarity_score"),
        conversion_potential_score= _swr(e, "conversion_potential_score"),
        brand_consistency_score=    _swr(e, "brand_consistency_score"),
        professionalism_score=      _swr(e, "professionalism_score"),
        overall_score=ScoreWithReason(
            score=_compute_overall(sub_scores),
            reason="Equal-weighted mean of six listing evaluation scores.",
        ),
        reasoning=e.get("reasoning", ""),
    )

    plan = ListingPlan(
        shop_section=        d["shop_section"],
        listing_type=        d["listing_type"],
        listing_title=       d["listing_title"],
        short_title=         d["short_title"],
        listing_description= d["listing_description"],
        bullet_features=     d["bullet_features"],
        seo_keywords=        d["seo_keywords"],
        etsy_tags=           d["etsy_tags"],
        materials=           d["materials"],
        primary_category=    d["primary_category"],
        secondary_category=  d["secondary_category"],
        room_styles=         d.get("room_styles", []),
        target_customer=     d["target_customer"],
        colour_palette=      d.get("colour_palette", []),
        image_order=         image_order,
        download_package=    download_package,
        customer_notes=      d["customer_notes"],
        faq=                 faq,
        evaluation=          evaluation,
        confidence_score=    int(d["confidence_score"]),
    )

    _validate(plan)
    return plan
