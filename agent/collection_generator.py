import json
import re
from dataclasses import dataclass
import anthropic
from agent.claude_client import ANTHROPIC_API_KEY, MODEL
from agent.vision_critic import ScoreWithReason, VisionReport

_MAX_TOKENS = 8192
_MIN_SIZE = 3
_MAX_SIZE = 8
_DEFAULT_SIZE = 5
_BATCH_SIZE = 2  # posters per Claude call

_PROHIBITED_PAPER_PHRASES = [
    "paper texture", "paper surface", "paper background", "paper edge",
    "cream paper", "textured paper", "watercolour paper", "watercolor paper",
    "washi paper", "deckled", "torn paper", "painted on paper",
]
_FULL_BLEED_TERMS = [
    "full bleed", "edge-to-edge", "edge to edge", "fills the entire",
    "entire canvas", "entire frame", "extends to", "composition extends",
]

# ── Prompts ────────────────────────────────────────────────────────────────────

_PRINTABLE_RULES = """\
PRINTABLE WALL ART RULES — apply to every prompt:
- Every poster must be full bleed: artwork fills the entire canvas edge to edge.
  Use natural phrasing: "full bleed composition", "artwork fills the entire frame",
  "edge-to-edge illustration", "composition extends naturally to all edges".
- Describe rendering technique only — NOT a physical surface:
  GOOD: "watercolour brushwork", "gouache texture", "ink linework", "painterly illustration".
  NEVER: cream paper, paper texture, washi paper, watercolour paper, paper surface,
         deckled edges, torn paper, painted on paper.
- Negative prompts must always include: border, white border, blank border, decorative border,
  paper edge, torn paper, deckled edge, blank corners, blank margins, canvas edge, frame,
  poster frame, picture frame, mockup, watermark, artist signature, logos, brand names,
  copyrighted characters.
- Do NOT introduce copyrighted characters, franchise names, artist names, or protected brands.
- Visible text must be fictional, generic, visually secondary, and minimal.\
"""

BIBLE_PROMPT = """\
You are a senior creative director designing a printable wall art collection for Etsy.

Your first task: create a Collection Bible — the single source of truth for the collection's visual identity.
Every poster will be generated from this bible.

Poster concept:
{concept}

Optimized image prompt:
{image_prompt}

Optimized negative prompt:
{negative_prompt}

{vision_notes}

THIS COLLECTION CONTAINS EXACTLY {collection_size} POSTERS — no more, no less.
- collection_story must introduce exactly {collection_size} distinct subjects or scenes.
- Do NOT describe, name, or imply characters or scenes that will not appear as actual posters.
- When stating the collection size anywhere in your response, use the digit {collection_size}.
  Do NOT use English number words (three, four, five, six, seven, eight) for the poster count.
- consistency_rules and storytelling_rules must not imply a count different from {collection_size}.
- Do not write phrases like "all five prints", "each of the five characters", or "across all posters"
  unless the number matches {collection_size}.

{printable_rules}

Return ONLY a valid JSON object — no markdown, no commentary:
{{
  "collection_bible": {{
    "collection_name": "string",
    "collection_story": "string — the narrative and commercial identity of the collection",
    "brand_identity": "string",
    "target_customer": "string",
    "recommended_room_style": "string",
    "visual_identity": "string",
    "shared_rendering_medium": "string — technique only, no paper surface references",
    "shared_linework": "string",
    "shared_lighting": "string",
    "shared_palette": ["colour name", "..."],
    "shared_accent_colour_rules": ["string", "..."],
    "shared_camera_angle": "string",
    "shared_perspective": "string",
    "shared_atmosphere": "string",
    "shared_detail_level": "string",
    "shared_print_treatment": "string",
    "shared_storytelling_rules": ["string", "..."],
    "shared_composition_rules": ["string", "..."],
    "shared_style_rules": ["string", "..."],
    "shared_negative_prompt": "string — comprehensive negative prompt for the entire collection",
    "style_dna": ["distinctive visual trait", "..."],
    "consistency_rules": ["rule every poster must obey", "..."],
    "forbidden_elements": ["thing that must never appear", "..."],
    "full_bleed_rules": ["full-bleed and printability rule", "..."]
  }}
}}
"""

POSTERS_PROMPT = """\
You are a senior creative director generating posters for a printable wall art collection.

The Collection Bible below is the single source of truth. Every poster must visibly belong to the same collection.

Collection Bible:
{bible}

Optimized image prompt (reference for style):
{image_prompt}

Collection size: {collection_size} posters

{printable_rules}

POSTER PROMPT RULES:
- Every image_prompt must be fully standalone — never reference other posters.
  Do NOT use: "same as previous", "continue the style", "matching the first image",
  "use the collection palette above", "similar to poster one".
- Each prompt must restate the essential shared art direction and print rules.
- Controlled variation allowed in: primary subject, specific location, environmental props,
  storytelling detail, weather, time of day, season, secondary accent colour.
- Do NOT vary: core art style, rendering medium, linework, palette family, lighting language,
  visual era, atmosphere, aspect ratio, composition system, print treatment.
- Every poster must have one clear unique hook.
- Avoid adjective overload and contradictions.

ETSY METADATA:
- suggested_etsy_tags: EXACTLY 13 tags, no duplicates, no copyrighted terms.
- Poster titles must differentiate each poster while preserving collection identity.

Return ONLY a valid JSON object — no markdown, no commentary:
{{
  "poster_items": [
    {{
      "index": 1,
      "title": "string",
      "subject": "string",
      "scene_concept": "string — detailed scene description",
      "storytelling_focus": "string",
      "unique_hook": "string",
      "image_prompt": "string — fully standalone, complete art direction, full-bleed",
      "negative_prompt": "string — shared_negative_prompt base plus poster-specific additions",
      "aspect_ratio": "string",
      "focal_point": "string",
      "foreground_elements": ["string", "..."],
      "midground_elements": ["string", "..."],
      "background_elements": ["string", "..."],
      "palette_variation": ["specific colour used in this poster", "..."],
      "lighting_variation": "string",
      "weather_or_time_variation": "string",
      "consistency_notes": ["how this poster maintains collection identity", "..."],
      "suggested_etsy_title": "string",
      "suggested_etsy_tags": ["exactly 13 unique tags"],
      "mockup_room_style": "string"
    }}
  ],
  "evaluation": {{
    "consistency_score":      {{"score": integer from 1 to 10, "reason": "string"}},
    "commercial_score":       {{"score": integer from 1 to 10, "reason": "string"}},
    "variation_score":        {{"score": integer from 1 to 10, "reason": "string"}},
    "brand_identity_score":   {{"score": integer from 1 to 10, "reason": "string"}},
    "print_collection_score": {{"score": integer from 1 to 10, "reason": "string"}},
    "market_uniqueness_score":{{"score": integer from 1 to 10, "reason": "string"}},
    "reasoning": "string — one paragraph overall assessment"
  }},
  "collection_consistency_notes": ["string", "..."],
  "confidence_score": integer from 1 to 10
}}
"""

POSTER_BATCH_PROMPT = """\
You are a senior creative director generating posters for a printable wall art collection.

The Collection Bible below is the single source of truth. Every poster must visibly belong to the same collection.

Collection Bible:
{bible}

Optimized image prompt (reference for style):
{image_prompt}

Total collection size: {collection_size} posters.
Generate ONLY these poster indexes in this response: {batch_indexes}

Previously generated posters — DO NOT duplicate any title, subject, scene_concept, storytelling_focus, or unique_hook:
{prior_summaries}

{printable_rules}

POSTER PROMPT RULES:
- Every image_prompt must be fully standalone — never reference other posters.
  Do NOT use: "same as previous", "continue the style", "matching the first image".
- Each prompt must restate the essential shared art direction and print rules.
- Controlled variation allowed in: primary subject, specific location, environmental props,
  storytelling detail, weather, time of day, season, secondary accent colour.
- Do NOT vary: core art style, rendering medium, linework, palette family, lighting language,
  visual era, atmosphere, aspect ratio, composition system, print treatment.
- Every poster must have one clear unique hook.
- Avoid adjective overload and contradictions.

ETSY METADATA:
- suggested_etsy_tags: EXACTLY 13 tags, no duplicates, no copyrighted terms.
- Poster titles must differentiate each poster while preserving collection identity.

Return ONLY a valid JSON object — no markdown, no commentary:
{{
  "poster_items": [
    {{
      "index": integer,
      "title": "string",
      "subject": "string",
      "scene_concept": "string — detailed scene description",
      "storytelling_focus": "string",
      "unique_hook": "string",
      "image_prompt": "string — fully standalone, complete art direction, full-bleed",
      "negative_prompt": "string",
      "aspect_ratio": "string",
      "focal_point": "string",
      "foreground_elements": ["string", "..."],
      "midground_elements": ["string", "..."],
      "background_elements": ["string", "..."],
      "palette_variation": ["string", "..."],
      "lighting_variation": "string",
      "weather_or_time_variation": "string",
      "consistency_notes": ["string", "..."],
      "suggested_etsy_title": "string",
      "suggested_etsy_tags": ["exactly 13 unique tags"],
      "mockup_room_style": "string"
    }}
  ]
}}
"""

EVAL_ONLY_PROMPT = """\
You are a senior creative director evaluating a completed printable wall art collection.

Collection identity:
{bible_summary}

All {collection_size} posters in this collection:
{poster_summaries}

Evaluate the complete collection across six dimensions.
Return ONLY a valid JSON object — no markdown, no commentary:
{{
  "evaluation": {{
    "consistency_score":       {{"score": integer from 1 to 10, "reason": "string"}},
    "commercial_score":        {{"score": integer from 1 to 10, "reason": "string"}},
    "variation_score":         {{"score": integer from 1 to 10, "reason": "string"}},
    "brand_identity_score":    {{"score": integer from 1 to 10, "reason": "string"}},
    "print_collection_score":  {{"score": integer from 1 to 10, "reason": "string"}},
    "market_uniqueness_score": {{"score": integer from 1 to 10, "reason": "string"}},
    "reasoning": "string — one paragraph overall assessment"
  }},
  "collection_consistency_notes": ["string", "..."],
  "confidence_score": integer from 1 to 10
}}
"""

_COMPACT_SUFFIX = """

RETRY — COMPACT MODE:
Return valid JSON only. No markdown. No commentary outside the JSON object.
Keep scene_concept under 30 words. Keep storytelling_focus under 20 words.
Keep each consistency_notes entry to one short sentence.
All required fields must still be present. Do not skip any requested poster index.
"""

_VISION_NOTES_TEMPLATE = """\
VISION CRITIC FINDINGS — apply across the collection:
Strengths to preserve:
{strengths}
Improvement areas to address:
{improvements}
Weaknesses to avoid propagating:
{weaknesses}
"""


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class CollectionBible:
    collection_name: str
    collection_story: str
    brand_identity: str
    target_customer: str
    recommended_room_style: str
    visual_identity: str
    shared_rendering_medium: str
    shared_linework: str
    shared_lighting: str
    shared_palette: list[str]
    shared_accent_colour_rules: list[str]
    shared_camera_angle: str
    shared_perspective: str
    shared_atmosphere: str
    shared_detail_level: str
    shared_print_treatment: str
    shared_storytelling_rules: list[str]
    shared_composition_rules: list[str]
    shared_style_rules: list[str]
    shared_negative_prompt: str
    style_dna: list[str]
    consistency_rules: list[str]
    forbidden_elements: list[str]
    full_bleed_rules: list[str]


@dataclass
class CollectionPoster:
    index: int
    title: str
    subject: str
    scene_concept: str
    storytelling_focus: str
    unique_hook: str
    image_prompt: str
    negative_prompt: str
    aspect_ratio: str
    focal_point: str
    foreground_elements: list[str]
    midground_elements: list[str]
    background_elements: list[str]
    palette_variation: list[str]
    lighting_variation: str
    weather_or_time_variation: str
    consistency_notes: list[str]
    suggested_etsy_title: str
    suggested_etsy_tags: list[str]
    mockup_room_style: str


@dataclass
class CollectionEvaluation:
    consistency_score:      ScoreWithReason
    commercial_score:       ScoreWithReason
    variation_score:        ScoreWithReason
    brand_identity_score:   ScoreWithReason
    print_collection_score: ScoreWithReason
    market_uniqueness_score: ScoreWithReason
    overall_score:          ScoreWithReason  # computed in Python
    reasoning: str


@dataclass
class CollectionPlan:
    collection_bible: CollectionBible
    collection_size: int
    poster_items: list[CollectionPoster]
    collection_consistency_notes: list[str]
    evaluation: CollectionEvaluation
    confidence_score: int


# ── Exceptions ─────────────────────────────────────────────────────────────────

class _TruncatedResponseError(RuntimeError):
    """Raised when Claude's JSON response is likely truncated by the token limit."""


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _infer_size(concept: dict, given: int | None) -> int:
    if given is not None:
        return max(_MIN_SIZE, min(_MAX_SIZE, given))
    s = concept.get("single_or_set", "").lower()
    notes = concept.get("set_consistency_notes", "").lower()
    for n in range(_MAX_SIZE, _MIN_SIZE - 1, -1):
        if f"set of {n}" in s or f"set of {n}" in notes or str(n) in notes:
            return n
    return _DEFAULT_SIZE


def _compute_overall(e: dict) -> int:
    keys = ["consistency_score", "commercial_score", "variation_score",
            "brand_identity_score", "print_collection_score", "market_uniqueness_score"]
    return round(sum(e[k]["score"] for k in keys) / len(keys))


def _swr(d: dict, key: str) -> ScoreWithReason:
    v = d[key]
    _validate_score(key, v["score"])
    return ScoreWithReason(score=v["score"], reason=v["reason"])


def _validate_score(label: str, value: int) -> None:
    if not isinstance(value, int) or not (1 <= value <= 10):
        raise ValueError(
            f"Score '{label}' is {value!r} — must be an integer from 1 to 10."
        )


def _make_batches(size: int, batch_size: int = _BATCH_SIZE) -> list[list[int]]:
    """Partition poster indexes 1..size into sequential batches of batch_size."""
    batches: list[list[int]] = []
    idx = 1
    while idx <= size:
        end = min(idx + batch_size - 1, size)
        batches.append(list(range(idx, end + 1)))
        idx = end + 1
    return batches


def _is_truncated_error(error_msg: str, raw: str) -> bool:
    """Return True when JSON failure is likely caused by output truncation."""
    signals = [
        "unterminated string",
        "unexpected end of data",
        "unexpected end of input",
        "unexpected eof",
    ]
    if any(s in error_msg.lower() for s in signals):
        return True
    # Raw text not ending with a closing JSON delimiter is another strong signal
    stripped = raw.rstrip()
    return bool(stripped) and stripped[-1] not in ("}", "]", '"')


def _raw_call(client: anthropic.Anthropic, prompt: str, on_usage=None) -> str:
    """Single Claude API call. Returns raw response text."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    if on_usage is not None:
        on_usage({
            "provider": "anthropic",
            "model": MODEL,
            "call_type": "text",
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        })
    return msg.content[0].text.strip()


def _parse_with_diagnostics(
    raw: str,
    stage: str,
    requested_indexes: list[int],
) -> dict:
    """Strip markdown fences, parse JSON, and raise with rich diagnostics on failure."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        likely = _is_truncated_error(str(exc), cleaned)
        msg = (
            f"[{stage}] JSON parse failed — batch {requested_indexes}\n"
            f"  likely_truncated: {likely}\n"
            f"  character_count:  {len(cleaned)}\n"
            f"  error:            {exc}\n"
            f"  first_300:        {cleaned[:300]!r}\n"
            f"  last_500:         {cleaned[-500:]!r}"
        )
        raise _TruncatedResponseError(msg) if likely else RuntimeError(msg)


def _claude(client: anthropic.Anthropic, prompt: str, on_usage=None) -> dict:
    """Parse and return JSON from Claude. Used for Bible and evaluation calls (no retry)."""
    raw = _raw_call(client, prompt, on_usage=on_usage)
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        preview = cleaned[:300].replace("\n", " ")
        raise RuntimeError(
            f"Claude returned invalid JSON: {exc}\n"
            f"Response preview (first 300 chars): {preview!r}"
        ) from exc


def _prior_summaries_text(prior_items: list[dict]) -> str:
    if not prior_items:
        return "None — this is the first batch."
    lines = []
    for p in prior_items:
        lines.append(f"  Poster {p['index']} — {p['title']}")
        lines.append(f"    subject:             {p['subject']}")
        lines.append(f"    scene_concept:       {p['scene_concept']}")
        lines.append(f"    storytelling_focus:  {p['storytelling_focus']}")
        lines.append(f"    unique_hook:         {p['unique_hook']}")
    return "\n".join(lines)


# ── Per-batch validation ───────────────────────────────────────────────────────

def _validate_batch(
    items: list[dict],
    requested_indexes: list[int],
    prior_items: list[dict],
    stage: str,
) -> None:
    """Validate a raw batch of poster dicts before appending to the accumulated list."""
    label = f"[{stage}] batch {requested_indexes}"

    if not isinstance(items, list):
        raise ValueError(f"{label}: poster_items is not a list.")

    # Count check first, then duplicate-within-batch, then index equality
    if len(items) != len(requested_indexes):
        raise ValueError(
            f"{label}: expected {len(requested_indexes)} poster(s), got {len(items)}."
        )
    if len(set(item["index"] for item in items)) != len(items):
        raise ValueError(f"{label}: duplicate indexes within batch.")
    returned = sorted(item["index"] for item in items)
    if returned != sorted(requested_indexes):
        raise ValueError(
            f"{label}: expected indexes {sorted(requested_indexes)}, got {returned}."
        )

    # Cross-batch uniqueness sets (built from prior batches)
    prior_titles =         {p["title"]             for p in prior_items}
    prior_subjects =       {p["subject"]           for p in prior_items}
    prior_scene_concepts = {p["scene_concept"]     for p in prior_items}
    prior_unique_hooks =   {p["unique_hook"]       for p in prior_items}
    prior_image_prompts =  {p["image_prompt"]      for p in prior_items}

    for item in items:
        idx = item["index"]
        item_label = f"{label} poster {idx}"

        # Required non-empty fields
        for field in [
            "title", "subject", "scene_concept", "storytelling_focus",
            "unique_hook", "image_prompt", "negative_prompt",
            "aspect_ratio", "suggested_etsy_title", "mockup_room_style",
        ]:
            if not item.get(field):
                raise ValueError(f"{item_label}: field '{field}' is empty or missing.")

        # Cross-batch duplicate checks
        if item["title"] in prior_titles:
            raise ValueError(f"{item_label}: duplicate title {item['title']!r}.")
        if item["subject"] in prior_subjects:
            raise ValueError(f"{item_label}: duplicate subject.")
        if item["scene_concept"] in prior_scene_concepts:
            raise ValueError(f"{item_label}: duplicate scene_concept.")
        if item["unique_hook"] in prior_unique_hooks:
            raise ValueError(f"{item_label}: duplicate unique_hook.")
        if item["image_prompt"] in prior_image_prompts:
            raise ValueError(f"{item_label}: duplicate image_prompt.")

        # Etsy tags
        tags = item.get("suggested_etsy_tags", [])
        if len(tags) != 13:
            raise ValueError(f"{item_label}: {len(tags)} Etsy tags — exactly 13 required.")
        if len(set(tags)) != 13:
            raise ValueError(f"{item_label}: duplicate Etsy tags.")

        # Full-bleed and prohibited-paper checks
        lower = item["image_prompt"].lower()
        if not any(term in lower for term in _FULL_BLEED_TERMS):
            raise ValueError(f"{item_label}: image_prompt missing full-bleed intent.")
        for phrase in _PROHIBITED_PAPER_PHRASES:
            if phrase in lower:
                raise ValueError(
                    f"{item_label}: image_prompt contains prohibited phrase '{phrase}'."
                )

        # Consistency notes non-empty
        if not item.get("consistency_notes"):
            raise ValueError(f"{item_label}: consistency_notes is empty.")


# ── Batch generation with retry ────────────────────────────────────────────────

def _generate_batch(
    client: anthropic.Anthropic,
    prompt: str,
    stage: str,
    requested_indexes: list[int],
    on_usage=None,
) -> dict:
    """
    Call Claude for one poster batch.
    On likely truncation: retry once with a compact-JSON suffix.
    Non-truncation parse failures propagate immediately.
    """
    raw = _raw_call(client, prompt, on_usage=on_usage)
    try:
        return _parse_with_diagnostics(raw, stage, requested_indexes)
    except _TruncatedResponseError:
        raw2 = _raw_call(client, prompt + _COMPACT_SUFFIX, on_usage=on_usage)
        return _parse_with_diagnostics(raw2, f"{stage}[retry]", requested_indexes)


# ── Full collection validation (post-assembly) ─────────────────────────────────

def _validate(plan: CollectionPlan) -> None:
    size = plan.collection_size
    posters = plan.poster_items
    bible = plan.collection_bible

    if not (_MIN_SIZE <= size <= _MAX_SIZE):
        raise ValueError(f"Collection size {size} outside allowed range {_MIN_SIZE}–{_MAX_SIZE}.")
    if len(posters) != size:
        raise ValueError(f"Expected {size} posters, got {len(posters)}.")
    if not bible.collection_name:
        raise ValueError("collection_name is empty.")
    if not bible.shared_palette:
        raise ValueError("shared_palette is empty.")
    if not bible.shared_lighting:
        raise ValueError("shared_lighting is empty.")
    if not bible.shared_composition_rules:
        raise ValueError("shared_composition_rules is empty.")
    if not bible.full_bleed_rules:
        raise ValueError("full_bleed_rules is empty.")

    indexes = [p.index for p in posters]
    if len(set(indexes)) != len(indexes):
        raise ValueError(f"Duplicate poster indexes: {indexes}.")
    if len({p.title for p in posters}) != len(posters):
        raise ValueError("Duplicate poster titles.")
    if len({p.scene_concept for p in posters}) != len(posters):
        raise ValueError("Duplicate scene concepts.")

    prompts = [p.image_prompt for p in posters]
    if any(not p for p in prompts):
        raise ValueError("One or more image prompts are empty.")
    if len(set(prompts)) != len(prompts):
        raise ValueError("Duplicate image prompts.")

    for p in posters:
        if not p.consistency_notes:
            raise ValueError(f"Poster '{p.title}' has no consistency notes.")
        if len(p.suggested_etsy_tags) != 13:
            raise ValueError(
                f"Poster '{p.title}' has {len(p.suggested_etsy_tags)} tags, expected 13."
            )
        if len(set(p.suggested_etsy_tags)) != 13:
            raise ValueError(f"Poster '{p.title}' has duplicate Etsy tags.")
        lower = p.image_prompt.lower()
        for phrase in _PROHIBITED_PAPER_PHRASES:
            if phrase in lower:
                raise ValueError(
                    f"Poster '{p.title}' prompt contains prohibited phrase: '{phrase}'."
                )
        if not any(term in lower for term in _FULL_BLEED_TERMS):
            raise ValueError(
                f"Poster '{p.title}' prompt missing full-bleed intent."
            )

    ev = plan.evaluation
    for label, swr in [
        ("consistency",       ev.consistency_score),
        ("commercial",        ev.commercial_score),
        ("variation",         ev.variation_score),
        ("brand_identity",    ev.brand_identity_score),
        ("print_collection",  ev.print_collection_score),
        ("market_uniqueness", ev.market_uniqueness_score),
        ("overall",           ev.overall_score),
    ]:
        _validate_score(label, swr.score)
    _validate_score("confidence_score", plan.confidence_score)

    # Cross-object: Bible must not claim wrong poster count
    _NUMBER_WORDS = {
        "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8,
    }
    _COUNT_PATTERN = re.compile(
        r"\b(three|four|five|six|seven|eight|\d+)\s+"
        r"(poster|print|artwork|illustration|piece|image|character)s?\b",
        re.IGNORECASE,
    )
    bible_text = " ".join([
        bible.collection_story,
        " ".join(bible.consistency_rules),
        " ".join(bible.shared_storytelling_rules),
    ])
    for match in _COUNT_PATTERN.finditer(bible_text):
        word = match.group(1).lower()
        claimed = _NUMBER_WORDS.get(word) or (int(word) if word.isdigit() else None)
        if claimed is not None and claimed != size:
            raise ValueError(
                f"Collection Bible claims {claimed} poster(s) "
                f"(found '{match.group()}') but collection_size is {size}."
            )


# ── Public interface ───────────────────────────────────────────────────────────

def generate_collection(
    poster_concept: dict,
    optimized_prompt: str,
    optimized_negative_prompt: str,
    vision_report: VisionReport | None = None,
    collection_size: int | None = None,
    on_usage=None,
) -> CollectionPlan:
    size = _infer_size(poster_concept, collection_size)

    vision_notes = ""
    if vision_report is not None:
        vision_notes = _VISION_NOTES_TEMPLATE.format(
            strengths="\n".join(f"- {s}" for s in vision_report.strengths) or "none noted",
            improvements="\n".join(f"- {s}" for s in vision_report.improvement_suggestions) or "none",
            weaknesses="\n".join(f"- {w}" for w in vision_report.weaknesses) or "none noted",
        )

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set in .env")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    batches = _make_batches(size)
    total_calls = 1 + len(batches) + 1  # bible + poster batches + evaluation
    bible_json: str = ""

    # ── Call 1: Collection Bible ───────────────────────────────────────────────
    print(f"  [1/{total_calls}] Generating Collection Bible...")
    bible_raw = _claude(client, BIBLE_PROMPT.format(
        concept=json.dumps(poster_concept, indent=2),
        image_prompt=optimized_prompt,
        negative_prompt=optimized_negative_prompt,
        vision_notes=vision_notes,
        collection_size=size,
        printable_rules=_PRINTABLE_RULES,
    ), on_usage=on_usage)
    b = bible_raw["collection_bible"]
    bible = CollectionBible(
        collection_name=b["collection_name"],
        collection_story=b["collection_story"],
        brand_identity=b["brand_identity"],
        target_customer=b["target_customer"],
        recommended_room_style=b["recommended_room_style"],
        visual_identity=b["visual_identity"],
        shared_rendering_medium=b["shared_rendering_medium"],
        shared_linework=b["shared_linework"],
        shared_lighting=b["shared_lighting"],
        shared_palette=b["shared_palette"],
        shared_accent_colour_rules=b.get("shared_accent_colour_rules", []),
        shared_camera_angle=b["shared_camera_angle"],
        shared_perspective=b["shared_perspective"],
        shared_atmosphere=b["shared_atmosphere"],
        shared_detail_level=b["shared_detail_level"],
        shared_print_treatment=b["shared_print_treatment"],
        shared_storytelling_rules=b.get("shared_storytelling_rules", []),
        shared_composition_rules=b.get("shared_composition_rules", []),
        shared_style_rules=b.get("shared_style_rules", []),
        shared_negative_prompt=b["shared_negative_prompt"],
        style_dna=b.get("style_dna", []),
        consistency_rules=b.get("consistency_rules", []),
        forbidden_elements=b.get("forbidden_elements", []),
        full_bleed_rules=b.get("full_bleed_rules", []),
    )
    bible_json = json.dumps(b, indent=2)

    # ── Calls 2..N: Poster batches ─────────────────────────────────────────────
    raw_poster_items: list[dict] = []

    for call_num, batch_indexes in enumerate(batches, start=2):
        print(f"  [{call_num}/{total_calls}] Generating posters {batch_indexes}...")
        prompt = POSTER_BATCH_PROMPT.format(
            bible=bible_json,
            image_prompt=optimized_prompt,
            collection_size=size,
            batch_indexes=", ".join(str(i) for i in batch_indexes),
            prior_summaries=_prior_summaries_text(raw_poster_items),
            printable_rules=_PRINTABLE_RULES,
        )
        result = _generate_batch(client, prompt, "posters", batch_indexes, on_usage=on_usage)
        batch_items = result.get("poster_items", [])
        _validate_batch(batch_items, batch_indexes, raw_poster_items, "posters")
        raw_poster_items.extend(batch_items)

    # ── Final call: Evaluation ─────────────────────────────────────────────────
    print(f"  [{total_calls}/{total_calls}] Generating evaluation...")
    poster_summaries = [
        {
            "index":             p["index"],
            "title":             p["title"],
            "subject":           p["subject"],
            "unique_hook":       p["unique_hook"],
            "consistency_notes": p.get("consistency_notes", []),
            "palette_variation": p.get("palette_variation", []),
        }
        for p in raw_poster_items
    ]
    raw_eval = _claude(client, EVAL_ONLY_PROMPT.format(
        bible_summary=json.dumps({
            "collection_name": b["collection_name"],
            "brand_identity":  b["brand_identity"],
            "style_dna":       b.get("style_dna", []),
            "shared_palette":  b["shared_palette"],
        }, indent=2),
        collection_size=size,
        poster_summaries=json.dumps(poster_summaries, indent=2),
    ), on_usage=on_usage)

    # ── Assemble ───────────────────────────────────────────────────────────────
    posters = [
        CollectionPoster(
            index=item["index"],
            title=item["title"],
            subject=item["subject"],
            scene_concept=item["scene_concept"],
            storytelling_focus=item["storytelling_focus"],
            unique_hook=item["unique_hook"],
            image_prompt=item["image_prompt"],
            negative_prompt=item["negative_prompt"],
            aspect_ratio=item["aspect_ratio"],
            focal_point=item["focal_point"],
            foreground_elements=item.get("foreground_elements", []),
            midground_elements=item.get("midground_elements", []),
            background_elements=item.get("background_elements", []),
            palette_variation=item.get("palette_variation", []),
            lighting_variation=item["lighting_variation"],
            weather_or_time_variation=item["weather_or_time_variation"],
            consistency_notes=item.get("consistency_notes", []),
            suggested_etsy_title=item["suggested_etsy_title"],
            suggested_etsy_tags=item["suggested_etsy_tags"],
            mockup_room_style=item["mockup_room_style"],
        )
        for item in raw_poster_items
    ]

    e = raw_eval["evaluation"]
    evaluation = CollectionEvaluation(
        consistency_score=      _swr(e, "consistency_score"),
        commercial_score=       _swr(e, "commercial_score"),
        variation_score=        _swr(e, "variation_score"),
        brand_identity_score=   _swr(e, "brand_identity_score"),
        print_collection_score= _swr(e, "print_collection_score"),
        market_uniqueness_score=_swr(e, "market_uniqueness_score"),
        overall_score=ScoreWithReason(
            score=_compute_overall(e),
            reason="Equal-weighted mean of six evaluation scores.",
        ),
        reasoning=e.get("reasoning", ""),
    )

    plan = CollectionPlan(
        collection_bible=bible,
        collection_size=size,
        poster_items=posters,
        collection_consistency_notes=raw_eval.get("collection_consistency_notes", []),
        evaluation=evaluation,
        confidence_score=raw_eval.get("confidence_score", 0),
    )

    _validate(plan)
    return plan
