import base64
import json
from dataclasses import dataclass
from pathlib import Path
import anthropic
from agent.claude_client import ANTHROPIC_API_KEY, MODEL

# Equal weights across the five primary scores
_WEIGHTS = {
    "composition":           0.20,
    "originality":           0.20,
    "commercial_appeal":     0.20,
    "print_quality":         0.20,
    "collection_consistency":0.20,
}

CRITIC_PROMPT = """\
You are a professional art director and Etsy market specialist reviewing AI-generated printable wall art.

You will receive:
1. The intended poster concept (JSON)
2. The optimized image generation prompt used
3. The optimized negative prompt used
4. The generated artwork image

Evaluate the artwork against the concept and return ONLY a valid JSON object — no markdown, no commentary.

Poster concept:
{concept}

Optimized image prompt:
{optimized_prompt}

Optimized negative prompt:
{optimized_negative_prompt}

Every numeric score MUST include a short reason field.

Important rules for retry_recommended:
- Only set retry_recommended = true when an issue is realistically fixable by improving the image generation prompt.
- DO NOT recommend retries for characteristics that are inherent limitations of the image model itself,
  such as: painterly rendering style, soft digital painting look, slight colour bleed, minor texture softness,
  or any other trait that cannot be fixed by changing the prompt.
- If retry_recommended = false, set retry_priority = [] (empty list).
- improvement_suggestions may still be populated when retry_recommended = false — treat them as
  recommendations for future generations, not reasons to retry the current image.

Return this exact JSON structure:
{{
  "composition_score":            {{"score": integer 1-10, "reason": "string — focal point, eye flow, spatial balance, negative space, visual hierarchy"}},
  "originality_score":            {{"score": integer 1-10, "reason": "string — distinctiveness, similarity to known artworks or common Etsy products"}},
  "commercial_appeal_score":      {{"score": integer 1-10, "reason": "string — Etsy thumbnail impact, target audience fit, printable wall art suitability"}},
  "print_quality_score":          {{"score": integer 1-10, "reason": "string — detail clarity, contrast, colour control, print readiness. Do not penalise inherent model rendering style."}},
  "collection_consistency_score": {{"score": integer 1-10, "reason": "string — alignment with intended style, palette, and collection rules from the concept"}},
  "trend_saturation_score":       {{"score": integer 1-10, "reason": "string — how competitive this niche currently is on Etsy; higher score = more saturated"}},
  "market_uniqueness_score":      {{"score": integer 1-10, "reason": "string — how different this artwork is from existing Etsy products in the same niche"}},
  "ip_similarity_risk":           "low | medium | high",
  "ip_similarity_reason":         "string — note any resemblance to copyrighted characters, logos, franchises, or protected artworks",
  "strengths":                    ["string", "..."],
  "weaknesses":                   ["string — do not list model rendering limitations as weaknesses unless the concept specifically required a different rendering style", "..."],
  "improvement_suggestions":      ["string — actionable, for future generations; include even when retry is not recommended", "..."],
  "reasoning":                    "string — one paragraph overall assessment",
  "retry_recommended":            true | false,
  "retry_priority":               ["composition | originality | lighting | colour | prompt_clarity | collection_consistency"],
  "confidence_score":             integer 1-10,
  "commercial_readiness":         integer 1-10,
  "print_readiness":              integer 1-10
}}

Evaluation checklist:
- Composition: focal point, foreground/background depth, visual hierarchy, negative space, balance
- Lighting: direction, mood, consistency, atmospheric quality
- Colour harmony: palette coherence, saturation control, contrast, print-friendliness
- Readability at thumbnail size (critical for Etsy search)
- Suitability as printable wall art (no frames, no rooms, no watermarks)
- Print readiness: detail level, edge clarity, resolution impression
- Originality: visual distinctiveness, avoidance of clichés
- IP risk: protected characters, logos, or obvious copies of specific artworks
- Concept consistency: style, subject, palette, composition rules from the brief
- Emotional impact and storytelling
- Commercial quality for the target Etsy niche
- Trend saturation: competitiveness of this niche on Etsy right now
- Market uniqueness: how distinctive this piece is versus existing Etsy listings

Be direct. Identify real weaknesses. Do not list model rendering traits as fixable issues.
Output JSON only.
"""


@dataclass
class ScoreWithReason:
    score: int
    reason: str


@dataclass
class VisionReport:
    # Five primary sub-scores (used to compute overall)
    composition_score:            ScoreWithReason
    originality_score:            ScoreWithReason
    commercial_appeal_score:      ScoreWithReason
    print_quality_score:          ScoreWithReason
    collection_consistency_score: ScoreWithReason
    # Computed from sub-scores — not generated by Claude
    overall_score:                ScoreWithReason
    # Market scores
    trend_saturation_score:       ScoreWithReason
    market_uniqueness_score:      ScoreWithReason
    # Risk
    ip_similarity_risk:           str
    ip_similarity_reason:         str
    # Qualitative
    strengths:                    list[str]
    weaknesses:                   list[str]
    improvement_suggestions:      list[str]
    reasoning:                    str
    # Retry metadata
    retry_recommended:            bool
    retry_priority:               list[str]  # always [] when retry_recommended is False
    # Readiness
    confidence_score:             int
    commercial_readiness:         int
    print_readiness:              int
    # Pipeline decision — consumed by Stage 13
    final_recommendation:         str  # PROCEED | PROCEED_WITH_IMPROVEMENTS | RETRY


def _compute_overall(scores: dict[str, int]) -> int:
    return round(sum(scores[k] * w for k, w in _WEIGHTS.items()))


def _final_recommendation(retry_recommended: bool, overall: int) -> str:
    if retry_recommended:
        return "RETRY"
    if overall >= 8:
        return "PROCEED"
    return "PROCEED_WITH_IMPROVEMENTS"


def review(
    poster_concept: dict,
    optimized_prompt: str,
    optimized_negative_prompt: str,
    image_path: str,
    on_usage=None,
) -> VisionReport:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set in .env")

    image_bytes = Path(image_path).read_bytes()
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    prompt_text = CRITIC_PROMPT.format(
        concept=json.dumps(poster_concept, indent=2),
        optimized_prompt=optimized_prompt,
        optimized_negative_prompt=optimized_negative_prompt,
    )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt_text},
            ],
        }],
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
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    d = json.loads(raw)

    def _swr(key: str) -> ScoreWithReason:
        v = d[key]
        return ScoreWithReason(score=v["score"], reason=v["reason"])

    comp  = _swr("composition_score")
    orig  = _swr("originality_score")
    comm  = _swr("commercial_appeal_score")
    print_q = _swr("print_quality_score")
    coll  = _swr("collection_consistency_score")

    overall_val = _compute_overall({
        "composition":           comp.score,
        "originality":           orig.score,
        "commercial_appeal":     comm.score,
        "print_quality":         print_q.score,
        "collection_consistency":coll.score,
    })

    retry = bool(d["retry_recommended"])
    # Enforce: no retry → no retry_priority
    retry_priority = d.get("retry_priority", []) if retry else []

    recommendation = _final_recommendation(retry, overall_val)

    return VisionReport(
        composition_score=comp,
        originality_score=orig,
        commercial_appeal_score=comm,
        print_quality_score=print_q,
        collection_consistency_score=coll,
        overall_score=ScoreWithReason(
            score=overall_val,
            reason="Weighted average of composition, originality, commercial appeal, print quality, collection consistency (20% each).",
        ),
        trend_saturation_score=_swr("trend_saturation_score"),
        market_uniqueness_score=_swr("market_uniqueness_score"),
        ip_similarity_risk=d["ip_similarity_risk"],
        ip_similarity_reason=d["ip_similarity_reason"],
        strengths=d["strengths"],
        weaknesses=d["weaknesses"],
        improvement_suggestions=d.get("improvement_suggestions", []),
        reasoning=d["reasoning"],
        retry_recommended=retry,
        retry_priority=retry_priority,
        confidence_score=d["confidence_score"],
        commercial_readiness=d["commercial_readiness"],
        print_readiness=d["print_readiness"],
        final_recommendation=recommendation,
    )
