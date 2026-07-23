import json
from agent.claude_client import ask

# These terms are appended to every optimized negative prompt in Python,
# guaranteeing they are always present regardless of Claude's output.
_NEGATIVE_ADDITIONS = (
    "border, white border, blank border, decorative border, "
    "paper edge, torn paper, deckled edge, deckled border, "
    "blank corners, blank margins, uneven margins, "
    "canvas edge, frame, poster frame, picture frame, "
    "paper texture background, cream paper, watercolor paper surface, textured paper"
)

OPTIMIZER_PROMPT = """\
You are an expert prompt engineer for AI image generation, specialising in printable wall art for Etsy.

You will receive a poster concept, a draft image generation prompt, and a negative prompt.
Your job is to return an improved version of both prompts plus a scoring report.

Poster concept:
{concept}

Draft image generation prompt:
{image_prompt}

Draft negative prompt:
{negative_prompt}

Optimisation goals:
- Increase originality — avoid visual clichés and similarity to well-known existing artworks.
- Remove unnecessary wording and redundant adjectives.
- Improve composition instructions so the AI model understands spatial layout clearly.
- Strengthen style descriptors using period, technique, and medium — never named artists.
- Ensure the prompt produces ONLY flat artwork: no frame, no room, no wall, no mockup, no watermark.
- Preserve the intended artistic style from the concept.
- Make the negative prompt tight and non-contradictory.

Printable wall art rules — apply these to every prompt:
- The artwork must fill the entire canvas edge to edge. Use natural phrasing such as:
  "full bleed composition", "artwork fills the entire frame", "composition extends to all edges",
  "edge-to-edge illustration". Choose the phrasing that fits the style naturally — do not repeat
  all of them mechanically.
- Do NOT describe the artwork as being painted ON paper.
  Describe only the artistic technique and medium, not the physical surface.
  GOOD: "watercolour brushwork", "gouache texture", "ink linework", "hand-painted illustration",
        "painterly cel-shading".
  AVOID: "cream paper surface", "watercolour paper", "textured paper background",
         "painted on washi paper", "paper texture", "deckled paper".
- Painterly aesthetics (watercolour, gouache, ink, cel-shading) are desirable and must be preserved.
  Only remove references that cause the model to render physical paper rather than printable artwork.

Return ONLY a valid JSON object — no markdown, no commentary:
{{
  "optimized_image_prompt": "string",
  "optimized_negative_prompt": "string",
  "optimization_report": {{
    "originality_score": integer 1–10,
    "commercial_appeal_score": integer 1–10,
    "print_quality_score": integer 1–10,
    "collection_consistency_score": integer 1–10,
    "prompt_clarity_score": integer 1–10,
    "ip_risk": "Low | Medium | High",
    "changes_made": ["string", "..."],
    "reasoning": "string — one paragraph"
  }}
}}
"""


def optimize(concept: dict, image_prompt: str, negative_prompt: str, on_usage=None) -> dict:
    prompt = OPTIMIZER_PROMPT.format(
        concept=json.dumps(concept, indent=2),
        image_prompt=image_prompt,
        negative_prompt=negative_prompt,
    )
    raw = ask(prompt, on_usage=on_usage).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    result = json.loads(raw)

    required = {"optimized_image_prompt", "optimized_negative_prompt", "optimization_report"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Optimizer response missing fields: {missing}")

    # Guarantee paper/border/frame exclusions are always present
    existing = result["optimized_negative_prompt"].rstrip(", ")
    result["optimized_negative_prompt"] = f"{existing}, {_NEGATIVE_ADDITIONS}"

    return result
