import json
from agent.claude_client import ask

PROMPT_TEMPLATE = """\
You are a market research analyst for a high-volume Etsy printable wall art shop.

Shops like TheWorldGallery and NeuralPrint sell single posters and coordinated sets of 2–3 posters \
across niches such as: vintage travel, Japanese ukiyo-e, botanical prints, animals and cats, \
food and drink, retro and cultural art, minimalist humour, and gallery wall sets.

Analyze the following research data and return ONLY a valid JSON object — no markdown, no commentary, nothing else.

Research data:
{dataset}

Return this exact JSON structure:
{{
  "niche": "string — one-line description of the wall art niche",
  "market_observations": ["string", "..."],
  "recurring_patterns": ["string", "..."],
  "potential_opportunities": ["string", "..."],
  "poster_concepts": [
    {{
      "name": "string — short collection name",
      "niche": "string — specific sub-niche (e.g. Japanese ukiyo-e, vintage botanical)",
      "concept": "string — what the artwork depicts and why it sells",
      "target_customer": "string",
      "art_style": "string — e.g. ukiyo-e woodblock, vintage retro, minimalist line art",
      "subject": "string — the main subject of the artwork",
      "color_palette": "string — 3–5 colours that define the palette",
      "composition": "string — layout and framing description",
      "aspect_ratio": "string — e.g. 2:3, 1:1, 3:2, 4:5",
      "single_or_set": "single | set of 2 | set of 3",
      "set_consistency_notes": "string — shared style/palette/subject rules for a set, or 'n/a' for single",
      "image_generation_prompt": "string — detailed prompt producing ONLY the artwork itself, no frame, no room, no wall, no mockup, no watermark, no UI elements, no text unless required",
      "negative_prompt": "string — what to exclude: frame, border, wall, room, mockup, watermark, text, signature",
      "suggested_etsy_title": "string",
      "suggested_etsy_tags": ["exactly 13 tag strings"],
      "mockup_room_style": "string — room style for the mockup stage only, e.g. Japandi living room, mid-century study"
    }}
  ]
}}

Rules:
- Every poster concept must be expandable into a coherent collection of 5–20 related prints.
- For sets, all prints must share the same visual style, compatible colour palette, consistent composition, and related subjects.
- image_generation_prompt must produce ONLY the flat artwork: no frame, no room, no wall, no mockup, no watermark.
- Do NOT name any specific artist in image_generation_prompt. Describe the historical period and visual technique instead (e.g. "late Edo period woodblock print style" not "Hokusai style").
- Do NOT use contradictory background instructions. If the artwork needs a plain background, specify "cream washi paper background" or "off-white paper texture" — never "no background".
- suggested_etsy_tags must contain EXACTLY 13 tags.
- aspect_ratio must be expressed as a ratio only (e.g. 2:3) — do not add paper size names.
- Return at least 3 poster concepts.
- Output JSON only.
"""


def analyze(products: list[dict]) -> dict:
    dataset_str = json.dumps(products, indent=2)
    prompt = PROMPT_TEMPLATE.format(dataset=dataset_str)
    raw = ask(prompt)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    result = json.loads(raw)

    required = {"niche", "market_observations", "recurring_patterns",
                "potential_opportunities", "poster_concepts"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Response missing fields: {missing}")

    return result
