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
- Fix all visual logic errors silently — scan the prompt for the following classes of issues and add corrective spatial/anatomical language without mentioning you changed anything:
  • Object-body overlap: held/worn objects must not obscure or merge with ears, head, face, limbs unless intentional. Add spatial anchors ("lantern held at side below eye level", "umbrella grip at waist height").
  • Limb & digit logic: if hands or paws are visible, anchor finger/toe count implicitly by describing the grip or pose ("one paw curled around the stem"). Never describe floating or detached limbs.
  • Scale coherence: objects a character holds must be proportional to their body. If a prop seems ambiguously sized, anchor it ("small paper lantern, fits in one paw").
  • Clothing & accessory physics: hats sit on the head, scarves drape around the neck, glasses rest on the nose — add these anchors if accessories are mentioned without positional grounding.
  • Gravity & physics: objects rest on surfaces unless explicitly floating. If something should be suspended, say so explicitly.
  • Occlusion logic: if two subjects overlap in space, specify which is in front ("fox standing before the torii gate, gate visible behind").
  • Light source consistency: if a light source is named (lantern, moon, window), ensure the prompt does not also describe contradictory lighting directions. Remove the contradiction.
  • Celestial uniqueness: there can only be one sun and one moon in a scene — never two suns, never two moons. This is a hard constraint. Scan the prompt for any phrasing that could produce multiple celestial bodies (dual sunset, twin suns, reflected sun treated as second sun). Reduce unconditionally to one and anchor its position explicitly ("a single setting sun, centre-right horizon", "a single full moon, upper right"). Add "only one sun" or "only one moon" as a direct instruction in the output prompt if the scene includes sky.
  • Shadow-light coherence: if a light source position is stated or implied (e.g. "morning sun from the left"), shadows must fall in the opposite direction. Add "shadows falling to the right" or equivalent to lock this. If no direction is given but shadows are mentioned, add a consistent light source.
  • Surface physics: characters cannot stand on liquid water unless it is frozen, shallow enough to wade, or the prompt explicitly invokes a supernatural/mythological context. If a character is described near water, anchor their position ("standing at the water's edge", "on a wooden dock", "on a stone stepping stone"). Remove ambiguous phrasing that implies standing on open water surface.
  • Reflection logic: if a reflective surface (water, mirror, polished floor) is present and a character or light source is described above it, add a brief reflection anchor ("soft reflection on the water below") so the model renders it consistently and doesn't omit or double the subject.
  • Weather coherence: clear sky and heavy rain cannot coexist. Sunset and dense fog reduce visibility — do not pair them with "crisp detail in the distance". Resolve contradictions toward the dominant mood.
  • Depth & atmosphere: if foreground and background subjects are both described with equal sharpness/saturation, add atmospheric perspective cues ("background elements slightly faded, muted in tone") to prevent a flat, confusing depth plane.
  • Tail, hair, fur blending: long appendages (tails, hair, sleeves) must not be described in a way that causes them to merge into the background or another figure. Add "clearly defined against background" if needed.
  • Character count: if the prompt implies a specific number of characters, add the count explicitly ("two ducks", "a single cat") to prevent the model from adding or removing figures.
  • Weapon and tool placement: swords, tools, and similar objects must rest in physically plausible positions — leaning against a post or wall, on a dedicated stand (katana-kake), sheathed at a figure's side, or laid flat on a horizontal surface with full contact. Never describe or allow a weapon to float diagonally across a railing, fence, or architectural element without full support underneath. If the prompt places a weapon on a thin surface (rail, branch, rope), replace with a stable alternative ("resting against the shrine post", "laid flat on a stone step").
  • Object intersection: solid objects cannot pass through each other. A tree branch cannot grow through a musical instrument; a fence post cannot intersect a body; a sword cannot clip through a surface. If two objects are described near each other and their prompts would cause spatial overlap, add separating language ("branch arching over the koto, not touching it", "figure standing beside the torii, gate behind them").
  • Indoor object outdoors: delicate indoor objects (koto, shamisen, tea-ceremony ware, calligraphy scrolls, folding screens) must not be placed directly outdoors in rain, snow, or open fields without a clear protective context. If the scene is exterior, either move the object to a covered veranda, engawa, or interior window view, or remove it entirely.
  • Suspended object attachment: every hanging object (paper lantern, banner, wind chime, rope) must have an explicit or implied attachment point — eaves, a branch, a post, a rope tied above. If the prompt implies hanging lanterns without naming their support, add "hanging from [eaves/branch/rope above]". Never render lanterns as floating freely in open sky unless the scene is explicitly fantastical or ceremonial with floating elements named as such.
  • Object scale anchored to landscape: a small decorative object (incense burner, ceramic vessel, stone lantern) placed in an open landscape must be anchored to human or architectural scale ("small bronze incense burner on a stone pedestal, waist-height, at the path's edge"). Without a scale anchor, the model may render it as a colossal presence dominating the landscape.
  • Water flow direction: moving water (river currents, tidal flow, rapids, rain puddles) must be described with directional flow language ("river current flowing toward lower right", "tide pulling seaward to the left", "ripples spreading outward from impact point"). Avoid vague "swirling" or "swirling patterns" without a direction — this causes the model to render abstract decorative spiral circles disconnected from natural water physics. Always add a flow direction vector.
  • Fish and wildlife in water: if multiple fish appear in a river or sea, they must swim in a consistent direction or natural loose school formation — never in decorative circular or spiral patterns. Add "fish swimming downstream toward lower right" or equivalent directional anchor. A single fish jumping or surfacing is fine; scattered fish orbiting in circles is not.
  • Fish body position: a fish in water is submerged with only the upper back and fin breaking the surface — it does not rest flat on top of the water surface like an object on a table. If a fish is described as surfacing or leaping, it must be mid-arc or partially breaking the surface ("salmon leaping upward, body arcing above the surface, tail still in water"). Never render a fish lying horizontally on top of still water.
  • Bird flock silhouettes: birds in flight shown as background flock must be rendered as recognisable bird shapes in loose V-formation or staggered diagonal line — not as scattered random black marks, bat-like shapes, or dense clouds of dots. Add "birds in loose V-formation" or "staggered diagonal flight silhouettes" to anchor the flock's visual form. If only a few birds (2–5), describe each individually ("three cranes in descending diagonal").
  • Lantern quantity and placement: limit lanterns in a scene to the minimum needed for the narrative — one held lantern per figure, one hanging lantern per structural attachment point (eave, branch, post). Do not allow lanterns to multiply freely. If the prompt implies many lanterns, specify an exact small number ("two paper lanterns hanging from the eave", "a single stone tōrō at the path entrance"). A lone lantern resting on an open rock or ground with no attachment context reads as floating — replace with a stone tōrō (traditional lantern pedestal) if a ground light source is needed.
  • Held lantern grip: if a figure carries a lantern, it must be explicitly attached to their hand — "held in the right hand, arm hanging at the figure's side, lantern at knee height" or "gripped by the handle at waist level". Never allow a lantern to float near a figure without contact. The lantern cord or handle must connect visibly to the hand or a carried pole.
  • Scene environment coherence: a single composition must belong to ONE environment type. Ocean waves, rice paddies, riverside docks, mountain forests, and city streets are distinct environments — they cannot coexist in the same scene. If the prompt mixes incompatible environments, reduce to the single most dominant one and remove all elements that belong to a different environment type.
  • Reflection vs shadow distinction: in rainy scenes or near reflective water, reflections appear directly below the subject, vertically mirrored — they are NOT cast diagonally like shadows. If a reflection is described or implied (rain puddles, river, wet ground), add "reflection directly below, vertically mirrored" to anchor the correct rendering. Never describe a reflection as if it were a cast shadow, and never describe a cast shadow as if it were a reflection.

Printable wall art rules — apply these to every prompt:
- The artwork must fill the entire canvas edge to edge, including all four corners — no blank or cream paper corners, no paper margin, no vignette border. Use natural phrasing such as:
  "full bleed composition", "artwork fills the entire frame", "composition extends to all edges and corners",
  "edge-to-edge illustration, background color fills every corner". Choose the phrasing that fits the style naturally — do not repeat
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
