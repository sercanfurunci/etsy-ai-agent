"""Procedural ukiyo-e prompt generator driven by data/ukiyoe_dataset.json.

No LLM calls — every prompt is assembled from the curated dataset using
genre-profile selection, tag-based compatibility, weighted history-aware
randomness, and batch-level diversity validation. See docs/ukiyoe-dataset.md.

Selection order: genre -> style -> subject -> environment -> season ->
time_of_day -> weather -> lighting -> composition -> perspective -> movement
-> palette -> mood -> symbolism -> print_technique -> surface_texture.
Season/time_of_day/weather/lighting are compatibility-checked together (see
_gather_bias / the hard bucket exclusion in generate_one).
"""
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import TypedDict

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "ukiyoe_dataset.json"
HISTORY_PATH = Path("outputs/.ukiyoe_prompt_history.json")

NEGATIVE_PROMPT = (
    "frame, border, white border, blank border, decorative border, paper edge, torn paper, "
    "deckled edge, canvas edge, poster frame, picture frame, wall, room, mockup, watermark, "
    "signature, text, UI elements, photorealism, 3D render, modern clothing, modern buildings, "
    "typography, logos, glossy digital painting, anime aesthetics, malformed anatomy, "
    "duplicated animals, excessive gradients, plastic texture, western oil painting style"
)

DRAMATIC_MOODS = {"dramatic", "powerful", "fierce", "stormy", "triumphant", "defiant"}
QUIET_MOODS = {"quiet", "tranquil", "serene", "contemplative", "peaceful", "meditative"}
NIGHT_LIGHTING_TAGS = {"night"}

# Season/daypart bucket vocabulary — must match scripts/build_dataset.py's
# SEASONS/TIMES_OF_DAY tagging (validate_dataset.py enforces exactly one
# bucket tag per entry in those two categories).
SEASON_BUCKET_TAGS = {"spring", "summer", "rainy-season", "autumn", "winter"}
DAYPART_TAGS = {"day", "night"}
OPPOSITE_SEASON = {"winter": "summer", "summer": "winter"}
OPPOSITE_DAYPART = {"day": "night", "night": "day"}

DEFAULT_TITLE_HISTORY_WINDOW = 20000
TITLE_STOPWORDS = {
    "a", "an", "the", "of", "and", "at", "in", "on", "under", "over", "hour", "study",
    "passing", "before", "during", "with", "through", "to", "for",
}


class GeneratedConcept(TypedDict):
    id: str
    title: str
    summary: str
    imagePrompt: str
    negativePrompt: str
    genre: str
    subjectId: str
    subjectRole: str
    environmentId: str
    seasonId: str
    timeOfDayId: str
    weatherId: str
    lightingId: str
    moodId: str
    symbolismId: str
    compositionId: str
    perspectiveId: str
    paletteId: str
    printTechniqueId: str
    surfaceTextureId: str
    styleId: str
    tags: list
    nocturnal: bool
    minimalComposition: bool
    titleRetries: int
    titleFallback: bool


# ──────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────

_dataset_cache: dict | None = None


def load_dataset() -> dict:
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found — run `python3 scripts/build_dataset.py` first."
        )
    _dataset_cache = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return _dataset_cache


def _merge_subjects(dataset: dict) -> list[dict]:
    merged = []
    animals = dataset["subjects"]["animals"]
    for sub in animals.values():
        for e in sub:
            merged.append({**e, "role": "animal"})
    people = dataset["subjects"]["people"]
    for sub in people.values():
        for e in sub:
            merged.append({**e, "role": "human"})
    myth = dataset["subjects"]["mythology"]
    for sub in myth.values():
        for e in sub:
            merged.append({**e, "role": "mythology"})
    for e in dataset["subjects"]["architecture"]:
        merged.append({**e, "role": "architectural"})
    for e in dataset["subjects"]["objects"]:
        merged.append({**e, "role": "object"})
    return merged


# ──────────────────────────────────────────────────────────────────────────
# History (recency-decay weighting), persisted across runs
# ──────────────────────────────────────────────────────────────────────────

def _load_history() -> dict:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_history(history: dict, window: int, title_window: int) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    trimmed = {}
    for key, values in history.items():
        # Title dedup keys use a much larger cap than the other axes: several of
        # the title templates only combine two low-cardinality fields (e.g.
        # time_of_day x environment = ~2,100 possible strings), so a small
        # window "forgets" them long before their pool is exhausted, causing
        # legitimate-looking repeats to reappear at scale. A large cap keeps
        # near-complete memory across a normal run without growing unbounded.
        w = title_window if key.startswith("title") else window
        trimmed[key] = values[-w:]
    HISTORY_PATH.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────
# Weighted selection with tag compatibility + cooldown-group decay + fallback
# ──────────────────────────────────────────────────────────────────────────

def _weighted_pick(
    entries: list[dict],
    recent_ids: list[str],
    window: int,
    boost_tags: set[str] | None = None,
    avoid_tags: set[str] | None = None,
) -> dict:
    id_counts = Counter(recent_ids[-window:])
    cg_counts = Counter(
        e.get("cooldownGroup")
        for eid in recent_ids[-window:]
        for e in entries
        if e["id"] == eid and e.get("cooldownGroup")
    )

    def score(e: dict) -> float | None:
        tags = {t.lower() for t in e.get("tags", [])}
        if avoid_tags and tags & avoid_tags:
            return None
        weight = e.get("weight", 1.0)
        if boost_tags and tags & boost_tags:
            weight *= 1.6
        weight *= 1.0 / (1.0 + id_counts.get(e["id"], 0) * 4)
        cg = e.get("cooldownGroup")
        if cg:
            weight *= 1.0 / (1.0 + cg_counts.get(cg, 0) * 2)
        return max(weight, 0.001)

    scored = [(e, score(e)) for e in entries]
    candidates = [(e, w) for e, w in scored if w is not None]
    if not candidates:
        # Fallback: filters left nothing — ignore tag constraints, keep recency decay.
        candidates = [(e, max(e.get("weight", 1.0) / (1.0 + id_counts.get(e["id"], 0) * 4), 0.001))
                      for e in entries]
    pool, weights = zip(*candidates)
    return random.choices(pool, weights=weights, k=1)[0]


def _gather_bias(subject: dict, dataset: dict, prefer_field: str, avoid_field: str | None = None) -> tuple[set[str], set[str]]:
    """Collects boost/avoid tag sets from compatibilityRules matching the subject's id/tags."""
    boost, avoid = set(), set()
    subject_tags = {t.lower() for t in subject.get("tags", [])}
    for rule in dataset.get("compatibilityRules", []):
        matches = (rule.get("ifId") == subject["id"]) or (rule.get("ifTag", "").lower() in subject_tags)
        if not matches:
            continue
        boost.update(t.lower() for t in rule.get(prefer_field, []))
        if avoid_field:
            avoid.update(t.lower() for t in rule.get(avoid_field, []))
    return boost, avoid


def _bucket_tag(entry: dict, bucket_vocab: set[str]) -> str | None:
    tags = {t.lower() for t in entry.get("tags", [])}
    hit = tags & bucket_vocab
    return next(iter(hit)) if hit else None


# ──────────────────────────────────────────────────────────────────────────
# Titles — normalization, dedup, cooldowns, deterministic fallback
# ──────────────────────────────────────────────────────────────────────────

def _title_case(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _normalize_title(title: str) -> str:
    """lowercase -> normalize unicode apostrophes/dashes -> strip punctuation -> collapse whitespace."""
    text = unicodedata.normalize("NFKD", title.lower())
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("—", "-").replace("–", "-").replace("‒", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _dedup_key(title: str) -> str:
    """Normalized form used for duplicate comparison — also drops a leading article."""
    normalized = _normalize_title(title)
    return re.sub(r"^(a|an|the)\s+", "", normalized)


def _significant_tokens(title: str) -> set[str]:
    return {t for t in _normalize_title(title).split() if len(t) > 2 and t not in TITLE_STOPWORDS}


TITLE_TEMPLATES = [
    lambda s, m, sym, e, w, se, t: f"{_title_case(s)} at {_title_case(e)}",
    lambda s, m, sym, e, w, se, t: f"The {_title_case(m)} {_title_case(s)}",
    lambda s, m, sym, e, w, se, t: f"{_title_case(sym)} and the {_title_case(s)}",
    lambda s, m, sym, e, w, se, t: f"{_title_case(e)}, {_title_case(m)}",
    lambda s, m, sym, e, w, se, t: f"A Study of {_title_case(s)} in {_title_case(w)}",
    lambda s, m, sym, e, w, se, t: f"{_title_case(s)} — {_title_case(sym)}",
    lambda s, m, sym, e, w, se, t: f"Passing {_title_case(e)} Under {_title_case(w)}",
    lambda s, m, sym, e, w, se, t: f"{_title_case(m)} Hour at {_title_case(e)}",
    lambda s, m, sym, e, w, se, t: f"{_title_case(w)} Over {_title_case(e)}",
    lambda s, m, sym, e, w, se, t: f"{_title_case(s)} in {_title_case(w)}",
    lambda s, m, sym, e, w, se, t: f"{_title_case(se)}: {_title_case(s)}",
    lambda s, m, sym, e, w, se, t: f"{_title_case(t)} at {_title_case(e)}",
]

MAX_TITLE_RETRIES = 24
TOKEN_STALE_THRESHOLD = 3


def _make_title(
    subject: dict, mood: dict, symbolism: dict, environment: dict, weather: dict,
    season: dict, time_of_day: dict, rng: random.Random, history: dict,
    title_window: int, cooldown_window: int,
) -> tuple[str, int, bool]:
    """Returns (title, retries_used, used_fallback).

    title_window (large, e.g. 20000) guards against exact/near-duplicate titles
    over a long run. cooldown_window (small, e.g. 50 — same as the other axes'
    recency window) softly discourages reusing the same (subject, template) or
    (environment, template) pairing or stale wording back-to-back — it must NOT
    share the large window, or the fixed-size pools of subject/env x template
    combinations exhaust quickly and force constant fallback (see docs).
    """
    subject_label, mood_label = subject["label"], mood["label"]
    symbolism_label, env_label = symbolism["label"], environment["label"]
    weather_label, time_label = weather["label"], time_of_day["label"]

    recent_normalized = set(history.get("titleNormalized", [])[-title_window:])
    recent_subj_template = set(history.get("titleSubjectTemplate", [])[-cooldown_window:])
    recent_env_template = set(history.get("titleEnvTemplate", [])[-cooldown_window:])
    token_counts = Counter(history.get("titleTokens", [])[-cooldown_window:])

    order = list(range(len(TITLE_TEMPLATES)))
    rng.shuffle(order)

    retries = 0
    candidate = None
    for attempt in range(MAX_TITLE_RETRIES):
        template_index = order[attempt % len(order)]
        candidate = TITLE_TEMPLATES[template_index](
            subject_label, mood_label, symbolism_label, env_label, weather_label,
            season["label"], time_label,
        )
        dedup_key = _dedup_key(candidate)
        subj_pair = f"{subject['id']}::{template_index}"
        env_pair = f"{environment['id']}::{template_index}"

        if dedup_key in recent_normalized:
            retries += 1
            continue
        if subj_pair in recent_subj_template:
            retries += 1
            continue
        if env_pair in recent_env_template:
            retries += 1
            continue

        tokens = _significant_tokens(candidate)
        if tokens and all(token_counts.get(tok, 0) >= TOKEN_STALE_THRESHOLD for tok in tokens):
            retries += 1
            continue

        # Accepted.
        history.setdefault("titleNormalized", []).append(dedup_key)
        history.setdefault("titleSubjectTemplate", []).append(subj_pair)
        history.setdefault("titleEnvTemplate", []).append(env_pair)
        history.setdefault("titleTokens", []).extend(tokens)
        return candidate, retries, False

    # Deterministic fallback — guaranteed to terminate and guaranteed unique.
    base = f"{_title_case(subject_label)} — {_title_case(env_label)} ({_title_case(season['label'])})"
    fallback = base
    suffix = 2
    while _dedup_key(fallback) in recent_normalized and suffix < 1000:
        fallback = f"{base} #{suffix}"
        suffix += 1

    dedup_key = _dedup_key(fallback)
    history.setdefault("titleNormalized", []).append(dedup_key)
    history.setdefault("titleSubjectTemplate", []).append(f"{subject['id']}::fallback")
    history.setdefault("titleEnvTemplate", []).append(f"{environment['id']}::fallback")
    history.setdefault("titleTokens", []).extend(_significant_tokens(fallback))
    return fallback, retries, True


# ──────────────────────────────────────────────────────────────────────────
# Prompt assembly (varied sentence structure, ~70-130 words)
# ──────────────────────────────────────────────────────────────────────────

def _assemble_prompt(parts: dict, rng: random.Random) -> str:
    templates = [
        (
            "{subject}, {environment}, during {season}. {time_of_day}, with {weather} at {lighting}. "
            "Composed as {composition}, rendered from {perspective}, with {movement}. The mood is "
            "{mood}, evoking symbolism of {symbolism}. {style} Coloured in {palette}, using "
            "{print_technique}, on {surface}."
        ),
        (
            "{environment}, where {subject} appears, in {season}. {time_of_day}; {weather}, {lighting}. "
            "Framed with {composition}, seen from {perspective}; {movement} moves through the scene. "
            "A {mood} atmosphere carries a sense of {symbolism}. {style} Rendered in {palette}, "
            "{print_technique}, on {surface}."
        ),
        (
            "{style} {subject}, set within {environment} during {season}. {time_of_day}, {weather} "
            "under {lighting}, composed as {composition}, viewed from {perspective}, with {movement}. "
            "{mood_cap} in feeling, symbolising {symbolism}. Coloured in {palette}, finished with "
            "{print_technique}, on {surface}."
        ),
        (
            "A quiet study: {subject}, {environment}, {season}. {time_of_day} — {weather}, {lighting}, "
            "{composition}, {perspective}. {movement_cap}. The overall mood is {mood}, touching on "
            "{symbolism}. {style} {palette}, using {print_technique}, on {surface}."
        ),
    ]
    template = rng.choice(templates)
    return template.format(
        **parts,
        mood_cap=parts["mood"][0].upper() + parts["mood"][1:],
        movement_cap=parts["movement"][0].upper() + parts["movement"][1:],
    )


# ──────────────────────────────────────────────────────────────────────────
# Single-item generation
# ──────────────────────────────────────────────────────────────────────────

def _pick_genre(dataset: dict, history: dict, window: int) -> tuple[str, dict]:
    genres = dataset["genreProfiles"]
    ids = list(genres.keys())
    recent = history.get("genre", [])[-window:]
    counts = Counter(recent)
    weights = [max(genres[g].get("weight", 1.0) / (1.0 + counts.get(g, 0) * 3), 0.001) for g in ids]
    genre_id = random.choices(ids, weights=weights, k=1)[0]
    return genre_id, genres[genre_id]


def generate_one(dataset: dict, history: dict, window: int, title_window: int, rng: random.Random) -> GeneratedConcept:
    subjects = _merge_subjects(dataset)

    # 1. genre
    genre_id, profile = _pick_genre(dataset, history, window)
    supernatural = bool(profile.get("supernaturalOverride"))

    # 2. style (only depends on genre)
    styles_tagged = [{**s, "tags": s.get("compatibleGenres", [])} for s in dataset["globalStyles"]]
    style = _weighted_pick(styles_tagged, history.get("style", []), window, {genre_id.lower()}, None)

    # 3. subject
    subj_boost = {t.lower() for t in profile.get("preferredSubjectTags", [])}
    subj_avoid = {t.lower() for t in profile.get("disallowedTags", [])}
    subject = _weighted_pick(subjects, history.get("subject", []), window, subj_boost, subj_avoid)

    # 4. environment
    env_boost, env_avoid = _gather_bias(subject, dataset, "preferEnvironmentTags", "avoidEnvironmentTags")
    environment = _weighted_pick(dataset["environments"], history.get("environment", []), window,
                                  env_boost, env_avoid)

    # 5. season
    season_boost, _ = _gather_bias(subject, dataset, "preferSeasonTags")
    season_boost |= {t.lower() for t in profile.get("preferredSeasonTags", [])}
    season = _weighted_pick(dataset["seasons"], history.get("season", []), window, season_boost, None)
    season_bucket = _bucket_tag(season, SEASON_BUCKET_TAGS)

    # 6. time_of_day
    time_boost, _ = _gather_bias(subject, dataset, "preferTimeTags")
    time_boost |= {t.lower() for t in profile.get("preferredTimeTags", [])}
    time_of_day = _weighted_pick(dataset["timesOfDay"], history.get("timeOfDay", []), window, time_boost, None)
    daypart_bucket = _bucket_tag(time_of_day, DAYPART_TAGS)

    # 7. weather — hard-excludes the opposite season bucket unless supernaturalOverride
    weather_avoid = set()
    if not supernatural and season_bucket in OPPOSITE_SEASON:
        weather_avoid = {OPPOSITE_SEASON[season_bucket]}
    weather_boost = {season_bucket} if season_bucket else set()
    weather = _weighted_pick(dataset["weather"], history.get("weather", []), window,
                              weather_boost, weather_avoid)

    # 8. lighting — hard-excludes the opposite daypart bucket unless supernaturalOverride
    lighting_avoid = set()
    if not supernatural and daypart_bucket in OPPOSITE_DAYPART:
        lighting_avoid = {OPPOSITE_DAYPART[daypart_bucket]}
    lighting_boost = {daypart_bucket} if daypart_bucket else set()
    lighting = _weighted_pick(dataset["lighting"], history.get("lighting", []), window,
                               lighting_boost, lighting_avoid)

    # 9. composition
    comp_avoid = _gather_bias(subject, dataset, "avoidCompositionTags")[0]
    comp_boost = {t.lower() for t in profile.get("preferredCompositions", [])}
    composition = _weighted_pick(dataset["compositions"], history.get("composition", []), window,
                                  comp_boost, comp_avoid)

    # 10. perspective
    perspective = _weighted_pick(dataset["perspectives"], history.get("perspective", []), window)

    # 11. movement
    movement = _weighted_pick(dataset["movement"], history.get("movement", []), window)

    # 12. palette
    palette_boost = {t.lower() for t in profile.get("preferredPalettes", [])}
    palette = _weighted_pick(dataset["palettes"], history.get("palette", []), window, palette_boost, None)

    # 13. mood
    mood_boost = {t.lower() for t in profile.get("preferredMoods", [])}
    mood = _weighted_pick(dataset["moods"], history.get("mood", []), window, mood_boost, None)

    # 14. symbolism
    symbolism = _weighted_pick(dataset["symbolism"], history.get("symbolism", []), window)

    # 15. print_technique
    print_technique = _weighted_pick(dataset["printTechniques"], history.get("printTechnique", []), window)

    # 16. surface_texture
    surface = _weighted_pick(dataset["surfaceTextures"], history.get("surfaceTexture", []), window)

    for key, e in (
        ("genre", {"id": genre_id}), ("style", style), ("subject", subject), ("environment", environment),
        ("season", season), ("timeOfDay", time_of_day), ("weather", weather), ("lighting", lighting),
        ("composition", composition), ("perspective", perspective), ("movement", movement),
        ("palette", palette), ("mood", mood), ("symbolism", symbolism),
        ("printTechnique", print_technique), ("surfaceTexture", surface),
    ):
        history.setdefault(key, []).append(e["id"])

    lighting_tags = {t.lower() for t in lighting.get("tags", [])}
    mood_tags_eerie = mood["id"] in {"eerie", "haunted", "mysterious"}
    nocturnal = bool(lighting_tags & NIGHT_LIGHTING_TAGS) or mood_tags_eerie
    minimal_composition = "minimal" in {t.lower() for t in composition.get("tags", [])}

    title, title_retries, title_fallback = _make_title(
        subject, mood, symbolism, environment, weather, season, time_of_day, rng, history,
        title_window, window,
    )

    image_prompt = _assemble_prompt({
        "subject": subject["promptText"],
        "environment": environment["promptText"],
        "season": season["promptText"],
        "time_of_day": _title_case(time_of_day["promptText"]),
        "weather": weather["promptText"],
        "lighting": lighting["promptText"],
        "composition": composition["promptText"],
        "perspective": perspective["promptText"],
        "movement": movement["promptText"],
        "mood": mood["label"],
        "symbolism": symbolism["label"],
        "style": style["promptText"],
        "palette": palette["promptText"],
        "print_technique": print_technique["promptText"],
        "surface": surface["promptText"],
    }, rng)

    return {
        "id": f"{subject['id']}-{environment['id']}-{rng.randint(1000, 9999)}",
        "title": title,
        "summary": f"{subject['label']} at {environment['label']}, {season['label']}, {mood['label']} in mood.",
        "imagePrompt": image_prompt,
        "negativePrompt": NEGATIVE_PROMPT,
        "genre": genre_id,
        "subjectId": subject["id"],
        "subjectRole": subject["role"],
        "environmentId": environment["id"],
        "seasonId": season["id"],
        "timeOfDayId": time_of_day["id"],
        "weatherId": weather["id"],
        "lightingId": lighting["id"],
        "moodId": mood["id"],
        "symbolismId": symbolism["id"],
        "compositionId": composition["id"],
        "perspectiveId": perspective["id"],
        "paletteId": palette["id"],
        "printTechniqueId": print_technique["id"],
        "surfaceTextureId": surface["id"],
        "styleId": style["id"],
        "tags": subject.get("tags", []),
        "nocturnal": nocturnal,
        "minimalComposition": minimal_composition,
        "titleRetries": title_retries,
        "titleFallback": title_fallback,
    }


# ──────────────────────────────────────────────────────────────────────────
# Batch generation with diversity validation + retry
# ──────────────────────────────────────────────────────────────────────────

def _validate_batch(items: list[GeneratedConcept]) -> tuple[list[str], list[str]]:
    """Returns (hard_problems, soft_problems). Only hard problems trigger a retry —
    soft ones are statistically hard to guarantee given the current dataset's tag
    distribution, so they're reported but not retried against (see docs)."""
    hard, soft = [], []

    def cap(field, limit, bucket):
        counts = Counter(item[field] for item in items)
        offenders = [k for k, v in counts.items() if v > limit]
        if offenders:
            bucket.append(f"{field} exceeds cap of {limit}: {offenders}")

    cap("genre", 2, hard)
    cap("paletteId", 2, hard)
    cap("compositionId", 2, hard)
    cap("environmentId", 2, hard)
    cap("symbolismId", 2, hard)

    nocturnal = sum(1 for item in items if item["nocturnal"])
    if nocturnal > 2:
        soft.append(f"too many nocturnal-feeling concepts: {nocturnal} > 2")

    minimal = sum(1 for item in items if item["minimalComposition"])
    if minimal > 2:
        hard.append(f"too many minimalist compositions: {minimal} > 2")

    roles = {item["subjectRole"] for item in items}
    for required_role in ("human", "animal", "architectural"):
        if required_role not in roles:
            hard.append(f"missing required role: {required_role}")

    moods = {item["moodId"] for item in items}
    if not (moods & DRAMATIC_MOODS):
        soft.append("missing a dramatic-mood concept")
    if not (moods & QUIET_MOODS):
        soft.append("missing a quiet-mood concept")

    if len({item["compositionId"] for item in items}) < 5:
        hard.append("fewer than 5 distinct composition families")
    if len({item["paletteId"] for item in items}) < 5:
        hard.append("fewer than 5 distinct palette families")

    dedup_seen = Counter(_dedup_key(item["title"]) for item in items)
    dupes = [t for t, c in dedup_seen.items() if c > 1]
    if dupes:
        hard.append(f"duplicate normalized titles: {dupes}")

    return hard, soft


def generate_batch(count: int = 10, max_attempts: int = 6, seed: int | None = None) -> list[GeneratedConcept]:
    dataset = load_dataset()
    settings = dataset.get("generationSettings", {})
    window = settings.get("historyWindow", 50)
    title_window = settings.get("titleHistoryWindow", DEFAULT_TITLE_HISTORY_WINDOW)
    rng = random.Random(seed) if seed is not None else random

    best_batch: list[GeneratedConcept] = []
    best_history: dict = {}
    best_hard: list[str] = None
    best_soft: list[str] = []

    for attempt in range(1, max_attempts + 1):
        history = _load_history()
        batch = [generate_one(dataset, history, window, title_window, rng) for _ in range(count)]
        hard, soft = _validate_batch(batch)
        if not hard:
            _save_history(history, window, title_window)
            if soft:
                print(f"[info] batch generated with {len(soft)} soft diversity note(s):")
                for p in soft:
                    print(f"  - {p}")
            return batch
        if best_hard is None or len(hard) < len(best_hard):
            best_batch, best_history, best_hard, best_soft = batch, history, hard, soft

    # Exhausted retries — persist history from the best attempt and report the gap.
    _save_history(best_history, window, title_window)
    print(f"[warn] batch diversity constraints not fully met after {max_attempts} attempts:")
    for p in best_hard + best_soft:
        print(f"  - {p}")
    return best_batch


def generate_single(seed: int | None = None) -> GeneratedConcept:
    return generate_batch(count=1, max_attempts=1, seed=seed)[0]
