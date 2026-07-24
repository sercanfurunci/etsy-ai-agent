"""Audit tests for agent/prompt_library — no network calls."""
import pytest
from agent.prompt_library import build, build_from_collection

AUDIT_CASES = [
    (1, "Tiny orange tiger swimming through giant emerald waves"),
    (1, "Tiny red fox crossing endless patterned cream grass"),
    (2, "Sleeping black cat beneath cascading willow branches"),
    (4, "Two cranes beneath dark pine branches on antique gold"),
    (10, "Two ducks drinking wine at a small dinner table"),
    (12, "Large vermilion sun with a single dark plum blossom branch"),
]


@pytest.fixture(params=AUDIT_CASES, ids=[f"c{c}-{i[:30]}" for c, i in AUDIT_CASES])
def built(request):
    collection_id, item = request.param
    prompt, neg = build_from_collection(collection_id, item)
    return prompt, neg, item


def test_subject_present(built):
    prompt, _, item = built
    # First significant word of the item must appear in the prompt
    first_word = item.split()[0].lower().rstrip(",")
    assert first_word in prompt.lower()


def test_no_duplicate_phrases(built):
    prompt, _, _ = built
    phrases = ["flat front-facing artwork", "print-ready", "no frame", "no mockup"]
    for phrase in phrases:
        assert prompt.lower().count(phrase) <= 1, f"'{phrase}' appears more than once"


def test_flat_print_constraint_present(built):
    prompt, _, _ = built
    assert "flat front-facing artwork" in prompt
    assert "print-ready" in prompt


def test_no_frame_room_wall_mockup(built):
    prompt, _, _ = built
    for term in ("no frame", "no wall", "no room", "no mockup", "no furniture"):
        assert term in prompt


def test_negative_prompt_not_duplicated(built):
    _, neg, _ = built
    # Each term in neg should appear exactly once
    for term in ["frame", "mockup", "watermark"]:
        assert neg.count(term) == 1


def test_generated_typography_disabled(built):
    _, neg, _ = built
    assert "readable english text" in neg.lower() or "gibberish typography" in neg.lower()


def test_negative_space_mentioned(built):
    prompt, _, _ = built
    assert "negative space" in prompt.lower()


def test_printmaking_technique_present(built):
    prompt, _, _ = built
    techniques = [
        "woodblock", "screen-print", "screenprint", "letterpress",
        "linocut", "silkscreen", "ink", "woodcut",
    ]
    assert any(t in prompt.lower() for t in techniques), \
        f"No printmaking technique found in: {prompt[:200]}"


def test_paper_texture_present(built):
    prompt, _, _ = built
    paper_terms = ["paper", "parchment", "washi", "aged", "linen"]
    assert any(t in prompt.lower() for t in paper_terms), \
        f"No paper texture found in: {prompt[:200]}"


def test_color_palette_present(built):
    prompt, _, _ = built
    color_terms = [
        "palette", "cream", "ivory", "gold", "teal", "coral", "charcoal",
        "vermilion", "emerald", "green", "red", "black", "white", "orange",
    ]
    assert any(t in prompt.lower() for t in color_terms), \
        f"No color/palette info found in: {prompt[:200]}"


def test_collection_body_is_specific(built):
    """Collection template part must contain meaningful specific content beyond the item string."""
    from agent.prompt_library import _load
    data = _load()
    prompt, _, item = built
    global_block = data["global_style_block"]
    collection_part = prompt.replace(f", {global_block}", "")
    # Strip the item itself — the remaining template body must be substantial
    template_body = collection_part.replace(item, "").strip(", ")
    assert len(template_body) > 80, (
        f"Collection template body too thin ({len(template_body)} chars after removing item): {template_body!r}"
    )


# Preset build() smoke test
def test_build_preset_returns_strings():
    prompt, neg = build(1)
    assert isinstance(prompt, str) and len(prompt) > 50
    assert isinstance(neg, str) and len(neg) > 50


def test_build_preset_no_duplicate_print_ready():
    prompt, _ = build(1)
    assert prompt.count("print-ready") == 1
