"""
Unit tests for the procedural ukiyo-e dataset + generator.
No API calls — everything here is pure dataset/random-selection logic.
"""
import pytest

import json
from unittest.mock import patch

from agent.ukiyoe_prompt_generator import (
    load_dataset,
    generate_batch,
    HISTORY_PATH,
    DATASET_PATH,
    DRAMATIC_MOODS,
    QUIET_MOODS,
    _dedup_key,
    _bucket_tag,
    _validate_batch,
    SEASON_BUCKET_TAGS,
    DAYPART_TAGS,
    OPPOSITE_SEASON,
    OPPOSITE_DAYPART,
)


@pytest.fixture(autouse=True)
def _reset_history():
    """Each test starts from a clean recency/title history file."""
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    yield
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()


def test_dataset_has_seasons_and_times_of_day():
    dataset = load_dataset()

    seasons = dataset["seasons"]
    times_of_day = dataset["timesOfDay"]

    assert 24 <= len(seasons) <= 32
    assert 24 <= len(times_of_day) <= 32

    for entry in seasons:
        bucket = SEASON_BUCKET_TAGS & {t.lower() for t in entry["tags"]}
        assert len(bucket) == 1, f"{entry['id']} must carry exactly one season-bucket tag"

    for entry in times_of_day:
        bucket = DAYPART_TAGS & {t.lower() for t in entry["tags"]}
        assert len(bucket) == 1, f"{entry['id']} must carry exactly one daypart tag"


def test_generate_batch_returns_requested_count_with_required_fields():
    batch = generate_batch(10, seed=1)
    assert len(batch) == 10

    required_fields = {
        "id", "title", "summary", "imagePrompt", "negativePrompt", "genre",
        "subjectId", "subjectRole", "environmentId", "seasonId", "timeOfDayId",
        "weatherId", "lightingId", "moodId", "symbolismId", "compositionId",
        "perspectiveId", "paletteId", "printTechniqueId", "surfaceTextureId",
        "styleId", "nocturnal", "minimalComposition", "titleRetries", "titleFallback",
    }
    for item in batch:
        assert required_fields <= item.keys()
        assert item["seasonId"]
        assert item["timeOfDayId"]


def test_no_incompatible_atmosphere_combinations_across_several_batches():
    dataset = load_dataset()
    seasons_by_id = {e["id"]: e for e in dataset["seasons"]}
    times_by_id = {e["id"]: e for e in dataset["timesOfDay"]}
    weather_by_id = {e["id"]: e for e in dataset["weather"]}
    lighting_by_id = {e["id"]: e for e in dataset["lighting"]}
    genres = dataset["genreProfiles"]

    for seed in range(5):
        for item in generate_batch(10, seed=seed):
            if genres.get(item["genre"], {}).get("supernaturalOverride"):
                continue  # yokai-folklore may intentionally break these rules

            season_bucket = _bucket_tag(seasons_by_id[item["seasonId"]], SEASON_BUCKET_TAGS)
            weather_tags = {t.lower() for t in weather_by_id[item["weatherId"]].get("tags", [])}
            if season_bucket in OPPOSITE_SEASON:
                assert OPPOSITE_SEASON[season_bucket] not in weather_tags, (
                    f"{item['weatherId']} conflicts with season bucket {season_bucket}"
                )

            daypart_bucket = _bucket_tag(times_by_id[item["timeOfDayId"]], DAYPART_TAGS)
            lighting_tags = {t.lower() for t in lighting_by_id[item["lightingId"]].get("tags", [])}
            if daypart_bucket in OPPOSITE_DAYPART:
                assert OPPOSITE_DAYPART[daypart_bucket] not in lighting_tags, (
                    f"{item['lightingId']} conflicts with time-of-day bucket {daypart_bucket}"
                )


def test_titles_are_unique_within_a_batch():
    batch = generate_batch(10, seed=2)
    normalized = [_dedup_key(item["title"]) for item in batch]
    assert len(normalized) == len(set(normalized))


def test_dedup_key_normalizes_punctuation_case_and_articles():
    assert _dedup_key("A Study of Fox") == _dedup_key("study of fox")
    assert _dedup_key("The Fox — At Dusk") == _dedup_key("fox at dusk")


# ── _validate_batch direct unit tests ─────────────────────────────────────────

def _item(**overrides):
    """Minimal valid GeneratedConcept dict for _validate_batch."""
    base = {
        "genre": "landscape",
        "paletteId": "pal-neutral",
        "compositionId": "comp-horizontal",
        "environmentId": "env-mountain",
        "symbolismId": "sym-perseverance",
        "nocturnal": False,
        "minimalComposition": False,
        "subjectRole": "human",
        "moodId": "dramatic",
        "title": "Untitled",
    }
    base.update(overrides)
    return base


def _valid_batch():
    """10 items that satisfy every hard constraint in _validate_batch."""
    roles = ["human", "animal", "architectural"] + ["human"] * 7
    moods = ["dramatic"] + ["tranquil"] * 9
    items = []
    for i in range(10):
        items.append(_item(
            title=f"Title {i}",
            subjectRole=roles[i],
            genre=f"genre-{i}",          # all distinct → no cap hit
            paletteId=f"pal-{i}",        # 10 distinct, no cap hit
            compositionId=f"comp-{i}",   # 10 distinct, no cap hit
            environmentId=f"env-{i}",    # all distinct
            symbolismId=f"sym-{i}",      # all distinct
            moodId=moods[i],
        ))
    return items


def test_validate_batch_passes_on_valid_batch():
    hard, _ = _validate_batch(_valid_batch())
    assert hard == []


def test_validate_batch_detects_duplicate_titles():
    items = _valid_batch()
    items[0]["title"] = items[1]["title"] = "Same Title"
    hard, _ = _validate_batch(items)
    assert any("duplicate" in msg for msg in hard)


def test_validate_batch_detects_missing_required_role():
    items = _valid_batch()
    for item in items:
        item["subjectRole"] = "human"  # remove animal and architectural
    hard, _ = _validate_batch(items)
    assert any("animal" in msg or "architectural" in msg for msg in hard)


def test_validate_batch_caps_genre_frequency():
    items = _valid_batch()
    for item in items:
        item["genre"] = "landscape"  # 10 of the same genre, cap is 2
    hard, _ = _validate_batch(items)
    assert any("genre" in msg for msg in hard)


def test_validate_batch_requires_five_distinct_compositions():
    items = _valid_batch()
    for item in items:
        item["compositionId"] = "comp-only"
    hard, _ = _validate_batch(items)
    assert any("composition" in msg for msg in hard)


# ── History persistence ────────────────────────────────────────────────────────

def test_history_file_written_after_batch():
    assert not HISTORY_PATH.exists()
    generate_batch(5, seed=7)
    assert HISTORY_PATH.exists()
    data = json.loads(HISTORY_PATH.read_text())
    assert "genre" in data or "titleNormalized" in data  # at least one axis tracked


# ── Dataset loading ────────────────────────────────────────────────────────────

def test_load_dataset_raises_for_missing_file(tmp_path):
    import agent.ukiyoe_prompt_generator as mod
    original = mod._dataset_cache
    mod._dataset_cache = None
    try:
        fake_path = tmp_path / "missing.json"
        with patch.object(mod, "DATASET_PATH", fake_path):
            with pytest.raises(FileNotFoundError, match="build_dataset"):
                load_dataset()
    finally:
        mod._dataset_cache = original
