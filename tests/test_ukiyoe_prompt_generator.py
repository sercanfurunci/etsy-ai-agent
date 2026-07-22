"""
Unit tests for the procedural ukiyo-e dataset + generator.
No API calls — everything here is pure dataset/random-selection logic.
"""
import pytest

from agent.ukiyoe_prompt_generator import (
    load_dataset,
    generate_batch,
    HISTORY_PATH,
    _dedup_key,
    _bucket_tag,
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
