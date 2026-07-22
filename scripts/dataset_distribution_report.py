"""Generates many batches locally (no API calls) and reports distribution stats.

Run: python3 scripts/dataset_distribution_report.py [num_batches]

By default runs a 1,000-concept pass (100 batches) followed by a 10,000-concept
stress pass (1,000 batches), reporting frequency distributions, exact and
normalized duplicate-title counts, duplicate subject/environment/season/time
combinations, season/time-of-day distribution, incompatible atmosphere
combinations (should always be zero — enforced by the generator itself, this
just double-checks), and title retry/fallback counts.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.ukiyoe_prompt_generator import (  # noqa: E402
    generate_batch, HISTORY_PATH, _dedup_key, _bucket_tag,
    SEASON_BUCKET_TAGS, DAYPART_TAGS, OPPOSITE_SEASON, OPPOSITE_DAYPART, load_dataset,
)

INDIGO_PALETTE_IDS = {"aizuri-e-blue-monochrome", "dark-nocturne-palette", "midnight-indigo-with-warm-ochre"}


def _is_incompatible(item: dict, dataset: dict) -> bool:
    """Mirrors the generator's own hard-exclusion definition exactly, so a
    healthy run reports zero here by construction, not by coincidence."""
    seasons_by_id = {e["id"]: e for e in dataset["seasons"]}
    times_by_id = {e["id"]: e for e in dataset["timesOfDay"]}
    weather_by_id = {e["id"]: e for e in dataset["weather"]}
    lighting_by_id = {e["id"]: e for e in dataset["lighting"]}
    genres = dataset["genreProfiles"]

    supernatural = bool(genres.get(item["genre"], {}).get("supernaturalOverride"))
    if supernatural:
        return False

    season_bucket = _bucket_tag(seasons_by_id[item["seasonId"]], SEASON_BUCKET_TAGS)
    weather_tags = {t.lower() for t in weather_by_id[item["weatherId"]].get("tags", [])}
    if season_bucket in OPPOSITE_SEASON and OPPOSITE_SEASON[season_bucket] in weather_tags:
        return True

    daypart_bucket = _bucket_tag(times_by_id[item["timeOfDayId"]], DAYPART_TAGS)
    lighting_tags = {t.lower() for t in lighting_by_id[item["lightingId"]].get("tags", [])}
    if daypart_bucket in OPPOSITE_DAYPART and OPPOSITE_DAYPART[daypart_bucket] in lighting_tags:
        return True

    return False


def _run_pass(num_batches: int, label: str) -> None:
    total = num_batches * 10
    dataset = load_dataset()

    subjects, palettes, genres, compositions = Counter(), Counter(), Counter(), Counter()
    seasons, times_of_day = Counter(), Counter()
    exact_titles, normalized_titles = Counter(), Counter()
    quad_pairs = Counter()
    nocturnal = minimal = indigo = 0
    incompatible = 0
    title_retries_total = 0
    title_fallback_count = 0

    for _ in range(num_batches):
        batch = generate_batch(10)
        for item in batch:
            subjects[item["subjectId"]] += 1
            palettes[item["paletteId"]] += 1
            genres[item["genre"]] += 1
            compositions[item["compositionId"]] += 1
            seasons[item["seasonId"]] += 1
            times_of_day[item["timeOfDayId"]] += 1
            exact_titles[item["title"]] += 1
            normalized_titles[_dedup_key(item["title"])] += 1
            quad_pairs[(item["subjectId"], item["environmentId"], item["seasonId"], item["timeOfDayId"])] += 1
            if item["nocturnal"]:
                nocturnal += 1
            if item["minimalComposition"]:
                minimal += 1
            if item["paletteId"] in INDIGO_PALETTE_IDS:
                indigo += 1
            if _is_incompatible(item, dataset):
                incompatible += 1
            title_retries_total += item["titleRetries"]
            if item["titleFallback"]:
                title_fallback_count += 1

    exact_dupes = sum(c - 1 for c in exact_titles.values() if c > 1)
    normalized_dupes = sum(c - 1 for c in normalized_titles.values() if c > 1)
    quad_dupes = sum(c - 1 for c in quad_pairs.values() if c > 1)

    print(f"\n{'=' * 70}\n{label}: {total} concepts across {num_batches} batches\n{'=' * 70}\n")

    print("Season distribution:")
    for sid, count in seasons.most_common():
        flag = "  [warn > 15%]" if count / total > 0.15 else ""
        print(f"  {sid:<32} {count:>5} ({count / total:.1%}){flag}")

    print("\nTime-of-day distribution:")
    for tid, count in times_of_day.most_common():
        flag = "  [warn > 12%]" if count / total > 0.12 else ""
        print(f"  {tid:<32} {count:>5} ({count / total:.1%}){flag}")

    print("\nTop 10 subjects:")
    for sid, count in subjects.most_common(10):
        print(f"  {sid:<35} {count} ({count / total:.1%})")

    print("\nGenre distribution:")
    for gid, count in genres.most_common():
        print(f"  {gid:<30} {count} ({count / total:.1%})")

    print(f"\nExact duplicate title count:       {exact_dupes} / {total} ({exact_dupes / total:.2%})")
    print(f"Normalized duplicate title count:  {normalized_dupes} / {total} ({normalized_dupes / total:.2%})"
          f"  [target < 0.1%]")
    print(f"Duplicate subject+env+season+time:  {quad_dupes} / {total} ({quad_dupes / total:.2%})")
    print(f"Incompatible atmosphere combos:      {incompatible} / {total}  [target 0]")
    print(f"Indigo-dominant palette share:       {indigo}/{total} ({indigo / total:.1%})  [target < 25%]")
    print(f"Nocturnal share:                     {nocturnal}/{total} ({nocturnal / total:.1%})")
    print(f"Minimalist-composition share:        {minimal}/{total} ({minimal / total:.1%})")
    print(f"Title retry count (sum):              {title_retries_total}")
    print(f"Title fallback count:                 {title_fallback_count} "
          f"({title_fallback_count / total:.2%})  [target < 1%]")

    print("\nWarnings:")
    warned = False
    if normalized_dupes / total > 0.001:
        print("  [warn] normalized duplicate title rate above 0.1% target")
        warned = True
    if incompatible > 0:
        print(f"  [warn] {incompatible} incompatible season/time/weather/lighting combination(s) found")
        warned = True
    if any(count / total > 0.15 for count in seasons.values()):
        print("  [warn] a season exceeds 15% of all generations")
        warned = True
    if any(count / total > 0.12 for count in times_of_day.values()):
        print("  [warn] a time-of-day state exceeds 12% of all generations")
        warned = True
    if title_fallback_count / total > 0.01:
        print("  [warn] title fallback usage above 1% target")
        warned = True
    if indigo / total > 0.25:
        print("  [warn] indigo-dominant palettes exceed 25% target")
        warned = True
    most_common_subject, most_common_count = subjects.most_common(1)[0]
    if most_common_count / total > 0.05:
        print(f"  [warn] subject '{most_common_subject}' exceeds 5% of all generations")
        warned = True
    if not warned:
        print("  none — all targets met")


def main() -> None:
    num_batches_1k = 100
    num_batches_10k = 1000
    if len(sys.argv) > 1:
        num_batches_1k = int(sys.argv[1])
        num_batches_10k = num_batches_1k * 10

    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()

    _run_pass(num_batches_1k, "PASS 1 — 1,000-concept run")
    # History persists across passes (not reset) — the 10k stress pass continues
    # to build on the same recency/title history, matching real usage.
    _run_pass(num_batches_10k, "PASS 2 — 10,000-concept stress run")


if __name__ == "__main__":
    main()
