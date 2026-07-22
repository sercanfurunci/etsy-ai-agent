# Ukiyo-e prompt dataset

A curated, tag-driven dataset for procedurally generating diverse, historically-informed
ukiyo-e poster concepts without calling an LLM. Replaces the earlier hardcoded category
lists in `agent/ukiyoe_prompt_generator.py`.

## Files

| File | Purpose |
|---|---|
| `scripts/build_dataset.py` | Source of truth. All content lives here as Python literals; running it writes `data/ukiyoe_dataset.json`. |
| `data/ukiyoe_dataset.json` | Generated artifact. Don't hand-edit — edit `build_dataset.py` and rebuild. |
| `scripts/validate_dataset.py` | Structural validation (parses, required keys, unique IDs, non-empty fields, positive weights, resolvable references, exactly-one season/daypart bucket tag). |
| `scripts/dataset_distribution_report.py` | Runs a 1,000-concept pass then a 10,000-concept stress pass, reporting frequency distributions, duplicate-title rates, incompatible-combination count, title retry/fallback counts. |
| `agent/ukiyoe_prompt_generator.py` | The generator: loads the dataset, does genre-first weighted selection with tag/bucket compatibility, assembles prompts, validates batch diversity, dedupes titles. |
| `tests/test_ukiyoe_prompt_generator.py` | Pytest coverage: dataset shape, required output fields, zero incompatible combos, title uniqueness. |

## Regenerating the dataset

```bash
source .venv/bin/activate
python3 scripts/build_dataset.py                    # writes data/ukiyoe_dataset.json
python3 scripts/validate_dataset.py                 # structural checks, exits non-zero on error
python3 scripts/dataset_distribution_report.py 100   # 1,000-concept pass + 10,000-concept stress pass
python3 -m pytest tests/test_ukiyoe_prompt_generator.py -v
```

## Dataset structure

```
metadata             — name, version, historical scope, disclaimer
globalStyles         — reusable style-sentence variants, appended once per prompt
genreProfiles         — 17 profiles (kacho-e, musha-e, yokai-folklore, ...) that bias selection
subjects.animals.*    — birds / mammals / fish / marineLife / insects / reptilesAndAmphibians
subjects.people.*     — travelers / craftspeople / performers / warriors / religiousFigures /
                         villagers / courtlyFigures
subjects.mythology.*  — yokai / spirits / legendaryAnimals / deities / ghostlyFigures
subjects.architecture, subjects.objects
flora, environments
seasons               — independent axis, 24–32 entries, each tagged with exactly one of
                         {spring, summer, rainy-season, autumn, winter}
timesOfDay            — independent axis, 24–32 entries, each tagged with exactly one of
                         {day, night}
weather, lighting, moods, symbolism, compositions, perspectives,
movement, palettes, printTechniques, surfaceTextures
iconicMotifs          — documents which entries carry a repetition penalty, and why
compatibilityRules    — tag-based "X favors/avoids Y" rules (including preferSeasonTags /
                         preferTimeTags, now actually wired into selection)
exclusionRules        — hard incompatible-tag pairs
generationSettings    — batch size, history window, title history window, per-batch caps
```

### Entry shape

```json
{
  "id": "red-crowned-crane",
  "label": "Red-crowned crane",
  "promptText": "a red-crowned crane standing among wind-bent reeds",
  "tags": ["bird", "wetland", "elegant", "winter"],
  "weight": 0.4,
  "rarity": 2.5,
  "cooldownGroup": "iconic-japanese-motifs"
}
```

`seasons`/`timesOfDay` entries additionally carry `compatibleGenres` and `avoidWith`
(built via `build_atmosphere_list()` in `build_dataset.py`, one fixed-shape tuple per row).

- `weight` — base selection probability multiplier. Iconic/overused motifs (Mount Fuji,
  cranes, koi, torii gates, cherry blossoms, samurai, dragons, foxes, full moons, indigo
  palettes, peak cherry-blossom/maple season, moonrise) are pre-weighted down (0.3–0.6) so
  they appear less often without being banned.
- `cooldownGroup` — entries sharing a group get an *additional* decay penalty when any
  member of the group was recently used, even if this exact entry wasn't.
- `tags` — the compatibility mechanism. Genre profiles and `compatibilityRules` reference
  tags, not individual IDs, so adding a new entry with sensible tags makes it participate
  in compatibility filtering automatically. `seasons`/`timesOfDay` entries additionally
  must carry exactly one **bucket tag** (season: spring/summer/rainy-season/autumn/winter;
  time: day/night) — `validate_dataset.py` enforces this.

## Selection order (`agent/ukiyoe_prompt_generator.py`)

```
genre → style → subject → environment → season → time_of_day → weather → lighting
→ composition → perspective → movement → palette → mood → symbolism
→ print_technique → surface_texture
```

1. **Genre-first.** A genre profile is picked (weighted, decayed by recent use).
2. **Style.** Picked right after genre (only depends on it), boosted toward styles whose
   `compatibleGenres` includes the chosen genre.
3. **Subject.** Merged animals+people+mythology+architecture+objects pool, boosted by the
   genre's `preferredSubjectTags`, hard-excluded by `disallowedTags`.
4. **Environment.** Boosted/excluded via `compatibilityRules` matched against the subject's
   tags/id (e.g. marine subjects favor coastal/river/pond environments).
5. **Season.** Boosted by the genre's `preferredSeasonTags` and by any `compatibilityRules`
   matching the subject (e.g. `cherry-blossom-favors-spring`, `maple-favors-autumn-season-too`).
6. **Time of day.** Same mechanism via `preferredTimeTags` (e.g. the `cicada` rule prefers
   `day`/`dusk`).
7. **Weather — hard compatibility with season.** If the chosen season's bucket is `winter`
   or `summer`, weather entries tagged with the *opposite* bucket (`summer`/`winter`) are
   hard-excluded — e.g. a `winter` season can never pull `sudden-summer-downpour`. Neutral
   and same-bucket weather is boosted, not required. Skipped entirely if the genre profile
   sets `supernaturalOverride` (currently only `yokai-folklore`).
8. **Lighting — hard compatibility with time of day.** If the time-of-day bucket is `day`,
   night-only lighting (`full-moon-brilliance`, `starlit-darkness`, ...) is hard-excluded,
   and vice versa. Also skipped under `supernaturalOverride`.
9. **Composition, movement, perspective** — genre-boosted where applicable; compositions
   also get hard-excluded per subject (e.g. warrior subjects avoid botanical compositions).
10. **Palette, mood, symbolism, print technique, surface texture** — weighted-random with
    recency decay, palette/mood boosted by the genre's preferred lists.

Every pick is scored as `weight × tag_boost × 1/(1 + recent_occurrences × 4) ×
1/(1 + cooldown_group_recent_occurrences × 2)`. If tag/bucket filtering leaves an empty
pool, filtering is dropped and the pick falls back to plain recency-decayed random choice —
the generator never hard-fails on a narrow filter.

## History and repetition control

`outputs/.ukiyoe_prompt_history.json` persists two different windows:

- **General axis history** (`historyWindow`, default 50) — subject, environment, season,
  time_of_day, weather, lighting, composition, perspective, movement, palette, mood,
  symbolism, print technique, surface texture, style, genre.
- **Title history** (`titleHistoryWindow`, default 20,000 — configurable, minimum 500) —
  kept far larger than the general window. Several title templates only combine two
  lower-cardinality fields (e.g. time-of-day × environment ≈ 2,100 possible strings); a
  small window "forgets" them long before their pool is exhausted, which reintroduces
  visible repeats at scale. A large title window was the fix (see *Title system* below).

Delete the file to reset everything.

## Title system

Titles are built from 12 templates (`TITLE_TEMPLATES`), each combining at least two of
{subject, mood, symbolism, environment, weather, season, time_of_day} so that recurring
subjects rarely collide on their own.

**Normalization** (`_normalize_title` / `_dedup_key`): lowercase → normalize unicode
apostrophes (`’ ‘` → `'`) and dashes (`— – ‒` → `-`) → strip all punctuation → collapse
whitespace → (for dedup comparison only) strip a leading article (`a`/`an`/`the`).

**Rejection and regeneration**, checked in this order, on every candidate:
1. Exact/near-duplicate — normalized dedup key seen in the last `titleHistoryWindow` titles.
2. Subject+template cooldown — this `(subject_id, template_index)` pairing used in the last
   `historyWindow` (small window — see below) titles.
3. Environment+template cooldown — same, for `(environment_id, template_index)`.
4. Token staleness — every significant word in the candidate (stopwords and words ≤2 chars
   excluded) has appeared ≥3 times in the last `historyWindow` titles.

Templates are tried in a shuffled order, cycling, up to `MAX_TITLE_RETRIES` (24) times.

**Why cooldowns #2–4 use the small window, not the title window:** they were originally
tried against the same large `titleHistoryWindow`. That broke generation at scale — with
only ~239 subjects × 12 templates ≈ 2,868 possible (subject, template) pairs, that pool is
exhausted after roughly 2,868 titles, after which *every* candidate fails the cooldown
check and the generator falls back almost 100% of the time (observed: 98.8% fallback rate
in an early 10k-concept test run). Splitting the checks — large window for exact-duplicate
only, small window for the softer cooldowns — fixed it: 0% duplicates *and* 0% fallback
at 10,000 concepts (see *Verified results* below).

**Deterministic fallback** (guarantees termination): if all retries collide,
`f"{subject} — {environment} ({season})"`, with a numeric `#2`, `#3`, ... suffix appended
until the dedup key is unique. Both the retry count and whether fallback was used are
returned on every generated concept (`titleRetries`, `titleFallback`).

## Batch-level diversity validation

`generate_batch(count=10)` generates a batch, then checks it against
`generationSettings` — hard constraints trigger a retry (up to 6 attempts, keeping the
best attempt if none fully pass):

- no genre / palette / composition / environment / symbolism used more than twice
- no more than 2 minimalist-tagged compositions
- at least one human, one animal, one architectural subject
- at least 5 distinct composition families and 5 distinct palette families
- no duplicate normalized titles

One constraint is **soft** (logged, not retried): "no more than 2 nocturnal-feeling
concepts." ~46% of generated concepts read as nocturnal (lighting tagged `night`, or an
eerie/haunted/mysterious mood) given the dataset's natural tag distribution, so enforcing
this as a hard cap made most batches fail all 6 retry attempts. It's reported via
`[info]`/`[warn]` prints instead of blocking.

## Adding a new entry

1. Open `scripts/build_dataset.py`, find the relevant category list (e.g. `ANIMAL_BIRDS`).
2. Add a tuple: `("label", "prompt text fragment", ["tag1", "tag2"])`. Add a 4th element
   (weight) and 5th element (extra-fields dict, e.g. `{"cooldownGroup": "..."}`) only if
   the entry needs non-default weight/cooldown.
3. For `seasons`/`timesOfDay` specifically, use `build_atmosphere_list()`'s fixed 7-tuple
   shape: `(label, prompt, tags, weight, compatible_genres, avoid_with, cooldown_group)` —
   `tags` **must** include exactly one bucket tag (season or day/night).
4. Run `python3 scripts/build_dataset.py && python3 scripts/validate_dataset.py`.

## Running validation and the distribution/stress tests

```bash
python3 scripts/validate_dataset.py                 # structural correctness
python3 scripts/dataset_distribution_report.py 100   # 1,000-concept pass + 10,000-concept stress pass
python3 -m pytest tests/test_ukiyoe_prompt_generator.py -v
```

The distribution report warns if: normalized duplicate-title rate exceeds 0.1%, any
incompatible atmosphere combination is found, any season exceeds 15% of all generations,
any time-of-day state exceeds 12%, title fallback usage exceeds 1%, any indigo-`cooldownGroup`
palette exceeds 25% of all generations, or any single subject exceeds 5%.

## Verified results (last run, 100-batch / 1,000-batch pass)

| Metric | 1,000 concepts | 10,000 concepts | Target |
|---|---|---|---|
| Normalized duplicate title rate | 0.00% | 0.00% | < 0.1% |
| Incompatible atmosphere combinations | 0 | 0 | 0 |
| Title fallback rate | 0.00% | 0.00% | < 1% |
| Max single season share | ~3.8% | ~3.8% | < 15% |
| Max single time-of-day share | ~3.8% | ~3.8% | < 12% |
| Indigo-dominant palette share | ~3% | ~3% | < 25% |
| Duplicate subject+env+season+time combos | 0 | 4 (0.04%) | — |

Re-run `python3 scripts/dataset_distribution_report.py` to reproduce; numbers vary slightly
run to run (unseeded `random`) but stay within these bounds.

## Known limitations (intentional)

- **Entry counts are ~55% of the originally requested numbers** for the 17 original
  categories (e.g. 35 birds vs. a requested 180–250 animals total). Quality-over-quantity
  was an explicit project decision; actual counts are printed by `build_dataset.py` and
  checked by `validate_dataset.py`'s minimums. `seasons` (29) and `timesOfDay` (30) hit
  their requested 24–32 range directly.
- **Nocturnal cap is soft, not hard** — the dataset's natural night/eerie tag distribution
  makes a hard 2-per-batch cap fail almost every retry; see *Batch-level diversity
  validation* above.
- **Season↔weather and time↔lighting hard exclusion only covers the clear-opposite pairs**
  (winter vs. summer; day vs. night) — spring/autumn/rainy-season and dawn/dusk shades are
  soft-boosted, not hard-filtered, to avoid over-constraining the pool. This matches the
  explicit example given ("heavy snow should favor winter unless supernatural override
  applies") without risking empty candidate pools for softer transitional cases.
- **Deities are presented as folklore/cultural motifs**, not doctrinal claims — entries
  avoid asserting religious fact and are phrased as scene elements (e.g. "a shrine to the
  river deity", not naming or depicting specific deities directly).
