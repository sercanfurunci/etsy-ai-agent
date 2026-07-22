"""Validates data/ukiyoe_dataset.json for structural correctness.

Run: python3 scripts/validate_dataset.py

Exits non-zero and prints every problem found if validation fails.
"""
import json
import sys
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ukiyoe_dataset.json"

REQUIRED_TOP_LEVEL_KEYS = {
    "metadata", "globalStyles", "genreProfiles", "subjects", "flora", "environments",
    "seasons", "timesOfDay", "weather", "lighting", "moods", "symbolism", "compositions",
    "perspectives", "movement", "palettes", "printTechniques", "surfaceTextures",
    "iconicMotifs", "compatibilityRules", "exclusionRules", "generationSettings",
}

MINIMUM_COUNTS = {
    "compositions": 30,
    "environments": 40,
    "flora": 30,
    "moods": 30,
    "symbolism": 30,
    "palettes": 25,
    "perspectives": 20,
    "movement": 20,
    "printTechniques": 20,
    "surfaceTextures": 20,
    "weather": 25,
    "lighting": 15,
    "seasons": 24,
    "timesOfDay": 24,
}

SEASON_BUCKET_TAGS = {"spring", "summer", "rainy-season", "autumn", "winter"}
DAYPART_TAGS = {"day", "night"}


RULE_KEYS = {"iconicMotifs", "compatibilityRules", "exclusionRules"}


def collect_entry_lists(node, path=""):
    """Yield (path, list_of_entries) for every list of dict-entries in the tree.

    Skips rule/metadata lists (iconicMotifs, compatibilityRules, exclusionRules) —
    those are rule descriptors, not content entries, and don't share the
    label/promptText/weight schema.
    """
    if path in RULE_KEYS:
        return
    if isinstance(node, list):
        if node and isinstance(node[0], dict) and "id" in node[0]:
            yield path, node
        return
    if isinstance(node, dict):
        for key, value in node.items():
            yield from collect_entry_lists(value, f"{path}.{key}" if path else key)


def main() -> int:
    errors = []
    warnings = []

    if not DATA_PATH.exists():
        print(f"FAIL: {DATA_PATH} does not exist.")
        return 1

    try:
        raw = DATA_PATH.read_text(encoding="utf-8")
        dataset = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"FAIL: JSON does not parse: {e}")
        return 1

    missing_keys = REQUIRED_TOP_LEVEL_KEYS - dataset.keys()
    if missing_keys:
        errors.append(f"Missing top-level keys: {sorted(missing_keys)}")

    all_ids_by_category = {}
    global_id_seen = {}

    for path, items in collect_entry_lists(dataset):
        ids_here = set()
        for item in items:
            item_id = item.get("id")
            label = item.get("label")
            prompt = item.get("promptText")
            weight = item.get("weight")

            if not item_id:
                errors.append(f"{path}: entry with empty id (label={label!r})")
            elif item_id in ids_here:
                errors.append(f"{path}: duplicate id '{item_id}' within category")
            else:
                ids_here.add(item_id)

            if item_id:
                if item_id in global_id_seen and global_id_seen[item_id] != path:
                    warnings.append(
                        f"id '{item_id}' appears in both '{global_id_seen[item_id]}' and '{path}' "
                        f"(fine if intentional, otherwise check for accidental duplication)"
                    )
                else:
                    global_id_seen[item_id] = path

            if not label or not str(label).strip():
                errors.append(f"{path}: entry '{item_id}' has empty label")
            if not prompt or not str(prompt).strip():
                errors.append(f"{path}: entry '{item_id}' has empty promptText")
            if not isinstance(weight, (int, float)) or weight <= 0:
                errors.append(f"{path}: entry '{item_id}' has invalid weight {weight!r}")

            if path == "seasons":
                bucket_tags = SEASON_BUCKET_TAGS & {t.lower() for t in item.get("tags", [])}
                if len(bucket_tags) != 1:
                    errors.append(
                        f"seasons: entry '{item_id}' must carry exactly one season-bucket tag "
                        f"from {sorted(SEASON_BUCKET_TAGS)}, found {sorted(bucket_tags)}"
                    )
            if path == "timesOfDay":
                daypart_tags = DAYPART_TAGS & {t.lower() for t in item.get("tags", [])}
                if len(daypart_tags) != 1:
                    errors.append(
                        f"timesOfDay: entry '{item_id}' must carry exactly one daypart tag "
                        f"from {sorted(DAYPART_TAGS)}, found {sorted(daypart_tags)}"
                    )

        all_ids_by_category[path] = ids_here

    for key, minimum in MINIMUM_COUNTS.items():
        actual = len(dataset.get(key, []))
        if actual < minimum:
            errors.append(f"'{key}' has {actual} entries, below minimum {minimum}")

    all_known_ids = set(global_id_seen.keys())
    for rule in dataset.get("compatibilityRules", []):
        ref_id = rule.get("ifId")
        if ref_id and ref_id not in all_known_ids:
            errors.append(f"compatibilityRules['{rule.get('id')}'] references unknown ifId '{ref_id}'")

    for genre_key, profile in dataset.get("genreProfiles", {}).items():
        for field in ("preferredCompositions", "preferredPalettes"):
            for ref_id in profile.get(field, []):
                if ref_id not in all_known_ids:
                    warnings.append(
                        f"genreProfiles['{genre_key}'].{field} references unknown id '{ref_id}'"
                    )

    for motif in dataset.get("iconicMotifs", []):
        if motif.get("id") not in all_known_ids:
            errors.append(f"iconicMotifs references unknown id '{motif.get('id')}'")

    total_entries = sum(len(v) for v in all_ids_by_category.values())

    print(f"Parsed OK: {DATA_PATH}")
    print(f"Leaf categories: {len(all_ids_by_category)}")
    print(f"Total entries: {total_entries}")
    print(f"Genre profiles: {len(dataset.get('genreProfiles', {}))}")
    print(f"Global styles: {len(dataset.get('globalStyles', []))}")
    print(f"Compatibility rules: {len(dataset.get('compatibilityRules', []))}")
    print(f"Exclusion rules: {len(dataset.get('exclusionRules', []))}")
    print(f"Iconic motifs flagged: {len(dataset.get('iconicMotifs', []))}")
    print()

    if warnings:
        print(f"{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  [warn] {w}")
        print()

    if errors:
        print(f"{len(errors)} error(s):")
        for e in errors:
            print(f"  [fail] {e}")
        return 1

    print("VALID — no errors found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
