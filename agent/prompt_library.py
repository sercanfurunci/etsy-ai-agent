import json
import sys
from pathlib import Path

_LIBRARY_PATH = Path(__file__).resolve().parent.parent / "data" / "prompt_library.json"
_data: dict | None = None


def _load() -> dict:
    global _data
    if _data is None:
        _data = json.loads(_LIBRARY_PATH.read_text(encoding="utf-8"))
    return _data


def build(preset_id: int | str) -> tuple[str, str]:
    """Returns (image_prompt, negative_prompt). Global style + print constraint appended."""
    data = _load()
    pid = int(preset_id)
    preset = next((p for p in data["presets"] if p["id"] == pid), None)
    if preset is None:
        raise KeyError(f"Preset {pid} not found")
    full_prompt = f"{preset['prompt']}, {data['global_style_block']}"
    return full_prompt, data["global_negative_prompt"]


def build_collection1(subject: str, environment: str, pattern_type: str, palette: str) -> tuple[str, str]:
    """Fill the Collection 1 master template and append globals."""
    data = _load()
    prompt = (data["collection_template"]
              .replace("{subject}", subject)
              .replace("{environment}", environment)
              .replace("{pattern_type}", pattern_type)
              .replace("{palette}", palette))
    full_prompt = f"{prompt}, {data['global_style_block']}"
    return full_prompt, data["global_negative_prompt"]


def list_collections() -> None:
    for c in _load()["collections"]:
        stars = "⭐" * c["rating"]
        primary = " ← ANA" if c.get("primary") else ""
        print(f"{c['id']:2d}. {c['name']} {stars}{primary}")
        print(f"    {c['description']}  ({len(c['items'])} item)")


def build_from_collection(collection_id: int, item: str) -> tuple[str, str]:
    """Fill collection template with item and append globals."""
    data = _load()
    col = next((c for c in data["collections"] if c["id"] == collection_id), None)
    if col is None:
        raise KeyError(f"Collection {collection_id} not found")
    prompt = col["template"].replace("{subject}", item)
    full_prompt = f"{prompt}, {data['global_style_block']}"
    return full_prompt, data["global_negative_prompt"]


def list_presets() -> None:
    for p in _load()["presets"]:
        print(f"{p['id']:2d}. {p['name']}")


def presets() -> list[dict]:
    return _load()["presets"]


def parameter_presets() -> dict:
    return _load()["parameter_presets"]


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "list":
        list_presets()
    elif args[0] == "build" and len(args) >= 2:
        prompt, neg = build(args[1])
        print("PROMPT:", prompt)
        print("\nNEGATIVE:", neg)
    elif args[0] == "collection1" and len(args) == 5:
        prompt, neg = build_collection1(args[1], args[2], args[3], args[4])
        print("PROMPT:", prompt)
        print("\nNEGATIVE:", neg)
    elif args[0] == "collections":
        list_collections()
    elif args[0] == "from-collection" and len(args) >= 3:
        prompt, neg = build_from_collection(int(args[1]), " ".join(args[2:]))
        print("PROMPT:", prompt)
        print("\nNEGATIVE:", neg)
    else:
        print("Usage:")
        print("  python3 -m agent.prompt_library list")
        print("  python3 -m agent.prompt_library build <1-48>")
        print("  python3 -m agent.prompt_library collection1 <subject> <environment> <pattern_type> <palette>")
        print("  python3 -m agent.prompt_library collections")
        print("  python3 -m agent.prompt_library from-collection <collection_id> <item>")
