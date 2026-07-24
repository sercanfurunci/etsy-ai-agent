"""
Batch image generator — optimizer always runs before BFL.

Modes:
  python3 generate_batch.py library          # prompt_library collections (default)
  python3 generate_batch.py ukiyoe [N]       # ukiyo-e procedural generator, N concepts (default 10)
  python3 generate_batch.py ukiyoe 5 --seed 42
"""
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from agent.prompt_optimizer import optimize
from image.bfl_provider import BFLImageProvider

provider = BFLImageProvider()


def _run(label: str, concept: dict, image_prompt: str, negative_prompt: str, idx: int, total: int) -> None:
    print(f"\n[{idx}/{total}] {label}")
    print("  Optimizing...")
    try:
        result = optimize(concept, image_prompt, negative_prompt)
        final_prompt = result["optimized_image_prompt"]
        r = result["optimization_report"]
        print(f"  Scores — originality:{r['originality_score']} "
              f"commercial:{r['commercial_appeal_score']} "
              f"print:{r['print_quality_score']}")
    except Exception as e:
        print(f"  Optimizer failed ({e}), using raw prompt")
        final_prompt = image_prompt

    print("  Generating...")
    try:
        path = provider.generate(final_prompt)
        print(f"  ✓ {path}")
    except Exception as e:
        print(f"  ✗ {e}")


def run_library() -> None:
    from agent.prompt_library import build_from_collection

    SELECTIONS = {
        1: [
            "Tiny orange tiger swimming through giant emerald waves",
            "Tiny red fox crossing endless patterned cream grass",
            "Tiny black cat sleeping beneath enormous willow branches",
            "Tiny whale inside abstract ocean currents",
            "Tiny frog holding an umbrella in the rain",
        ],
        2: [
            "Sleeping cat beneath willow tree",
            "Red fox resting in tall grass",
            "Elegant red-crowned crane",
            "Red panda with lantern",
            "Koi swimming peacefully",
        ],
        3: ["Cherry Blossom", "Lotus", "Ginkgo", "Japanese Maple", "Plum Blossom"],
        4: ["Two cranes beneath pine branches", "Crane with golden moon",
            "Flying cranes", "Crane in snowfall", "Crane beside lotus pond"],
        5: ["Vinyl record poster", "Tokyo street", "Jazz poster",
            "Japanese magazine cover", "Japanese café"],
        6: ["Japanese Cappadocia", "Japanese Istanbul", "Japanese Paris",
            "Japanese Venice", "Japanese Iceland"],
    }

    total = sum(len(v) for v in SELECTIONS.values())
    done = 0
    for cid, items in SELECTIONS.items():
        print(f"\n{'='*60}\nCOLLECTION {cid}\n{'='*60}")
        for item in items:
            done += 1
            prompt, neg = build_from_collection(cid, item)
            _run(item, {"title": item, "collection_id": cid}, prompt, neg, done, total)


def run_ukiyoe(count: int = 10, seed: int | None = None) -> None:
    from agent.ukiyoe_prompt_generator import generate_batch, NEGATIVE_PROMPT
    concepts = generate_batch(count=count, seed=seed)
    for i, c in enumerate(concepts, 1):
        _run(c["title"], c, c["imagePrompt"], NEGATIVE_PROMPT, i, len(concepts))


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0] if args else "library"

    if mode == "ukiyoe":
        count = int(args[1]) if len(args) > 1 and args[1].isdigit() else 10
        seed = None
        if "--seed" in args:
            seed = int(args[args.index("--seed") + 1])
        run_ukiyoe(count=count, seed=seed)
    else:
        run_library()
