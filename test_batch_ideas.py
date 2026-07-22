# ponytail: temporary test — generates a batch of poster ideas, then an image for each, sequentially
from agent.ukiyoe_prompt_generator import generate_batch
from image.openai_provider import OpenAIImageProvider

TOTAL_IDEAS = 10

if __name__ == "__main__":
    print(f"\nGenerating {TOTAL_IDEAS} ukiyo-e poster ideas...\n")
    concepts = generate_batch(TOTAL_IDEAS)

    print(f"{len(concepts)} ideas:\n")
    for i, c in enumerate(concepts, 1):
        print(f"  [{i}] {c['title']}  ({c['genre']}, {c['subjectRole']})")
        print(f"      {c['environmentId']} · {c['seasonId']} · {c['timeOfDayId']} · "
              f"{c['weatherId']} · {c['compositionId']}")

    confirm = input(f"\nGenerate {len(concepts)} images now? [y/N]: ").strip().lower()
    if confirm != "y":
        raise SystemExit("Aborted.")

    provider = OpenAIImageProvider()
    results = []

    print()
    for i, concept in enumerate(concepts, 1):
        title = concept["title"]
        print(f"[{i}/{len(concepts)}] {title}")
        print(f"  {concept['imagePrompt']}")

        try:
            path = provider.generate(concept["imagePrompt"])
            print(f"  [OK] {path}")
            results.append((title, path, None))
        except Exception as e:
            print(f"  [skip] image generation error: {e}")
            results.append((title, None, f"image error: {e}"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for title, path, error in results:
        status = path if path else f"FAILED — {error}"
        print(f"  {title}: {status}")

    ok_count = sum(1 for _, path, _ in results if path)
    print(f"\n  {ok_count}/{len(results)} images generated successfully.")
