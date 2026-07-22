from agent.pipeline import run_pipeline


def run(topic: str) -> None:
    print(f"\nResearching: '{topic}'\n")
    try:
        result = run_pipeline(topic)
    except Exception as e:
        print(f"[Error] {e}")
        return

    print(f"Niche: {result['niche']}\n")
    for c in result.get("poster_concepts", []):
        print(f"  • {c.get('name', '—')} [{c.get('single_or_set', '—')}] — {c.get('suggested_etsy_title', '—')}")
