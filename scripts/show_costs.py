"""
Show cost summary for a production run.

Usage:
  python3 scripts/show_costs.py <run_dir>
  python3 scripts/show_costs.py <run_dir> --json
  python3 scripts/show_costs.py <run_dir> --by-stage
  python3 scripts/show_costs.py <run_dir> --by-poster
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show cost summary for a production run directory."
    )
    parser.add_argument("run_dir", help="Path to production run directory")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON")
    parser.add_argument("--by-stage", action="store_true", help="Show per-stage breakdown")
    parser.add_argument("--by-poster", action="store_true", help="Show per-poster image calls")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary_path = run_dir / "costs" / "summary.json"

    if not summary_path.exists():
        print(f"Error: no costs/summary.json found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(summary_path.read_text(encoding="utf-8"))

    if args.as_json:
        print(json.dumps(data, indent=2))
        return

    cost_str = data.get("total_cost")
    cost_display = f"${cost_str}" if cost_str is not None else "unavailable (null pricing)"

    print(f"Run:            {data.get('run_id', '?')}")
    print(f"Total cost:     {cost_display}")
    print(f"Input tokens:   {data.get('total_input_tokens', 0):,}")
    print(f"Output tokens:  {data.get('total_output_tokens', 0):,}")
    print(f"Images:         {data.get('total_images', 0)}")
    print(f"API calls:      {data.get('total_calls', 0)}")

    if args.by_stage:
        print("\nPer-stage breakdown:")
        for s in data.get("by_stage", []):
            cost = f"${s['cost']}" if s.get("cost") is not None else "?"
            print(
                f"  {s['stage']:<30}  calls={s['call_count']}"
                f"  in={s['input_tokens']:,}  out={s['output_tokens']:,}"
                f"  images={s['image_count']}  cost={cost}"
            )

    if args.by_poster:
        jsonl = run_dir / "costs" / "usage_records.jsonl"
        if jsonl.exists():
            print("\nPer-poster image calls:")
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("call_type") == "image":
                    meta = rec.get("metadata", {})
                    poster = meta.get("poster_index", "?")
                    print(
                        f"  poster={poster}  stage={rec['stage']}"
                        f"  model={rec['model']}  size={rec.get('image_size', '?')}"
                    )


if __name__ == "__main__":
    main()
