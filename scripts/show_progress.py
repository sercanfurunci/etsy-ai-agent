"""
Show progress for a production run or queue directory.

Usage:
  python3 scripts/show_progress.py outputs/<run_dir>
  python3 scripts/show_progress.py queues/<queue_dir>
  python3 scripts/show_progress.py outputs/<run_dir> --json
  python3 scripts/show_progress.py outputs/<run_dir> --watch
  python3 scripts/show_progress.py queues/<queue_dir> --watch --interval 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _fmt_seconds(s: float | None) -> str:
    if s is None:
        return "unknown"
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m {sec}s"
    h, mins = divmod(m, 60)
    return f"{h}h {mins}m"


def _is_queue(path: Path) -> bool:
    return (path / "queue.json").exists()


def _print_production(p, file=sys.stdout) -> None:
    print(f"Run:            {p.run_id}", file=file)
    print(f"Status:         {p.status}", file=file)
    print(f"Progress:       {p.percent:.1f}%", file=file)
    if p.current_stage:
        print(f"Current stage:  {p.current_stage}", file=file)
    if p.poster_total > 0:
        print(f"Posters:        {p.poster_completed} / {p.poster_total} completed", file=file)
    if p.current_poster is not None:
        print(f"Current poster: {p.current_poster}", file=file)
    if p.current_attempt is not None:
        print(f"Attempt:        {p.current_attempt}", file=file)
    print(f"Elapsed:        {_fmt_seconds(p.elapsed_seconds)}", file=file)
    if p.eta_available and p.eta_seconds is not None:
        print(f"ETA:            about {_fmt_seconds(p.eta_seconds)}", file=file)
        print(f"Confidence:     {p.eta_confidence}", file=file)
    else:
        print("ETA:            unavailable", file=file)


def _print_queue(q, file=sys.stdout) -> None:
    print(f"Queue:          {q.queue_dir}", file=file)
    print(f"Status:         {q.status}", file=file)
    print(f"Progress:       {q.percent:.1f}%", file=file)
    print(f"Jobs:           total={q.total_jobs}  completed={q.completed_jobs}"
          f"  failed={q.failed_jobs}  pending={q.pending_jobs}"
          f"  cancelled={q.cancelled_jobs}", file=file)
    if q.active_job_id:
        print(f"Active job:     {q.active_job_id} (position {q.active_job_position})", file=file)
    print(f"Elapsed:        {_fmt_seconds(q.elapsed_seconds)}", file=file)
    if q.eta_available and q.eta_seconds is not None:
        print(f"ETA:            about {_fmt_seconds(q.eta_seconds)}", file=file)
        print(f"Confidence:     {q.eta_confidence}", file=file)
    else:
        print("ETA:            unavailable", file=file)
    if q.current_job_progress:
        print("\nActive job progress:", file=file)
        p = q.current_job_progress
        print(f"  Progress:     {p.percent:.1f}%  stage={p.current_stage}", file=file)


def _production_to_dict(p) -> dict:
    return {
        "schema_version": p.schema_version,
        "run_id": p.run_id,
        "status": p.status,
        "percent": p.percent,
        "current_stage": p.current_stage,
        "poster_completed": p.poster_completed,
        "poster_total": p.poster_total,
        "current_poster": p.current_poster,
        "current_attempt": p.current_attempt,
        "elapsed_seconds": p.elapsed_seconds,
        "eta_seconds": p.eta_seconds,
        "eta_available": p.eta_available,
        "eta_confidence": p.eta_confidence,
        "eta_basis": p.eta_basis,
        "estimated_completion_at": p.estimated_completion_at,
        "created_at": p.created_at,
        "last_updated_at": p.last_updated_at,
        "stages": [
            {
                "stage_name": s.stage_name,
                "status": s.status,
                "weight": s.weight,
                "started_at": s.started_at,
                "completed_at": s.completed_at,
                "duration_seconds": s.duration_seconds,
            }
            for s in p.stages
        ],
    }


def _queue_to_dict(q) -> dict:
    d: dict = {
        "schema_version": q.schema_version,
        "queue_dir": q.queue_dir,
        "status": q.status,
        "total_jobs": q.total_jobs,
        "completed_jobs": q.completed_jobs,
        "failed_jobs": q.failed_jobs,
        "pending_jobs": q.pending_jobs,
        "running_jobs": q.running_jobs,
        "cancelled_jobs": q.cancelled_jobs,
        "percent": q.percent,
        "active_job_id": q.active_job_id,
        "active_job_position": q.active_job_position,
        "elapsed_seconds": q.elapsed_seconds,
        "eta_seconds": q.eta_seconds,
        "eta_available": q.eta_available,
        "eta_confidence": q.eta_confidence,
        "estimated_completion_at": q.estimated_completion_at,
    }
    if q.current_job_progress:
        d["current_job_progress"] = _production_to_dict(q.current_job_progress)
    return d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show progress for a production run or queue directory."
    )
    parser.add_argument("path", help="Path to run directory or queue directory")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON")
    parser.add_argument("--watch", action="store_true", help="Poll and redraw until complete")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Polling interval in seconds (default: 1.0)")
    args = parser.parse_args()

    from agent.progress_tracking import get_production_progress, get_queue_progress

    path = Path(args.path)
    is_q = _is_queue(path)

    def fetch():
        if is_q:
            return get_queue_progress(path)
        return get_production_progress(path)

    def emit(obj) -> None:
        if args.as_json:
            d = _queue_to_dict(obj) if is_q else _production_to_dict(obj)
            print(json.dumps(d, indent=2))
        else:
            if is_q:
                _print_queue(obj)
            else:
                _print_production(obj)

    if not args.watch:
        emit(fetch())
        return

    try:
        while True:
            # Simple clear: print blank lines then reposition (no external deps)
            print("\033[2J\033[H", end="")
            obj = fetch()
            emit(obj)
            # Stop when done
            status = obj.status if is_q else obj.status
            if status in ("completed", "completed_with_failures", "failed"):
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
