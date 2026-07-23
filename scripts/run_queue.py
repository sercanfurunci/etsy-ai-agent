"""
CLI for the persistent job queue.

Usage:
  python3 scripts/run_queue.py create  queues/my_queue
  python3 scripts/run_queue.py add     queues/my_queue --query "Vintage Japanese..." --collection-size 4
  python3 scripts/run_queue.py run     queues/my_queue [--stop-on-failure]
  python3 scripts/run_queue.py resume  queues/my_queue [--stop-on-failure]
  python3 scripts/run_queue.py list    queues/my_queue
  python3 scripts/run_queue.py cancel  queues/my_queue job_002
  python3 scripts/run_queue.py unlock  queues/my_queue --force
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Helpers ────────────────────────────────────────────────────────────────────

def _import_queue():
    from agent.job_queue import (
        QueueError, cancel_job, enqueue_job, force_unlock,
        list_jobs, resume_queue, run_queue,
    )
    return {
        "enqueue_job": enqueue_job, "run_queue": run_queue,
        "resume_queue": resume_queue, "list_jobs": list_jobs,
        "cancel_job": cancel_job, "force_unlock": force_unlock,
        "QueueError": QueueError,
    }


def _import_request():
    from agent.production_orchestrator import ProductionRequest
    return ProductionRequest


# ── Sub-command handlers ───────────────────────────────────────────────────────

def cmd_create(args):
    q = Path(args.queue_dir)
    q.mkdir(parents=True, exist_ok=True)
    queue_file = q / "queue.json"
    if queue_file.exists():
        print(f"Queue already exists: {q}")
        return
    # Create an empty queue by importing and using the module's internals
    import json
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    queue_file.write_text(json.dumps({
        "queue_id": q.name,
        "schema_version": 1,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "jobs": [],
    }, indent=2), encoding="utf-8")
    print(f"Created queue: {q}")


def cmd_add(args):
    m = _import_queue()
    Req = _import_request()
    req = Req(
        query=args.query,
        collection_size=args.collection_size,
        output_root=args.output_root or str(Path(args.queue_dir).parent / "outputs"),
        selected_concept_index=args.selected_concept_index,
        max_image_retries=args.max_image_retries,
        skip_mockups=args.skip_mockups,
        skip_listing=args.skip_listing,
    )
    job = m["enqueue_job"](args.queue_dir, req)
    print(f"Enqueued {job.job_id}: {args.query!r}")


def cmd_run(args):
    m = _import_queue()
    cb = _progress_printer() if getattr(args, "show_progress", False) else None
    result = m["run_queue"](args.queue_dir, stop_on_failure=args.stop_on_failure,
                            _progress_callback=cb)
    _print_result(result)


def cmd_resume(args):
    m = _import_queue()
    cb = _progress_printer() if getattr(args, "show_progress", False) else None
    result = m["resume_queue"](args.queue_dir, stop_on_failure=args.stop_on_failure,
                               _progress_callback=cb)
    _print_result(result)


def cmd_list(args):
    m = _import_queue()
    jobs = m["list_jobs"](args.queue_dir)
    if not jobs:
        print("Queue is empty.")
        return
    w = max(len(j.job_id) for j in jobs)
    for j in jobs:
        run = f"  run_dir={j.run_dir}" if j.run_dir else ""
        err = f"  error={j.error_message!r}" if j.error_message else ""
        cost = f"  cost={j.total_cost}" if j.total_cost is not None else ""
        pct = ""
        if j.run_dir:
            try:
                from agent.progress_tracking import get_production_progress
                prog = get_production_progress(j.run_dir)
                pct = f"  progress={prog.percent:.0f}%"
            except Exception:
                pass
        print(f"{j.job_id:<{w}}  {j.status:<12}  {j.request.get('query','')!r}{cost}{pct}{run}{err}")


def cmd_cancel(args):
    m = _import_queue()
    job = m["cancel_job"](args.queue_dir, args.job_id)
    print(f"Cancelled {job.job_id}")


def cmd_unlock(args):
    if not args.force:
        print("Pass --force to confirm removing the lock file.", file=sys.stderr)
        sys.exit(1)
    m = _import_queue()
    m["force_unlock"](args.queue_dir)


def _progress_printer():
    """Return a callback that prints one concise line per ProgressEvent."""
    def cb(event):
        pct = f"{event.percent:.1f}%" if event.percent is not None else "?"
        stage = event.stage_name or event.event_type
        poster = ""
        if event.poster_index is not None and event.poster_total is not None:
            poster = f" · poster {event.poster_index}/{event.poster_total}"
        attempt = ""
        if event.attempt_number is not None:
            attempt = f" · attempt {event.attempt_number}"
        job = f"{event.job_id} · " if event.job_id else ""
        print(f"[{pct}] {job}{stage}{poster}{attempt}", flush=True)
    return cb


def _print_result(result):
    print(f"\nQueue {result.status.upper()}")
    cost_str = f"  total_cost={result.total_cost}" if result.total_cost is not None else ""
    print(f"  total={result.total_jobs}  completed={result.completed_jobs}"
          f"  failed={result.failed_jobs}  pending={result.pending_jobs}"
          f"  cancelled={result.cancelled_jobs}{cost_str}")
    for s in result.job_summaries:
        marker = "✓" if s.status == "completed" else ("✗" if s.status == "failed" else "·")
        err = f"  {s.error_message!r}" if s.error_message else ""
        print(f"  {marker} {s.job_id}  {s.status}{err}")


# ── Argument parser ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_queue.py",
        description="Persistent job queue for etsy-ai-agent production runs.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # create
    c = sub.add_parser("create", help="Initialise a new empty queue directory")
    c.add_argument("queue_dir")

    # add
    a = sub.add_parser("add", help="Add a production job to the queue")
    a.add_argument("queue_dir")
    a.add_argument("--query", required=True, help="Search query / niche")
    a.add_argument("--collection-size", type=int, default=3, dest="collection_size")
    a.add_argument("--output-root", default=None, dest="output_root",
                   help="Root dir for production outputs (default: <queue_dir>/../outputs)")
    a.add_argument("--selected-concept-index", type=int, default=None,
                   dest="selected_concept_index")
    a.add_argument("--max-image-retries", type=int, default=1, dest="max_image_retries")
    a.add_argument("--skip-mockups", action="store_true", dest="skip_mockups")
    a.add_argument("--skip-listing", action="store_true", dest="skip_listing")

    # run
    r = sub.add_parser("run", help="Process all pending jobs in the queue")
    r.add_argument("queue_dir")
    r.add_argument("--stop-on-failure", action="store_true", dest="stop_on_failure")
    r.add_argument("--show-progress", action="store_true", dest="show_progress",
                   help="Print one line per progress event (does not affect execution)")

    # resume
    rs = sub.add_parser("resume", help="Resume an interrupted queue")
    rs.add_argument("queue_dir")
    rs.add_argument("--stop-on-failure", action="store_true", dest="stop_on_failure")
    rs.add_argument("--show-progress", action="store_true", dest="show_progress",
                    help="Print one line per progress event (does not affect execution)")

    # list
    ls = sub.add_parser("list", help="List all jobs in the queue")
    ls.add_argument("queue_dir")

    # cancel
    ca = sub.add_parser("cancel", help="Cancel a pending or failed job")
    ca.add_argument("queue_dir")
    ca.add_argument("job_id")

    # unlock
    ul = sub.add_parser("unlock", help="Remove a stale queue lock (use after process crash)")
    ul.add_argument("queue_dir")
    ul.add_argument("--force", action="store_true",
                    help="Required to confirm lock removal")

    return p


_HANDLERS = {
    "create": cmd_create,
    "add":    cmd_add,
    "run":    cmd_run,
    "resume": cmd_resume,
    "list":   cmd_list,
    "cancel": cmd_cancel,
    "unlock": cmd_unlock,
}


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        _HANDLERS[args.command](args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
