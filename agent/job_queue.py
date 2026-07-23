"""
Persistent job queue for sequentially processing production runs.

Directory layout:
  <queue_dir>/
    queue.json       ← single source of truth (canonical)
    .queue.lock      ← held during run_queue / resume_queue

Production outputs stay under ProductionRequest.output_root; they are never
duplicated inside queue_dir.  queue.json stores the run_dir path once created.

Schema version: 1

State machines
--------------
Job statuses:  pending → running → completed | failed | cancelled
Queue statuses: pending → running → completed | completed_with_failures | failed | paused
"""
import dataclasses
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ── Constants ──────────────────────────────────────────────────────────────────

SCHEMA_VERSION = 1
_LOCK_FILE = ".queue.lock"
_QUEUE_FILE = "queue.json"

_JOB_STATUSES = frozenset({"pending", "running", "completed", "failed", "cancelled"})
_QUEUE_STATUSES = frozenset({
    "pending", "running", "completed", "completed_with_failures", "failed", "paused",
})


# ── Exceptions ─────────────────────────────────────────────────────────────────

class QueueError(RuntimeError):
    """Base error for all queue operations."""


class QueueLockedError(QueueError):
    """Raised when attempting to run a queue that is already locked."""


class QueueSchemaError(QueueError):
    """Raised when queue.json schema version is unknown or the file is malformed."""


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class QueueJob:
    job_id: str
    position: int
    status: str                          # pending|running|completed|failed|cancelled
    request: dict                        # serialized ProductionRequest
    run_dir: str | None = None           # set when run_production creates the dir
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str = ""
    error_message: str | None = None
    attempt_count: int = 0
    # Stage 10.4 cost fields (loaded from run costs/summary.json after each job)
    total_cost: str | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_images: int | None = None


@dataclass
class QueueManifest:
    queue_id: str
    schema_version: int
    status: str
    created_at: str
    updated_at: str
    jobs: list[QueueJob] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None


@dataclass
class QueueJobSummary:
    job_id: str
    status: str
    run_dir: str | None
    error_message: str | None


@dataclass
class QueueResult:
    queue_dir: str
    status: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    pending_jobs: int
    cancelled_jobs: int
    job_summaries: list[QueueJobSummary]
    started_at: str | None
    completed_at: str | None
    # Stage 10.4 cost aggregate across completed jobs
    total_cost: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_images: int = 0


@dataclass
class QueueDependencies:
    """
    Callable fields for queue-level operations.
    Injected in tests; production uses _default_queue_deps().

    run_production(request, _on_run_dir=None) -> ProductionResult
    resume_production(run_dir) -> ProductionResult
    """
    run_production: Callable
    resume_production: Callable


# ── Serialization helpers ──────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_id(position: int) -> str:
    return f"job_{position:03d}"


def _to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(x) for x in obj]
    return obj


# ── Atomic persistence ─────────────────────────────────────────────────────────

def _write_queue_atomic(queue_dir: Path, manifest: QueueManifest) -> None:
    path = queue_dir / _QUEUE_FILE
    tmp = path.with_suffix(".tmp")
    data = json.dumps(_to_dict(manifest), indent=2, ensure_ascii=False)
    try:
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ── Loading and validation ─────────────────────────────────────────────────────

def _load_queue(queue_dir: Path) -> QueueManifest:
    path = queue_dir / _QUEUE_FILE
    if not path.exists():
        raise FileNotFoundError(f"queue.json not found in {queue_dir}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QueueSchemaError(f"Malformed queue.json: {exc}") from exc

    if not isinstance(raw, dict):
        raise QueueSchemaError("queue.json must be a JSON object")

    sv = raw.get("schema_version")
    if sv is None:
        raise QueueSchemaError("queue.json missing schema_version")
    if sv != SCHEMA_VERSION:
        raise QueueSchemaError(
            f"Unsupported schema_version {sv!r}; expected {SCHEMA_VERSION}"
        )

    for key in ("queue_id", "status", "created_at", "updated_at", "jobs"):
        if key not in raw:
            raise QueueError(f"queue.json missing required field: {key!r}")

    if raw["status"] not in _QUEUE_STATUSES:
        raise QueueError(f"Invalid queue status: {raw['status']!r}")

    jobs_raw = raw["jobs"]
    if not isinstance(jobs_raw, list):
        raise QueueSchemaError("queue.json 'jobs' must be a list")

    jobs: list[QueueJob] = []
    seen_ids: set[str] = set()
    seen_positions: set[int] = set()

    for i, jd in enumerate(jobs_raw):
        if not isinstance(jd, dict):
            raise QueueSchemaError(f"Job entry {i} is not a dict")
        for key in ("job_id", "position", "status", "request"):
            if key not in jd:
                raise QueueError(f"Job entry {i} missing required field: {key!r}")
        jid = jd["job_id"]
        pos = jd["position"]
        if jd["status"] not in _JOB_STATUSES:
            raise QueueError(f"Job {jid!r} has invalid status: {jd['status']!r}")
        if not isinstance(jd["request"], dict):
            raise QueueError(f"Job {jid!r} 'request' must be a dict")
        if jid in seen_ids:
            raise QueueError(f"Duplicate job_id: {jid!r}")
        if pos in seen_positions:
            raise QueueError(f"Duplicate position: {pos!r}")
        seen_ids.add(jid)
        seen_positions.add(pos)
        jobs.append(QueueJob(
            job_id=jid,
            position=pos,
            status=jd["status"],
            request=jd["request"],
            run_dir=jd.get("run_dir"),
            created_at=jd.get("created_at", ""),
            started_at=jd.get("started_at"),
            completed_at=jd.get("completed_at"),
            updated_at=jd.get("updated_at", ""),
            error_message=jd.get("error_message"),
            attempt_count=jd.get("attempt_count", 0),
            total_cost=jd.get("total_cost"),
            total_input_tokens=jd.get("total_input_tokens"),
            total_output_tokens=jd.get("total_output_tokens"),
            total_images=jd.get("total_images"),
        ))

    return QueueManifest(
        queue_id=raw["queue_id"],
        schema_version=raw["schema_version"],
        status=raw["status"],
        created_at=raw["created_at"],
        updated_at=raw["updated_at"],
        jobs=jobs,
        started_at=raw.get("started_at"),
        completed_at=raw.get("completed_at"),
    )


def _get_job(manifest: QueueManifest, job_id: str) -> QueueJob:
    for job in manifest.jobs:
        if job.job_id == job_id:
            return job
    raise QueueError(f"Job not found: {job_id!r}")


def _validate_completed_job_outputs(job: QueueJob) -> str | None:
    """
    Returns an error string if a completed job's run_dir is invalid, else None.
    A completed job is only trusted if its run_dir exists and has status=completed.
    """
    if not job.run_dir:
        return "run_dir is not set"
    run_dir = Path(job.run_dir)
    if not run_dir.exists():
        return f"run_dir does not exist: {run_dir}"
    manifest_file = run_dir / "manifest.json"
    if not manifest_file.exists():
        return "run_dir missing manifest.json"
    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        if data.get("status") != "completed":
            return f"production status is {data.get('status')!r}, not 'completed'"
    except Exception as exc:
        return f"cannot read production manifest: {exc}"
    return None


# ── Request serialization ──────────────────────────────────────────────────────

def _request_to_dict(request: Any) -> dict:
    return dataclasses.asdict(request)


def _dict_to_request(d: dict) -> Any:
    from agent.production_orchestrator import ProductionRequest
    return ProductionRequest(
        query=d["query"],
        collection_size=d["collection_size"],
        output_root=d["output_root"],
        selected_concept_index=d.get("selected_concept_index"),
        max_image_retries=d.get("max_image_retries", 1),
        skip_mockups=d.get("skip_mockups", False),
        skip_listing=d.get("skip_listing", False),
        enable_cost_tracking=d.get("enable_cost_tracking", True),
    )


def _refresh_job_cost(job: QueueJob) -> None:
    """Load cost summary from completed job's run directory into job fields."""
    if not job.run_dir:
        return
    summary_path = Path(job.run_dir) / "costs" / "summary.json"
    if not summary_path.exists():
        return
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        job.total_cost = data.get("total_cost")
        job.total_input_tokens = data.get("total_input_tokens")
        job.total_output_tokens = data.get("total_output_tokens")
        job.total_images = data.get("total_images")
    except Exception:
        pass


# ── Lock ───────────────────────────────────────────────────────────────────────

def _acquire_lock(queue_dir: Path) -> None:
    lock = queue_dir / _LOCK_FILE
    if lock.exists():
        try:
            data = json.loads(lock.read_text(encoding="utf-8"))
            pid = data.get("pid", "?")
            ts = data.get("timestamp", "?")
        except Exception:
            pid, ts = "?", "?"
        raise QueueLockedError(
            f"Queue is locked by PID {pid} since {ts}. "
            f"If that process has exited, run: unlock --force"
        )
    lock.write_text(
        json.dumps({"pid": os.getpid(), "timestamp": _now()}),
        encoding="utf-8",
    )


def _release_lock(queue_dir: Path) -> None:
    lock = queue_dir / _LOCK_FILE
    lock.unlink(missing_ok=True)


# ── Default real deps ──────────────────────────────────────────────────────────

def _default_queue_deps() -> QueueDependencies:
    from agent.production_orchestrator import run_production, resume_production
    return QueueDependencies(
        run_production=run_production,
        resume_production=resume_production,
    )


# ── Result builder ─────────────────────────────────────────────────────────────

def _build_result(queue_dir: Path, manifest: QueueManifest) -> QueueResult:
    from decimal import Decimal
    completed = sum(1 for j in manifest.jobs if j.status == "completed")
    failed    = sum(1 for j in manifest.jobs if j.status == "failed")
    pending   = sum(1 for j in manifest.jobs if j.status == "pending")
    cancelled = sum(1 for j in manifest.jobs if j.status == "cancelled")

    total_cost_dec = Decimal("0")
    any_null_cost = False
    total_in = 0
    total_out = 0
    total_img = 0
    for j in manifest.jobs:
        if j.status == "completed":
            if j.total_cost is None:
                any_null_cost = True
            else:
                total_cost_dec += Decimal(j.total_cost)
            total_in += j.total_input_tokens or 0
            total_out += j.total_output_tokens or 0
            total_img += j.total_images or 0

    return QueueResult(
        queue_dir=str(queue_dir),
        status=manifest.status,
        total_jobs=len(manifest.jobs),
        completed_jobs=completed,
        failed_jobs=failed,
        pending_jobs=pending,
        cancelled_jobs=cancelled,
        job_summaries=[
            QueueJobSummary(
                job_id=j.job_id,
                status=j.status,
                run_dir=j.run_dir,
                error_message=j.error_message,
            )
            for j in manifest.jobs
        ],
        started_at=manifest.started_at,
        completed_at=manifest.completed_at,
        total_cost=None if (any_null_cost or not completed) else f"{total_cost_dec:.8f}",
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_images=total_img,
    )


# ── Queue status finalization ──────────────────────────────────────────────────

def _finalize_queue_status(manifest: QueueManifest, queue_dir: Path) -> None:
    now = _now()
    manifest.updated_at = now
    manifest.completed_at = manifest.completed_at or now

    completed = sum(1 for j in manifest.jobs if j.status == "completed")
    failed    = sum(1 for j in manifest.jobs if j.status == "failed")
    pending   = sum(1 for j in manifest.jobs if j.status == "pending")

    if pending > 0:
        manifest.status = "paused"
    elif failed > 0 and completed > 0:
        manifest.status = "completed_with_failures"
    elif failed > 0:
        manifest.status = "failed"
    else:
        manifest.status = "completed"

    _write_queue_atomic(queue_dir, manifest)


# ── Single job execution ───────────────────────────────────────────────────────

def _execute_job(
    job: QueueJob,
    manifest: QueueManifest,
    queue_dir: Path,
    deps: QueueDependencies,
    is_resume: bool,
    progress_callback: Any = None,
) -> None:
    """
    Execute one job; updates job in-place and writes manifest after each transition.
    Raises on production failure (caller decides whether to continue or stop).
    """
    now = _now()
    job.status = "running"
    job.started_at = job.started_at or now
    job.updated_at = now
    job.attempt_count += 1
    manifest.updated_at = now
    _write_queue_atomic(queue_dir, manifest)

    request = _dict_to_request(job.request)

    def on_run_dir(path: Path) -> None:
        """Called by run_production the moment the run directory is created."""
        job.run_dir = str(path)
        job.updated_at = _now()
        manifest.updated_at = job.updated_at
        _write_queue_atomic(queue_dir, manifest)

    try:
        if is_resume and job.run_dir and Path(job.run_dir).exists():
            deps.resume_production(job.run_dir, _progress_callback=progress_callback)
        else:
            deps.run_production(request, _on_run_dir=on_run_dir, _progress_callback=progress_callback)

        now = _now()
        job.status = "completed"
        job.completed_at = now
        job.updated_at = now
        job.error_message = None
        manifest.updated_at = now
        _refresh_job_cost(job)
        _write_queue_atomic(queue_dir, manifest)

    except Exception as exc:
        now = _now()
        job.status = "failed"
        job.error_message = str(exc)
        job.updated_at = now
        manifest.updated_at = now
        _write_queue_atomic(queue_dir, manifest)
        raise


# ── Core shared runner ─────────────────────────────────────────────────────────

def _run_or_resume_queue(
    queue_dir: Path,
    deps: QueueDependencies,
    stop_on_failure: bool,
    is_resume: bool,
    progress_callback: Any = None,
) -> QueueResult:
    _acquire_lock(queue_dir)
    try:
        manifest = _load_queue(queue_dir)

        # Validate completed jobs on resume — untrusted if run_dir is gone
        if is_resume:
            changed = False
            for job in manifest.jobs:
                if job.status != "completed":
                    continue
                err = _validate_completed_job_outputs(job)
                if err:
                    print(f"[queue] {job.job_id} marked completed but invalid ({err}) — resetting to failed")
                    job.status = "failed"
                    job.error_message = f"Validation failed on resume: {err}"
                    job.updated_at = _now()
                    changed = True
            if changed:
                manifest.updated_at = _now()
                _write_queue_atomic(queue_dir, manifest)

        # Fast path: nothing left to do
        actionable_statuses = {"pending"} if not is_resume else {"pending", "failed", "running"}
        if not any(j.status in actionable_statuses for j in manifest.jobs):
            return _build_result(queue_dir, manifest)

        # Start queue
        now = _now()
        manifest.status = "running"
        manifest.started_at = manifest.started_at or now
        manifest.updated_at = now
        _write_queue_atomic(queue_dir, manifest)

        for job in sorted(manifest.jobs, key=lambda j: j.position):
            if job.status in ("cancelled", "completed"):
                continue
            if job.status not in actionable_statuses:
                continue
            try:
                _execute_job(job, manifest, queue_dir, deps, is_resume=is_resume,
                             progress_callback=progress_callback)
            except Exception:
                if stop_on_failure:
                    _finalize_queue_status(manifest, queue_dir)
                    raise
                # stop_on_failure=False: persist failure, continue with next job

        _finalize_queue_status(manifest, queue_dir)
        return _build_result(queue_dir, manifest)

    finally:
        _release_lock(queue_dir)


# ── Public API ─────────────────────────────────────────────────────────────────

def enqueue_job(
    queue_dir: str | Path,
    request: Any,
) -> QueueJob:
    """
    Append a new pending job to the queue.

    Creates queue_dir and queue.json if they do not exist.
    Never executes the job or calls any production API.
    Repeated identical requests create separate jobs (no deduplication).
    """
    from agent.production_orchestrator import _validate_request
    _validate_request(request)

    queue_dir = Path(queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)

    queue_file = queue_dir / _QUEUE_FILE
    if queue_file.exists():
        manifest = _load_queue(queue_dir)
    else:
        now = _now()
        manifest = QueueManifest(
            queue_id=queue_dir.name,
            schema_version=SCHEMA_VERSION,
            status="pending",
            created_at=now,
            updated_at=now,
        )

    position = len(manifest.jobs) + 1
    now = _now()
    job = QueueJob(
        job_id=_job_id(position),
        position=position,
        status="pending",
        request=_request_to_dict(request),
        created_at=now,
        updated_at=now,
    )
    manifest.jobs.append(job)
    manifest.updated_at = now
    _write_queue_atomic(queue_dir, manifest)
    return job


def run_queue(
    queue_dir: str | Path,
    _deps: QueueDependencies | None = None,
    stop_on_failure: bool = False,
    _progress_callback: Any = None,
) -> QueueResult:
    """
    Process all pending jobs in insertion order.

    Failed or running jobs from a previous run are left untouched; use
    resume_queue() to retry them.

    stop_on_failure=True  → persist failure state and re-raise the exception
    stop_on_failure=False → record failure, continue with next pending job
    _progress_callback    → optional Callable[[ProgressEvent], None] for live events
    """
    queue_dir = Path(queue_dir)
    deps = _deps or _default_queue_deps()
    return _run_or_resume_queue(queue_dir, deps, stop_on_failure, is_resume=False,
                                progress_callback=_progress_callback)


def resume_queue(
    queue_dir: str | Path,
    _deps: QueueDependencies | None = None,
    stop_on_failure: bool = False,
    _progress_callback: Any = None,
) -> QueueResult:
    """
    Resume an interrupted queue.

    Per-job behaviour:
      completed → validates run_dir; if invalid, marks failed and retries
      cancelled → skipped permanently
      pending   → run_production(request)
      failed    → resume_production(run_dir) if run_dir exists, else run_production
      running   → treated as interrupted; same logic as failed
    _progress_callback → optional Callable[[ProgressEvent], None] for live events
    """
    queue_dir = Path(queue_dir)
    deps = _deps or _default_queue_deps()
    return _run_or_resume_queue(queue_dir, deps, stop_on_failure, is_resume=True,
                                progress_callback=_progress_callback)


def list_jobs(queue_dir: str | Path) -> list[QueueJob]:
    """Return all jobs in position order. Does not lock."""
    manifest = _load_queue(Path(queue_dir))
    return sorted(manifest.jobs, key=lambda j: j.position)


def cancel_job(queue_dir: str | Path, job_id: str) -> QueueJob:
    """
    Cancel a pending or failed job.

    Idempotent for already-cancelled jobs.
    Cannot cancel completed or running jobs.
    Does not delete production outputs.
    """
    queue_dir = Path(queue_dir)
    manifest = _load_queue(queue_dir)
    job = _get_job(manifest, job_id)

    if job.status == "cancelled":
        return job  # idempotent
    if job.status in ("completed", "running"):
        raise QueueError(f"Cannot cancel job {job_id!r} with status {job.status!r}")

    now = _now()
    job.status = "cancelled"
    job.updated_at = now
    manifest.updated_at = now
    _write_queue_atomic(queue_dir, manifest)
    return job


def force_unlock(queue_dir: str | Path) -> None:
    """Remove the queue lock file. Only call when the locking process has exited."""
    queue_dir = Path(queue_dir)
    lock = queue_dir / _LOCK_FILE
    if lock.exists():
        lock.unlink()
        print(f"[queue] Lock removed from {queue_dir}")
    else:
        print(f"[queue] No lock file found in {queue_dir}")
