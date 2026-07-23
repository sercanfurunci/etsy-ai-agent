"""
Stage 10.5 — Progress tracking and ETA for production runs and queues.

Progress files (per run):
  <run_dir>/progress/events.jsonl   — append-only event history
  <run_dir>/progress/snapshot.json  — current progress snapshot (atomic write)

Source of truth is always manifest.json + image attempt files.
Progress files are an optimized projection and can be reconstructed.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

# ── Schema ─────────────────────────────────────────────────────────────────────
SCHEMA_VERSION = 1

# ── Stage weights (total = 100) ────────────────────────────────────────────────
# vision_critique and retry_generation run inside the image loop, so their
# weights combine with image_generation into one "image block" (total 60).
# When image_generation is running, partial progress comes from poster count.
STAGE_WEIGHTS: dict[str, int] = {
    "research": 5,
    "concept_generation": 5,
    "concept_selection": 2,
    "prompt_optimization": 5,
    "collection_generation": 8,
    "image_generation": 45,
    "vision_critique": 10,
    "retry_generation": 5,
    "mockup_generation": 7,
    "listing_generation": 6,
    "finalize": 2,
}
assert sum(STAGE_WEIGHTS.values()) == 100

_IMAGE_BLOCK_STAGES = frozenset({"image_generation", "vision_critique", "retry_generation"})
_IMAGE_BLOCK_WEIGHT = sum(STAGE_WEIGHTS[s] for s in _IMAGE_BLOCK_STAGES)  # 60

# ── Fallback durations (seconds) — estimates only, not factual provider speeds ─
FALLBACK_DURATIONS: dict[str, float] = {
    "research": 5.0,
    "concept_generation": 12.0,
    "concept_selection": 0.5,
    "prompt_optimization": 10.0,
    "collection_generation": 20.0,
    "image_attempt": 30.0,
    "vision_critique": 8.0,
    "retry_generation": 5.0,
    "mockup_generation": 15.0,
    "listing_generation": 15.0,
    "finalize": 1.0,
}


# ── Event / scope constants ────────────────────────────────────────────────────

class EventScope:
    PRODUCTION = "production"
    QUEUE = "queue"


class EventType:
    # Production
    RUN_STARTED = "run_started"
    RUN_RESUMED = "run_resumed"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    STAGE_SKIPPED = "stage_skipped"
    POSTER_STARTED = "poster_started"
    IMAGE_ATTEMPT_STARTED = "image_attempt_started"
    IMAGE_ATTEMPT_COMPLETED = "image_attempt_completed"
    POSTER_COMPLETED = "poster_completed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    # Queue
    QUEUE_STARTED = "queue_started"
    QUEUE_RESUMED = "queue_resumed"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"
    QUEUE_COMPLETED = "queue_completed"
    QUEUE_PAUSED = "queue_paused"
    QUEUE_FAILED = "queue_failed"


class ProgressSchemaError(ValueError):
    """Raised when a progress file has an unsupported schema_version."""


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class ProgressEvent:
    event_id: str
    scope: str
    run_id: str | None
    queue_id: str | None
    job_id: str | None
    event_type: str
    stage_name: str | None
    stage_status: str | None
    poster_index: int | None
    poster_total: int | None
    attempt_number: int | None
    completed_units: int | None
    total_units: int | None
    percent: float | None
    message: str
    timestamp: str
    sequence: int
    schema_version: int = SCHEMA_VERSION
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "scope": self.scope,
            "run_id": self.run_id,
            "queue_id": self.queue_id,
            "job_id": self.job_id,
            "event_type": self.event_type,
            "stage_name": self.stage_name,
            "stage_status": self.stage_status,
            "poster_index": self.poster_index,
            "poster_total": self.poster_total,
            "attempt_number": self.attempt_number,
            "completed_units": self.completed_units,
            "total_units": self.total_units,
            "percent": self.percent,
            "message": self.message,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProgressEvent":
        sv = d.get("schema_version", 1)
        if sv != SCHEMA_VERSION:
            raise ProgressSchemaError(f"Unsupported event schema_version: {sv}")
        return cls(
            schema_version=sv,
            event_id=d.get("event_id", ""),
            scope=d.get("scope", ""),
            run_id=d.get("run_id"),
            queue_id=d.get("queue_id"),
            job_id=d.get("job_id"),
            event_type=d.get("event_type", ""),
            stage_name=d.get("stage_name"),
            stage_status=d.get("stage_status"),
            poster_index=d.get("poster_index"),
            poster_total=d.get("poster_total"),
            attempt_number=d.get("attempt_number"),
            completed_units=d.get("completed_units"),
            total_units=d.get("total_units"),
            percent=d.get("percent"),
            message=d.get("message", ""),
            timestamp=d.get("timestamp", ""),
            sequence=d.get("sequence", 0),
            metadata=d.get("metadata", {}),
        )


@dataclass
class StageProgress:
    stage_name: str
    status: str
    weight: int
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None


@dataclass
class ProductionProgress:
    schema_version: int
    run_id: str
    status: str
    percent: float
    current_stage: str | None
    poster_completed: int
    poster_total: int
    current_poster: int | None
    current_attempt: int | None
    elapsed_seconds: float
    eta_seconds: float | None
    eta_available: bool
    eta_confidence: str          # low | medium | high
    eta_basis: str
    estimated_completion_at: str | None
    stages: list[StageProgress]
    created_at: str
    first_started_at: str | None
    last_updated_at: str
    last_event_sequence: int = 0


@dataclass
class QueueProgress:
    schema_version: int
    queue_dir: str
    status: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    pending_jobs: int
    running_jobs: int
    cancelled_jobs: int
    percent: float
    active_job_id: str | None
    active_job_position: int | None
    current_job_progress: ProductionProgress | None
    elapsed_seconds: float
    eta_seconds: float | None
    eta_available: bool
    eta_confidence: str
    estimated_completion_at: str | None


# ── Time helpers ───────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _duration_seconds(start: str | None, end: str | None = None) -> float | None:
    t0 = _parse_iso(start)
    if t0 is None:
        return None
    t1 = _parse_iso(end) if end else datetime.now(timezone.utc)
    delta = (t1 - t0).total_seconds()
    return max(0.0, delta)


# ── Image block partial progress ───────────────────────────────────────────────

def _count_poster_dirs(images_dir: Path) -> tuple[int, int]:
    """Return (completed_count, active_partial_count) from images directory.

    completed_count: dirs with final.png
    active_partial_count: dirs with original.png but no final.png (0 or 1 partial unit)
    """
    if not images_dir.exists():
        return 0, 0
    completed = 0
    has_active = False
    for d in images_dir.iterdir():
        if not d.is_dir():
            continue
        if (d / "final.png").exists():
            completed += 1
        elif (d / "original.png").exists():
            has_active = True
    return completed, (1 if has_active else 0)


def _image_block_earned(
    img_status: str,
    prod_dir: Path,
    collection_size: int,
) -> float:
    """Earned weight for the image block (image_generation + vision_critique + retry_generation).

    Image partial-progress formula:
      - Each poster is one required unit.
      - Completed poster (has final.png) = 1.0 unit.
      - Active poster (has original.png, no final.png) = 0.5 unit (partial credit, capped < 1).
      - Untouched poster = 0 units.
      - fraction = (completed + 0.5 * has_active) / total_posters
      - earned = fraction * IMAGE_BLOCK_WEIGHT (60)

    A retry never causes progress to decrease because completed_posters is monotonic.
    """
    if img_status in ("completed", "skipped"):
        return float(_IMAGE_BLOCK_WEIGHT)
    if img_status in ("pending",):
        return 0.0
    # running or failed — scan filesystem
    if collection_size <= 0:
        return 0.0
    completed, has_active = _count_poster_dirs(prod_dir / "images")
    fraction = (completed + 0.5 * has_active) / collection_size
    fraction = min(fraction, 0.99)  # never reach 100% while still running
    return fraction * _IMAGE_BLOCK_WEIGHT


# ── Percent computation ────────────────────────────────────────────────────────

def _compute_percent(
    stages: list[dict],
    prod_dir: Path,
    collection_size: int,
    run_status: str,
) -> float:
    """Compute overall run percent from manifest stage list."""
    if run_status == "completed":
        return 100.0

    img_status = "pending"
    earned = 0.0

    for s in stages:
        name = s["stage_name"]
        status = s["status"]
        if name in _IMAGE_BLOCK_STAGES:
            if name == "image_generation":
                img_status = status
            continue
        w = STAGE_WEIGHTS.get(name, 0)
        if status in ("completed", "skipped"):
            earned += w
        # running/pending/failed contribute 0 (no partial for non-image stages)

    earned += _image_block_earned(img_status, prod_dir, collection_size)
    return min(100.0, max(0.0, earned))


# ── Stage durations from manifest ──────────────────────────────────────────────

def _stage_durations(stages: list[dict]) -> dict[str, float]:
    """Return {stage_name: duration_seconds} for completed/skipped stages."""
    result: dict[str, float] = {}
    for s in stages:
        dur = _duration_seconds(s.get("started_at") or None, s.get("completed_at"))
        if dur is not None and s["status"] in ("completed", "skipped"):
            result[s["stage_name"]] = dur
    return result


def _avg_poster_duration(prod_dir: Path) -> float | None:
    """Average seconds per completed poster from attempts.json timestamps."""
    images_dir = prod_dir / "images"
    if not images_dir.exists():
        return None
    durations: list[float] = []
    for d in images_dir.iterdir():
        if not d.is_dir() or not (d / "final.png").exists():
            continue
        att_file = d / "attempts.json"
        if not att_file.exists():
            continue
        try:
            attempts = json.loads(att_file.read_text(encoding="utf-8"))
            times = [a["created_at"] for a in attempts if a.get("created_at")]
            if len(times) >= 1:
                # duration = last attempt time minus first attempt time + buffer
                t0 = _parse_iso(times[0])
                t1 = _parse_iso(times[-1])
                if t0 and t1:
                    dur = max(1.0, (t1 - t0).total_seconds())
                    durations.append(dur)
        except Exception:
            continue
    if not durations:
        return None
    return sum(durations) / len(durations)


def _avg_retry_factor(prod_dir: Path) -> float:
    """Average attempts per completed poster (>= 1.0)."""
    images_dir = prod_dir / "images"
    if not images_dir.exists():
        return 1.0
    attempt_counts: list[int] = []
    for d in images_dir.iterdir():
        if not d.is_dir() or not (d / "final.png").exists():
            continue
        att_file = d / "attempts.json"
        if not att_file.exists():
            continue
        try:
            attempts = json.loads(att_file.read_text(encoding="utf-8"))
            attempt_counts.append(len(attempts))
        except Exception:
            continue
    if not attempt_counts:
        return 1.0
    return sum(attempt_counts) / len(attempt_counts)


# ── ETA computation ────────────────────────────────────────────────────────────

def _compute_eta(
    stages: list[dict],
    prod_dir: Path,
    collection_size: int,
    run_status: str,
    percent: float,
    now_dt: datetime | None = None,
) -> tuple[float | None, bool, str, str]:
    """Return (eta_seconds, eta_available, confidence, basis)."""
    if run_status == "completed":
        return 0.0, True, "high", "run_completed"

    if now_dt is None:
        now_dt = datetime.now(timezone.utc)

    durations = _stage_durations(stages)
    has_history = bool(durations)

    remaining = 0.0

    for s in stages:
        name = s["stage_name"]
        status = s["status"]
        if status in ("completed", "skipped"):
            continue
        if name in _IMAGE_BLOCK_STAGES:
            continue  # handled below
        fallback = FALLBACK_DURATIONS.get(name, 10.0)
        remaining += durations.get(name, fallback)

    # Image block remaining
    img_status = next((s["status"] for s in stages if s["stage_name"] == "image_generation"), "pending")
    if img_status not in ("completed", "skipped"):
        completed_posters, _ = _count_poster_dirs(prod_dir / "images")
        remaining_posters = max(0, collection_size - completed_posters)
        avg_dur = _avg_poster_duration(prod_dir)
        retry_factor = _avg_retry_factor(prod_dir) if has_history else 1.2

        if avg_dur is not None:
            has_history = True
            per_poster = avg_dur * retry_factor
        else:
            per_poster = (
                FALLBACK_DURATIONS["image_attempt"]
                + FALLBACK_DURATIONS["vision_critique"]
                + FALLBACK_DURATIONS["retry_generation"]
            ) * 1.2
        remaining += remaining_posters * per_poster

    if remaining < 0:
        remaining = 0.0

    # Confidence
    if percent >= 95:
        confidence = "high"
    elif has_history and percent >= 20:
        confidence = "medium"
    else:
        confidence = "low"

    basis = "historical_stage_durations" if has_history else "static_fallback"
    eta_available = has_history or collection_size > 0

    # Estimated completion timestamp
    est_at: str | None = None
    if eta_available:
        try:
            from datetime import timedelta
            est_dt = now_dt + timedelta(seconds=remaining)
            est_at = est_dt.isoformat()
        except Exception:
            pass

    return remaining, eta_available, confidence, basis


# ── Build ProductionProgress from manifest ────────────────────────────────────

def _build_production_progress(
    manifest: dict,
    prod_dir: Path,
    now: datetime | None = None,
    last_event_sequence: int = 0,
) -> ProductionProgress:
    run_id = manifest.get("production_id", prod_dir.name)
    status = manifest.get("status", "pending")
    collection_size = manifest.get("collection_size", 0)
    stages_raw = manifest.get("stages", [])

    percent = _compute_percent(stages_raw, prod_dir, collection_size, status)

    # Current stage
    current_stage = manifest.get("current_stage") or None
    if status == "completed":
        current_stage = "finalize"

    # Poster counts
    images_dir = prod_dir / "images"
    completed_posters, _ = _count_poster_dirs(images_dir)
    poster_total = collection_size

    # Current poster / attempt (scan for active)
    current_poster: int | None = None
    current_attempt: int | None = None
    if images_dir.exists() and current_stage == "image_generation":
        for d in sorted(images_dir.iterdir()):
            if not d.is_dir():
                continue
            if not (d / "final.png").exists() and (d / "original.png").exists():
                # Active poster — parse index from dir name
                try:
                    current_poster = int(d.name.split("_")[-1])
                except Exception:
                    pass
                att_file = d / "attempts.json"
                if att_file.exists():
                    try:
                        ats = json.loads(att_file.read_text(encoding="utf-8"))
                        current_attempt = len(ats)
                    except Exception:
                        pass
                break

    # Elapsed
    now_dt = now or datetime.now(timezone.utc)
    first_started = manifest.get("created_at")
    elapsed = _duration_seconds(first_started) or 0.0
    if status == "completed":
        completed_at = manifest.get("updated_at")
        elapsed = _duration_seconds(first_started, completed_at) or elapsed

    # ETA
    eta_secs, eta_avail, confidence, basis = _compute_eta(
        stages_raw, prod_dir, collection_size, status, percent, now_dt
    )

    est_at: str | None = None
    if eta_avail and eta_secs is not None:
        try:
            from datetime import timedelta
            est_at = (now_dt + timedelta(seconds=eta_secs)).isoformat()
        except Exception:
            pass

    stage_list = [
        StageProgress(
            stage_name=s["stage_name"],
            status=s["status"],
            weight=STAGE_WEIGHTS.get(s["stage_name"], 0),
            started_at=s.get("started_at") or None,
            completed_at=s.get("completed_at"),
            duration_seconds=_duration_seconds(
                s.get("started_at") or None, s.get("completed_at")
            ),
        )
        for s in stages_raw
    ]

    return ProductionProgress(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        status=status,
        percent=percent,
        current_stage=current_stage,
        poster_completed=completed_posters,
        poster_total=poster_total,
        current_poster=current_poster,
        current_attempt=current_attempt,
        elapsed_seconds=elapsed,
        eta_seconds=eta_secs,
        eta_available=eta_avail,
        eta_confidence=confidence,
        eta_basis=basis,
        estimated_completion_at=est_at,
        stages=stage_list,
        created_at=manifest.get("created_at", ""),
        first_started_at=manifest.get("created_at"),
        last_updated_at=manifest.get("updated_at", ""),
        last_event_sequence=last_event_sequence,
    )


# ── ProgressTracker ────────────────────────────────────────────────────────────

class ProgressTracker:
    """
    Emits ProgressEvents, appends them to events.jsonl, and writes atomic snapshots.

    Failures in callback or persistence never propagate to the caller.
    """

    def __init__(
        self,
        run_id: str,
        progress_dir: Path,
        callback: Callable[[ProgressEvent], None] | None = None,
        initial_sequence: int = 0,
    ) -> None:
        self.run_id = run_id
        self.progress_dir = Path(progress_dir)
        self._callback = callback
        self._sequence = initial_sequence
        self.progress_dir.mkdir(parents=True, exist_ok=True)

    # ── Sequence ──────────────────────────────────────────────────────────────

    @classmethod
    def load_sequence_from_dir(cls, progress_dir: Path) -> int:
        snap = progress_dir / "snapshot.json"
        if snap.exists():
            try:
                data = json.loads(snap.read_text(encoding="utf-8"))
                return int(data.get("last_event_sequence", 0))
            except Exception:
                pass
        return 0

    # ── Event emission ────────────────────────────────────────────────────────

    def emit(
        self,
        event_type: str,
        *,
        stage_name: str | None = None,
        stage_status: str | None = None,
        poster_index: int | None = None,
        poster_total: int | None = None,
        attempt_number: int | None = None,
        completed_units: int | None = None,
        total_units: int | None = None,
        percent: float | None = None,
        message: str = "",
        metadata: dict | None = None,
        job_id: str | None = None,
        queue_id: str | None = None,
    ) -> ProgressEvent:
        self._sequence += 1
        event = ProgressEvent(
            event_id=uuid.uuid4().hex,
            scope=EventScope.PRODUCTION,
            run_id=self.run_id,
            queue_id=queue_id,
            job_id=job_id,
            event_type=event_type,
            stage_name=stage_name,
            stage_status=stage_status,
            poster_index=poster_index,
            poster_total=poster_total,
            attempt_number=attempt_number,
            completed_units=completed_units,
            total_units=total_units,
            percent=percent,
            message=message,
            timestamp=_utcnow(),
            sequence=self._sequence,
            metadata=metadata or {},
        )
        self._persist_event(event)
        if self._callback is not None:
            try:
                self._callback(event)
            except Exception:
                pass
        return event

    def _persist_event(self, event: ProgressEvent) -> None:
        try:
            jsonl = self.progress_dir / "events.jsonl"
            with open(jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception:
            pass

    def save_snapshot(self, progress: ProductionProgress) -> None:
        try:
            path = self.progress_dir / "snapshot.json"
            tmp = path.with_suffix(".tmp")
            data = _production_progress_to_dict(progress)
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            pass


def _production_progress_to_dict(p: ProductionProgress) -> dict:
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
        "created_at": p.created_at,
        "first_started_at": p.first_started_at,
        "last_updated_at": p.last_updated_at,
        "last_event_sequence": p.last_event_sequence,
    }


# ── Public API: production progress ───────────────────────────────────────────

def get_production_progress(
    run_dir: str | Path,
    *,
    now: datetime | None = None,
) -> ProductionProgress:
    """
    Return current ProductionProgress for a run directory.

    Reconstruction hierarchy:
      1. Try progress/snapshot.json (fast path)
      2. Reconstruct from manifest.json + filesystem (always works if manifest exists)

    If events.jsonl is corrupt, the current snapshot lookup still works.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"manifest.json is not valid JSON: {e}") from e

    # Try snapshot for fast path (e.g. current sequence number)
    last_seq = 0
    snap_path = run_dir / "progress" / "snapshot.json"
    if snap_path.exists():
        try:
            snap = json.loads(snap_path.read_text(encoding="utf-8"))
            sv = snap.get("schema_version", 1)
            if sv != SCHEMA_VERSION:
                raise ProgressSchemaError(f"Unsupported snapshot schema_version: {sv}")
            last_seq = snap.get("last_event_sequence", 0)
        except ProgressSchemaError:
            raise
        except Exception:
            last_seq = 0  # corrupt snapshot → reconstruct

    return _build_production_progress(manifest, run_dir, now=now, last_event_sequence=last_seq)


def load_events(run_dir: str | Path) -> tuple[list[ProgressEvent], bool]:
    """
    Load events from progress/events.jsonl.

    Returns (events, had_errors). Corrupt lines are skipped; had_errors=True warns callers.
    """
    run_dir = Path(run_dir)
    jsonl = run_dir / "progress" / "events.jsonl"
    if not jsonl.exists():
        return [], False
    events: list[ProgressEvent] = []
    had_errors = False
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            events.append(ProgressEvent.from_dict(d))
        except Exception:
            had_errors = True
    return events, had_errors


# ── Public API: queue progress ─────────────────────────────────────────────────

def get_queue_progress(
    queue_dir: str | Path,
    *,
    now: datetime | None = None,
) -> QueueProgress:
    """
    Return current QueueProgress aggregated from queue.json + per-job production progress.

    Queue percent model:
      - completed job = 1 full unit
      - failed job = 1 full unit (resolved for this queue pass)
      - cancelled job = 1 full unit (resolved)
      - pending job = 0
      - running job = its production progress fraction

    This gives percent=100 when all jobs are resolved (AQ requirement).
    """
    from agent.job_queue import _load_queue  # avoid circular at module level

    queue_dir = Path(queue_dir)
    manifest = _load_queue(queue_dir)

    now_dt = now or datetime.now(timezone.utc)
    jobs = sorted(manifest.jobs, key=lambda j: j.position)

    total = len(jobs)
    completed = sum(1 for j in jobs if j.status == "completed")
    failed = sum(1 for j in jobs if j.status == "failed")
    pending = sum(1 for j in jobs if j.status == "pending")
    running = sum(1 for j in jobs if j.status == "running")
    cancelled = sum(1 for j in jobs if j.status == "cancelled")

    # Percent
    resolved_units = completed + failed + cancelled
    active_fraction = 0.0
    active_job: Any = None
    active_prod_progress: ProductionProgress | None = None

    for j in jobs:
        if j.status == "running":
            active_job = j
            if j.run_dir and Path(j.run_dir).exists():
                try:
                    active_prod_progress = get_production_progress(j.run_dir, now=now_dt)
                    active_fraction = active_prod_progress.percent / 100.0
                except Exception:
                    active_fraction = 0.0
            break

    if total > 0:
        percent = min(100.0, max(0.0, (resolved_units + active_fraction) / total * 100.0))
    else:
        percent = 0.0

    # Elapsed
    elapsed = _duration_seconds(manifest.started_at) or 0.0
    if manifest.status in ("completed", "completed_with_failures", "failed"):
        elapsed = _duration_seconds(manifest.started_at, manifest.completed_at) or elapsed

    # ETA
    completed_job_durations: list[float] = []
    for j in jobs:
        if j.status == "completed" and j.started_at and j.completed_at:
            dur = _duration_seconds(j.started_at, j.completed_at)
            if dur is not None:
                completed_job_durations.append(dur)

    if manifest.status in ("completed", "completed_with_failures"):
        eta_secs: float | None = 0.0
        eta_avail = True
        eta_conf = "high"
    else:
        remaining_jobs = pending
        if completed_job_durations:
            med = sorted(completed_job_durations)[len(completed_job_durations) // 2]
            job_eta = remaining_jobs * med
            # Add active job remaining
            if active_prod_progress and active_prod_progress.eta_seconds is not None:
                job_eta += active_prod_progress.eta_seconds
            eta_secs = max(0.0, job_eta)
            eta_avail = True
            eta_conf = "medium" if completed_job_durations else "low"
        elif active_prod_progress and active_prod_progress.eta_seconds is not None:
            # Only active job ETA available
            fallback_per_job = sum(FALLBACK_DURATIONS.values()) * 1.5
            eta_secs = max(0.0, active_prod_progress.eta_seconds + remaining_jobs * fallback_per_job)
            eta_avail = True
            eta_conf = "low"
        else:
            eta_secs = None
            eta_avail = False
            eta_conf = "low"

    est_at: str | None = None
    if eta_avail and eta_secs is not None:
        try:
            from datetime import timedelta
            est_at = (now_dt + timedelta(seconds=eta_secs)).isoformat()
        except Exception:
            pass

    return QueueProgress(
        schema_version=SCHEMA_VERSION,
        queue_dir=str(queue_dir),
        status=manifest.status,
        total_jobs=total,
        completed_jobs=completed,
        failed_jobs=failed,
        pending_jobs=pending,
        running_jobs=running,
        cancelled_jobs=cancelled,
        percent=percent,
        active_job_id=active_job.job_id if active_job else None,
        active_job_position=active_job.position if active_job else None,
        current_job_progress=active_prod_progress,
        elapsed_seconds=elapsed,
        eta_seconds=eta_secs,
        eta_available=eta_avail,
        eta_confidence=eta_conf,
        estimated_completion_at=est_at,
    )


# ── Watch generators ───────────────────────────────────────────────────────────

def watch_production(
    run_dir: str | Path,
    *,
    interval_seconds: float = 1.0,
) -> Iterator[ProductionProgress]:
    """Poll run_dir for progress updates. Yields ProductionProgress on each tick."""
    run_dir = Path(run_dir)
    while True:
        try:
            yield get_production_progress(run_dir)
        except Exception:
            pass
        time.sleep(interval_seconds)


def watch_queue(
    queue_dir: str | Path,
    *,
    interval_seconds: float = 1.0,
) -> Iterator[QueueProgress]:
    """Poll queue_dir for progress updates. Yields QueueProgress on each tick."""
    queue_dir = Path(queue_dir)
    while True:
        try:
            yield get_queue_progress(queue_dir)
        except Exception:
            pass
        time.sleep(interval_seconds)
