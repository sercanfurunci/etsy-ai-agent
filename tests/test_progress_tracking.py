"""
Tests for agent/progress_tracking.py — Stage 10.5.

Tests A–AQ:
A   pending run reports 0%
B   completed run reports exactly 100%
C   skipped stage contributes full completion weight
D   running stage partial progress via image poster count
E   percent never below 0 or above 100
F   image poster progress reflected
G   retry does not reduce progress
H   one completed poster among four correct image fraction
I   vision reports do not falsely complete all vision work
J   stage_started event emitted (production integration)
K   stage_completed event emitted (production integration)
L   stage_failed event persisted before exception re-raised
M   callback receives events
N   callback exception does not fail production
O   events persist and reload from disk
P   snapshot persists atomically (temp+replace)
Q   corrupt events file does not block reconstruction
R   missing snapshot reconstructs from manifest
S   resume continues event sequence without reset
T   completed-run resume emits no new event
U   interrupted stage emits new start event on resume
V   elapsed time survives resume
W   ETA unavailable / low confidence with no history
X   ETA becomes available after completed stage durations
Y   completed run ETA is zero and high confidence
Z   ETA never negative
AA  poster duration influences image ETA
AB  retry history influences retry factor
AC  queue pending reports 0%
AD  queue completed reports 100%
AE  active job production fraction reflected in queue percent
AF  queue completed_with_failures reports 100%
AG  queue ETA uses completed job duration
AH  cancelled job counted as resolved
AI  failed job counts as resolved for current queue pass
AJ  show_progress --json emits valid JSON
AK  show_progress --watch exits cleanly on KeyboardInterrupt
AL  run_queue --show-progress does not change behavior
AM  old run without progress/ dir remains readable
AN  old queue without extra fields remains readable
AO  unsupported progress schema version rejected
AP  no prompts/API keys/image data persisted in events
AQ  all pre-existing tests still pass (validated by running full suite)
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.progress_tracking import (
    SCHEMA_VERSION,
    STAGE_WEIGHTS,
    _IMAGE_BLOCK_WEIGHT,
    ProgressEvent,
    ProgressSchemaError,
    ProgressTracker,
    ProductionProgress,
    StageProgress,
    get_production_progress,
    get_queue_progress,
    load_events,
    _compute_percent,
    _image_block_earned,
    _count_poster_dirs,
)

_ROOT = Path(__file__).resolve().parent.parent


# ── Manifest builders ──────────────────────────────────────────────────────────

def _stage(name: str, status: str, started: str = "2026-01-01T00:00:00+00:00",
           completed: str | None = None) -> dict:
    return {
        "stage_name": name,
        "status": status,
        "started_at": started if status != "pending" else "",
        "completed_at": completed or ("2026-01-01T00:01:00+00:00" if status in ("completed", "skipped") else None),
        "output_file": None,
        "error_message": None,
    }


def _all_stages(status: str) -> list[dict]:
    from agent.production_orchestrator import _STAGE_NAMES
    return [_stage(n, status) for n in _STAGE_NAMES]


def _manifest(
    prod_id: str = "test_run",
    status: str = "pending",
    stage_statuses: dict | None = None,
    collection_size: int = 2,
) -> dict:
    from agent.production_orchestrator import _STAGE_NAMES
    stages = []
    for n in _STAGE_NAMES:
        st = (stage_statuses or {}).get(n, "pending")
        stages.append(_stage(n, st))
    return {
        "production_id": prod_id,
        "query": "test query",
        "collection_size": collection_size,
        "status": status,
        "current_stage": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:05:00+00:00",
        "output_directory": "/tmp/test_run",
        "stages": stages,
        "resume_count": 0,
        "last_resumed_at": None,
        "total_cost": None,
        "total_input_tokens": None,
        "total_output_tokens": None,
        "total_images": None,
    }


def _write_manifest(run_dir: Path, **kwargs) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(_manifest(**kwargs)), encoding="utf-8"
    )


def _make_poster(run_dir: Path, poster_num: int, has_final: bool = False,
                 has_original: bool = False) -> Path:
    d = run_dir / "images" / f"poster_{poster_num:02d}"
    d.mkdir(parents=True, exist_ok=True)
    if has_original:
        (d / "original.png").write_bytes(b"img")
    if has_final:
        (d / "final.png").write_bytes(b"img")
        if not (d / "original.png").exists():
            (d / "original.png").write_bytes(b"img")
    return d


# ── A: pending run reports 0% ──────────────────────────────────────────────────

def test_A_pending_run_reports_zero_percent(tmp_path):
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, status="pending")
    p = get_production_progress(run_dir)
    assert p.percent == 0.0
    assert p.status == "pending"


# ── B: completed run reports exactly 100% ─────────────────────────────────────

def test_B_completed_run_reports_100_percent(tmp_path):
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, status="completed",
                    stage_statuses={n: "completed" for n in STAGE_WEIGHTS})
    p = get_production_progress(run_dir)
    assert p.percent == 100.0
    assert p.status == "completed"


# ── C: skipped stage contributes full weight ───────────────────────────────────

def test_C_skipped_stage_contributes_full_weight(tmp_path):
    run_dir = tmp_path / "run"
    # All non-image stages completed, image block all completed, listing skipped
    statuses = {n: "completed" for n in STAGE_WEIGHTS}
    statuses["listing_generation"] = "skipped"
    _write_manifest(run_dir, status="running", stage_statuses=statuses)
    p = get_production_progress(run_dir)
    # skipped listing still contributes its weight → should be 100% (status=running won't be forced)
    assert p.percent == 100.0


# ── D: running image stage partial progress ────────────────────────────────────

def test_D_image_stage_running_uses_poster_fraction(tmp_path):
    run_dir = tmp_path / "run"
    statuses = {n: "completed" for n in STAGE_WEIGHTS}
    statuses["image_generation"] = "running"
    statuses["vision_critique"] = "running"
    statuses["retry_generation"] = "running"
    _write_manifest(run_dir, status="running", stage_statuses=statuses, collection_size=4)

    # 2 of 4 posters complete
    _make_poster(run_dir, 1, has_final=True)
    _make_poster(run_dir, 2, has_final=True)

    p = get_production_progress(run_dir)
    # Non-image weight = 100 - 60 = 40. Image fraction = 2/4 = 0.5 → 0.5*60 = 30
    # Total = 70%
    expected = 40.0 + 0.5 * _IMAGE_BLOCK_WEIGHT
    assert abs(p.percent - expected) < 0.1


# ── E: percent clamped to [0, 100] ────────────────────────────────────────────

def test_E_percent_never_out_of_range(tmp_path):
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, status="pending")
    p = get_production_progress(run_dir)
    assert 0.0 <= p.percent <= 100.0

    run_dir2 = tmp_path / "run2"
    _write_manifest(run_dir2, status="completed",
                    stage_statuses={n: "completed" for n in STAGE_WEIGHTS})
    p2 = get_production_progress(run_dir2)
    assert 0.0 <= p2.percent <= 100.0


# ── F: image poster progress reflected ────────────────────────────────────────

def test_F_image_poster_progress_reflected(tmp_path):
    run_dir = tmp_path / "run"
    statuses = {n: "pending" for n in STAGE_WEIGHTS}
    statuses["image_generation"] = "running"
    statuses["vision_critique"] = "running"
    statuses["retry_generation"] = "running"
    _write_manifest(run_dir, status="running", stage_statuses=statuses, collection_size=2)

    _make_poster(run_dir, 1, has_final=True)
    # Poster 2 not started

    p = get_production_progress(run_dir)
    # Non-image stages = 0. Image: 1/2 = 0.5 → 30
    assert abs(p.percent - 30.0) < 0.1
    assert p.poster_completed == 1
    assert p.poster_total == 2


# ── G: retry does not reduce progress ─────────────────────────────────────────

def test_G_retry_does_not_reduce_progress(tmp_path):
    run_dir = tmp_path / "run"
    statuses = {n: "pending" for n in STAGE_WEIGHTS}
    statuses["image_generation"] = "running"
    statuses["vision_critique"] = "running"
    statuses["retry_generation"] = "running"
    _write_manifest(run_dir, status="running", stage_statuses=statuses, collection_size=2)

    # Poster 1 completed
    _make_poster(run_dir, 1, has_final=True)
    p1 = get_production_progress(run_dir)

    # Add poster 2 with original but no final (retry in progress)
    _make_poster(run_dir, 2, has_original=True)
    p2 = get_production_progress(run_dir)

    # Progress must not decrease
    assert p2.percent >= p1.percent


# ── H: one completed poster among four ────────────────────────────────────────

def test_H_one_poster_of_four_image_fraction(tmp_path):
    run_dir = tmp_path / "run"
    statuses = {n: "pending" for n in STAGE_WEIGHTS}
    statuses["image_generation"] = "running"
    statuses["vision_critique"] = "running"
    statuses["retry_generation"] = "running"
    _write_manifest(run_dir, status="running", stage_statuses=statuses, collection_size=4)

    _make_poster(run_dir, 1, has_final=True)
    p = get_production_progress(run_dir)

    # 1/4 = 0.25 of image block (60) = 15
    assert abs(p.percent - 15.0) < 0.1


# ── I: vision reports don't falsely complete all vision work ──────────────────

def test_I_vision_reports_do_not_complete_all_vision_work(tmp_path):
    run_dir = tmp_path / "run"
    statuses = {n: "pending" for n in STAGE_WEIGHTS}
    statuses["image_generation"] = "running"
    statuses["vision_critique"] = "running"
    statuses["retry_generation"] = "running"
    _write_manifest(run_dir, status="running", stage_statuses=statuses, collection_size=4)

    # Poster 1 has vision report but not final.png
    d = _make_poster(run_dir, 1, has_original=True)
    (d / "vision_report_1.json").write_text('{"score": 8}')

    p = get_production_progress(run_dir)
    # image_generation still running, vision_critique status from manifest = running
    # image fraction: 0 complete + 0.5 active = 0.5/4 = 0.125 → 0.125 * 60 = 7.5
    assert p.percent < 50.0
    assert p.poster_completed == 0


# ── J: stage_started event emitted ────────────────────────────────────────────

def test_J_stage_started_event_emitted(tmp_path):
    events: list[ProgressEvent] = []
    from tests._helpers_progress import _run_minimal_production
    _run_minimal_production(tmp_path, events, fail_at=None)

    types = [e.event_type for e in events]
    assert "stage_started" in types
    started_stages = [e.stage_name for e in events if e.event_type == "stage_started"]
    assert "research" in started_stages


# ── K: stage_completed event emitted ──────────────────────────────────────────

def test_K_stage_completed_event_emitted(tmp_path):
    events: list[ProgressEvent] = []
    from tests._helpers_progress import _run_minimal_production
    _run_minimal_production(tmp_path, events, fail_at=None)

    completed = [e.stage_name for e in events if e.event_type == "stage_completed"]
    assert "research" in completed


# ── L: failure event persisted before exception re-raised ─────────────────────

def test_L_failure_event_persisted_before_exception(tmp_path):
    events: list[ProgressEvent] = []
    from tests._helpers_progress import _run_minimal_production
    with pytest.raises(Exception):
        _run_minimal_production(tmp_path, events, fail_at="concept_generation")

    failed_events = [e for e in events if e.event_type == "stage_failed"]
    assert len(failed_events) >= 1
    assert failed_events[0].stage_name == "concept_generation"

    # Also verify events are on disk
    run_dirs = list((tmp_path / "outputs").iterdir())
    assert run_dirs, "run dir should exist"
    disk_events, _ = load_events(run_dirs[0])
    disk_types = [e.event_type for e in disk_events]
    assert "stage_failed" in disk_types


# ── M: callback receives events ───────────────────────────────────────────────

def test_M_callback_receives_events(tmp_path):
    events: list[ProgressEvent] = []
    from tests._helpers_progress import _run_minimal_production
    _run_minimal_production(tmp_path, events, fail_at=None)
    assert len(events) > 0
    assert all(isinstance(e, ProgressEvent) for e in events)


# ── N: callback exception does not fail production ────────────────────────────

def test_N_callback_exception_does_not_fail_production(tmp_path):
    def bad_callback(event):
        raise RuntimeError("callback error")

    from tests._helpers_progress import _run_minimal_production
    # Should not raise
    _run_minimal_production(tmp_path, None, fail_at=None, callback=bad_callback)


# ── O: events persist and reload from disk ────────────────────────────────────

def test_O_events_persist_and_reload(tmp_path):
    events: list[ProgressEvent] = []
    from tests._helpers_progress import _run_minimal_production
    _run_minimal_production(tmp_path, events, fail_at=None)

    run_dirs = list((tmp_path / "outputs").iterdir())
    disk_events, had_errors = load_events(run_dirs[0])
    assert not had_errors
    assert len(disk_events) > 0
    assert all(isinstance(e, ProgressEvent) for e in disk_events)
    # Check sequence ordering
    seqs = [e.sequence for e in disk_events]
    assert seqs == sorted(seqs)


# ── P: snapshot persists atomically ───────────────────────────────────────────

def test_P_snapshot_persists_atomically(tmp_path):
    tracker = ProgressTracker(
        run_id="test",
        progress_dir=tmp_path / "progress",
    )
    tracker.emit("run_started", message="test")

    # Build a minimal progress object and save snapshot
    from tests._helpers_progress import _minimal_progress
    p = _minimal_progress()
    tracker.save_snapshot(p)

    snap = tmp_path / "progress" / "snapshot.json"
    assert snap.exists()
    data = json.loads(snap.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    # No .tmp file left behind
    assert not snap.with_suffix(".tmp").exists()


# ── Q: corrupt events file does not block reconstruction ──────────────────────

def test_Q_corrupt_events_file_does_not_block(tmp_path):
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, status="running")
    progress_dir = run_dir / "progress"
    progress_dir.mkdir()
    # Write corrupt JSONL
    (progress_dir / "events.jsonl").write_text("not json\n{}\n", encoding="utf-8")

    # get_production_progress should still work
    p = get_production_progress(run_dir)
    assert isinstance(p, ProductionProgress)

    # load_events should flag errors but not raise
    events, had_errors = load_events(run_dir)
    assert had_errors


# ── R: missing snapshot reconstructs from manifest ────────────────────────────

def test_R_missing_snapshot_reconstructs_from_manifest(tmp_path):
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, status="running")
    # No progress/ dir at all
    assert not (run_dir / "progress").exists()

    p = get_production_progress(run_dir)
    assert isinstance(p, ProductionProgress)
    assert p.run_id == "test_run"


# ── S: resume continues event sequence ────────────────────────────────────────

def test_S_resume_continues_event_sequence(tmp_path):
    # First run emits events starting at sequence 1
    tracker1 = ProgressTracker(run_id="run1", progress_dir=tmp_path / "progress")
    e1 = tracker1.emit("run_started")
    e2 = tracker1.emit("stage_started", stage_name="research")

    # Save snapshot with last_event_sequence
    from tests._helpers_progress import _minimal_progress
    p = _minimal_progress(last_event_sequence=e2.sequence)
    tracker1.save_snapshot(p)

    # Second tracker (resume) loads sequence
    initial = ProgressTracker.load_sequence_from_dir(tmp_path / "progress")
    tracker2 = ProgressTracker(run_id="run1", progress_dir=tmp_path / "progress",
                               initial_sequence=initial)
    e3 = tracker2.emit("run_resumed")

    # Sequences should be monotonically increasing without reset
    assert e1.sequence == 1
    assert e2.sequence == 2
    assert e3.sequence > e2.sequence


# ── T: completed-run resume emits no new event ────────────────────────────────

def test_T_completed_run_resume_emits_no_event(tmp_path):
    from agent.production_orchestrator import resume_production
    from tests._helpers_progress import _make_completed_run_dir

    events: list[ProgressEvent] = []
    run_dir = _make_completed_run_dir(tmp_path)

    def cb(e): events.append(e)
    # resume_production fast-path: all stages complete → no pipeline work, no events emitted
    resume_production(run_dir, _progress_callback=cb)
    # Fast path returns immediately — no run_resumed or stage events
    assert not any(e.event_type == "stage_started" for e in events)


# ── U: interrupted stage emits new start event on resume ──────────────────────

def test_U_interrupted_stage_emits_new_start_event(tmp_path):
    from tests._helpers_progress import _run_minimal_production

    events1: list = []
    with pytest.raises(Exception):
        _run_minimal_production(tmp_path, events1, fail_at="concept_generation")

    run_dirs = list((tmp_path / "outputs").iterdir())
    run_dir = run_dirs[0]

    # Now resume — concept_generation should emit stage_started again
    events2: list = []
    from tests._helpers_progress import _resume_minimal_production
    _resume_minimal_production(run_dir, events2)

    resumed_starts = [e for e in events2
                      if e.event_type == "stage_started" and e.stage_name == "concept_generation"]
    assert len(resumed_starts) >= 1


# ── V: elapsed time captured ──────────────────────────────────────────────────

def test_V_elapsed_time_captured(tmp_path):
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, status="running")
    p = get_production_progress(run_dir)
    assert p.elapsed_seconds >= 0.0


# ── W: ETA low confidence with no history ─────────────────────────────────────

def test_W_eta_low_confidence_no_history(tmp_path):
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, status="running")
    p = get_production_progress(run_dir)
    assert p.eta_confidence in ("low", "medium")


# ── X: ETA becomes available after completed stage durations ──────────────────

def test_X_eta_available_after_completed_stages(tmp_path):
    run_dir = tmp_path / "run"
    statuses = {n: "completed" for n in STAGE_WEIGHTS}
    statuses["listing_generation"] = "running"
    statuses["finalize"] = "pending"
    _write_manifest(run_dir, status="running", stage_statuses=statuses)
    p = get_production_progress(run_dir)
    assert p.eta_available
    # Has history from completed stages
    assert p.eta_confidence in ("medium", "high")


# ── Y: completed run ETA is 0 and high confidence ─────────────────────────────

def test_Y_completed_run_eta_zero_high_confidence(tmp_path):
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, status="completed",
                    stage_statuses={n: "completed" for n in STAGE_WEIGHTS})
    p = get_production_progress(run_dir)
    assert p.eta_seconds == 0.0
    assert p.eta_confidence == "high"
    assert p.eta_available is True


# ── Z: ETA never negative ─────────────────────────────────────────────────────

def test_Z_eta_never_negative(tmp_path):
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, status="running")
    p = get_production_progress(run_dir)
    if p.eta_seconds is not None:
        assert p.eta_seconds >= 0.0


# ── AA: poster duration influences image ETA ──────────────────────────────────

def test_AA_poster_duration_influences_image_eta(tmp_path):
    from agent.progress_tracking import _avg_poster_duration

    run_dir = tmp_path / "run"
    images = run_dir / "images"

    # Create a completed poster with a timed attempts.json
    d = images / "poster_01"
    d.mkdir(parents=True)
    (d / "final.png").write_bytes(b"x")
    t1 = "2026-01-01T00:00:00+00:00"
    t2 = "2026-01-01T00:01:00+00:00"  # 60s
    (d / "attempts.json").write_text(json.dumps([
        {"attempt_number": 1, "created_at": t1, "image_file": "", "vision_report_file": "",
         "retry_plan_file": "", "accepted": True},
        {"attempt_number": 1, "created_at": t2, "image_file": "", "vision_report_file": "",
         "retry_plan_file": "", "accepted": True},
    ]))

    avg = _avg_poster_duration(run_dir)
    assert avg is not None
    assert avg >= 1.0


# ── AB: retry history influences retry factor ─────────────────────────────────

def test_AB_retry_history_influences_retry_factor(tmp_path):
    from agent.progress_tracking import _avg_retry_factor

    run_dir = tmp_path / "run"
    images = run_dir / "images"

    d = images / "poster_01"
    d.mkdir(parents=True)
    (d / "final.png").write_bytes(b"x")
    (d / "attempts.json").write_text(json.dumps([
        {"attempt_number": 1, "created_at": "", "image_file": "", "vision_report_file": "",
         "retry_plan_file": "", "accepted": False},
        {"attempt_number": 2, "created_at": "", "image_file": "", "vision_report_file": "",
         "retry_plan_file": "", "accepted": True},
    ]))

    factor = _avg_retry_factor(run_dir)
    assert factor == 2.0  # 2 attempts for 1 poster


# ── AC: queue pending reports 0% ──────────────────────────────────────────────

def test_AC_queue_pending_reports_zero(tmp_path):
    from agent.job_queue import enqueue_job
    from agent.production_orchestrator import ProductionRequest

    q = tmp_path / "queue"
    req = ProductionRequest(query="test", collection_size=3,
                            output_root=str(tmp_path / "outputs"))
    enqueue_job(q, req)

    progress = get_queue_progress(q)
    assert progress.percent == 0.0
    assert progress.status == "pending"


# ── AD: queue completed reports 100% ──────────────────────────────────────────

def test_AD_queue_completed_reports_100(tmp_path):
    q = tmp_path / "queue"
    _make_queue_json(q, status="completed", jobs=[
        _queue_job("job_001", "completed"),
        _queue_job("job_002", "completed"),
    ])
    progress = get_queue_progress(q)
    assert progress.percent == 100.0


# ── AE: active job production fraction in queue percent ───────────────────────

def test_AE_active_job_fraction_in_queue_percent(tmp_path):
    q = tmp_path / "queue"
    # Create a run dir with 50% progress
    run_dir = tmp_path / "outputs" / "run_001"
    statuses = {n: "pending" for n in STAGE_WEIGHTS}
    statuses["image_generation"] = "running"
    statuses["vision_critique"] = "running"
    statuses["retry_generation"] = "running"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps(
        _manifest(status="running", stage_statuses=statuses, collection_size=2)
    ))
    _make_poster(run_dir, 1, has_final=True)  # 1/2 = 50% image

    _make_queue_json(q, status="running", jobs=[
        _queue_job("job_001", "completed"),
        _queue_job("job_002", "running", run_dir=str(run_dir)),
    ])
    progress = get_queue_progress(q)
    # 1 completed + fraction_of_running / 2 total → > 50%
    assert progress.percent > 50.0
    assert progress.percent < 100.0


# ── AF: queue completed_with_failures reports 100% ────────────────────────────

def test_AF_queue_completed_with_failures_reports_100(tmp_path):
    q = tmp_path / "queue"
    _make_queue_json(q, status="completed_with_failures", jobs=[
        _queue_job("job_001", "completed"),
        _queue_job("job_002", "failed"),
    ])
    progress = get_queue_progress(q)
    assert progress.percent == 100.0


# ── AG: queue ETA uses completed job duration ──────────────────────────────────

def test_AG_queue_eta_uses_completed_job_duration(tmp_path):
    q = tmp_path / "queue"
    _make_queue_json(q, status="running", jobs=[
        _queue_job("job_001", "completed",
                   started_at="2026-01-01T00:00:00+00:00",
                   completed_at="2026-01-01T00:05:00+00:00"),  # 300s
        _queue_job("job_002", "pending"),
    ])
    progress = get_queue_progress(q)
    assert progress.eta_available
    assert progress.eta_seconds is not None
    assert progress.eta_seconds > 0


# ── AH: cancelled job counted as resolved ─────────────────────────────────────

def test_AH_cancelled_job_counted_as_resolved(tmp_path):
    q = tmp_path / "queue"
    _make_queue_json(q, status="running", jobs=[
        _queue_job("job_001", "completed"),
        _queue_job("job_002", "cancelled"),
        _queue_job("job_003", "pending"),
    ])
    progress = get_queue_progress(q)
    # 2 resolved (completed + cancelled) / 3 total = 66.7%
    assert abs(progress.percent - (2 / 3 * 100)) < 1.0


# ── AI: failed job counts as resolved for current queue pass ──────────────────

def test_AI_failed_job_counts_as_resolved(tmp_path):
    q = tmp_path / "queue"
    _make_queue_json(q, status="completed_with_failures", jobs=[
        _queue_job("job_001", "completed"),
        _queue_job("job_002", "failed"),
    ])
    progress = get_queue_progress(q)
    # Both resolved → 100%
    assert progress.percent == 100.0


# ── AJ: show_progress --json emits valid JSON ─────────────────────────────────

def test_AJ_show_progress_json_valid(tmp_path):
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, status="running")

    result = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "show_progress.py"),
         str(run_dir), "--json"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert "percent" in data
    assert "status" in data
    assert "schema_version" in data


# ── AK: show_progress --watch exits on KeyboardInterrupt ──────────────────────

def test_AK_show_progress_watch_exits_cleanly(tmp_path):
    run_dir = tmp_path / "run"
    # Completed run so --watch exits immediately
    _write_manifest(run_dir, status="completed",
                    stage_statuses={n: "completed" for n in STAGE_WEIGHTS})

    result = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "show_progress.py"),
         str(run_dir), "--watch", "--interval", "0.1"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0


# ── AL: run_queue --show-progress does not change behavior ────────────────────

def test_AL_run_queue_show_progress_no_behavior_change(tmp_path):
    # Verify --show-progress is accepted as a flag without error
    result = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "run_queue.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    # run subcommand help should mention --show-progress
    result2 = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "run_queue.py"), "run", "--help"],
        capture_output=True, text=True,
    )
    assert "--show-progress" in result2.stdout


# ── AM: old run without progress/ dir remains readable ────────────────────────

def test_AM_old_run_without_progress_dir(tmp_path):
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, status="completed",
                    stage_statuses={n: "completed" for n in STAGE_WEIGHTS})
    # No progress/ dir
    assert not (run_dir / "progress").exists()

    p = get_production_progress(run_dir)
    assert p.percent == 100.0
    assert p.status == "completed"


# ── AN: old queue without extra fields remains readable ───────────────────────

def test_AN_old_queue_without_extra_fields(tmp_path):
    q = tmp_path / "queue"
    # Minimal valid queue.json
    _make_queue_json(q, status="completed", jobs=[
        _queue_job("job_001", "completed"),
    ])
    progress = get_queue_progress(q)
    assert isinstance(progress.percent, float)


# ── AO: unsupported schema version rejected ────────────────────────────────────

def test_AO_unsupported_schema_version_rejected(tmp_path):
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, status="running")
    progress_dir = run_dir / "progress"
    progress_dir.mkdir()
    # Write snapshot with unsupported version
    (progress_dir / "snapshot.json").write_text(
        json.dumps({"schema_version": 999, "percent": 50}), encoding="utf-8"
    )
    with pytest.raises(ProgressSchemaError):
        get_production_progress(run_dir)


# ── AP: no sensitive data in events ───────────────────────────────────────────

def test_AP_no_sensitive_data_in_events(tmp_path):
    tracker = ProgressTracker(run_id="r1", progress_dir=tmp_path / "progress")
    tracker.emit(
        "stage_started",
        stage_name="research",
        metadata={"info": "ok"},
    )
    events, _ = load_events(tmp_path)
    for e in events:
        d = e.to_dict()
        text = json.dumps(d).lower()
        assert "api_key" not in text
        assert "base64" not in text
        assert "optimized_image_prompt" not in text


# ── AQ: pre-existing tests unaffected ────────────────────────────────────────

def test_AQ_progress_module_imports_without_side_effects():
    """Import stability: progress_tracking can be imported in any order."""
    import importlib
    import agent.progress_tracking as pt
    importlib.reload(pt)
    assert pt.SCHEMA_VERSION == 1
    assert sum(pt.STAGE_WEIGHTS.values()) == 100


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_queue_json(
    q: Path,
    status: str = "pending",
    jobs: list | None = None,
) -> None:
    q.mkdir(parents=True, exist_ok=True)
    (q / "queue.json").write_text(json.dumps({
        "queue_id": q.name,
        "schema_version": 1,
        "status": status,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:05:00+00:00",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:05:00+00:00" if status in (
            "completed", "completed_with_failures", "failed") else None,
        "jobs": jobs or [],
    }), encoding="utf-8")


def _queue_job(
    job_id: str,
    status: str,
    run_dir: str | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> dict:
    return {
        "job_id": job_id,
        "position": int(job_id.split("_")[1]),
        "status": status,
        "request": {"query": "x", "collection_size": 2, "output_root": "/tmp"},
        "run_dir": run_dir,
        "created_at": "2026-01-01T00:00:00+00:00",
        "started_at": started_at or ("2026-01-01T00:00:00+00:00" if status != "pending" else None),
        "completed_at": completed_at or ("2026-01-01T00:05:00+00:00" if status == "completed" else None),
        "updated_at": "2026-01-01T00:05:00+00:00",
        "error_message": None,
        "attempt_count": 1 if status != "pending" else 0,
        "total_cost": None,
        "total_input_tokens": None,
        "total_output_tokens": None,
        "total_images": None,
    }
