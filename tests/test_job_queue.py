"""
Unit tests for agent/job_queue.py — no live API calls.

Tests A–AD map to the spec requirements:
A  enqueue creates persistent pending job
B  sequential IDs and insertion order
C  identical requests create separate jobs
D  run_queue processes jobs in order
E  successful jobs become completed
F  failed job recorded and next job continues when stop_on_failure=False
G  stop_on_failure=True persists failure and re-raises
H  completed queue makes zero dependency calls
I  completed jobs are not rerun
J  failed job with run_dir uses resume_production
K  failed job without run_dir uses run_production
L  interrupted running job with run_dir resumes
M  interrupted running job without run_dir restarts
N  production failure still captures created run_dir
O  cancelled job is skipped
P  completed/running jobs reject cancellation
Q  cancelling cancelled job is idempotent
R  malformed queue JSON raises clear error
S  duplicate IDs rejected
T  invalid status rejected
U  unknown schema version rejected
V  completed job with missing run_dir is not trusted
W  queue lock prevents second runner
X  lock released after success
Y  lock released after exception
Z  CLI create/add/list uses no paid APIs
AA fully completed resume is idempotent
AB old production runs remain compatible (run_production still accepts no _on_run_dir)
AC request settings survive serialization exactly
AD queue summary counts are accurate
"""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.job_queue import (
    QueueDependencies,
    QueueError,
    QueueLockedError,
    QueueSchemaError,
    cancel_job,
    enqueue_job,
    list_jobs,
    resume_queue,
    run_queue,
)
from agent.production_orchestrator import ProductionRequest, run_production


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _req(tmp_path: Path, query: str = "cozy wall art", n: int = 3) -> ProductionRequest:
    return ProductionRequest(
        query=query,
        collection_size=n,
        output_root=str(tmp_path / "outputs"),
    )


def _prod_manifest_data(status: str = "completed") -> dict:
    return {"status": status, "stages": [], "schema_version": 1}


def _write_prod_manifest(run_dir: Path, status: str = "completed") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(_prod_manifest_data(status)), encoding="utf-8"
    )


def _make_deps(
    tmp_path: Path,
    *,
    fail_queries: set[str] | None = None,
) -> QueueDependencies:
    """
    Returns QueueDependencies where each run_production call:
      1. Creates a unique run directory with a completed production manifest
      2. Calls _on_run_dir if provided
      3. Raises RuntimeError if request.query is in fail_queries
    """
    counter = [0]
    fail_queries = fail_queries or set()

    def fake_run(request, _on_run_dir=None, **kwargs):
        counter[0] += 1
        rd = tmp_path / "outputs" / f"run_{counter[0]:03d}"
        _write_prod_manifest(rd)
        if _on_run_dir is not None:
            _on_run_dir(rd)
        if request.query in fail_queries:
            raise RuntimeError(f"deliberate failure: {request.query}")

    def fake_resume(run_dir, **kwargs):
        pass  # pretend success

    return QueueDependencies(run_production=fake_run, resume_production=fake_resume)


def _queue_data(queue_dir: Path) -> dict:
    return json.loads((queue_dir / "queue.json").read_text())


def _set_queue_data(queue_dir: Path, data: dict) -> None:
    (queue_dir / "queue.json").write_text(json.dumps(data), encoding="utf-8")


# ── Tests A–C: enqueue ─────────────────────────────────────────────────────────

def test_A_enqueue_creates_pending_job(tmp_path):
    q = tmp_path / "queue"
    job = enqueue_job(q, _req(tmp_path))
    assert job.status == "pending"
    assert job.job_id == "job_001"
    assert job.position == 1
    # Persisted to disk
    jobs = list_jobs(q)
    assert len(jobs) == 1
    assert jobs[0].status == "pending"
    assert jobs[0].job_id == "job_001"


def test_B_sequential_ids_and_insertion_order(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path, query="first"))
    enqueue_job(q, _req(tmp_path, query="second"))
    enqueue_job(q, _req(tmp_path, query="third"))
    jobs = list_jobs(q)
    assert [j.job_id for j in jobs] == ["job_001", "job_002", "job_003"]
    assert [j.position for j in jobs] == [1, 2, 3]
    assert [j.request["query"] for j in jobs] == ["first", "second", "third"]


def test_C_identical_requests_create_separate_jobs(tmp_path):
    q = tmp_path / "queue"
    req = _req(tmp_path)
    enqueue_job(q, req)
    enqueue_job(q, req)
    jobs = list_jobs(q)
    assert len(jobs) == 2
    assert jobs[0].job_id != jobs[1].job_id


# ── Tests D–H: run_queue execution ────────────────────────────────────────────

def test_D_run_queue_processes_jobs_in_order(tmp_path):
    q = tmp_path / "queue"
    order = []

    counter = [0]
    def fake_run(request, _on_run_dir=None, **kwargs):
        counter[0] += 1
        order.append(request.query)
        rd = tmp_path / "outputs" / f"r{counter[0]}"
        _write_prod_manifest(rd)
        if _on_run_dir:
            _on_run_dir(rd)

    for query in ["alpha", "beta", "gamma"]:
        enqueue_job(q, _req(tmp_path, query=query))

    run_queue(q, _deps=QueueDependencies(run_production=fake_run, resume_production=MagicMock()))
    assert order == ["alpha", "beta", "gamma"]


def test_E_successful_jobs_become_completed(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))
    result = run_queue(q, _deps=_make_deps(tmp_path))
    assert result.status == "completed"
    assert result.completed_jobs == 1
    assert result.failed_jobs == 0
    jobs = list_jobs(q)
    assert jobs[0].status == "completed"
    assert jobs[0].run_dir is not None
    assert jobs[0].attempt_count == 1


def test_F_failed_job_continues_when_not_stop_on_failure(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path, query="fail"))
    enqueue_job(q, _req(tmp_path, query="ok"))

    result = run_queue(
        q,
        _deps=_make_deps(tmp_path, fail_queries={"fail"}),
        stop_on_failure=False,
    )

    assert result.status == "completed_with_failures"
    assert result.failed_jobs == 1
    assert result.completed_jobs == 1
    jobs = list_jobs(q)
    assert jobs[0].status == "failed"
    assert jobs[1].status == "completed"
    assert jobs[0].error_message is not None


def test_G_stop_on_failure_reraises_and_stops(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path, query="fail"))
    enqueue_job(q, _req(tmp_path, query="should_not_run"))

    with pytest.raises(RuntimeError, match="deliberate failure"):
        run_queue(q, _deps=_make_deps(tmp_path, fail_queries={"fail"}), stop_on_failure=True)

    jobs = list_jobs(q)
    assert jobs[0].status == "failed"
    assert jobs[1].status == "pending"  # second job not touched


def test_H_completed_queue_zero_deps(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))
    run_queue(q, _deps=_make_deps(tmp_path))  # first run

    mock_deps = QueueDependencies(run_production=MagicMock(), resume_production=MagicMock())
    run_queue(q, _deps=mock_deps)

    mock_deps.run_production.assert_not_called()
    mock_deps.resume_production.assert_not_called()


def test_I_completed_jobs_not_rerun(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path, query="done"))
    run_queue(q, _deps=_make_deps(tmp_path))  # runs "done"

    # Add a new job
    enqueue_job(q, _req(tmp_path, query="new"))
    second_run_calls = []

    counter = [1]
    def fake_run(request, _on_run_dir=None, **kwargs):
        second_run_calls.append(request.query)
        counter[0] += 1
        rd = tmp_path / "outputs" / f"r_new_{counter[0]}"
        _write_prod_manifest(rd)
        if _on_run_dir:
            _on_run_dir(rd)

    run_queue(q, _deps=QueueDependencies(run_production=fake_run, resume_production=MagicMock()))
    assert second_run_calls == ["new"]  # only the new job ran


# ── Tests J–M: resume_queue job routing ───────────────────────────────────────

def test_J_failed_job_with_run_dir_uses_resume_production(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))

    # Simulate: first run created run_dir then failed
    run_dir = tmp_path / "outputs" / "run_001"
    _write_prod_manifest(run_dir, status="failed")

    data = _queue_data(q)
    data["jobs"][0]["status"] = "failed"
    data["jobs"][0]["run_dir"] = str(run_dir)
    _set_queue_data(q, data)

    resume_calls = []
    def _fake_resume_j(rd, **kwargs): resume_calls.append(rd)
    deps = QueueDependencies(
        run_production=MagicMock(),
        resume_production=_fake_resume_j,
    )
    resume_queue(q, _deps=deps)

    assert resume_calls == [str(run_dir)]
    deps.run_production.assert_not_called()


def test_K_failed_job_without_run_dir_uses_run_production(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))

    data = _queue_data(q)
    data["jobs"][0]["status"] = "failed"
    # run_dir intentionally absent
    _set_queue_data(q, data)

    run_calls = []
    counter = [0]
    def fake_run(request, _on_run_dir=None, **kwargs):
        run_calls.append(request.query)
        counter[0] += 1
        rd = tmp_path / "outputs" / f"restart_{counter[0]}"
        _write_prod_manifest(rd)
        if _on_run_dir:
            _on_run_dir(rd)

    resume_mock = MagicMock()
    resume_queue(q, _deps=QueueDependencies(run_production=fake_run, resume_production=resume_mock))

    assert len(run_calls) == 1
    resume_mock.assert_not_called()


def test_L_interrupted_running_job_with_run_dir_resumes(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))

    run_dir = tmp_path / "outputs" / "run_crash"
    _write_prod_manifest(run_dir, status="running")

    data = _queue_data(q)
    data["jobs"][0]["status"] = "running"
    data["jobs"][0]["run_dir"] = str(run_dir)
    _set_queue_data(q, data)

    resume_calls = []
    def _fake_resume_l(rd, **kwargs): resume_calls.append(rd)
    deps = QueueDependencies(
        run_production=MagicMock(),
        resume_production=_fake_resume_l,
    )
    resume_queue(q, _deps=deps)

    assert resume_calls == [str(run_dir)]
    deps.run_production.assert_not_called()


def test_M_interrupted_running_job_without_run_dir_restarts(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))

    data = _queue_data(q)
    data["jobs"][0]["status"] = "running"
    # no run_dir
    _set_queue_data(q, data)

    run_calls = []
    counter = [0]
    def fake_run(request, _on_run_dir=None, **kwargs):
        run_calls.append(request.query)
        counter[0] += 1
        rd = tmp_path / "outputs" / f"restart_{counter[0]}"
        _write_prod_manifest(rd)
        if _on_run_dir:
            _on_run_dir(rd)

    resume_mock = MagicMock()
    resume_queue(q, _deps=QueueDependencies(run_production=fake_run, resume_production=resume_mock))

    assert len(run_calls) == 1
    resume_mock.assert_not_called()


# ── Test N: run_dir captured even on production failure ────────────────────────

def test_N_failed_job_stores_run_dir(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))

    captured = [None]

    def fail_after_create(request, _on_run_dir=None, **kwargs):
        rd = tmp_path / "outputs" / "run_N"
        rd.mkdir(parents=True, exist_ok=True)
        if _on_run_dir:
            _on_run_dir(rd)
            captured[0] = str(rd)
        raise RuntimeError("failure after dir created")

    with pytest.raises(RuntimeError, match="failure after dir created"):
        run_queue(
            q,
            _deps=QueueDependencies(run_production=fail_after_create, resume_production=MagicMock()),
            stop_on_failure=True,
        )

    jobs = list_jobs(q)
    assert jobs[0].status == "failed"
    assert jobs[0].run_dir == captured[0]
    assert captured[0] is not None


# ── Tests O–Q: cancel_job ─────────────────────────────────────────────────────

def test_O_cancelled_job_is_skipped(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path, query="skip"))
    enqueue_job(q, _req(tmp_path, query="run"))

    cancel_job(q, "job_001")

    run_calls = []
    counter = [0]
    def fake_run(request, _on_run_dir=None, **kwargs):
        run_calls.append(request.query)
        counter[0] += 1
        rd = tmp_path / "outputs" / f"r{counter[0]}"
        _write_prod_manifest(rd)
        if _on_run_dir:
            _on_run_dir(rd)

    run_queue(q, _deps=QueueDependencies(run_production=fake_run, resume_production=MagicMock()))
    assert run_calls == ["run"]  # "skip" was cancelled


def test_P_completed_and_running_reject_cancellation(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))
    enqueue_job(q, _req(tmp_path, query="b"))

    # Mark job_001 completed
    data = _queue_data(q)
    data["jobs"][0]["status"] = "completed"
    data["jobs"][0]["run_dir"] = str(tmp_path / "outputs" / "run_001")
    # Mark job_002 running
    data["jobs"][1]["status"] = "running"
    _set_queue_data(q, data)

    with pytest.raises(QueueError, match="Cannot cancel"):
        cancel_job(q, "job_001")

    with pytest.raises(QueueError, match="Cannot cancel"):
        cancel_job(q, "job_002")


def test_Q_cancelling_cancelled_job_is_idempotent(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))

    cancel_job(q, "job_001")
    job = cancel_job(q, "job_001")  # second call
    assert job.status == "cancelled"  # no exception, no change


# ── Tests R–U: validation errors ──────────────────────────────────────────────

def test_R_malformed_queue_json_raises(tmp_path):
    q = tmp_path / "queue"
    q.mkdir()
    (q / "queue.json").write_text("{{{not json", encoding="utf-8")
    with pytest.raises(QueueSchemaError, match="Malformed"):
        list_jobs(q)


def test_S_duplicate_job_ids_rejected(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))
    data = _queue_data(q)
    # duplicate job_001
    data["jobs"].append({**data["jobs"][0], "position": 2})
    _set_queue_data(q, data)
    with pytest.raises(QueueError, match="Duplicate job_id"):
        list_jobs(q)


def test_T_invalid_status_rejected(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))
    data = _queue_data(q)
    data["jobs"][0]["status"] = "flying"
    _set_queue_data(q, data)
    with pytest.raises(QueueError, match="invalid status"):
        list_jobs(q)


def test_U_unknown_schema_version_rejected(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))
    data = _queue_data(q)
    data["schema_version"] = 99
    _set_queue_data(q, data)
    with pytest.raises(QueueSchemaError, match="schema_version"):
        list_jobs(q)


# ── Test V: completed job with missing run_dir is not trusted ──────────────────

def test_V_completed_job_missing_run_dir_not_trusted(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))

    data = _queue_data(q)
    data["jobs"][0]["status"] = "completed"
    data["jobs"][0]["run_dir"] = str(tmp_path / "outputs" / "nonexistent_run")
    _set_queue_data(q, data)

    run_calls = []
    counter = [0]
    def fake_run(request, _on_run_dir=None, **kwargs):
        run_calls.append(request.query)
        counter[0] += 1
        rd = tmp_path / "outputs" / f"new_{counter[0]}"
        _write_prod_manifest(rd)
        if _on_run_dir:
            _on_run_dir(rd)

    resume_queue(q, _deps=QueueDependencies(run_production=fake_run, resume_production=MagicMock()))
    # Job was "completed" but run_dir missing → should have been rerun
    assert len(run_calls) == 1


# ── Tests W–Y: locking ────────────────────────────────────────────────────────

def test_W_lock_prevents_second_runner(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))

    # Manually place a lock file
    (q / ".queue.lock").write_text('{"pid": 9999, "timestamp": "t"}', encoding="utf-8")

    with pytest.raises(QueueLockedError):
        run_queue(q, _deps=_make_deps(tmp_path))


def test_X_lock_released_after_success(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))
    run_queue(q, _deps=_make_deps(tmp_path))
    assert not (q / ".queue.lock").exists()


def test_Y_lock_released_after_exception(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))

    def always_fail(request, _on_run_dir=None, **kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_queue(
            q,
            _deps=QueueDependencies(run_production=always_fail, resume_production=MagicMock()),
            stop_on_failure=True,
        )

    assert not (q / ".queue.lock").exists()


# ── Test Z: CLI (no paid APIs) ────────────────────────────────────────────────

def test_Z_cli_create_add_list_no_paid_apis(tmp_path):
    cli = str(Path(__file__).parent.parent / "scripts" / "run_queue.py")
    q = str(tmp_path / "my_queue")
    out_root = str(tmp_path / "outputs")

    def run(args):
        return subprocess.run(
            [sys.executable, cli, *args],
            capture_output=True, text=True, check=True,
        )

    run(["create", q])
    assert (Path(q) / "queue.json").exists()

    run(["add", q, "--query", "mountain mist", "--collection-size", "3",
         "--output-root", out_root])
    run(["add", q, "--query", "cherry blossom", "--collection-size", "3",
         "--output-root", out_root])

    result = run(["list", q])
    assert "mountain mist" in result.stdout
    assert "cherry blossom" in result.stdout


# ── Test AA: fully completed resume is idempotent ────────────────────────────

def test_AA_completed_resume_is_idempotent(tmp_path):
    q = tmp_path / "queue"
    enqueue_job(q, _req(tmp_path))
    deps1 = _make_deps(tmp_path)
    run_queue(q, _deps=deps1)

    before = (q / "queue.json").read_text()
    mock_deps = QueueDependencies(run_production=MagicMock(), resume_production=MagicMock())
    result = resume_queue(q, _deps=mock_deps)

    mock_deps.run_production.assert_not_called()
    mock_deps.resume_production.assert_not_called()
    assert result.status == "completed"
    # queue.json unchanged (fast path — no writes)
    assert (q / "queue.json").read_text() == before


# ── Test AB: run_production backward compatibility ────────────────────────────

def test_AB_run_production_accepts_no_on_run_dir(tmp_path):
    """Existing callers that don't pass _on_run_dir must still work."""
    from tests.test_production_orchestrator import _make_deps as _make_orch_deps
    from tests.test_production_orchestrator import _request as _make_orch_req

    deps = _make_orch_deps(tmp_path)
    req = _make_orch_req(tmp_path)
    result = run_production(req, _deps=deps)
    assert result.manifest.status == "completed"


# ── Test AC: request settings survive serialization ──────────────────────────

def test_AC_request_settings_survive_serialization(tmp_path):
    q = tmp_path / "queue"
    req = ProductionRequest(
        query="vintage botanical",
        collection_size=4,
        output_root=str(tmp_path / "out"),
        selected_concept_index=2,
        max_image_retries=0,
        skip_mockups=True,
        skip_listing=True,
    )
    enqueue_job(q, req)

    jobs = list_jobs(q)
    r = jobs[0].request
    assert r["query"] == "vintage botanical"
    assert r["collection_size"] == 4
    assert r["selected_concept_index"] == 2
    assert r["max_image_retries"] == 0
    assert r["skip_mockups"] is True
    assert r["skip_listing"] is True


# ── Test AD: summary counts are accurate ─────────────────────────────────────

def test_AD_queue_summary_counts_accurate(tmp_path):
    q = tmp_path / "queue"
    for query in ["ok1", "fail", "ok2", "cancel"]:
        enqueue_job(q, _req(tmp_path, query=query))

    cancel_job(q, "job_004")

    result = run_queue(
        q,
        _deps=_make_deps(tmp_path, fail_queries={"fail"}),
        stop_on_failure=False,
    )

    assert result.total_jobs == 4
    assert result.completed_jobs == 2
    assert result.failed_jobs == 1
    assert result.cancelled_jobs == 1
    assert result.pending_jobs == 0
    assert result.status == "completed_with_failures"
