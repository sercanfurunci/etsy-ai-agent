"""
Shared test helpers for progress tracking integration tests.
All fakes use **kwargs to absorb on_usage and on_progress callbacks.
"""
import json
import shutil
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

from agent.production_orchestrator import (
    ProductionDependencies,
    ProductionRequest,
    run_production,
    resume_production,
)
from agent.progress_tracking import ProductionProgress, SCHEMA_VERSION, STAGE_WEIGHTS


# ── Minimal fake deps ──────────────────────────────────────────────────────────

def _make_research_provider():
    p = MagicMock()
    p.search.return_value = [{"title": "P1"}]
    return p


def _make_deps(
    tmp_path: Path,
    fail_at: str | None = None,
) -> ProductionDependencies:
    """Fake deps. If fail_at == stage_name, that stage raises RuntimeError."""

    def analyze(products, **kwargs):
        if fail_at == "concept_generation":
            raise RuntimeError("deliberate concept fail")
        return {
            "niche": "test",
            "market_observations": "ok",
            "recurring_patterns": "ok",
            "potential_opportunities": "ok",
            "poster_concepts": [{"name": "P1", "image_generation_prompt": "img",
                                 "negative_prompt": "neg"}],
        }

    def optimize(concept, prompt, neg, **kwargs):
        return {"optimized_image_prompt": "opt", "optimized_negative_prompt": "neg"}

    def generate_collection(concept, prompt, neg, collection_size=3, **kwargs):
        plan = MagicMock()
        plan.collection_size = collection_size
        plan.collection_bible.collection_name = "Test"
        posters = []
        for i in range(1, collection_size + 1):
            p = MagicMock()
            p.index = i
            p.image_prompt = "img"
            p.negative_prompt = "neg"
            posters.append(p)
        plan.poster_items = posters
        return plan

    def generate_image(prompt, **kwargs) -> str:
        img = tmp_path / "_tmp_img.png"
        img.write_bytes(b"PNG")
        return str(img)

    def vision_review(concept, prompt, neg, path, **kwargs):
        vr = MagicMock()
        vr.overall_score.score = 9
        vr.retry_recommended = False
        return vr

    def prepare_retry(concept, prompt, neg, vr, **kwargs):
        rp = MagicMock()
        rp.should_retry = False
        rp.revised_image_prompt = None
        rp.revised_negative_prompt = None
        return rp

    def generate_mockup_plan(collection_plan, **kwargs):
        return MagicMock()

    def generate_listing_plan(collection_plan, mockup_plan, **kwargs):
        return MagicMock()

    provider = MagicMock()
    provider.generate = generate_image

    return ProductionDependencies(
        research_provider_factory=_make_research_provider,
        analyze=analyze,
        optimize=optimize,
        generate_collection=generate_collection,
        image_provider_factory=lambda: provider,
        vision_review=vision_review,
        prepare_retry=prepare_retry,
        generate_mockup_plan=generate_mockup_plan,
        generate_listing_plan=generate_listing_plan,
    )


def _run_minimal_production(
    tmp_path: Path,
    events_list: list | None,
    fail_at: str | None,
    callback: Callable | None = None,
) -> None:
    """Run a production pipeline with fake deps, optionally failing at a stage."""
    deps = _make_deps(tmp_path, fail_at=fail_at)

    collected = events_list if events_list is not None else []

    def cb(event):
        collected.append(event)
        if callback:
            callback(event)

    req = ProductionRequest(
        query="test art",
        collection_size=3,
        output_root=str(tmp_path / "outputs"),
        enable_cost_tracking=False,
        skip_mockups=True,
        skip_listing=True,
        max_image_retries=0,
    )
    run_production(req, _deps=deps, _progress_callback=cb)


def _resume_minimal_production(run_dir: Path, events_list: list) -> None:
    """Resume a production run, collecting events."""
    tmp_path = run_dir.parent.parent

    # Re-make deps that succeed at all stages
    deps = _make_deps(tmp_path, fail_at=None)

    def cb(event):
        events_list.append(event)

    resume_production(run_dir, _deps=deps, _progress_callback=cb)


def _make_completed_run_dir(tmp_path: Path) -> Path:
    """Create a fully completed production run directory for resume testing."""
    from agent.production_orchestrator import _STAGE_NAMES

    run_dir = tmp_path / "outputs" / "completed_run"
    run_dir.mkdir(parents=True)

    stages = [
        {
            "stage_name": n,
            "status": "completed",
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:01:00+00:00",
            "output_file": None,
            "error_message": None,
        }
        for n in _STAGE_NAMES
    ]

    (run_dir / "manifest.json").write_text(json.dumps({
        "production_id": "completed_run",
        "query": "test",
        "collection_size": 1,
        "status": "completed",
        "current_stage": "finalize",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:05:00+00:00",
        "output_directory": str(run_dir),
        "stages": stages,
        "resume_count": 0,
        "last_resumed_at": None,
        "total_cost": None,
        "total_input_tokens": None,
        "total_output_tokens": None,
        "total_images": None,
        "final_listing_file": None,
        "error_message": None,
    }), encoding="utf-8")

    # Write request.json
    (run_dir / "request.json").write_text(json.dumps({
        "query": "test",
        "collection_size": 1,
        "output_root": str(tmp_path / "outputs"),
        "selected_concept_index": None,
        "max_image_retries": 1,
        "skip_mockups": True,
        "skip_listing": True,
        "enable_cost_tracking": False,
    }), encoding="utf-8")

    # Create output files that _validate_completed_stages checks
    (run_dir / "research").mkdir()
    (run_dir / "research" / "result.json").write_text(
        json.dumps({"products": []}), encoding="utf-8"
    )
    (run_dir / "concepts").mkdir()
    (run_dir / "concepts" / "concepts.json").write_text(
        json.dumps([{"name": "C1"}]), encoding="utf-8"
    )
    (run_dir / "concepts" / "selected_concept.json").write_text(
        json.dumps({"name": "C1"}), encoding="utf-8"
    )
    (run_dir / "prompts").mkdir()
    (run_dir / "prompts" / "optimized_prompt.json").write_text(
        json.dumps({"optimized_image_prompt": "x", "optimized_negative_prompt": "y"}),
        encoding="utf-8"
    )
    (run_dir / "collection").mkdir()
    (run_dir / "collection" / "collection_plan.json").write_text(
        json.dumps({"collection_bible": {}, "poster_items": []}), encoding="utf-8"
    )
    images = run_dir / "images" / "poster_01"
    images.mkdir(parents=True)
    (images / "final.png").write_bytes(b"PNG")

    return run_dir


def _minimal_progress(last_event_sequence: int = 0) -> ProductionProgress:
    """Build a minimal ProductionProgress for snapshot testing."""
    from agent.production_orchestrator import _STAGE_NAMES
    return ProductionProgress(
        schema_version=SCHEMA_VERSION,
        run_id="test",
        status="running",
        percent=0.0,
        current_stage=None,
        poster_completed=0,
        poster_total=2,
        current_poster=None,
        current_attempt=None,
        elapsed_seconds=0.0,
        eta_seconds=None,
        eta_available=False,
        eta_confidence="low",
        eta_basis="static_fallback",
        estimated_completion_at=None,
        stages=[],
        created_at="2026-01-01T00:00:00+00:00",
        first_started_at="2026-01-01T00:00:00+00:00",
        last_updated_at="2026-01-01T00:00:00+00:00",
        last_event_sequence=last_event_sequence,
    )
