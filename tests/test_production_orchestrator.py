"""
Unit tests for production_orchestrator.run_production.
No live API calls — all dependencies are injected as fakes.
"""
import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import agent.production_orchestrator as _orch_module
from agent.production_orchestrator import (
    ProductionDependencies,
    ProductionRequest,
    run_production,
)


# ── Fake builders ──────────────────────────────────────────────────────────────

def _fake_product(i: int):
    p = MagicMock()
    p.to_dict.return_value = {"title": f"Product {i}", "description": f"Desc {i}"}
    return p


def _fake_concepts():
    return [
        {
            "name": "Concept Alpha",
            "niche": "ukiyo-e",
            "art_style": "woodblock",
            "image_generation_prompt": "full bleed ukiyo-e forest",
            "negative_prompt": "border frame",
        },
        {
            "name": "Concept Beta",
            "niche": "botanical",
            "art_style": "watercolour",
            "image_generation_prompt": "full bleed botanical illustration",
            "negative_prompt": "border frame",
        },
    ]


def _fake_collection_plan(n: int = 2):
    plan = MagicMock()
    plan.collection_size = n
    plan.collection_bible.collection_name = "Test Collection"
    posters = []
    for i in range(1, n + 1):
        p = MagicMock()
        p.index = i
        p.title = f"Poster {i}"
        p.image_prompt = f"full bleed composition poster {i}"
        p.negative_prompt = f"test negative {i}"
        posters.append(p)
    plan.poster_items = posters
    return plan


def _image_factory(tmp_path: Path):
    """Returns a factory callable that creates a provider writing real temp files."""
    counter = [0]

    def factory():
        provider = MagicMock()

        def generate(prompt: str) -> str:
            counter[0] += 1
            path = tmp_path / f"generated_{counter[0]}.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return str(path)

        provider.generate = generate
        return provider

    return factory


def _vr_seq(*retry_flags: bool):
    """vision_review returning retry_recommended=flags in sequence (last repeats)."""
    calls = [0]

    def vision_review(concept, prompt, neg, path):
        idx = min(calls[0], len(retry_flags) - 1)
        retry = retry_flags[idx]
        calls[0] += 1
        vr = MagicMock()
        vr.retry_recommended = retry
        vr.overall_score.score = 6 if retry else 9
        vr.final_recommendation = "RETRY" if retry else "PROCEED"
        return vr

    return vision_review


def _rp_seq(*should_retry_flags: bool):
    """prepare_retry returning should_retry=flags in sequence (last repeats)."""
    calls = [0]

    def prepare_retry(concept, prompt, neg, vr):
        idx = min(calls[0], len(should_retry_flags) - 1)
        retry = should_retry_flags[idx]
        calls[0] += 1
        rp = MagicMock()
        rp.should_retry = retry
        rp.revised_image_prompt = "revised full bleed prompt" if retry else None
        rp.revised_negative_prompt = "revised negative" if retry else None
        return rp

    return prepare_retry


def _make_deps(
    tmp_path: Path,
    n_posters: int = 2,
    vr_flags=(False,),
    rp_flags=(False,),
) -> ProductionDependencies:
    concepts = _fake_concepts()
    collection_plan = _fake_collection_plan(n_posters)

    return ProductionDependencies(
        research_provider_factory=lambda: MagicMock(
            search=MagicMock(return_value=[_fake_product(i) for i in range(3)])
        ),
        analyze=MagicMock(return_value={"poster_concepts": concepts}),
        optimize=MagicMock(return_value={
            "optimized_image_prompt": "full bleed optimized prompt",
            "optimized_negative_prompt": "optimized negative",
        }),
        generate_collection=MagicMock(return_value=collection_plan),
        image_provider_factory=_image_factory(tmp_path),
        vision_review=_vr_seq(*vr_flags),
        prepare_retry=_rp_seq(*rp_flags),
        generate_mockup_plan=MagicMock(return_value=MagicMock()),
        generate_listing_plan=MagicMock(return_value=MagicMock()),
    )


def _request(tmp_path: Path, **overrides) -> ProductionRequest:
    base = dict(
        query="cozy anime wall art",
        collection_size=3,
        output_root=str(tmp_path / "outputs"),
        max_image_retries=1,
    )
    base.update(overrides)
    return ProductionRequest(**base)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_successful_full_pipeline(tmp_path):
    deps = _make_deps(tmp_path, n_posters=2)
    req = _request(tmp_path)
    result = run_production(req, _deps=deps)

    assert result.manifest.status == "completed"
    assert result.selected_concept is not None
    assert result.collection_plan is not None

    out = Path(result.manifest.output_directory)
    assert (out / "manifest.json").exists()
    assert (out / "request.json").exists()
    assert (out / "research/result.json").exists()
    assert (out / "concepts/concepts.json").exists()
    assert (out / "concepts/selected_concept.json").exists()
    assert (out / "prompts/optimized_prompt.json").exists()
    assert (out / "collection/collection_plan.json").exists()
    assert (out / "images/poster_01/original.png").exists()
    assert (out / "images/poster_01/final.png").exists()
    assert (out / "images/poster_01/attempts.json").exists()
    assert (out / "mockups/mockup_plan.json").exists()
    assert (out / "listing/listing_plan.json").exists()


def test_manifest_ends_as_completed(tmp_path):
    deps = _make_deps(tmp_path)
    result = run_production(_request(tmp_path), _deps=deps)
    manifest_data = json.loads(
        (Path(result.manifest.output_directory) / "manifest.json").read_text()
    )
    assert manifest_data["status"] == "completed"
    completed = [s for s in manifest_data["stages"] if s["status"] == "completed"]
    assert len(completed) >= 9  # at least research through finalize (minus retry_generation if skipped)


def test_automatic_concept_selection_picks_first(tmp_path):
    deps = _make_deps(tmp_path)
    req = _request(tmp_path, selected_concept_index=None)
    result = run_production(req, _deps=deps)
    assert result.selected_concept["name"] == "Concept Alpha"


def test_explicit_concept_selection(tmp_path):
    deps = _make_deps(tmp_path)
    req = _request(tmp_path, selected_concept_index=2)
    result = run_production(req, _deps=deps)
    assert result.selected_concept["name"] == "Concept Beta"


def test_invalid_concept_index_fails_at_selection_stage(tmp_path):
    deps = _make_deps(tmp_path)
    req = _request(tmp_path, selected_concept_index=99)
    with pytest.raises(ValueError, match="out of range"):
        run_production(req, _deps=deps)

    out = Path(deps.generate_collection.call_args or "x")  # not called
    deps.generate_collection.assert_not_called()

    # Manifest reflects failure at concept_selection
    output_dirs = list((tmp_path / "outputs").iterdir())
    assert output_dirs, "output dir should exist"
    manifest = json.loads((output_dirs[0] / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    failed = [s for s in manifest["stages"] if s["status"] == "failed"]
    assert failed[0]["stage_name"] == "concept_selection"


def test_skipped_mockup_stage(tmp_path):
    deps = _make_deps(tmp_path)
    req = _request(tmp_path, skip_mockups=True)
    result = run_production(req, _deps=deps)

    manifest = json.loads(
        (Path(result.manifest.output_directory) / "manifest.json").read_text()
    )
    stage_map = {s["stage_name"]: s["status"] for s in manifest["stages"]}
    assert stage_map["mockup_generation"] == "skipped"
    assert stage_map["listing_generation"] == "skipped"  # auto-skipped when mockups skipped
    deps.generate_mockup_plan.assert_not_called()
    deps.generate_listing_plan.assert_not_called()


def test_skipped_listing_stage(tmp_path):
    deps = _make_deps(tmp_path)
    req = _request(tmp_path, skip_listing=True)
    result = run_production(req, _deps=deps)

    manifest = json.loads(
        (Path(result.manifest.output_directory) / "manifest.json").read_text()
    )
    stage_map = {s["stage_name"]: s["status"] for s in manifest["stages"]}
    assert stage_map["listing_generation"] == "skipped"
    assert stage_map["mockup_generation"] == "completed"
    deps.generate_listing_plan.assert_not_called()


def test_stage_failure_updates_manifest(tmp_path):
    deps = _make_deps(tmp_path)
    deps.generate_collection.side_effect = RuntimeError("Claude API timeout")

    with pytest.raises(RuntimeError, match="Claude API timeout"):
        run_production(_request(tmp_path), _deps=deps)

    output_dirs = list((tmp_path / "outputs").iterdir())
    manifest = json.loads((output_dirs[0] / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "Claude API timeout" in manifest["error_message"]
    stage_map = {s["stage_name"]: s["status"] for s in manifest["stages"]}
    assert stage_map["collection_generation"] == "failed"


def test_original_exception_is_reraised(tmp_path):
    deps = _make_deps(tmp_path)
    sentinel = RuntimeError("sentinel error")
    deps.analyze.side_effect = sentinel

    with pytest.raises(RuntimeError) as exc_info:
        run_production(_request(tmp_path), _deps=deps)

    assert exc_info.value is sentinel


def test_output_files_persisted_before_later_stage_fails(tmp_path):
    """Files from completed stages exist even when a later stage fails."""
    deps = _make_deps(tmp_path)
    deps.generate_collection.side_effect = ValueError("collection failed")

    with pytest.raises(ValueError):
        run_production(_request(tmp_path), _deps=deps)

    output_dirs = list((tmp_path / "outputs").iterdir())
    out = output_dirs[0]
    # Stages 1-4 completed before stage 5 failed
    assert (out / "research/result.json").exists()
    assert (out / "concepts/concepts.json").exists()
    assert (out / "concepts/selected_concept.json").exists()
    assert (out / "prompts/optimized_prompt.json").exists()
    # Stage 5 output must not exist
    assert not (out / "collection/collection_plan.json").exists()


def test_image_accepted_on_first_attempt(tmp_path):
    deps = _make_deps(tmp_path, n_posters=1, vr_flags=(False,), rp_flags=(False,))
    result = run_production(_request(tmp_path), _deps=deps)

    out = Path(result.manifest.output_directory)
    attempts = json.loads((out / "images/poster_01/attempts.json").read_text())
    assert len(attempts) == 1
    assert attempts[0]["attempt_number"] == 1
    assert attempts[0]["accepted"] is True
    assert (out / "images/poster_01/original.png").exists()
    assert (out / "images/poster_01/final.png").exists()
    # No attempt_2.png should exist
    assert not (out / "images/poster_01/attempt_2.png").exists()


def test_image_retried_once_then_accepted(tmp_path):
    # First vision: retry=True, second: retry=False
    deps = _make_deps(
        tmp_path, n_posters=1,
        vr_flags=(True, False),
        rp_flags=(True, False),
    )
    result = run_production(_request(tmp_path, max_image_retries=2), _deps=deps)

    out = Path(result.manifest.output_directory)
    attempts = json.loads((out / "images/poster_01/attempts.json").read_text())
    assert len(attempts) == 2
    assert attempts[0]["accepted"] is False
    assert attempts[1]["accepted"] is True
    assert (out / "images/poster_01/original.png").exists()
    assert (out / "images/poster_01/attempt_2.png").exists()
    assert (out / "images/poster_01/final.png").exists()

    # retry_generation stage should be completed
    manifest = json.loads((out / "manifest.json").read_text())
    stage_map = {s["stage_name"]: s["status"] for s in manifest["stages"]}
    assert stage_map["retry_generation"] == "completed"


def test_max_retry_reached(tmp_path):
    # vision always says retry; max_image_retries=1 → 2 total attempts then stop
    deps = _make_deps(
        tmp_path, n_posters=1,
        vr_flags=(True,),
        rp_flags=(True,),
    )
    result = run_production(_request(tmp_path, max_image_retries=1), _deps=deps)

    out = Path(result.manifest.output_directory)
    attempts = json.loads((out / "images/poster_01/attempts.json").read_text())
    # 1 original + 1 retry = 2 total, then stopped
    assert len(attempts) == 2
    # Neither attempt was organically accepted (stopped by max retries)
    assert attempts[-1]["accepted"] is False
    # final.png still exists (last attempt)
    assert (out / "images/poster_01/final.png").exists()
    # Overall production still completes
    assert result.manifest.status == "completed"


def test_no_terminal_input_in_orchestrator():
    src = inspect.getsource(_orch_module)
    assert "input(" not in src, (
        "production_orchestrator.py must never call input() — "
        "all user interaction belongs in test_production.py"
    )


def test_request_validation_before_deps_called(tmp_path):
    deps = MagicMock()
    req = ProductionRequest(
        query="",  # invalid: empty query
        collection_size=3,
        output_root=str(tmp_path),
    )
    with pytest.raises(ValueError, match="query"):
        run_production(req, _deps=deps)

    deps.research_provider_factory.assert_not_called()
    deps.analyze.assert_not_called()
    deps.generate_collection.assert_not_called()


def test_invalid_collection_size_rejected(tmp_path):
    req = ProductionRequest(query="test", collection_size=99, output_root=str(tmp_path))
    with pytest.raises(ValueError, match="collection_size"):
        run_production(req, _deps=MagicMock())


def test_invalid_max_retries_rejected(tmp_path):
    req = ProductionRequest(
        query="test", collection_size=3,
        output_root=str(tmp_path), max_image_retries=10
    )
    with pytest.raises(ValueError, match="max_image_retries"):
        run_production(req, _deps=MagicMock())


def test_selected_concept_index_zero_rejected(tmp_path):
    req = ProductionRequest(
        query="test", collection_size=3,
        output_root=str(tmp_path), selected_concept_index=0
    )
    with pytest.raises(ValueError, match="selected_concept_index"):
        run_production(req, _deps=MagicMock())
