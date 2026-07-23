"""
Unit tests for resume_production (Stage 10.2).
No live API calls — all dependencies injected as fakes.

Tests A–R map to the spec requirements:
A  completed research is not called again
B  failed concept generation resumes from concept generation
C  running stage is treated as interrupted and rerun
D  completed run performs zero dependency calls
E  corrupt completed-stage JSON invalidates that stage and downstream
F  earlier valid completed stages remain untouched
G  skipped mockup/listing stages remain skipped
H  missing request.json raises clear error
I  missing manifest.json raises clear error
J  old Stage 10.1 manifest without resume fields loads successfully
K  resume_count increments
L  request settings are loaded from disk unchanged
M  completed posters are not regenerated
N  incomplete poster resumes from next attempt
O  retry limit counts previous attempts
P  one corrupt poster does not regenerate other valid posters
Q  original exception is re-raised after manifest failure update
R  resume on already completed run is idempotent
"""
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from agent.collection_generator import (
    CollectionBible,
    CollectionEvaluation,
    CollectionPlan,
    CollectionPoster,
)
from agent.vision_critic import ScoreWithReason
from agent.production_orchestrator import (
    AttemptRecord,
    ProductionDependencies,
    ProductionManifest,
    ProductionRequest,
    StageRecord,
    _STAGE_NAMES,
    _invalidate_from,
    _load_manifest,
    _load_request,
    resume_production,
    run_production,
)


# ── Shared fake builders ───────────────────────────────────────────────────────

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
        }
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
        p.negative_prompt = f"negative {i}"
        posters.append(p)
    plan.poster_items = posters
    return plan


def _real_collection_plan(n: int = 2) -> CollectionPlan:
    def _swr() -> ScoreWithReason:
        return ScoreWithReason(score=8, reason="test")

    bible = CollectionBible(
        collection_name="Test Collection",
        collection_story="story",
        brand_identity="brand",
        target_customer="customer",
        recommended_room_style="modern",
        visual_identity="visual",
        shared_rendering_medium="woodblock",
        shared_linework="thin",
        shared_lighting="natural",
        shared_palette=["indigo"],
        shared_accent_colour_rules=[],
        shared_camera_angle="eye level",
        shared_perspective="one point",
        shared_atmosphere="calm",
        shared_detail_level="high",
        shared_print_treatment="flat",
        shared_storytelling_rules=[],
        shared_composition_rules=[],
        shared_style_rules=[],
        shared_negative_prompt="border",
        style_dna=["ukiyo-e"],
        consistency_rules=[],
        forbidden_elements=[],
        full_bleed_rules=[],
    )
    posters = [
        CollectionPoster(
            index=i,
            title=f"Poster {i}",
            subject=f"subject {i}",
            scene_concept=f"scene {i}",
            storytelling_focus=f"focus {i}",
            unique_hook=f"hook {i}",
            image_prompt=f"full bleed composition poster {i}",
            negative_prompt=f"negative {i}",
            aspect_ratio="2:3",
            focal_point="center",
            foreground_elements=[],
            midground_elements=[],
            background_elements=[],
            palette_variation=[],
            lighting_variation="natural",
            weather_or_time_variation="clear",
            consistency_notes=[],
            suggested_etsy_title=f"Poster {i}",
            suggested_etsy_tags=[],
            mockup_room_style="modern",
        )
        for i in range(1, n + 1)
    ]
    evaluation = CollectionEvaluation(
        consistency_score=_swr(),
        commercial_score=_swr(),
        variation_score=_swr(),
        brand_identity_score=_swr(),
        print_collection_score=_swr(),
        market_uniqueness_score=_swr(),
        overall_score=_swr(),
        reasoning="test",
    )
    return CollectionPlan(
        collection_bible=bible,
        collection_size=n,
        poster_items=posters,
        collection_consistency_notes=[],
        evaluation=evaluation,
        confidence_score=9,
    )


def _image_factory(tmp_path: Path, counter=None):
    if counter is None:
        counter = [0]

    def factory():
        provider = MagicMock()

        def generate(prompt, **kwargs) -> str:
            counter[0] += 1
            path = tmp_path / f"raw_{counter[0]}.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return str(path)

        provider.generate = generate
        return provider

    return factory


def _vr_accept():
    vr = MagicMock()
    vr.retry_recommended = False
    vr.overall_score.score = 9
    vr.final_recommendation = "PROCEED"
    return vr


def _vr_retry():
    vr = MagicMock()
    vr.retry_recommended = True
    vr.overall_score.score = 4
    vr.final_recommendation = "RETRY"
    return vr


def _rp_accept():
    rp = MagicMock()
    rp.should_retry = False
    rp.revised_image_prompt = None
    rp.revised_negative_prompt = None
    return rp


def _rp_retry():
    rp = MagicMock()
    rp.should_retry = True
    rp.revised_image_prompt = "revised full bleed prompt"
    rp.revised_negative_prompt = "revised negative"
    return rp


def _make_deps(tmp_path: Path, n_posters: int = 2) -> ProductionDependencies:
    collection_plan = _real_collection_plan(n_posters)
    return ProductionDependencies(
        research_provider_factory=lambda: MagicMock(
            search=MagicMock(return_value=[_fake_product(i) for i in range(3)])
        ),
        analyze=MagicMock(return_value={"poster_concepts": _fake_concepts()}),
        optimize=MagicMock(return_value={
            "optimized_image_prompt": "full bleed optimized",
            "optimized_negative_prompt": "optimized negative",
        }),
        generate_collection=MagicMock(return_value=collection_plan),
        image_provider_factory=_image_factory(tmp_path),
        vision_review=MagicMock(return_value=_vr_accept()),
        prepare_retry=MagicMock(return_value=_rp_accept()),
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


def _run_dir(result) -> Path:
    return Path(result.manifest.output_directory)


def _get_manifest_data(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text())


def _stage_status(run_dir: Path, stage_name: str) -> str:
    data = _get_manifest_data(run_dir)
    for s in data["stages"]:
        if s["stage_name"] == stage_name:
            return s["status"]
    raise KeyError(stage_name)


def _write_fake_poster(prod_dir: Path, poster_index: int, n_attempts: int = 1, last_accepted: bool = True):
    """Write fake poster files simulating a completed poster."""
    poster_dir = prod_dir / "images" / f"poster_{poster_index:02d}"
    poster_dir.mkdir(parents=True, exist_ok=True)

    attempts = []
    for i in range(1, n_attempts + 1):
        img_name = "original.png" if i == 1 else f"attempt_{i}.png"
        img_file = poster_dir / img_name
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n")

        vr_file = poster_dir / f"vision_report_{i}.json"
        vr_file.write_text(json.dumps({"overall_score": {"score": 8, "reason": "ok"}}))

        accepted = last_accepted if i == n_attempts else False
        rp_file = poster_dir / f"retry_plan_{i}.json"
        rp_file.write_text(json.dumps({
            "should_retry": not accepted,
            "revised_image_prompt": "revised prompt" if not accepted else None,
            "revised_negative_prompt": "revised neg" if not accepted else None,
        }))
        attempts.append({
            "attempt_number": i,
            "image_file": str(img_file.relative_to(prod_dir)),
            "vision_report_file": str(vr_file.relative_to(prod_dir)),
            "retry_plan_file": str(rp_file.relative_to(prod_dir)),
            "accepted": accepted,
            "created_at": "2026-01-01T00:00:00+00:00",
        })

    (poster_dir / "attempts.json").write_text(json.dumps(attempts))
    last_img_name = "original.png" if n_attempts == 1 else f"attempt_{n_attempts}.png"
    shutil.copy2(str(poster_dir / last_img_name), str(poster_dir / "final.png"))
    return poster_dir


# ── Test A: completed research is not called again ─────────────────────────────

def test_A_completed_research_not_called_again(tmp_path):
    # Run until concept_generation fails
    deps1 = _make_deps(tmp_path)
    deps1.analyze.side_effect = RuntimeError("api error")
    req = _request(tmp_path)

    with pytest.raises(RuntimeError, match="api error"):
        run_production(req, _deps=deps1)

    run_dirs = list((tmp_path / "outputs").iterdir())
    run_dir = run_dirs[0]
    assert _stage_status(run_dir, "research") == "completed"

    # Resume — research must not be called again
    gen_counter = [0]

    deps2 = _make_deps(tmp_path, n_posters=3)
    original_factory = deps2.research_provider_factory

    def counting_factory():
        gen_counter[0] += 1
        return original_factory()

    deps2.research_provider_factory = counting_factory
    resume_production(run_dir, _deps=deps2)
    assert gen_counter[0] == 0, "research_provider_factory must not be called on resume"


# ── Test B: failed concept generation resumes from there ──────────────────────

def test_B_failed_stage_resumes_from_that_stage(tmp_path):
    deps1 = _make_deps(tmp_path)
    deps1.analyze.side_effect = RuntimeError("analyze failed")
    req = _request(tmp_path)

    with pytest.raises(RuntimeError):
        run_production(req, _deps=deps1)

    run_dir = list((tmp_path / "outputs").iterdir())[0]
    assert _stage_status(run_dir, "concept_generation") == "failed"

    deps2 = _make_deps(tmp_path, n_posters=3)
    result = resume_production(run_dir, _deps=deps2)

    assert result.manifest.status == "completed"
    deps2.analyze.assert_called_once()


# ── Test C: running stage is rerun ────────────────────────────────────────────

def test_C_running_stage_is_reset_and_rerun(tmp_path):
    deps1 = _make_deps(tmp_path)
    req = _request(tmp_path)

    with pytest.raises(RuntimeError):
        deps1.analyze.side_effect = RuntimeError("mid-run crash")
        run_production(req, _deps=deps1)

    run_dir = list((tmp_path / "outputs").iterdir())[0]

    # Manually set concept_generation to "running" (simulates crash mid-execution)
    data = _get_manifest_data(run_dir)
    for s in data["stages"]:
        if s["stage_name"] == "concept_generation":
            s["status"] = "running"
    (run_dir / "manifest.json").write_text(json.dumps(data))

    deps2 = _make_deps(tmp_path, n_posters=3)
    result = resume_production(run_dir, _deps=deps2)

    assert result.manifest.status == "completed"
    deps2.analyze.assert_called_once()


# ── Test D: completed run performs zero dependency calls ──────────────────────

def test_D_completed_run_zero_calls(tmp_path):
    deps1 = _make_deps(tmp_path, n_posters=3)
    req = _request(tmp_path)
    result1 = run_production(req, _deps=deps1)

    run_dir = _run_dir(result1)

    # Fresh mock deps for the resume
    deps2 = ProductionDependencies(
        research_provider_factory=MagicMock(),
        analyze=MagicMock(),
        optimize=MagicMock(),
        generate_collection=MagicMock(),
        image_provider_factory=MagicMock(),
        vision_review=MagicMock(),
        prepare_retry=MagicMock(),
        generate_mockup_plan=MagicMock(),
        generate_listing_plan=MagicMock(),
    )
    result2 = resume_production(run_dir, _deps=deps2)

    deps2.research_provider_factory.assert_not_called()
    deps2.analyze.assert_not_called()
    deps2.optimize.assert_not_called()
    deps2.generate_collection.assert_not_called()
    deps2.image_provider_factory.assert_not_called()
    deps2.vision_review.assert_not_called()
    deps2.prepare_retry.assert_not_called()
    deps2.generate_mockup_plan.assert_not_called()
    deps2.generate_listing_plan.assert_not_called()
    assert result2.manifest.status == "completed"


# ── Test E: corrupt completed-stage JSON invalidates downstream ───────────────

def test_E_corrupt_stage_json_invalidates_downstream(tmp_path):
    deps1 = _make_deps(tmp_path, n_posters=3)
    req = _request(tmp_path)
    result1 = run_production(req, _deps=deps1)

    run_dir = _run_dir(result1)

    # Corrupt the collection_plan output
    (run_dir / "collection/collection_plan.json").write_text("not json {{{")

    # Mark collection_generation as completed (it was), others remain completed
    # Resume: should detect corruption, invalidate collection_generation and downstream
    deps2 = _make_deps(tmp_path, n_posters=3)
    result2 = resume_production(run_dir, _deps=deps2)

    assert result2.manifest.status == "completed"
    deps2.generate_collection.assert_called_once()  # was re-run


# ── Test F: earlier completed stages remain untouched ─────────────────────────

def test_F_earlier_stages_untouched_on_invalidation(tmp_path):
    deps1 = _make_deps(tmp_path, n_posters=3)
    req = _request(tmp_path)
    result1 = run_production(req, _deps=deps1)

    run_dir = _run_dir(result1)
    original_concepts_mtime = (run_dir / "concepts/concepts.json").stat().st_mtime

    # Corrupt collection_plan to force re-run from that stage
    (run_dir / "collection/collection_plan.json").write_text("{}")

    deps2 = _make_deps(tmp_path, n_posters=3)
    resume_production(run_dir, _deps=deps2)

    # Stages before collection_generation must not have re-run
    deps2.analyze.assert_not_called()
    deps2.optimize.assert_not_called()
    new_mtime = (run_dir / "concepts/concepts.json").stat().st_mtime
    assert new_mtime == original_concepts_mtime


# ── Test G: skipped stages remain skipped ─────────────────────────────────────

def test_G_skipped_stages_remain_skipped(tmp_path):
    deps1 = _make_deps(tmp_path, n_posters=3)
    req = _request(tmp_path, skip_mockups=True)
    result1 = run_production(req, _deps=deps1)
    run_dir = _run_dir(result1)

    # Fail at the finalize stage by corrupting its concept (forces re-run of something)
    # Simpler: just resume a completed run with skip_mockups — skipped stays skipped
    deps2 = ProductionDependencies(
        research_provider_factory=MagicMock(),
        analyze=MagicMock(),
        optimize=MagicMock(),
        generate_collection=MagicMock(),
        image_provider_factory=MagicMock(),
        vision_review=MagicMock(),
        prepare_retry=MagicMock(),
        generate_mockup_plan=MagicMock(),
        generate_listing_plan=MagicMock(),
    )
    result2 = resume_production(run_dir, _deps=deps2)

    data = _get_manifest_data(run_dir)
    stage_map = {s["stage_name"]: s["status"] for s in data["stages"]}
    assert stage_map["mockup_generation"] == "skipped"
    assert stage_map["listing_generation"] == "skipped"
    deps2.generate_mockup_plan.assert_not_called()


# ── Test H: missing request.json raises clear error ───────────────────────────

def test_H_missing_request_json_raises(tmp_path):
    fake_dir = tmp_path / "nonexistent_run"
    fake_dir.mkdir()
    (fake_dir / "manifest.json").write_text(json.dumps({
        "production_id": "x",
        "query": "q",
        "collection_size": 3,
        "status": "failed",
        "current_stage": "",
        "created_at": "t",
        "updated_at": "t",
        "output_directory": str(fake_dir),
        "stages": [],
    }))
    with pytest.raises(FileNotFoundError, match="request.json"):
        resume_production(fake_dir)


# ── Test I: missing manifest.json raises clear error ──────────────────────────

def test_I_missing_manifest_json_raises(tmp_path):
    fake_dir = tmp_path / "nonexistent_run"
    fake_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        resume_production(fake_dir)


# ── Test J: old manifest without resume fields loads successfully ──────────────

def test_J_old_manifest_without_resume_fields(tmp_path):
    # Create a directory with Stage 10.1 style manifest (no resume_count / last_resumed_at)
    deps1 = _make_deps(tmp_path, n_posters=3)
    req = _request(tmp_path)
    result1 = run_production(req, _deps=deps1)
    run_dir = _run_dir(result1)

    # Strip new fields to simulate old manifest
    data = _get_manifest_data(run_dir)
    data.pop("resume_count", None)
    data.pop("last_resumed_at", None)
    data["status"] = "failed"  # force resume to do something
    # Also mark concept_generation failed so it reruns
    for s in data["stages"]:
        if s["stage_name"] == "concept_generation":
            s["status"] = "failed"
    (run_dir / "manifest.json").write_text(json.dumps(data))

    deps2 = _make_deps(tmp_path, n_posters=3)
    result2 = resume_production(run_dir, _deps=deps2)
    assert result2.manifest.status == "completed"
    assert result2.manifest.resume_count == 1  # default was 0, incremented to 1


# ── Test K: resume_count increments ───────────────────────────────────────────

def test_K_resume_count_increments(tmp_path):
    # Initial run fails at concept_generation
    deps1 = _make_deps(tmp_path)
    deps1.analyze.side_effect = RuntimeError("fail")
    req = _request(tmp_path)
    with pytest.raises(RuntimeError):
        run_production(req, _deps=deps1)

    run_dir = list((tmp_path / "outputs").iterdir())[0]

    # First resume also fails
    deps2 = _make_deps(tmp_path, n_posters=3)
    deps2.analyze.side_effect = RuntimeError("fail again")
    with pytest.raises(RuntimeError):
        resume_production(run_dir, _deps=deps2)

    # Second resume succeeds
    deps3 = _make_deps(tmp_path, n_posters=3)
    result = resume_production(run_dir, _deps=deps3)
    assert result.manifest.status == "completed"
    assert result.manifest.resume_count == 2


# ── Test L: request settings loaded from disk unchanged ───────────────────────

def test_L_request_loaded_from_disk(tmp_path):
    deps1 = _make_deps(tmp_path)
    deps1.analyze.side_effect = RuntimeError("fail")
    req = _request(tmp_path, query="vintage botanical art", collection_size=3, max_image_retries=0)

    with pytest.raises(RuntimeError):
        run_production(req, _deps=deps1)

    run_dir = list((tmp_path / "outputs").iterdir())[0]

    deps2 = _make_deps(tmp_path, n_posters=3)
    result = resume_production(run_dir, _deps=deps2)

    # The request reconstructed from disk must match the original
    loaded = _load_request(run_dir)
    assert loaded.query == "vintage botanical art"
    assert loaded.collection_size == 3
    assert loaded.max_image_retries == 0


# ── Test M: completed posters are not regenerated ─────────────────────────────

def test_M_completed_posters_not_regenerated(tmp_path):
    # Run up to image_generation failing at poster 2
    deps1 = _make_deps(tmp_path, n_posters=2)
    req = _request(tmp_path)
    result1 = run_production(req, _deps=deps1)
    run_dir = _run_dir(result1)

    # Wipe poster 2's attempts.json to simulate a partial run
    poster2_dir = run_dir / "images/poster_02"
    (poster2_dir / "attempts.json").unlink()
    (poster2_dir / "final.png").unlink()

    # Mark image_generation as failed
    data = _get_manifest_data(run_dir)
    data["status"] = "failed"
    for s in data["stages"]:
        if s["stage_name"] in ("image_generation", "vision_critique", "retry_generation", "finalize"):
            s["status"] = "pending"
    (run_dir / "manifest.json").write_text(json.dumps(data))

    generate_calls = []
    generate_counter = [0]

    def counting_factory():
        provider = MagicMock()

        def generate(prompt, **kwargs) -> str:
            generate_calls.append(prompt)
            generate_counter[0] += 1
            path = tmp_path / f"raw_resume_{generate_counter[0]}.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return str(path)

        provider.generate = generate
        return provider

    deps2 = _make_deps(tmp_path, n_posters=2)
    deps2.image_provider_factory = counting_factory

    resume_production(run_dir, _deps=deps2)

    # Only poster 2 should have been regenerated (1 generate call)
    assert len(generate_calls) == 1


# ── Test N: incomplete poster resumes from next attempt ───────────────────────

def test_N_incomplete_poster_resumes_from_next_attempt(tmp_path):
    """Poster with original.png + retry_plan_1.json but no attempts.json resumes at attempt 2."""
    deps1 = _make_deps(tmp_path, n_posters=1)
    req = _request(tmp_path, max_image_retries=1)
    result1 = run_production(req, _deps=deps1)
    run_dir = _run_dir(result1)

    # Simulate a crash after attempt 1 was generated/critiqued/planned but before attempts.json
    poster_dir = run_dir / "images/poster_01"
    (poster_dir / "attempts.json").unlink()
    (poster_dir / "final.png").unlink()

    # Make retry_plan_1 say should_retry=True
    rp1 = poster_dir / "retry_plan_1.json"
    rp1.write_text(json.dumps({
        "should_retry": True,
        "revised_image_prompt": "revised prompt for attempt 2",
        "revised_negative_prompt": "revised neg",
    }))

    # Mark image_generation failed
    data = _get_manifest_data(run_dir)
    data["status"] = "failed"
    for s in data["stages"]:
        if s["stage_name"] in ("image_generation", "vision_critique", "retry_generation", "finalize"):
            s["status"] = "pending"
    (run_dir / "manifest.json").write_text(json.dumps(data))

    generate_calls = []
    gen_counter = [0]

    def counting_factory():
        provider = MagicMock()

        def generate(prompt, **kwargs) -> str:
            generate_calls.append(prompt)
            gen_counter[0] += 1
            path = tmp_path / f"raw_n_{gen_counter[0]}.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return str(path)

        provider.generate = generate
        return provider

    deps2 = _make_deps(tmp_path, n_posters=1)
    deps2.image_provider_factory = counting_factory
    resume_production(run_dir, _deps=deps2)

    # Should have generated exactly 1 image (attempt 2, not attempt 1 again)
    assert len(generate_calls) == 1
    assert generate_calls[0] == "revised prompt for attempt 2"


# ── Test O: retry limit counts previous attempts ──────────────────────────────

def test_O_retry_limit_counts_previous_attempts(tmp_path):
    """With max_retries=1 and attempt 1 already on disk, only attempt 2 runs then stops."""
    deps1 = _make_deps(tmp_path, n_posters=1)
    req = _request(tmp_path, max_image_retries=1)
    result1 = run_production(req, _deps=deps1)
    run_dir = _run_dir(result1)

    poster_dir = run_dir / "images/poster_01"
    (poster_dir / "attempts.json").unlink()
    (poster_dir / "final.png").unlink()

    # attempt 1: should_retry=True
    (poster_dir / "retry_plan_1.json").write_text(json.dumps({
        "should_retry": True,
        "revised_image_prompt": "revised",
        "revised_negative_prompt": "revised neg",
    }))

    data = _get_manifest_data(run_dir)
    data["status"] = "failed"
    for s in data["stages"]:
        if s["stage_name"] in ("image_generation", "vision_critique", "retry_generation", "finalize"):
            s["status"] = "pending"
    (run_dir / "manifest.json").write_text(json.dumps(data))

    gen_calls = [0]

    def counting_factory():
        provider = MagicMock()

        def generate(prompt, **kwargs):
            gen_calls[0] += 1
            path = tmp_path / f"raw_o_{gen_calls[0]}.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return str(path)

        provider.generate = generate
        return provider

    deps2 = _make_deps(tmp_path, n_posters=1)
    deps2.image_provider_factory = counting_factory
    # Vision always says retry, but max_retries=1 means we stop after attempt 2
    deps2.vision_review = MagicMock(return_value=_vr_retry())
    deps2.prepare_retry = MagicMock(return_value=_rp_retry())

    resume_production(run_dir, _deps=deps2)

    # Only 1 generate call (attempt 2 only — attempt 1 was reconstructed from disk)
    assert gen_calls[0] == 1

    attempts = json.loads((poster_dir / "attempts.json").read_text())
    assert len(attempts) == 2  # attempt 1 (reconstructed) + attempt 2 (new)


# ── Test P: corrupt poster does not affect other posters ──────────────────────

def test_P_corrupt_poster_does_not_affect_valid_posters(tmp_path):
    deps1 = _make_deps(tmp_path, n_posters=2)
    req = _request(tmp_path)
    result1 = run_production(req, _deps=deps1)
    run_dir = _run_dir(result1)

    # Corrupt poster 2: bad attempts.json, no final.png, no retry_plan so reconstruction fails
    (run_dir / "images/poster_02/attempts.json").write_text("{{NOT JSON}}")
    (run_dir / "images/poster_02/final.png").unlink()
    (run_dir / "images/poster_02/retry_plan_1.json").unlink()

    data = _get_manifest_data(run_dir)
    data["status"] = "failed"
    for s in data["stages"]:
        if s["stage_name"] in ("image_generation", "vision_critique", "retry_generation", "finalize"):
            s["status"] = "pending"
    (run_dir / "manifest.json").write_text(json.dumps(data))

    prompts_generated = []
    gen_counter = [0]

    def counting_factory():
        provider = MagicMock()

        def generate(prompt, **kwargs):
            prompts_generated.append(prompt)
            gen_counter[0] += 1
            path = tmp_path / f"raw_p_{gen_counter[0]}.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return str(path)

        provider.generate = generate
        return provider

    deps2 = _make_deps(tmp_path, n_posters=2)
    deps2.image_provider_factory = counting_factory
    resume_production(run_dir, _deps=deps2)

    # Poster 1 was valid (has final.png and valid attempts.json) → skipped
    # Poster 2 was corrupt → regenerated
    assert len(prompts_generated) == 1


# ── Test Q: original exception is re-raised ───────────────────────────────────

def test_Q_original_exception_reraised(tmp_path):
    deps1 = _make_deps(tmp_path)
    deps1.analyze.side_effect = RuntimeError("fail")
    req = _request(tmp_path)

    with pytest.raises(RuntimeError):
        run_production(req, _deps=deps1)

    run_dir = list((tmp_path / "outputs").iterdir())[0]

    sentinel = ValueError("sentinel on resume")
    deps2 = _make_deps(tmp_path, n_posters=3)
    deps2.analyze.side_effect = sentinel

    with pytest.raises(ValueError) as exc_info:
        resume_production(run_dir, _deps=deps2)

    assert exc_info.value is sentinel
    data = _get_manifest_data(run_dir)
    assert data["status"] == "failed"


# ── Test R: resume on completed run is idempotent ─────────────────────────────

def test_R_resume_idempotent_on_completed_run(tmp_path):
    deps1 = _make_deps(tmp_path, n_posters=3)
    req = _request(tmp_path)
    result1 = run_production(req, _deps=deps1)
    run_dir = _run_dir(result1)

    manifest_before = (run_dir / "manifest.json").read_text()

    deps2 = ProductionDependencies(
        research_provider_factory=MagicMock(),
        analyze=MagicMock(),
        optimize=MagicMock(),
        generate_collection=MagicMock(),
        image_provider_factory=MagicMock(),
        vision_review=MagicMock(),
        prepare_retry=MagicMock(),
        generate_mockup_plan=MagicMock(),
        generate_listing_plan=MagicMock(),
    )
    result2 = resume_production(run_dir, _deps=deps2)

    manifest_after = (run_dir / "manifest.json").read_text()

    # No deps called
    for attr in ("analyze", "optimize", "generate_collection",
                 "image_provider_factory", "vision_review", "prepare_retry",
                 "generate_mockup_plan", "generate_listing_plan"):
        getattr(deps2, attr).assert_not_called()

    assert result2.manifest.status == "completed"
    # Manifest unchanged (no writes on idempotent fast path)
    assert manifest_before == manifest_after


# ── _invalidate_from unit tests ────────────────────────────────────────────────

def _make_manifest(statuses: dict[str, str]) -> ProductionManifest:
    stages = [
        StageRecord(stage_name=n, status=statuses.get(n, "pending"), started_at="")
        for n in _STAGE_NAMES
    ]
    return ProductionManifest(
        production_id="test",
        query="q",
        collection_size=3,
        status="running",
        current_stage="",
        created_at="t",
        updated_at="t",
        output_directory="/tmp/x",
        stages=stages,
    )


def test_invalidate_from_resets_stage_and_downstream():
    m = _make_manifest({
        "research": "completed",
        "concept_generation": "completed",
        "concept_selection": "completed",
        "prompt_optimization": "completed",
        "collection_generation": "completed",
        "image_generation": "completed",
        "vision_critique": "completed",
        "retry_generation": "skipped",
        "mockup_generation": "completed",
        "listing_generation": "completed",
        "finalize": "completed",
    })
    _invalidate_from(m, "collection_generation")

    def status(name):
        return next(s.status for s in m.stages if s.stage_name == name)

    assert status("research") == "completed"
    assert status("concept_generation") == "completed"
    assert status("concept_selection") == "completed"
    assert status("prompt_optimization") == "completed"
    assert status("collection_generation") == "pending"
    assert status("image_generation") == "pending"
    assert status("vision_critique") == "pending"
    assert status("retry_generation") == "skipped"  # skipped preserved
    assert status("mockup_generation") == "pending"
    assert status("listing_generation") == "pending"
    assert status("finalize") == "pending"
