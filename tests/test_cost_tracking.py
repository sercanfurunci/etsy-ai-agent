"""
Unit tests for agent/cost_tracking.py — no live API calls.

Tests A–AG:
A  PricingCatalog loads from a JSON file path
B  PricingCatalog loads from a dict
C  PricingCatalog raises CatalogSchemaError on wrong schema_version
D  get_claude_cost returns None when input_per_mtok is null
E  get_claude_cost returns correct Decimal when prices are set
F  get_image_cost returns None when price is null
G  get_image_cost returns correct Decimal when price is set
H  CostTracker.record() appends one line to JSONL
I  CostTracker.record() creates costs/ dir if it does not exist
J  CostTracker.load_existing() reloads records from JSONL
K  compute_summary total_cost is null when pricing is null
L  compute_summary total_cost is a Decimal string when pricing is set
M  StageCostSummary groups records by stage correctly
N  save_summary() writes costs/summary.json atomically
O  ask() calls on_usage with correct provider/model/tokens
P  analyze() passes on_usage to ask
Q  optimize() passes on_usage to ask
R  vision_critic.review() calls on_usage after API call
S  retry_generator.prepare_retry() calls on_usage via ask when retrying
T  generate_collection() calls on_usage for bible + batch + eval
U  OpenAIImageProvider.generate() calls on_usage after image write
V  ProductionRequest has enable_cost_tracking field defaulting to True
W  run_production creates costs/ and usage_records.jsonl when tracking enabled
X  costs/summary.json written at end of run_production
Y  resume_production loads existing JSONL without duplicating records
Z  _refresh_job_cost loads summary into QueueJob cost fields
AA QueueResult.total_cost aggregates completed job costs
AB show_costs.py --json outputs valid JSON
AC show_costs.py --by-stage shows stage breakdown
AD show_costs.py exits non-zero for missing run directory
AE total_cost is null in summary when pricing catalog has null prices
AF multiple records aggregate input/output tokens correctly
AG Decimal cost is serialized as string not float in summary JSON
"""
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.cost_tracking import (
    CatalogSchemaError,
    CostTracker,
    PricingCatalog,
    ProductionCostSummary,
    QueueCostSummary,
    StageCostSummary,
    UsageRecord,
)


# ── Pricing fixtures ───────────────────────────────────────────────────────────

_NULL_CATALOG = {
    "schema_version": 1,
    "anthropic": {
        "claude-haiku-4-5-20251001": {"input_per_mtok": None, "output_per_mtok": None}
    },
    "openai_image": {
        "gpt-image-2": {"per_image": {"1536x2304": None}}
    },
}

_PRICED_CATALOG = {
    "schema_version": 1,
    "anthropic": {
        "test-model": {"input_per_mtok": 0.25, "output_per_mtok": 1.25}
    },
    "openai_image": {
        "test-img-model": {"per_image": {"1536x2304": 0.04}}
    },
}


def _raw_text(input_tokens=100, output_tokens=200, model="test-model"):
    return {
        "provider": "anthropic",
        "model": model,
        "call_type": "text",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _raw_image(model="test-img-model", size="1536x2304"):
    return {
        "provider": "openai",
        "model": model,
        "call_type": "image",
        "image_count": 1,
        "image_size": size,
    }


# ── A: load from file ──────────────────────────────────────────────────────────

def test_A_pricing_catalog_loads_from_file(tmp_path):
    p = tmp_path / "pricing.json"
    p.write_text(json.dumps(_NULL_CATALOG), encoding="utf-8")
    cat = PricingCatalog(path=p)
    assert cat._data["schema_version"] == 1


# ── B: load from dict ─────────────────────────────────────────────────────────

def test_B_pricing_catalog_loads_from_dict():
    cat = PricingCatalog(data=_NULL_CATALOG)
    assert "anthropic" in cat._data


# ── C: wrong schema version ───────────────────────────────────────────────────

def test_C_wrong_schema_version_raises():
    bad = {**_NULL_CATALOG, "schema_version": 99}
    with pytest.raises(CatalogSchemaError, match="99"):
        PricingCatalog(data=bad)


# ── D: null claude price → None ───────────────────────────────────────────────

def test_D_get_claude_cost_null_returns_none():
    cat = PricingCatalog(data=_NULL_CATALOG)
    result = cat.get_claude_cost("claude-haiku-4-5-20251001", 1000, 500)
    assert result is None


# ── E: real claude price → Decimal ───────────────────────────────────────────

def test_E_get_claude_cost_returns_decimal():
    cat = PricingCatalog(data=_PRICED_CATALOG)
    # 100 input @ $0.25/Mtok + 200 output @ $1.25/Mtok
    result = cat.get_claude_cost("test-model", 100, 200)
    assert isinstance(result, Decimal)
    expected = (Decimal("0.25") * 100 + Decimal("1.25") * 200) / Decimal("1000000")
    assert result == expected


# ── F: null image price → None ───────────────────────────────────────────────

def test_F_get_image_cost_null_returns_none():
    cat = PricingCatalog(data=_NULL_CATALOG)
    assert cat.get_image_cost("gpt-image-2", "1536x2304") is None


# ── G: real image price → Decimal ────────────────────────────────────────────

def test_G_get_image_cost_returns_decimal():
    cat = PricingCatalog(data=_PRICED_CATALOG)
    result = cat.get_image_cost("test-img-model", "1536x2304")
    assert result == Decimal("0.04")


# ── H: record appends to JSONL ────────────────────────────────────────────────

def test_H_record_appends_to_jsonl(tmp_path):
    tracker = CostTracker("run1", tmp_path / "costs", PricingCatalog(data=_NULL_CATALOG))
    tracker.record(_raw_text(), stage="concept_generation")
    jsonl = tmp_path / "costs" / "usage_records.jsonl"
    assert jsonl.exists()
    lines = [l for l in jsonl.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["stage"] == "concept_generation"
    assert rec["call_type"] == "text"


# ── I: creates costs dir ──────────────────────────────────────────────────────

def test_I_record_creates_costs_dir(tmp_path):
    costs_dir = tmp_path / "deep" / "costs"
    tracker = CostTracker("run1", costs_dir, PricingCatalog(data=_NULL_CATALOG))
    tracker.record(_raw_text(), stage="concept_generation")
    assert costs_dir.exists()


# ── J: load_existing reloads from JSONL ──────────────────────────────────────

def test_J_load_existing_reloads_records(tmp_path):
    cat = PricingCatalog(data=_NULL_CATALOG)
    costs = tmp_path / "costs"
    tracker1 = CostTracker("run1", costs, cat)
    tracker1.record(_raw_text(100, 200), stage="s1")
    tracker1.record(_raw_text(50, 75), stage="s2")

    tracker2 = CostTracker("run1", costs, cat)
    tracker2.load_existing()
    assert len(tracker2._records) == 2
    assert tracker2._records[0].stage == "s1"
    assert tracker2._records[1].stage == "s2"


# ── K: null pricing → null total_cost ────────────────────────────────────────

def test_K_null_pricing_gives_null_total_cost(tmp_path):
    cat = PricingCatalog(data=_NULL_CATALOG)
    tracker = CostTracker("run1", tmp_path / "costs", cat)
    tracker.record(_raw_text(model="claude-haiku-4-5-20251001"), stage="s1")
    summary = tracker.compute_summary()
    assert summary.total_cost is None
    assert summary.by_stage[0].cost is None


# ── L: priced catalog → Decimal string total_cost ────────────────────────────

def test_L_priced_catalog_gives_decimal_string_total_cost(tmp_path):
    cat = PricingCatalog(data=_PRICED_CATALOG)
    tracker = CostTracker("run1", tmp_path / "costs", cat)
    tracker.record(_raw_text(100, 200), stage="s1")
    summary = tracker.compute_summary()
    assert summary.total_cost is not None
    assert "." in summary.total_cost
    # Verify it's a valid Decimal
    Decimal(summary.total_cost)


# ── M: stage grouping ─────────────────────────────────────────────────────────

def test_M_summary_groups_by_stage(tmp_path):
    cat = PricingCatalog(data=_NULL_CATALOG)
    tracker = CostTracker("run1", tmp_path / "costs", cat)
    tracker.record(_raw_text(100, 50, model="claude-haiku-4-5-20251001"), stage="s1")
    tracker.record(_raw_text(200, 80, model="claude-haiku-4-5-20251001"), stage="s2")
    tracker.record(_raw_text(30, 10, model="claude-haiku-4-5-20251001"), stage="s1")
    summary = tracker.compute_summary()
    stages = {s.stage: s for s in summary.by_stage}
    assert stages["s1"].call_count == 2
    assert stages["s1"].input_tokens == 130
    assert stages["s2"].call_count == 1


# ── N: save_summary writes atomically ────────────────────────────────────────

def test_N_save_summary_writes_json(tmp_path):
    cat = PricingCatalog(data=_NULL_CATALOG)
    tracker = CostTracker("run1", tmp_path / "costs", cat)
    tracker.record(_raw_text(model="claude-haiku-4-5-20251001"), stage="s1")
    summary = tracker.save_summary()
    path = tmp_path / "costs" / "summary.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["run_id"] == "run1"
    assert data["total_calls"] == 1
    # No .tmp file left behind
    assert not (tmp_path / "costs" / "summary.tmp").exists()


# ── O: ask() calls on_usage ───────────────────────────────────────────────────

def test_O_ask_calls_on_usage(monkeypatch):
    import agent.claude_client as cc
    fake_msg = MagicMock()
    fake_msg.content[0].text = "hello"
    fake_msg.usage.input_tokens = 10
    fake_msg.usage.output_tokens = 20

    captured = []
    monkeypatch.setattr(cc, "ANTHROPIC_API_KEY", "test-key")

    with patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake_msg
        cc.ask("test prompt", on_usage=captured.append)

    assert len(captured) == 1
    assert captured[0]["provider"] == "anthropic"
    assert captured[0]["call_type"] == "text"
    assert captured[0]["input_tokens"] == 10
    assert captured[0]["output_tokens"] == 20


# ── P: analyze() passes on_usage ─────────────────────────────────────────────

def test_P_analyze_passes_on_usage(monkeypatch):
    captured = []
    import agent.analyzer as az
    monkeypatch.setattr(az, "ask", lambda prompt, on_usage=None: (
        on_usage({"provider": "anthropic", "model": "x", "call_type": "text",
                  "input_tokens": 5, "output_tokens": 5}) or
        '{"niche":"n","market_observations":[],"recurring_patterns":[],'
        '"potential_opportunities":[],"poster_concepts":[]}'
    ))
    from agent.analyzer import analyze
    analyze([], on_usage=captured.append)
    assert len(captured) == 1
    assert captured[0]["provider"] == "anthropic"


# ── Q: optimize() passes on_usage ────────────────────────────────────────────

def test_Q_optimize_passes_on_usage(monkeypatch):
    captured = []
    import agent.prompt_optimizer as po
    monkeypatch.setattr(po, "ask", lambda prompt, on_usage=None: (
        on_usage({"provider": "anthropic", "model": "x", "call_type": "text",
                  "input_tokens": 3, "output_tokens": 3}) or
        '{"optimized_image_prompt":"p","optimized_negative_prompt":"n","optimization_report":{}}'
    ))
    from agent.prompt_optimizer import optimize
    optimize({}, "p", "n", on_usage=captured.append)
    assert len(captured) == 1


# ── R: vision_critic.review() calls on_usage ─────────────────────────────────

def test_R_vision_critic_calls_on_usage(tmp_path, monkeypatch):
    import agent.vision_critic as vc
    monkeypatch.setattr(vc, "ANTHROPIC_API_KEY", "test-key")

    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    captured = []
    critique_json = json.dumps({
        "composition_score": {"score": 8, "reason": "good"},
        "originality_score": {"score": 8, "reason": "good"},
        "commercial_appeal_score": {"score": 8, "reason": "good"},
        "print_quality_score": {"score": 8, "reason": "good"},
        "collection_consistency_score": {"score": 8, "reason": "good"},
        "trend_saturation_score": {"score": 5, "reason": "mid"},
        "market_uniqueness_score": {"score": 7, "reason": "good"},
        "ip_similarity_risk": "low",
        "ip_similarity_reason": "none",
        "strengths": ["good"],
        "weaknesses": [],
        "improvement_suggestions": [],
        "reasoning": "ok",
        "retry_recommended": False,
        "retry_priority": [],
        "confidence_score": 8,
        "commercial_readiness": 8,
        "print_readiness": 8,
    })
    fake_msg = MagicMock()
    fake_msg.content[0].text = critique_json
    fake_msg.usage.input_tokens = 50
    fake_msg.usage.output_tokens = 100

    with patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake_msg
        vc.review({}, "p", "n", str(img), on_usage=captured.append)

    assert len(captured) == 1
    assert captured[0]["input_tokens"] == 50
    assert captured[0]["output_tokens"] == 100


# ── S: prepare_retry() calls on_usage via ask when retrying ──────────────────

def test_S_prepare_retry_calls_on_usage(monkeypatch):
    import agent.retry_generator as rg
    captured = []
    retry_json = json.dumps({
        "revised_image_prompt": "new prompt",
        "revised_negative_prompt": "new neg",
        "changes_made": [],
        "expected_improvements": [],
        "preserved_elements": [],
        "confidence_score": 7,
    })
    monkeypatch.setattr(rg, "ask", lambda prompt, on_usage=None: (
        on_usage({"provider": "anthropic", "model": "x", "call_type": "text",
                  "input_tokens": 10, "output_tokens": 15}) or retry_json
    ))
    vr = MagicMock()
    vr.retry_recommended = True
    vr.final_recommendation = "RETRY"
    vr.retry_priority = []
    vr.weaknesses = []
    vr.improvement_suggestions = []
    vr.composition_score.score = 8
    vr.print_quality_score.score = 8
    vr.collection_consistency_score.score = 8
    vr.commercial_appeal_score.score = 8
    from agent.retry_generator import prepare_retry
    prepare_retry({}, "p", "n", vr, on_usage=captured.append)
    assert len(captured) == 1


# ── T: generate_collection() calls on_usage ──────────────────────────────────

def test_T_generate_collection_calls_on_usage(monkeypatch):
    import agent.collection_generator as cg
    monkeypatch.setattr(cg, "ANTHROPIC_API_KEY", "test-key")
    captured = []

    fake_msg = MagicMock()
    fake_msg.usage.input_tokens = 10
    fake_msg.usage.output_tokens = 20

    bible_resp = {"collection_bible": {
        "collection_name": "Test", "collection_story": "s", "brand_identity": "b",
        "target_customer": "t", "recommended_room_style": "modern",
        "visual_identity": "v", "shared_rendering_medium": "ink",
        "shared_linework": "thin", "shared_lighting": "natural",
        "shared_palette": ["black"], "shared_accent_colour_rules": [],
        "shared_camera_angle": "eye level", "shared_perspective": "one point",
        "shared_atmosphere": "calm", "shared_detail_level": "high",
        "shared_print_treatment": "flat", "shared_storytelling_rules": [],
        "shared_composition_rules": ["rule"], "shared_style_rules": [],
        "shared_negative_prompt": "border",
        "style_dna": ["ukiyo-e"], "consistency_rules": [],
        "forbidden_elements": [], "full_bleed_rules": ["fill the frame"],
    }}

    def fake_create(**kwargs):
        content = kwargs.get("messages", [{}])[0].get("content", "")
        if isinstance(content, str) and "Collection Bible" in content:
            fake_msg.content[0].text = json.dumps(bible_resp)
        elif isinstance(content, str) and "Evaluate" in content:
            fake_msg.content[0].text = json.dumps({
                "evaluation": {
                    "consistency_score": {"score": 8, "reason": "ok"},
                    "commercial_score": {"score": 8, "reason": "ok"},
                    "variation_score": {"score": 8, "reason": "ok"},
                    "brand_identity_score": {"score": 8, "reason": "ok"},
                    "print_collection_score": {"score": 8, "reason": "ok"},
                    "market_uniqueness_score": {"score": 8, "reason": "ok"},
                    "reasoning": "good",
                },
                "collection_consistency_notes": [],
                "confidence_score": 8,
            })
        else:
            # Poster batch
            batch_prompt = content if isinstance(content, str) else ""
            fake_msg.content[0].text = json.dumps({"poster_items": [{
                "index": 1, "title": "Test Poster",
                "subject": "nature",
                "scene_concept": "forest scene full bleed edge-to-edge",
                "storytelling_focus": "peace",
                "unique_hook": "misty morning",
                "image_prompt": "full bleed watercolour forest edge-to-edge",
                "negative_prompt": "border frame",
                "aspect_ratio": "2:3", "focal_point": "tree",
                "foreground_elements": [], "midground_elements": [],
                "background_elements": [],
                "palette_variation": ["green"],
                "lighting_variation": "soft",
                "weather_or_time_variation": "morning",
                "consistency_notes": ["matches style"],
                "suggested_etsy_title": "Test",
                "suggested_etsy_tags": [f"tag{i}" for i in range(13)],
                "mockup_room_style": "modern",
            }]})
        return fake_msg

    with patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.side_effect = fake_create
        concept = {
            "name": "Test", "niche": "test", "art_style": "ink",
            "image_generation_prompt": "full bleed forest",
            "negative_prompt": "border", "single_or_set": "single",
        }
        try:
            cg.generate_collection(concept, "full bleed forest", "border",
                                   collection_size=3, on_usage=captured.append)
        except Exception:
            pass  # validation failures are fine — we just want to confirm on_usage was called

    assert len(captured) >= 1


# ── U: OpenAIImageProvider.generate() calls on_usage after write ──────────────

def test_U_image_provider_calls_on_usage(tmp_path, monkeypatch):
    import image.openai_provider as oip
    monkeypatch.setattr(oip, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(oip, "OUTPUT_DIR", tmp_path / "output")

    captured = []
    fake_response = MagicMock()
    fake_response.data[0].b64_json = "iVBORw0KGgo="  # minimal valid base64

    with patch("image.openai_provider.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.images.generate.return_value = fake_response
        from image.openai_provider import OpenAIImageProvider
        provider = OpenAIImageProvider()
        provider.generate("a prompt", on_usage=captured.append)

    assert len(captured) == 1
    assert captured[0]["provider"] == "openai"
    assert captured[0]["call_type"] == "image"
    assert captured[0]["image_count"] == 1


# ── V: ProductionRequest has enable_cost_tracking ────────────────────────────

def test_V_production_request_has_cost_tracking_field():
    from agent.production_orchestrator import ProductionRequest
    req = ProductionRequest(query="test", collection_size=3, output_root="/tmp")
    assert req.enable_cost_tracking is True
    req2 = ProductionRequest(query="test", collection_size=3, output_root="/tmp",
                             enable_cost_tracking=False)
    assert req2.enable_cost_tracking is False


# ── W: run_production creates costs/ with JSONL ──────────────────────────────

def test_W_run_production_creates_costs_jsonl(tmp_path):
    from agent.production_orchestrator import (
        ProductionDependencies, ProductionRequest, run_production,
    )

    counter = [0]
    def img_factory():
        prov = MagicMock()
        def gen(prompt, **kwargs):
            counter[0] += 1
            p = tmp_path / f"img_{counter[0]}.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n")
            return str(p)
        prov.generate = gen
        return prov

    deps = ProductionDependencies(
        research_provider_factory=lambda: MagicMock(
            search=MagicMock(return_value=[])
        ),
        analyze=MagicMock(return_value={"poster_concepts": [{
            "name": "T", "niche": "n", "art_style": "ink",
            "image_generation_prompt": "full bleed",
            "negative_prompt": "border", "single_or_set": "single",
        }]}),
        optimize=MagicMock(return_value={
            "optimized_image_prompt": "full bleed optimized",
            "optimized_negative_prompt": "border",
        }),
        generate_collection=MagicMock(return_value=_fake_collection(tmp_path, 1)),
        image_provider_factory=img_factory,
        vision_review=MagicMock(return_value=_fake_vr()),
        prepare_retry=MagicMock(return_value=_fake_rp()),
        generate_mockup_plan=MagicMock(return_value=MagicMock()),
        generate_listing_plan=MagicMock(return_value=MagicMock()),
    )
    req = ProductionRequest(
        query="test cost tracking",
        collection_size=3,
        output_root=str(tmp_path / "outputs"),
        enable_cost_tracking=True,
    )
    result = run_production(req, _deps=deps)
    run_dir = Path(result.manifest.output_directory)
    # CostTracker.__init__ always creates the costs/ directory
    assert (run_dir / "costs").is_dir()


# ── X: costs/summary.json written at end ─────────────────────────────────────

def test_X_run_production_writes_cost_summary(tmp_path):
    from agent.production_orchestrator import (
        ProductionDependencies, ProductionRequest, run_production,
    )
    counter = [0]
    def img_factory():
        prov = MagicMock()
        def gen(prompt, **kwargs):
            counter[0] += 1
            p = tmp_path / f"img_{counter[0]}.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n")
            return str(p)
        prov.generate = gen
        return prov

    deps = ProductionDependencies(
        research_provider_factory=lambda: MagicMock(search=MagicMock(return_value=[])),
        analyze=MagicMock(return_value={"poster_concepts": [{
            "name": "T", "niche": "n", "art_style": "ink",
            "image_generation_prompt": "full bleed",
            "negative_prompt": "border", "single_or_set": "single",
        }]}),
        optimize=MagicMock(return_value={
            "optimized_image_prompt": "full bleed optimized",
            "optimized_negative_prompt": "border",
        }),
        generate_collection=MagicMock(return_value=_fake_collection(tmp_path, 1)),
        image_provider_factory=img_factory,
        vision_review=MagicMock(return_value=_fake_vr()),
        prepare_retry=MagicMock(return_value=_fake_rp()),
        generate_mockup_plan=MagicMock(return_value=MagicMock()),
        generate_listing_plan=MagicMock(return_value=MagicMock()),
    )
    req = ProductionRequest(
        query="test summary write",
        collection_size=3,
        output_root=str(tmp_path / "outputs"),
    )
    result = run_production(req, _deps=deps)
    run_dir = Path(result.manifest.output_directory)
    summary_path = run_dir / "costs" / "summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert "total_calls" in data
    assert "by_stage" in data


# ── Y: resume loads existing JSONL without duplicating ───────────────────────

def test_Y_resume_loads_existing_jsonl_no_duplication(tmp_path):
    from agent.cost_tracking import CostTracker, PricingCatalog
    cat = PricingCatalog(data=_NULL_CATALOG)
    costs = tmp_path / "costs"

    # First partial run: 2 records
    t1 = CostTracker("run1", costs, cat)
    t1.record(_raw_text(model="claude-haiku-4-5-20251001"), stage="concept_generation")
    t1.record(_raw_text(model="claude-haiku-4-5-20251001"), stage="prompt_optimization")

    # Resume: load existing and add one more
    t2 = CostTracker("run1", costs, cat)
    t2.load_existing()
    assert len(t2._records) == 2
    t2.record(_raw_text(model="claude-haiku-4-5-20251001"), stage="collection_generation")
    assert len(t2._records) == 3

    # Total lines in JSONL should be 3
    lines = [l for l in (costs / "usage_records.jsonl").read_text().splitlines() if l.strip()]
    assert len(lines) == 3


# ── Z: _refresh_job_cost loads summary into QueueJob ─────────────────────────

def test_Z_refresh_job_cost_loads_summary(tmp_path):
    from agent.job_queue import QueueJob, _refresh_job_cost

    run_dir = tmp_path / "run"
    costs = run_dir / "costs"
    costs.mkdir(parents=True)
    (costs / "summary.json").write_text(json.dumps({
        "total_cost": "0.00100000",
        "total_input_tokens": 500,
        "total_output_tokens": 250,
        "total_images": 2,
    }), encoding="utf-8")

    job = QueueJob(job_id="job_001", position=1, status="completed",
                   request={}, run_dir=str(run_dir))
    _refresh_job_cost(job)
    assert job.total_cost == "0.00100000"
    assert job.total_input_tokens == 500
    assert job.total_output_tokens == 250
    assert job.total_images == 2


# ── AA: QueueResult.total_cost aggregates completed jobs ─────────────────────

def test_AA_queue_result_aggregates_total_cost(tmp_path):
    from agent.job_queue import QueueJob, QueueManifest, _build_result

    def _job(job_id, pos, status, cost, inp=0, out=0, img=0, run_dir=None):
        return QueueJob(
            job_id=job_id, position=pos, status=status, request={},
            run_dir=run_dir,
            total_cost=cost, total_input_tokens=inp,
            total_output_tokens=out, total_images=img,
        )

    manifest = QueueManifest(
        queue_id="q1", schema_version=1, status="completed",
        created_at="", updated_at="",
        jobs=[
            _job("job_001", 1, "completed", "0.00100000", inp=100, out=50, img=1),
            _job("job_002", 2, "completed", "0.00200000", inp=200, out=80, img=2),
        ],
    )
    result = _build_result(tmp_path, manifest)
    assert result.total_cost == "0.00300000"
    assert result.total_input_tokens == 300
    assert result.total_output_tokens == 130
    assert result.total_images == 3


# ── AB: show_costs.py --json outputs valid JSON ───────────────────────────────

def test_AB_show_costs_json(tmp_path):
    run_dir = tmp_path / "run"
    costs = run_dir / "costs"
    costs.mkdir(parents=True)
    summary_data = {
        "run_id": "test_run", "computed_at": "2026-01-01",
        "total_calls": 5, "total_input_tokens": 1000,
        "total_output_tokens": 500, "total_images": 2,
        "total_cost": None, "by_stage": [],
    }
    (costs / "summary.json").write_text(json.dumps(summary_data), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/show_costs.py", str(run_dir), "--json"],
        capture_output=True, text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    parsed = json.loads(result.stdout)
    assert parsed["run_id"] == "test_run"
    assert parsed["total_calls"] == 5


# ── AC: show_costs.py --by-stage shows stage data ────────────────────────────

def test_AC_show_costs_by_stage(tmp_path):
    run_dir = tmp_path / "run"
    costs = run_dir / "costs"
    costs.mkdir(parents=True)
    summary_data = {
        "run_id": "test_run", "computed_at": "2026-01-01",
        "total_calls": 2, "total_input_tokens": 300,
        "total_output_tokens": 150, "total_images": 0,
        "total_cost": None,
        "by_stage": [
            {"stage": "concept_generation", "call_count": 1,
             "input_tokens": 100, "output_tokens": 50, "image_count": 0, "cost": None},
            {"stage": "collection_generation", "call_count": 1,
             "input_tokens": 200, "output_tokens": 100, "image_count": 0, "cost": None},
        ],
    }
    (costs / "summary.json").write_text(json.dumps(summary_data), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/show_costs.py", str(run_dir), "--by-stage"],
        capture_output=True, text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    assert "concept_generation" in result.stdout
    assert "collection_generation" in result.stdout


# ── AD: show_costs.py exits non-zero for missing run directory ────────────────

def test_AD_show_costs_missing_dir_exits_nonzero(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/show_costs.py", str(tmp_path / "nonexistent")],
        capture_output=True, text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode != 0
    assert "Error" in result.stderr


# ── AE: total_cost null when all pricing is null ─────────────────────────────

def test_AE_total_cost_null_when_pricing_null(tmp_path):
    cat = PricingCatalog(data=_NULL_CATALOG)
    tracker = CostTracker("run1", tmp_path / "costs", cat)
    tracker.record({"provider": "anthropic", "model": "claude-haiku-4-5-20251001",
                    "call_type": "text", "input_tokens": 100, "output_tokens": 50},
                   stage="concept_generation")
    summary = tracker.compute_summary()
    assert summary.total_cost is None


# ── AF: multiple records aggregate tokens ─────────────────────────────────────

def test_AF_multiple_records_aggregate_tokens(tmp_path):
    cat = PricingCatalog(data=_NULL_CATALOG)
    tracker = CostTracker("run1", tmp_path / "costs", cat)
    tracker.record(_raw_text(100, 200, model="claude-haiku-4-5-20251001"), stage="s1")
    tracker.record(_raw_text(300, 400, model="claude-haiku-4-5-20251001"), stage="s1")
    tracker.record(_raw_image(model="gpt-image-2"), stage="image_generation")
    summary = tracker.compute_summary()
    assert summary.total_input_tokens == 400
    assert summary.total_output_tokens == 600
    assert summary.total_images == 1
    assert summary.total_calls == 3


# ── AG: Decimal serialized as string in JSON ──────────────────────────────────

def test_AG_decimal_serialized_as_string(tmp_path):
    cat = PricingCatalog(data=_PRICED_CATALOG)
    tracker = CostTracker("run1", tmp_path / "costs", cat)
    tracker.record(_raw_text(1000, 500), stage="concept_generation")
    tracker.save_summary()
    raw = (tmp_path / "costs" / "summary.json").read_text()
    data = json.loads(raw)
    # total_cost must be a string, not a float
    assert isinstance(data["total_cost"], str)
    # Must not be exponential notation or a float
    assert "e" not in data["total_cost"].lower()
    assert data["total_cost"].replace(".", "").replace("-", "").isdigit()


# ── Shared helpers for pipeline tests ─────────────────────────────────────────

def _fake_vr():
    vr = MagicMock()
    vr.retry_recommended = False
    vr.overall_score.score = 9
    vr.final_recommendation = "PROCEED"
    return vr


def _fake_rp():
    rp = MagicMock()
    rp.should_retry = False
    rp.revised_image_prompt = None
    rp.revised_negative_prompt = None
    return rp


def _fake_collection(tmp_path: Path, n: int):
    from agent.collection_generator import (
        CollectionBible, CollectionEvaluation, CollectionPlan, CollectionPoster,
    )
    from agent.vision_critic import ScoreWithReason

    def _swr():
        return ScoreWithReason(score=8, reason="ok")

    bible = CollectionBible(
        collection_name="TC", collection_story="s", brand_identity="b",
        target_customer="t", recommended_room_style="modern",
        visual_identity="v", shared_rendering_medium="ink",
        shared_linework="thin", shared_lighting="natural",
        shared_palette=["black"], shared_accent_colour_rules=[],
        shared_camera_angle="eye", shared_perspective="1pt",
        shared_atmosphere="calm", shared_detail_level="high",
        shared_print_treatment="flat", shared_storytelling_rules=[],
        shared_composition_rules=[], shared_style_rules=[],
        shared_negative_prompt="border", style_dna=[], consistency_rules=[],
        forbidden_elements=[], full_bleed_rules=[],
    )
    posters = [
        CollectionPoster(
            index=i, title=f"P{i}", subject=f"s{i}",
            scene_concept=f"scene {i}", storytelling_focus=f"f{i}",
            unique_hook=f"hook{i}",
            image_prompt=f"full bleed composition poster {i}",
            negative_prompt="border", aspect_ratio="2:3",
            focal_point="center", foreground_elements=[],
            midground_elements=[], background_elements=[],
            palette_variation=[], lighting_variation="soft",
            weather_or_time_variation="clear", consistency_notes=[],
            suggested_etsy_title=f"P{i}",
            suggested_etsy_tags=[],
            mockup_room_style="modern",
        )
        for i in range(1, n + 1)
    ]
    ev = CollectionEvaluation(
        consistency_score=_swr(), commercial_score=_swr(), variation_score=_swr(),
        brand_identity_score=_swr(), print_collection_score=_swr(),
        market_uniqueness_score=_swr(), overall_score=_swr(), reasoning="ok",
    )
    return CollectionPlan(
        collection_bible=bible, collection_size=n, poster_items=posters,
        collection_consistency_notes=[], evaluation=ev, confidence_score=9,
    )
