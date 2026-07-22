import dataclasses
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from agent.collection_generator import CollectionPlan, _MIN_SIZE, _MAX_SIZE
from agent.mockup_generator import MockupPlan
from agent.listing_generator import ListingPlan

_MAX_IMAGE_RETRIES_LIMIT = 3

_STAGE_NAMES = [
    "research",
    "concept_generation",
    "concept_selection",
    "prompt_optimization",
    "collection_generation",
    "image_generation",   # covers per-poster loop
    "vision_critique",    # runs within image_generation loop
    "retry_generation",   # runs within image_generation loop
    "mockup_generation",
    "listing_generation",
    "finalize",
]
_TOTAL_STAGES = len(_STAGE_NAMES)


# ── Public dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ProductionRequest:
    query: str
    collection_size: int
    output_root: str
    selected_concept_index: int | None = None
    max_image_retries: int = 1
    skip_mockups: bool = False
    skip_listing: bool = False


@dataclass
class StageRecord:
    stage_name: str
    status: str          # pending | running | completed | failed | skipped
    started_at: str
    completed_at: str | None = None
    output_file: str | None = None
    error_message: str | None = None


@dataclass
class ProductionManifest:
    production_id: str
    query: str
    collection_size: int
    status: str          # pending | running | completed | failed
    current_stage: str
    created_at: str
    updated_at: str
    output_directory: str
    stages: list[StageRecord]
    final_listing_file: str | None = None
    error_message: str | None = None


@dataclass
class ProductionResult:
    manifest: ProductionManifest
    research_result: dict | None = None
    selected_concept: dict | None = None
    collection_plan: CollectionPlan | None = None
    mockup_plan: MockupPlan | None = None
    listing_plan: ListingPlan | None = None


@dataclass
class AttemptRecord:
    attempt_number: int
    image_file: str
    vision_report_file: str
    retry_plan_file: str
    accepted: bool
    created_at: str


@dataclass
class ProductionDependencies:
    """
    Callable fields for each pipeline stage.
    Default instance uses real project functions (_default_deps).
    Tests inject fakes without touching the network or API keys.
    """
    research_provider_factory: Callable   # () -> provider with .search(query, limit)
    analyze: Callable                     # (list[dict]) -> dict
    optimize: Callable                    # (concept, prompt, neg) -> dict
    generate_collection: Callable         # (concept, prompt, neg, collection_size) -> CollectionPlan
    image_provider_factory: Callable      # () -> provider with .generate(prompt) -> str path
    vision_review: Callable               # (concept, prompt, neg, image_path) -> VisionReport
    prepare_retry: Callable               # (concept, prompt, neg, vr) -> RetryPlan
    generate_mockup_plan: Callable        # (collection_plan) -> MockupPlan
    generate_listing_plan: Callable       # (collection_plan, mockup_plan) -> ListingPlan


# ── Serialization ──────────────────────────────────────────────────────────────

def _to_serializable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_serializable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return str(obj)  # fallback: MagicMock in tests, unknown types in production


def _write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_serializable(obj), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Manifest lifecycle ─────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_index(name: str) -> int:
    return _STAGE_NAMES.index(name) + 1


def _get_stage(manifest: ProductionManifest, stage_name: str) -> StageRecord:
    for s in manifest.stages:
        if s.stage_name == stage_name:
            return s
    raise KeyError(f"Stage {stage_name!r} not in manifest.")


def _write_manifest(manifest: ProductionManifest, prod_dir: Path) -> None:
    _write_json(prod_dir / "manifest.json", manifest)


def _stage_start(
    manifest: ProductionManifest,
    stage_name: str,
    prod_dir: Path,
    print_header: bool = True,
) -> None:
    s = _get_stage(manifest, stage_name)
    s.status = "running"
    s.started_at = _now()
    if print_header:
        manifest.current_stage = stage_name
        n = _stage_index(stage_name)
        label = stage_name.replace("_", " ").title()
        print(f"[{n}/{_TOTAL_STAGES}] {label}...")
    manifest.updated_at = _now()
    _write_manifest(manifest, prod_dir)


def _stage_complete(
    manifest: ProductionManifest,
    stage_name: str,
    prod_dir: Path,
    output_file: str | None = None,
) -> None:
    s = _get_stage(manifest, stage_name)
    s.status = "completed"
    s.completed_at = _now()
    if output_file:
        s.output_file = output_file
    manifest.updated_at = _now()
    _write_manifest(manifest, prod_dir)


def _stage_skip(
    manifest: ProductionManifest,
    stage_name: str,
    prod_dir: Path,
) -> None:
    s = _get_stage(manifest, stage_name)
    s.status = "skipped"
    s.started_at = _now()
    s.completed_at = _now()
    manifest.updated_at = _now()
    n = _stage_index(stage_name)
    label = stage_name.replace("_", " ").title()
    print(f"[{n}/{_TOTAL_STAGES}] {label} — skipped")
    _write_manifest(manifest, prod_dir)


def _stage_fail(
    manifest: ProductionManifest,
    stage_name: str,
    prod_dir: Path,
    exc: Exception,
) -> None:
    s = _get_stage(manifest, stage_name)
    s.status = "failed"
    s.completed_at = _now()
    s.error_message = f"{type(exc).__name__}: {str(exc)[:300]}"
    manifest.status = "failed"
    manifest.error_message = s.error_message
    manifest.updated_at = _now()
    _write_manifest(manifest, prod_dir)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _production_id(query: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower().strip())[:30].rstrip("-")
    return f"{ts}_{slug}" if slug else ts


def _create_manifest(
    request: ProductionRequest,
    prod_id: str,
    prod_dir: Path,
) -> ProductionManifest:
    now = _now()
    stages = [
        StageRecord(stage_name=name, status="pending", started_at="")
        for name in _STAGE_NAMES
    ]
    return ProductionManifest(
        production_id=prod_id,
        query=request.query,
        collection_size=request.collection_size,
        status="pending",
        current_stage="",
        created_at=now,
        updated_at=now,
        output_directory=str(prod_dir),
        stages=stages,
    )


def _select_concept(concepts: list[dict], index: int | None) -> dict:
    if index is not None:
        if not (1 <= index <= len(concepts)):
            raise ValueError(
                f"selected_concept_index {index} is out of range (1–{len(concepts)})."
            )
        return concepts[index - 1]
    # Auto-select: first concept deterministically (no score field in analyzer output)
    return concepts[0]


def _log(stage_num: int, msg: str) -> None:
    print(f"[{stage_num}/{_TOTAL_STAGES}] {msg}")


def _validate_request(request: ProductionRequest) -> None:
    if not request.query.strip():
        raise ValueError("query must not be empty.")
    if not (_MIN_SIZE <= request.collection_size <= _MAX_SIZE):
        raise ValueError(
            f"collection_size must be {_MIN_SIZE}–{_MAX_SIZE}, got {request.collection_size}."
        )
    if not (0 <= request.max_image_retries <= _MAX_IMAGE_RETRIES_LIMIT):
        raise ValueError(
            f"max_image_retries must be 0–{_MAX_IMAGE_RETRIES_LIMIT}, "
            f"got {request.max_image_retries}."
        )
    if not request.output_root.strip():
        raise ValueError("output_root must not be empty.")
    if request.selected_concept_index is not None and request.selected_concept_index < 1:
        raise ValueError(
            f"selected_concept_index must be ≥ 1 when provided, "
            f"got {request.selected_concept_index}."
        )


def _default_deps() -> ProductionDependencies:
    from research.web_provider import WebResearchProvider
    from agent.analyzer import analyze
    from agent.prompt_optimizer import optimize
    from agent.collection_generator import generate_collection
    from image.openai_provider import OpenAIImageProvider
    from agent.vision_critic import review as vision_review
    from agent.retry_generator import prepare_retry
    from agent.mockup_generator import generate_mockup_plan
    from agent.listing_generator import generate_listing_plan

    return ProductionDependencies(
        research_provider_factory=WebResearchProvider,
        analyze=analyze,
        optimize=optimize,
        generate_collection=generate_collection,
        image_provider_factory=OpenAIImageProvider,
        vision_review=vision_review,
        prepare_retry=prepare_retry,
        generate_mockup_plan=generate_mockup_plan,
        generate_listing_plan=generate_listing_plan,
    )


# ── Main public function ───────────────────────────────────────────────────────

def run_production(
    request: ProductionRequest,
    _deps: ProductionDependencies | None = None,
) -> ProductionResult:
    """
    Run the full production pipeline for one collection.

    _deps is intentionally underscored — it is a test seam only.
    Production callers must never pass _deps.

    # resume integration point (Stage 10.2):
    #   Add resume: bool = False parameter here.
    #   When True, load existing manifest from output_directory before starting.
    #   Skip stages whose status is already "completed".
    """
    _validate_request(request)
    deps = _deps or _default_deps()

    prod_id = _production_id(request.query)
    prod_dir = Path(request.output_root) / prod_id
    prod_dir.mkdir(parents=True, exist_ok=True)

    manifest = _create_manifest(request, prod_id, prod_dir)
    manifest.status = "running"
    _write_manifest(manifest, prod_dir)
    _write_json(prod_dir / "request.json", request)

    result = ProductionResult(manifest=manifest)

    # ── Stage 1: Research ──────────────────────────────────────────────────────
    # resume integration point: check if "research" is already "completed" and skip
    products: list = []
    try:
        _stage_start(manifest, "research", prod_dir)
        provider = deps.research_provider_factory()
        products = provider.search(request.query, limit=20)
        product_dicts = [p.to_dict() if hasattr(p, "to_dict") else p for p in products]
        research_data = {"products": product_dicts}
        _write_json(prod_dir / "research/result.json", research_data)
        result.research_result = research_data
        _stage_complete(manifest, "research", prod_dir, "research/result.json")
    except Exception as exc:
        _stage_fail(manifest, "research", prod_dir, exc)
        raise

    # ── Stage 2: Concept Generation ────────────────────────────────────────────
    concepts: list[dict] = []
    try:
        _stage_start(manifest, "concept_generation", prod_dir)
        product_dicts = [p.to_dict() if hasattr(p, "to_dict") else p for p in products]
        analysis = deps.analyze(product_dicts)
        concepts = analysis.get("poster_concepts", [])
        if not concepts:
            raise ValueError("Analyzer returned no poster_concepts.")
        _write_json(prod_dir / "concepts/concepts.json", concepts)
        result.research_result = result.research_result or {}
        _stage_complete(manifest, "concept_generation", prod_dir, "concepts/concepts.json")
    except Exception as exc:
        _stage_fail(manifest, "concept_generation", prod_dir, exc)
        raise

    # ── Stage 3: Concept Selection ─────────────────────────────────────────────
    selected_concept: dict = {}
    try:
        _stage_start(manifest, "concept_selection", prod_dir)
        selected_concept = _select_concept(concepts, request.selected_concept_index)
        _write_json(prod_dir / "concepts/selected_concept.json", selected_concept)
        result.selected_concept = selected_concept
        _stage_complete(manifest, "concept_selection", prod_dir, "concepts/selected_concept.json")
    except Exception as exc:
        _stage_fail(manifest, "concept_selection", prod_dir, exc)
        raise

    # ── Stage 4: Prompt Optimization ───────────────────────────────────────────
    opt: dict = {}
    try:
        _stage_start(manifest, "prompt_optimization", prod_dir)
        opt = deps.optimize(
            selected_concept,
            selected_concept.get("image_generation_prompt", ""),
            selected_concept.get("negative_prompt", ""),
        )
        _write_json(prod_dir / "prompts/optimized_prompt.json", opt)
        _stage_complete(manifest, "prompt_optimization", prod_dir, "prompts/optimized_prompt.json")
    except Exception as exc:
        _stage_fail(manifest, "prompt_optimization", prod_dir, exc)
        raise

    # ── Stage 5: Collection Generation ────────────────────────────────────────
    collection_plan: CollectionPlan | None = None
    try:
        _stage_start(manifest, "collection_generation", prod_dir)
        collection_plan = deps.generate_collection(
            selected_concept,
            opt["optimized_image_prompt"],
            opt["optimized_negative_prompt"],
            collection_size=request.collection_size,
        )
        _write_json(prod_dir / "collection/collection_plan.json", collection_plan)
        result.collection_plan = collection_plan
        _stage_complete(manifest, "collection_generation", prod_dir, "collection/collection_plan.json")
    except Exception as exc:
        _stage_fail(manifest, "collection_generation", prod_dir, exc)
        raise

    # ── Stages 6/7/8: Image Generation + Vision Critique + Retry ──────────────
    # resume integration point: load existing attempts.json per poster and skip completed ones
    img_stage = _stage_index("image_generation")
    try:
        _stage_start(manifest, "image_generation", prod_dir)
        _stage_start(manifest, "vision_critique", prod_dir, print_header=False)
        _stage_start(manifest, "retry_generation", prod_dir, print_header=False)

        img_provider = deps.image_provider_factory()
        any_retry_ran = False
        n_posters = collection_plan.collection_size

        for i, poster in enumerate(collection_plan.poster_items, start=1):
            poster_dir = prod_dir / "images" / f"poster_{poster.index:02d}"
            poster_dir.mkdir(parents=True, exist_ok=True)
            poster_label = f"Poster {i}/{n_posters}"

            attempt_num = 0
            current_prompt = poster.image_prompt
            current_negative = poster.negative_prompt
            attempts: list[AttemptRecord] = []

            while True:
                attempt_num += 1
                _log(img_stage, f"{poster_label} — generating attempt {attempt_num}")

                raw_path = img_provider.generate(current_prompt)
                img_name = "original.png" if attempt_num == 1 else f"attempt_{attempt_num}.png"
                dest = poster_dir / img_name
                shutil.copy2(str(raw_path), str(dest))

                _log(img_stage, f"{poster_label} — running vision critique")
                vr = deps.vision_review(
                    selected_concept, current_prompt, current_negative, str(dest)
                )
                vr_file = poster_dir / f"vision_report_{attempt_num}.json"
                _write_json(vr_file, vr)
                _log(img_stage, f"{poster_label} — vision score {vr.overall_score.score}/10")

                rp = deps.prepare_retry(
                    selected_concept, current_prompt, current_negative, vr
                )
                rp_file = poster_dir / f"retry_plan_{attempt_num}.json"
                _write_json(rp_file, rp)

                attempt_accepted = not rp.should_retry
                attempts.append(AttemptRecord(
                    attempt_number=attempt_num,
                    image_file=str(dest.relative_to(prod_dir)),
                    vision_report_file=str(vr_file.relative_to(prod_dir)),
                    retry_plan_file=str(rp_file.relative_to(prod_dir)),
                    accepted=attempt_accepted,
                    created_at=_now(),
                ))

                if attempt_accepted or attempt_num > request.max_image_retries:
                    status_word = "accepted" if attempt_accepted else "max retries reached"
                    _log(img_stage, f"{poster_label} — {status_word}")
                    break

                any_retry_ran = True
                _log(img_stage, f"{poster_label} — retrying (attempt {attempt_num + 1})")
                current_prompt = rp.revised_image_prompt or current_prompt
                current_negative = rp.revised_negative_prompt or current_negative

            # Save final.png as copy of last attempt (never overwrites original.png)
            last_name = "original.png" if attempt_num == 1 else f"attempt_{attempt_num}.png"
            shutil.copy2(str(poster_dir / last_name), str(poster_dir / "final.png"))
            _write_json(poster_dir / "attempts.json", attempts)

        _stage_complete(manifest, "image_generation", prod_dir, "images/")
        _stage_complete(manifest, "vision_critique", prod_dir)
        if any_retry_ran:
            _stage_complete(manifest, "retry_generation", prod_dir)
        else:
            _stage_skip(manifest, "retry_generation", prod_dir)

    except Exception as exc:
        _stage_fail(manifest, "image_generation", prod_dir, exc)
        raise

    # ── Stage 9: Mockup Generation ─────────────────────────────────────────────
    mockup_plan: MockupPlan | None = None
    # Listing requires mockups; auto-skip listing if mockups are skipped
    effective_skip_listing = request.skip_listing or request.skip_mockups

    if request.skip_mockups:
        _stage_skip(manifest, "mockup_generation", prod_dir)
    else:
        try:
            _stage_start(manifest, "mockup_generation", prod_dir)
            mockup_plan = deps.generate_mockup_plan(collection_plan)
            _write_json(prod_dir / "mockups/mockup_plan.json", mockup_plan)
            result.mockup_plan = mockup_plan
            _stage_complete(manifest, "mockup_generation", prod_dir, "mockups/mockup_plan.json")
        except Exception as exc:
            _stage_fail(manifest, "mockup_generation", prod_dir, exc)
            raise

    # ── Stage 10: Listing Generation ───────────────────────────────────────────
    if effective_skip_listing:
        _stage_skip(manifest, "listing_generation", prod_dir)
    else:
        try:
            _stage_start(manifest, "listing_generation", prod_dir)
            listing_plan = deps.generate_listing_plan(collection_plan, mockup_plan)
            _write_json(prod_dir / "listing/listing_plan.json", listing_plan)
            result.listing_plan = listing_plan
            manifest.final_listing_file = "listing/listing_plan.json"
            _stage_complete(manifest, "listing_generation", prod_dir, "listing/listing_plan.json")
        except Exception as exc:
            _stage_fail(manifest, "listing_generation", prod_dir, exc)
            raise

    # ── Stage 11: Finalize ─────────────────────────────────────────────────────
    _stage_start(manifest, "finalize", prod_dir)
    manifest.status = "completed"
    _stage_complete(manifest, "finalize", prod_dir)
    print(f"\n[OK] Production complete → {prod_dir}")

    return result
