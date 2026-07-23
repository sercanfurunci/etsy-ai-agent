import dataclasses
import json
import re
import shutil
import types as _types
import typing
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
    enable_cost_tracking: bool = True


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
    # Stage 10.2 resume fields (defaulted for backward compat)
    resume_count: int = 0
    last_resumed_at: str | None = None
    # Stage 10.4 cost fields (defaulted for backward compat)
    total_cost: str | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_images: int | None = None


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
    cost_tracker: Any = None              # CostTracker | None — set by run_production when enabled
    progress_tracker: Any = None          # ProgressTracker | None — set by run_production when callback provided


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


def _write_json_atomic(path: Path | str, obj: Any) -> None:
    """Write JSON via a temp file then rename — atomic on POSIX."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(_to_serializable(obj), indent=2, ensure_ascii=False)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ── Generic dataclass deserializer ────────────────────────────────────────────

def _from_dict(cls: type, data: Any) -> Any:
    """Reconstruct a dataclass from its _to_serializable() JSON output."""
    if data is None:
        return None
    if not (dataclasses.is_dataclass(cls) and isinstance(cls, type)):
        return data
    if not isinstance(data, dict):
        return data

    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {}

    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(hints.get(f.name), data[f.name])
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            kwargs[f.name] = f.default_factory()
        else:
            kwargs[f.name] = None
    return cls(**kwargs)


def _coerce(hint: Any, value: Any) -> Any:
    """Recursively coerce a JSON value to match its type hint."""
    if value is None:
        return None

    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if origin is list and args:
        elem = args[0]
        if isinstance(value, list) and dataclasses.is_dataclass(elem) and isinstance(elem, type):
            return [_from_dict(elem, x) for x in value]
        return value

    # Union[X, Y] or X | Y (Python 3.10+)
    is_union = origin is typing.Union or (
        hasattr(_types, "UnionType") and isinstance(hint, _types.UnionType)
    )
    if is_union and args:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _coerce(non_none[0], value)
        return value

    if hint and dataclasses.is_dataclass(hint) and isinstance(hint, type):
        return _from_dict(hint, value)

    return value


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
    _write_json_atomic(prod_dir / "manifest.json", manifest)


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


# ── Resume helpers ─────────────────────────────────────────────────────────────

def _invalidate_from(manifest: ProductionManifest, stage_name: str) -> None:
    """Reset stage_name and all downstream non-skipped stages to pending."""
    start_idx = _STAGE_NAMES.index(stage_name)
    for s in manifest.stages:
        if _STAGE_NAMES.index(s.stage_name) >= start_idx and s.status != "skipped":
            s.status = "pending"
            s.started_at = ""
            s.completed_at = None
            s.output_file = None
            s.error_message = None
    manifest.error_message = None


def _load_manifest(prod_dir: Path) -> ProductionManifest:
    f = prod_dir / "manifest.json"
    if not f.exists():
        raise FileNotFoundError(f"manifest.json not found in {prod_dir}")
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"manifest.json is not valid JSON: {e}") from e

    stages = [
        StageRecord(
            stage_name=s["stage_name"],
            status=s["status"],
            started_at=s.get("started_at", ""),
            completed_at=s.get("completed_at"),
            output_file=s.get("output_file"),
            error_message=s.get("error_message"),
        )
        for s in data["stages"]
    ]
    return ProductionManifest(
        production_id=data["production_id"],
        query=data["query"],
        collection_size=data["collection_size"],
        status=data["status"],
        current_stage=data.get("current_stage", ""),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        output_directory=data["output_directory"],
        stages=stages,
        final_listing_file=data.get("final_listing_file"),
        error_message=data.get("error_message"),
        resume_count=data.get("resume_count", 0),
        last_resumed_at=data.get("last_resumed_at"),
        total_cost=data.get("total_cost"),
        total_input_tokens=data.get("total_input_tokens"),
        total_output_tokens=data.get("total_output_tokens"),
        total_images=data.get("total_images"),
    )


def _load_request(prod_dir: Path) -> ProductionRequest:
    f = prod_dir / "request.json"
    if not f.exists():
        raise FileNotFoundError(f"request.json not found in {prod_dir}")
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"request.json is not valid JSON: {e}") from e
    return ProductionRequest(
        query=data["query"],
        collection_size=data["collection_size"],
        output_root=data["output_root"],
        selected_concept_index=data.get("selected_concept_index"),
        max_image_retries=data.get("max_image_retries", 1),
        skip_mockups=data.get("skip_mockups", False),
        skip_listing=data.get("skip_listing", False),
        enable_cost_tracking=data.get("enable_cost_tracking", True),
    )


# ── Stage output loaders ───────────────────────────────────────────────────────

def _load_research_result(prod_dir: Path) -> dict:
    return json.loads((prod_dir / "research/result.json").read_text(encoding="utf-8"))


def _load_concepts(prod_dir: Path) -> list[dict]:
    return json.loads((prod_dir / "concepts/concepts.json").read_text(encoding="utf-8"))


def _load_selected_concept(prod_dir: Path) -> dict:
    return json.loads((prod_dir / "concepts/selected_concept.json").read_text(encoding="utf-8"))


def _load_optimized_prompt(prod_dir: Path) -> dict:
    return json.loads((prod_dir / "prompts/optimized_prompt.json").read_text(encoding="utf-8"))


def _load_collection_plan(prod_dir: Path) -> CollectionPlan:
    data = json.loads((prod_dir / "collection/collection_plan.json").read_text(encoding="utf-8"))
    return _from_dict(CollectionPlan, data)


def _load_mockup_plan(prod_dir: Path) -> MockupPlan:
    data = json.loads((prod_dir / "mockups/mockup_plan.json").read_text(encoding="utf-8"))
    return _from_dict(MockupPlan, data)


def _load_listing_plan(prod_dir: Path) -> ListingPlan:
    data = json.loads((prod_dir / "listing/listing_plan.json").read_text(encoding="utf-8"))
    return _from_dict(ListingPlan, data)


def _load_poster_attempts(poster_dir: Path) -> list[AttemptRecord] | None:
    """Return parsed attempts or None if missing/corrupt."""
    f = poster_dir / "attempts.json"
    if not f.exists():
        return None
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
        return [AttemptRecord(**a) for a in raw]
    except Exception:
        return None


# ── Completed-stage validators ─────────────────────────────────────────────────

def _require_json(path: Path, required_keys: list[str] | None = None) -> dict | list:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    if required_keys:
        if not isinstance(data, dict):
            raise ValueError(f"{path.name}: expected JSON object, got {type(data).__name__}")
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"missing keys: {missing}")
    return data


def _validate_completed_stages(manifest: ProductionManifest, prod_dir: Path) -> None:
    """Validate completed stage output files. Invalidates downstream on failure."""
    checks: list[tuple[str, Any]] = [
        ("research", lambda: _require_json(prod_dir / "research/result.json", ["products"])),
        ("concept_generation", lambda: _require_json(prod_dir / "concepts/concepts.json")),
        ("concept_selection", lambda: _require_json(prod_dir / "concepts/selected_concept.json")),
        ("prompt_optimization", lambda: _require_json(
            prod_dir / "prompts/optimized_prompt.json",
            ["optimized_image_prompt", "optimized_negative_prompt"],
        )),
        ("collection_generation", lambda: _require_json(
            prod_dir / "collection/collection_plan.json",
            ["collection_bible", "poster_items"],
        )),
        ("image_generation", lambda: _validate_image_outputs(prod_dir)),
    ]
    for stage_name, check in checks:
        stage = _get_stage(manifest, stage_name)
        if stage.status != "completed":
            continue
        try:
            check()
        except Exception as e:
            print(f"[resume] Stage {stage_name!r} output invalid ({e}) — rerunning from here")
            _invalidate_from(manifest, stage_name)


def _validate_image_outputs(prod_dir: Path) -> None:
    images_dir = prod_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError("images/ directory missing")
    if not any(images_dir.iterdir()):
        raise FileNotFoundError("images/ directory is empty")


# ── Per-poster resume state ────────────────────────────────────────────────────

def _get_poster_resume_state(
    poster_dir: Path,
    prod_dir: Path,
    poster: Any,
    max_image_retries: int,
) -> tuple[list[AttemptRecord], int | None, str | None, str | None]:
    """
    Returns (existing_attempts, next_attempt_num, prompt, negative).
    next_attempt_num is None when the poster is complete (skip it).
    """
    final_png = poster_dir / "final.png"

    # Case 1: attempts.json exists
    existing = _load_poster_attempts(poster_dir)
    if existing is not None:
        if final_png.exists():
            return existing, None, None, None  # complete
        # attempts.json exists but no final.png → corrupted, rerun
        return [], 1, poster.image_prompt, poster.negative_prompt

    # Case 2: no attempts.json — try to reconstruct from individual files
    reconstructed: list[AttemptRecord] = []
    last_rp_data: dict = {}

    for attempt_num in range(1, max_image_retries + 2):
        img_name = "original.png" if attempt_num == 1 else f"attempt_{attempt_num}.png"
        img_file = poster_dir / img_name
        vr_file = poster_dir / f"vision_report_{attempt_num}.json"
        rp_file = poster_dir / f"retry_plan_{attempt_num}.json"

        if not img_file.exists():
            break  # no more attempts on disk

        if not vr_file.exists() or not rp_file.exists():
            break  # attempt incomplete — restart from here

        try:
            last_rp_data = json.loads(rp_file.read_text(encoding="utf-8"))
            if not isinstance(last_rp_data, dict):
                break  # unexpected format — restart from this attempt
        except Exception:
            break  # corrupt retry plan

        accepted = not last_rp_data.get("should_retry", True)
        reconstructed.append(AttemptRecord(
            attempt_number=attempt_num,
            image_file=str(img_file.relative_to(prod_dir)),
            vision_report_file=str(vr_file.relative_to(prod_dir)),
            retry_plan_file=str(rp_file.relative_to(prod_dir)),
            accepted=accepted,
            created_at="",
        ))

        if accepted or attempt_num > max_image_retries:
            # This attempt was terminal — finalise and mark done
            _write_json_atomic(poster_dir / "attempts.json", reconstructed)
            shutil.copy2(str(img_file), str(poster_dir / "final.png"))
            return reconstructed, None, None, None

    if reconstructed:
        next_prompt = last_rp_data.get("revised_image_prompt") or poster.image_prompt
        next_negative = last_rp_data.get("revised_negative_prompt") or poster.negative_prompt
        return reconstructed, len(reconstructed) + 1, next_prompt, next_negative

    return [], 1, poster.image_prompt, poster.negative_prompt


# ── Core pipeline (shared by run_production and resume_production) ─────────────

def _run_pipeline(
    request: ProductionRequest,
    manifest: ProductionManifest,
    prod_dir: Path,
    deps: ProductionDependencies,
) -> ProductionResult:
    """
    Execute or resume the production pipeline.
    For each stage, checks manifest status before running:
      completed → load outputs from disk, skip execution
      skipped   → skip execution
      pending   → execute normally
    """
    result = ProductionResult(manifest=manifest)

    # ── Cost tracking helper ───────────────────────────────────────────────────
    def _ou(stage: str) -> Callable | None:
        if deps.cost_tracker is None:
            return None
        def cb(raw: dict) -> None:
            deps.cost_tracker.record(raw, stage)
        return cb

    # ── Progress tracking helper ───────────────────────────────────────────────
    def _ep(event_type: str, **kwargs) -> None:
        if deps.progress_tracker is None:
            return
        try:
            deps.progress_tracker.emit(event_type, **kwargs)
        except Exception:
            pass

    n_posters_total = 0  # set when collection_plan is known

    # ── Stage 1: Research ──────────────────────────────────────────────────────
    product_dicts: list[dict] = []
    s1 = _get_stage(manifest, "research")
    if s1.status == "completed":
        research_data = _load_research_result(prod_dir)
        product_dicts = research_data.get("products", [])
        result.research_result = research_data
    else:
        try:
            _stage_start(manifest, "research", prod_dir)
            _ep("stage_started", stage_name="research")
            provider = deps.research_provider_factory()
            products = provider.search(request.query, limit=20)
            product_dicts = [p.to_dict() if hasattr(p, "to_dict") else p for p in products]
            research_data = {"products": product_dicts}
            _write_json(prod_dir / "research/result.json", research_data)
            result.research_result = research_data
            _stage_complete(manifest, "research", prod_dir, "research/result.json")
            _ep("stage_completed", stage_name="research")
        except Exception as exc:
            _stage_fail(manifest, "research", prod_dir, exc)
            _ep("stage_failed", stage_name="research")
            raise

    # ── Stage 2: Concept Generation ────────────────────────────────────────────
    concepts: list[dict] = []
    s2 = _get_stage(manifest, "concept_generation")
    if s2.status == "completed":
        concepts = _load_concepts(prod_dir)
    else:
        try:
            _stage_start(manifest, "concept_generation", prod_dir)
            _ep("stage_started", stage_name="concept_generation")
            analysis = deps.analyze(product_dicts, user_request=request.query, on_usage=_ou("concept_generation"))
            concepts = analysis.get("poster_concepts", [])
            if not concepts:
                raise ValueError("Analyzer returned no poster_concepts.")
            _write_json(prod_dir / "concepts/concepts.json", concepts)
            result.research_result = result.research_result or {}
            _stage_complete(manifest, "concept_generation", prod_dir, "concepts/concepts.json")
            _ep("stage_completed", stage_name="concept_generation")
        except Exception as exc:
            _stage_fail(manifest, "concept_generation", prod_dir, exc)
            _ep("stage_failed", stage_name="concept_generation")
            raise

    # ── Stage 3: Concept Selection ─────────────────────────────────────────────
    selected_concept: dict = {}
    s3 = _get_stage(manifest, "concept_selection")
    if s3.status == "completed":
        selected_concept = _load_selected_concept(prod_dir)
        result.selected_concept = selected_concept
    else:
        try:
            _stage_start(manifest, "concept_selection", prod_dir)
            _ep("stage_started", stage_name="concept_selection")
            selected_concept = _select_concept(concepts, request.selected_concept_index)
            _write_json(prod_dir / "concepts/selected_concept.json", selected_concept)
            result.selected_concept = selected_concept
            _stage_complete(manifest, "concept_selection", prod_dir, "concepts/selected_concept.json")
            _ep("stage_completed", stage_name="concept_selection")
        except Exception as exc:
            _stage_fail(manifest, "concept_selection", prod_dir, exc)
            _ep("stage_failed", stage_name="concept_selection")
            raise

    # ── Stage 4: Prompt Optimization ───────────────────────────────────────────
    opt: dict = {}
    s4 = _get_stage(manifest, "prompt_optimization")
    if s4.status == "completed":
        opt = _load_optimized_prompt(prod_dir)
    else:
        try:
            _stage_start(manifest, "prompt_optimization", prod_dir)
            _ep("stage_started", stage_name="prompt_optimization")
            opt = deps.optimize(
                selected_concept,
                selected_concept.get("image_generation_prompt", ""),
                selected_concept.get("negative_prompt", ""),
                on_usage=_ou("prompt_optimization"),
            )
            _write_json(prod_dir / "prompts/optimized_prompt.json", opt)
            _stage_complete(manifest, "prompt_optimization", prod_dir, "prompts/optimized_prompt.json")
            _ep("stage_completed", stage_name="prompt_optimization")
        except Exception as exc:
            _stage_fail(manifest, "prompt_optimization", prod_dir, exc)
            _ep("stage_failed", stage_name="prompt_optimization")
            raise

    # ── Stage 5: Collection Generation ────────────────────────────────────────
    collection_plan: CollectionPlan | None = None
    s5 = _get_stage(manifest, "collection_generation")
    if s5.status == "completed":
        collection_plan = _load_collection_plan(prod_dir)
        result.collection_plan = collection_plan
    else:
        try:
            _stage_start(manifest, "collection_generation", prod_dir)
            _ep("stage_started", stage_name="collection_generation")
            collection_plan = deps.generate_collection(
                selected_concept,
                opt["optimized_image_prompt"],
                opt["optimized_negative_prompt"],
                collection_size=request.collection_size,
                on_usage=_ou("collection_generation"),
            )
            _write_json(prod_dir / "collection/collection_plan.json", collection_plan)
            result.collection_plan = collection_plan
            _stage_complete(manifest, "collection_generation", prod_dir, "collection/collection_plan.json")
            _ep("stage_completed", stage_name="collection_generation")
        except Exception as exc:
            _stage_fail(manifest, "collection_generation", prod_dir, exc)
            _ep("stage_failed", stage_name="collection_generation")
            raise

    # ── Stages 6/7/8: Image Generation + Vision Critique + Retry ──────────────
    img_stage = _stage_index("image_generation")
    s6 = _get_stage(manifest, "image_generation")
    if s6.status != "completed":
        try:
            _stage_start(manifest, "image_generation", prod_dir)
            _ep("stage_started", stage_name="image_generation")
            _stage_start(manifest, "vision_critique", prod_dir, print_header=False)
            _stage_start(manifest, "retry_generation", prod_dir, print_header=False)

            img_provider = None  # lazy init — avoid call if all posters already done
            any_retry_ran = False
            n_posters = collection_plan.collection_size
            n_posters_total = n_posters

            for i, poster in enumerate(collection_plan.poster_items, start=1):
                poster_dir = prod_dir / "images" / f"poster_{poster.index:02d}"
                poster_dir.mkdir(parents=True, exist_ok=True)
                poster_label = f"Poster {i}/{n_posters}"

                existing, start_num, resume_prompt, resume_negative = \
                    _get_poster_resume_state(
                        poster_dir, prod_dir, poster, request.max_image_retries
                    )

                if start_num is None:
                    # Poster already complete
                    if len(existing) > 1:
                        any_retry_ran = True
                    _log(img_stage, f"{poster_label} — already complete, skipping")
                    continue

                if img_provider is None:
                    img_provider = deps.image_provider_factory()

                _ep("poster_started", stage_name="image_generation",
                    poster_index=poster.index, poster_total=n_posters)

                attempts = list(existing)
                attempt_num = start_num - 1
                current_prompt = resume_prompt
                current_negative = resume_negative

                while True:
                    attempt_num += 1
                    _log(img_stage, f"{poster_label} — generating attempt {attempt_num}")
                    _ep("image_attempt_started", stage_name="image_generation",
                        poster_index=poster.index, poster_total=n_posters,
                        attempt_number=attempt_num)

                    raw_path = img_provider.generate(current_prompt, on_usage=_ou("image_generation"))
                    img_name = "original.png" if attempt_num == 1 else f"attempt_{attempt_num}.png"
                    dest = poster_dir / img_name
                    shutil.copy2(str(raw_path), str(dest))

                    _ep("image_attempt_completed", stage_name="image_generation",
                        poster_index=poster.index, poster_total=n_posters,
                        attempt_number=attempt_num)

                    _log(img_stage, f"{poster_label} — running vision critique")
                    vr = deps.vision_review(
                        selected_concept, current_prompt, current_negative, str(dest),
                        on_usage=_ou("vision_critique"),
                    )
                    vr_file = poster_dir / f"vision_report_{attempt_num}.json"
                    _write_json(vr_file, vr)
                    _log(img_stage, f"{poster_label} — vision score {vr.overall_score.score}/10")

                    rp = deps.prepare_retry(
                        selected_concept, current_prompt, current_negative, vr,
                        on_usage=_ou("retry_generation"),
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

                last_name = "original.png" if attempt_num == 1 else f"attempt_{attempt_num}.png"
                shutil.copy2(str(poster_dir / last_name), str(poster_dir / "final.png"))
                _write_json_atomic(poster_dir / "attempts.json", attempts)
                _ep("poster_completed", stage_name="image_generation",
                    poster_index=poster.index, poster_total=n_posters)

            _stage_complete(manifest, "image_generation", prod_dir, "images/")
            _stage_complete(manifest, "vision_critique", prod_dir)
            if any_retry_ran:
                _stage_complete(manifest, "retry_generation", prod_dir)
            else:
                _stage_skip(manifest, "retry_generation", prod_dir)
            _ep("stage_completed", stage_name="image_generation")

        except Exception as exc:
            _stage_fail(manifest, "image_generation", prod_dir, exc)
            _ep("stage_failed", stage_name="image_generation")
            raise

    # ── Stage 9: Mockup Generation ─────────────────────────────────────────────
    mockup_plan: MockupPlan | None = None
    effective_skip_listing = request.skip_listing or request.skip_mockups

    s9 = _get_stage(manifest, "mockup_generation")
    if s9.status == "completed":
        mockup_plan = _load_mockup_plan(prod_dir)
        result.mockup_plan = mockup_plan
    elif s9.status == "skipped" or request.skip_mockups:
        if s9.status != "skipped":
            _stage_skip(manifest, "mockup_generation", prod_dir)
            _ep("stage_skipped", stage_name="mockup_generation")
    else:
        try:
            _stage_start(manifest, "mockup_generation", prod_dir)
            _ep("stage_started", stage_name="mockup_generation")
            mockup_plan = deps.generate_mockup_plan(collection_plan, on_usage=_ou("mockup_generation"))
            _write_json(prod_dir / "mockups/mockup_plan.json", mockup_plan)
            result.mockup_plan = mockup_plan
            _stage_complete(manifest, "mockup_generation", prod_dir, "mockups/mockup_plan.json")
            _ep("stage_completed", stage_name="mockup_generation")
        except Exception as exc:
            _stage_fail(manifest, "mockup_generation", prod_dir, exc)
            _ep("stage_failed", stage_name="mockup_generation")
            raise

    # ── Stage 10: Listing Generation ───────────────────────────────────────────
    s10 = _get_stage(manifest, "listing_generation")
    if s10.status == "completed":
        result.listing_plan = _load_listing_plan(prod_dir)
    elif s10.status == "skipped" or effective_skip_listing:
        if s10.status != "skipped":
            _stage_skip(manifest, "listing_generation", prod_dir)
            _ep("stage_skipped", stage_name="listing_generation")
    else:
        try:
            _stage_start(manifest, "listing_generation", prod_dir)
            _ep("stage_started", stage_name="listing_generation")
            listing_plan = deps.generate_listing_plan(collection_plan, mockup_plan, on_usage=_ou("listing_generation"))
            _write_json(prod_dir / "listing/listing_plan.json", listing_plan)
            result.listing_plan = listing_plan
            manifest.final_listing_file = "listing/listing_plan.json"
            _stage_complete(manifest, "listing_generation", prod_dir, "listing/listing_plan.json")
            _ep("stage_completed", stage_name="listing_generation")
        except Exception as exc:
            _stage_fail(manifest, "listing_generation", prod_dir, exc)
            _ep("stage_failed", stage_name="listing_generation")
            raise

    # ── Stage 11: Finalize ─────────────────────────────────────────────────────
    s11 = _get_stage(manifest, "finalize")
    if s11.status != "completed":
        _stage_start(manifest, "finalize", prod_dir)
        _ep("stage_started", stage_name="finalize")
        manifest.status = "completed"
        _stage_complete(manifest, "finalize", prod_dir)
        _ep("stage_completed", stage_name="finalize")
        _ep("run_completed")
    else:
        if manifest.status != "completed":
            manifest.status = "completed"
            manifest.updated_at = _now()
            _write_manifest(manifest, prod_dir)
    print(f"\n[OK] Production complete → {prod_dir}")

    return result


# ── Fast-path result builder (no API calls) ────────────────────────────────────

def _build_result_from_manifest(manifest: ProductionManifest, prod_dir: Path) -> ProductionResult:
    """Build ProductionResult for an already-completed run without any API calls."""
    result = ProductionResult(manifest=manifest)

    rr = prod_dir / "research/result.json"
    if rr.exists():
        result.research_result = json.loads(rr.read_text(encoding="utf-8"))

    sc = prod_dir / "concepts/selected_concept.json"
    if sc.exists():
        result.selected_concept = json.loads(sc.read_text(encoding="utf-8"))

    cp = prod_dir / "collection/collection_plan.json"
    if cp.exists():
        try:
            result.collection_plan = _load_collection_plan(prod_dir)
        except Exception:
            pass

    mp = prod_dir / "mockups/mockup_plan.json"
    if mp.exists():
        try:
            result.mockup_plan = _load_mockup_plan(prod_dir)
        except Exception:
            pass

    lp = prod_dir / "listing/listing_plan.json"
    if lp.exists():
        try:
            result.listing_plan = _load_listing_plan(prod_dir)
        except Exception:
            pass

    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def run_production(
    request: ProductionRequest,
    _deps: ProductionDependencies | None = None,
    _on_run_dir: Callable[[Path], None] | None = None,
    _progress_callback: Callable | None = None,
) -> ProductionResult:
    """
    Run the full production pipeline for one collection.

    _deps is a test seam; production callers must not pass it.
    _on_run_dir is called with the prod_dir Path immediately after it is created,
    before any API calls — used by the job queue to persist run_dir early.
    _progress_callback receives ProgressEvent objects; errors in it are swallowed.
    """
    _validate_request(request)
    deps = _deps or _default_deps()

    prod_id = _production_id(request.query)
    prod_dir = Path(request.output_root) / prod_id
    prod_dir.mkdir(parents=True, exist_ok=True)

    if _on_run_dir is not None:
        _on_run_dir(prod_dir)

    if request.enable_cost_tracking and deps.cost_tracker is None:
        from agent.cost_tracking import CostTracker
        deps = dataclasses.replace(deps, cost_tracker=CostTracker(
            run_id=prod_id, costs_dir=prod_dir / "costs"
        ))

    if deps.progress_tracker is None:
        from agent.progress_tracking import ProgressTracker
        deps = dataclasses.replace(deps, progress_tracker=ProgressTracker(
            run_id=prod_id,
            progress_dir=prod_dir / "progress",
            callback=_progress_callback,
        ))

    manifest = _create_manifest(request, prod_id, prod_dir)
    manifest.status = "running"
    _write_manifest(manifest, prod_dir)
    _write_json(prod_dir / "request.json", request)

    deps.progress_tracker.emit("run_started", message=f"Production {prod_id} started")

    try:
        result = _run_pipeline(request, manifest, prod_dir, deps)
    except Exception:
        deps.progress_tracker.emit("run_failed", message="Production failed")
        raise
    finally:
        if deps.cost_tracker is not None:
            try:
                summary = deps.cost_tracker.save_summary()
                manifest.total_cost = summary.total_cost
                manifest.total_input_tokens = summary.total_input_tokens
                manifest.total_output_tokens = summary.total_output_tokens
                manifest.total_images = summary.total_images
                _write_manifest(manifest, prod_dir)
            except Exception:
                pass
    return result


def resume_production(
    run_dir: str | Path,
    _deps: ProductionDependencies | None = None,
    _progress_callback: Callable | None = None,
) -> ProductionResult:
    """
    Resume a failed or interrupted production run from its existing output directory.

    Completed stages are not re-executed; their outputs are loaded from disk.
    Failed or interrupted stages are reset and rerun from the point of failure.

    Example:
        result = resume_production("outputs/20260722_173000_cozy-anime-wall-art")

    _deps is a test seam; production callers must not pass it.
    """
    prod_dir = Path(run_dir)

    request = _load_request(prod_dir)
    manifest = _load_manifest(prod_dir)

    # Validate completed stages; invalidate any with missing/corrupt outputs
    _validate_completed_stages(manifest, prod_dir)

    # Fast path: if all stages still valid/complete, no work needed — no writes, no API calls
    if all(s.status in ("completed", "skipped") for s in manifest.stages):
        return _build_result_from_manifest(manifest, prod_dir)

    # Reset interrupted stages to pending so the pipeline can rerun them
    for s in manifest.stages:
        if s.status in ("running", "failed"):
            s.status = "pending"
            s.started_at = ""
            s.completed_at = None
            s.error_message = None

    # Update resume metadata
    manifest.resume_count += 1
    manifest.last_resumed_at = _now()
    manifest.status = "running"
    manifest.error_message = None
    _write_manifest(manifest, prod_dir)

    deps = _deps or _default_deps()

    if request.enable_cost_tracking and deps.cost_tracker is None:
        from agent.cost_tracking import CostTracker
        tracker = CostTracker(run_id=manifest.production_id, costs_dir=prod_dir / "costs")
        tracker.load_existing()
        deps = dataclasses.replace(deps, cost_tracker=tracker)

    if deps.progress_tracker is None:
        from agent.progress_tracking import ProgressTracker
        initial_seq = ProgressTracker.load_sequence_from_dir(prod_dir / "progress")
        deps = dataclasses.replace(deps, progress_tracker=ProgressTracker(
            run_id=manifest.production_id,
            progress_dir=prod_dir / "progress",
            callback=_progress_callback,
            initial_sequence=initial_seq,
        ))

    deps.progress_tracker.emit("run_resumed", message=f"Production {manifest.production_id} resumed")

    try:
        result = _run_pipeline(request, manifest, prod_dir, deps)
    except Exception:
        deps.progress_tracker.emit("run_failed", message="Production failed on resume")
        raise
    finally:
        if deps.cost_tracker is not None:
            try:
                summary = deps.cost_tracker.save_summary()
                manifest.total_cost = summary.total_cost
                manifest.total_input_tokens = summary.total_input_tokens
                manifest.total_output_tokens = summary.total_output_tokens
                manifest.total_images = summary.total_images
                _write_manifest(manifest, prod_dir)
            except Exception:
                pass
    return result
