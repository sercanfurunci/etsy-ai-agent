"""Cost tracking for the etsy-ai-agent pipeline."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

CATALOG_SCHEMA_VERSION = 1


class CatalogSchemaError(ValueError):
    """Raised when the pricing catalog has an unsupported schema version."""


@dataclass
class UsageRecord:
    record_id: str
    run_id: str
    stage: str
    provider: str           # "anthropic" | "openai"
    model: str
    call_type: str          # "text" | "image"
    recorded_at: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    image_count: int | None = None
    image_size: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "run_id": self.run_id,
            "stage": self.stage,
            "provider": self.provider,
            "model": self.model,
            "call_type": self.call_type,
            "recorded_at": self.recorded_at,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "image_count": self.image_count,
            "image_size": self.image_size,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> UsageRecord:
        return cls(
            record_id=d["record_id"],
            run_id=d["run_id"],
            stage=d["stage"],
            provider=d["provider"],
            model=d["model"],
            call_type=d["call_type"],
            recorded_at=d["recorded_at"],
            input_tokens=d.get("input_tokens"),
            output_tokens=d.get("output_tokens"),
            image_count=d.get("image_count"),
            image_size=d.get("image_size"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class StageCostSummary:
    stage: str
    call_count: int
    input_tokens: int
    output_tokens: int
    image_count: int
    cost: str | None    # Decimal serialized as "0.00000000", or null


@dataclass
class ProductionCostSummary:
    run_id: str
    computed_at: str
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_images: int
    total_cost: str | None  # null when any pricing is unavailable
    by_stage: list[StageCostSummary]


@dataclass
class QueueCostSummary:
    total_cost: str | None
    total_input_tokens: int
    total_output_tokens: int
    total_images: int
    job_count: int


class PricingCatalog:
    def __init__(
        self,
        data: dict | None = None,
        path: str | Path | None = None,
    ) -> None:
        if path is not None:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        elif data is None:
            default = Path(__file__).parent.parent / "data" / "model_pricing.json"
            if default.exists():
                data = json.loads(default.read_text(encoding="utf-8"))
            else:
                data = {
                    "schema_version": CATALOG_SCHEMA_VERSION,
                    "anthropic": {},
                    "openai_image": {},
                }

        version = data.get("schema_version")
        if version != CATALOG_SCHEMA_VERSION:
            raise CatalogSchemaError(
                f"Unsupported pricing catalog schema_version {version!r} "
                f"(expected {CATALOG_SCHEMA_VERSION})"
            )
        self._data = data

    def get_claude_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> Decimal | None:
        """Return total cost in USD for one Claude text call, or None if pricing unavailable."""
        pricing = self._data.get("anthropic", {}).get(model)
        if not pricing:
            return None
        inp = pricing.get("input_per_mtok")
        out = pricing.get("output_per_mtok")
        if inp is None or out is None:
            return None
        return (
            Decimal(str(inp)) * input_tokens + Decimal(str(out)) * output_tokens
        ) / Decimal("1000000")

    def get_image_cost(self, model: str, size: str) -> Decimal | None:
        """Return cost per image in USD, or None if pricing unavailable."""
        per_image = self._data.get("openai_image", {}).get(model, {}).get("per_image", {})
        price = per_image.get(size)
        if price is None:
            return None
        return Decimal(str(price))


def _record_cost(rec: UsageRecord, catalog: PricingCatalog) -> Decimal | None:
    if rec.call_type == "text":
        if rec.input_tokens is None:
            return None
        return catalog.get_claude_cost(rec.model, rec.input_tokens, rec.output_tokens or 0)
    if rec.call_type == "image":
        if not rec.image_count or not rec.image_size:
            return None
        per_img = catalog.get_image_cost(rec.model, rec.image_size)
        return None if per_img is None else per_img * rec.image_count
    return None


class CostTracker:
    def __init__(
        self,
        run_id: str,
        costs_dir: Path,
        catalog: PricingCatalog | None = None,
    ) -> None:
        self.run_id = run_id
        self.costs_dir = Path(costs_dir)
        self.catalog = catalog or PricingCatalog()
        self._records: list[UsageRecord] = []
        self.costs_dir.mkdir(parents=True, exist_ok=True)

    def record(self, raw: dict, stage: str) -> UsageRecord:
        """Append one API call record to JSONL and in-memory list."""
        rec = UsageRecord(
            record_id=f"{self.run_id}_{uuid.uuid4().hex[:8]}",
            run_id=self.run_id,
            stage=stage,
            provider=raw["provider"],
            model=raw["model"],
            call_type=raw["call_type"],
            recorded_at=datetime.now(timezone.utc).isoformat(),
            input_tokens=raw.get("input_tokens"),
            output_tokens=raw.get("output_tokens"),
            image_count=raw.get("image_count"),
            image_size=raw.get("image_size"),
            metadata=raw.get("metadata", {}),
        )
        self._records.append(rec)
        jsonl = self.costs_dir / "usage_records.jsonl"
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")
        return rec

    def load_existing(self) -> None:
        """Load records already written to JSONL (call on resume to avoid double-counting)."""
        jsonl = self.costs_dir / "usage_records.jsonl"
        if not jsonl.exists():
            return
        records: list[UsageRecord] = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(UsageRecord.from_dict(json.loads(line)))
            except Exception:
                pass
        self._records = records

    def compute_summary(self) -> ProductionCostSummary:
        stage_data: dict[str, dict] = {}
        for rec in self._records:
            s = stage_data.setdefault(rec.stage, {
                "call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "image_count": 0,
                "costs": [],
            })
            s["call_count"] += 1
            s["input_tokens"] += rec.input_tokens or 0
            s["output_tokens"] += rec.output_tokens or 0
            s["image_count"] += rec.image_count or 0
            s["costs"].append(_record_cost(rec, self.catalog))

        by_stage: list[StageCostSummary] = []
        total_known = Decimal("0")
        any_null = False

        for stage, s in stage_data.items():
            costs = s["costs"]
            if any(c is None for c in costs):
                stage_cost_str: str | None = None
                any_null = True
            else:
                stage_dec = sum(costs, Decimal("0"))
                total_known += stage_dec
                stage_cost_str = f"{stage_dec:.8f}"

            by_stage.append(StageCostSummary(
                stage=stage,
                call_count=s["call_count"],
                input_tokens=s["input_tokens"],
                output_tokens=s["output_tokens"],
                image_count=s["image_count"],
                cost=stage_cost_str,
            ))

        return ProductionCostSummary(
            run_id=self.run_id,
            computed_at=datetime.now(timezone.utc).isoformat(),
            total_calls=len(self._records),
            total_input_tokens=sum(r.input_tokens or 0 for r in self._records),
            total_output_tokens=sum(r.output_tokens or 0 for r in self._records),
            total_images=sum(r.image_count or 0 for r in self._records),
            total_cost=None if any_null else f"{total_known:.8f}",
            by_stage=by_stage,
        )

    def save_summary(self) -> ProductionCostSummary:
        """Compute summary, write costs/summary.json atomically, return summary."""
        summary = self.compute_summary()
        data = {
            "run_id": summary.run_id,
            "computed_at": summary.computed_at,
            "total_calls": summary.total_calls,
            "total_input_tokens": summary.total_input_tokens,
            "total_output_tokens": summary.total_output_tokens,
            "total_images": summary.total_images,
            "total_cost": summary.total_cost,
            "by_stage": [
                {
                    "stage": s.stage,
                    "call_count": s.call_count,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "image_count": s.image_count,
                    "cost": s.cost,
                }
                for s in summary.by_stage
            ],
        }
        path = self.costs_dir / "summary.json"
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return summary
