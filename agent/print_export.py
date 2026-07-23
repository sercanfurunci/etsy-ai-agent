"""
Stage 11.1 — Print Export System

Exports poster final.png images to standard print sizes at 300 DPI.
No network calls, no AI, no API calls — Pillow only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from PIL import Image

# ── Print size registry ────────────────────────────────────────────────────────

@dataclass
class PrintSize:
    name: str
    width: float    # physical width
    height: float   # physical height
    unit: str       # "inch" or "mm"
    px_width: int   # at 300 DPI
    px_height: int  # at 300 DPI


# All sizes at 300 DPI.  mm sizes: round(mm / 25.4 * 300) for pixel counts.
PRINT_SIZES: dict[str, PrintSize] = {
    "2x3":   PrintSize("2x3",   2,   3,   "inch", 600,   900),
    "3x4":   PrintSize("3x4",   3,   4,   "inch", 900,   1200),
    "4x5":   PrintSize("4x5",   4,   5,   "inch", 1200,  1500),
    "5x7":   PrintSize("5x7",   5,   7,   "inch", 1500,  2100),
    "11x14": PrintSize("11x14", 11,  14,  "inch", 3300,  4200),
    "16x20": PrintSize("16x20", 16,  20,  "inch", 4800,  6000),
    "18x24": PrintSize("18x24", 18,  24,  "inch", 5400,  7200),
    "24x36": PrintSize("24x36", 24,  36,  "inch", 7200,  10800),
    "A5":    PrintSize("A5",    148, 210, "mm",   1748,  2480),
    "A4":    PrintSize("A4",    210, 297, "mm",   2480,  3508),
    "A3":    PrintSize("A3",    297, 420, "mm",   3508,  4961),
    "A2":    PrintSize("A2",    420, 594, "mm",   4961,  7016),
}

_DPI = 300

# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ExportRecord:
    size: str
    physical_width: float
    physical_height: float
    unit: str
    target_width: int
    target_height: int
    target_ratio: str       # e.g. "11:14"
    crop_mode: str
    output_format: str
    output_path: str        # absolute path
    dpi: int
    upscale_enabled: bool
    upscaled: bool
    scale_factor: float
    background_color: str
    warnings: list[str]


@dataclass
class PosterExportResult:
    poster_id: str
    source_path: str
    source_width: int
    source_height: int
    source_sha256: str
    exports: list[ExportRecord]
    failed: list[dict]   # {"size": ..., "error": ...}


@dataclass
class ExportResult:
    run_dir: str
    created_at: str         # ISO8601
    crop_mode: str
    output_format: str
    upscale: bool
    requested_sizes: list[str]
    poster_results: list[PosterExportResult]
    warnings: list[str]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_hex_color(color: str) -> tuple[int, int, int]:
    """Parse #RRGGBB → (R, G, B).  Raises ValueError on invalid input."""
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
        raise ValueError(
            f"background_color must be a valid hex color (#RRGGBB), got: {color!r}"
        )
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    return (r, g, b)


def _ratio_str(w: int, h: int) -> str:
    from math import gcd
    d = gcd(w, h)
    return f"{w // d}:{h // d}"


def _natural_sort_key(name: str):
    """Sort poster_1, poster_2, ..., poster_10 in natural order."""
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]


def _discover_posters(images_dir: Path) -> list[Path]:
    """Return poster_XX directories in natural sort order."""
    poster_dirs = [
        d for d in images_dir.iterdir()
        if d.is_dir() and re.fullmatch(r"poster_\d+", d.name)
    ]
    poster_dirs.sort(key=lambda d: _natural_sort_key(d.name))
    return poster_dirs


# ── Image rendering ────────────────────────────────────────────────────────────

def _render_fit_or_pad(
    src: Image.Image,
    target_w: int,
    target_h: int,
    bg_rgb: tuple[int, int, int],
    upscale: bool,
) -> tuple[Image.Image, float, bool]:
    """
    fit / pad crop mode:
    Resize proportionally to fit WITHIN target. Never crop, never stretch.
    Add padding with bg_rgb when ratios differ.

    Note on semantic distinction:
      - fit:  padding is a side-effect of ratio mismatch; white is the convention.
      - pad:  caller explicitly chooses a background canvas colour.
    Both render identically in code; the semantic difference is in the caller's intent.

    Returns (result_image, scale_factor, was_upscaled).
    """
    src_w, src_h = src.size

    # Compute scale factor to fit inside target
    scale = min(target_w / src_w, target_h / src_h)

    was_upscaled = False
    if scale > 1.0 and not upscale:
        # Cap: never enlarge beyond original pixels
        scale = 1.0
    elif scale > 1.0:
        was_upscaled = True

    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))

    resized = src.resize((new_w, new_h), Image.LANCZOS)

    # Build canvas in correct mode
    src_mode = src.mode
    if src_mode == "RGBA":
        canvas = Image.new("RGBA", (target_w, target_h), bg_rgb + (255,))
    else:
        canvas = Image.new("RGB", (target_w, target_h), bg_rgb)

    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2

    if src_mode == "RGBA":
        canvas.paste(resized, (paste_x, paste_y), resized)
    else:
        if resized.mode != canvas.mode:
            resized = resized.convert(canvas.mode)
        canvas.paste(resized, (paste_x, paste_y))

    return canvas, scale, was_upscaled


def _render_fill(
    src: Image.Image,
    target_w: int,
    target_h: int,
    bg_rgb: tuple[int, int, int],
    upscale: bool,
) -> tuple[Image.Image, float, bool]:
    """
    fill crop mode:
    Resize proportionally to COVER target fully, then center-crop.
    Never stretch.

    Returns (result_image, scale_factor, was_upscaled).
    """
    src_w, src_h = src.size

    # Scale to cover: use the LARGER ratio
    scale = max(target_w / src_w, target_h / src_h)

    was_upscaled = False
    if scale > 1.0 and not upscale:
        # If we can't upscale, fall back to fit behaviour — place at original size
        # centered on target canvas (may not fill completely, but never crops less than src)
        scale = 1.0
    elif scale > 1.0:
        was_upscaled = True

    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))

    resized = src.resize((new_w, new_h), Image.LANCZOS)

    # Center crop to target
    crop_x = (new_w - target_w) // 2
    crop_y = (new_h - target_h) // 2
    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)

    # Build canvas (in case resized is smaller than target without upscale)
    src_mode = src.mode
    if src_mode == "RGBA":
        canvas = Image.new("RGBA", (target_w, target_h), bg_rgb + (255,))
    else:
        canvas = Image.new("RGB", (target_w, target_h), bg_rgb)

    cropped = resized.crop((
        crop_x,
        crop_y,
        crop_x + min(target_w, new_w),
        crop_y + min(target_h, new_h),
    ))

    paste_x = (target_w - cropped.width) // 2
    paste_y = (target_h - cropped.height) // 2

    if src_mode == "RGBA":
        canvas.paste(cropped, (paste_x, paste_y), cropped)
    else:
        if cropped.mode != canvas.mode:
            cropped = cropped.convert(canvas.mode)
        canvas.paste(cropped, (paste_x, paste_y))

    return canvas, scale, was_upscaled


def _save_image(
    img: Image.Image,
    dest: Path,
    output_format: str,
) -> None:
    """Save image to dest with correct format, DPI, quality settings."""
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    try:
        if output_format == "jpg":
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            img.save(
                tmp_path,
                format="JPEG",
                quality=95,
                subsampling=0,
                dpi=(_DPI, _DPI),
            )
        else:  # png
            img.save(tmp_path, format="PNG", dpi=(_DPI, _DPI))
        os.replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _export_one_size(
    src_img: Image.Image,
    size_key: str,
    ps: PrintSize,
    crop_mode: str,
    output_format: str,
    upscale: bool,
    bg_rgb: tuple[int, int, int],
    bg_color: str,
    dest_dir: Path,
    overwrite: bool,
) -> tuple[ExportRecord | None, dict | None]:
    """
    Export src_img to one print size.
    Returns (ExportRecord, None) on success, (None, failed_dict) on skip/error.
    """
    ext = "jpg" if output_format == "jpg" else "png"
    out_path = dest_dir / f"poster.{ext}"

    # Overwrite check
    if out_path.exists() and not overwrite:
        return None, {
            "size": size_key,
            "error": "file exists, use overwrite=True",
        }

    target_w = ps.px_width
    target_h = ps.px_height

    warnings: list[str] = []

    # Upscale warning: if source is smaller than target and upscale is off
    src_w, src_h = src_img.size
    scale_needed = min(target_w / src_w, target_h / src_h)
    if scale_needed > 1.0 and not upscale:
        warnings.append(
            f"Source image ({src_w}×{src_h}px) is smaller than target "
            f"({target_w}×{target_h}px) at 300 DPI. "
            "Output will be centered on canvas without enlargement. "
            "Use upscale=True to enable LANCZOS enlargement."
        )

    try:
        if crop_mode in ("fit", "pad"):
            result_img, scale_factor, was_upscaled = _render_fit_or_pad(
                src_img, target_w, target_h, bg_rgb, upscale
            )
        else:  # fill
            result_img, scale_factor, was_upscaled = _render_fill(
                src_img, target_w, target_h, bg_rgb, upscale
            )

        dest_dir.mkdir(parents=True, exist_ok=True)
        _save_image(result_img, out_path, output_format)

    except Exception as exc:
        return None, {"size": size_key, "error": str(exc)}

    record = ExportRecord(
        size=size_key,
        physical_width=ps.width,
        physical_height=ps.height,
        unit=ps.unit,
        target_width=target_w,
        target_height=target_h,
        target_ratio=_ratio_str(target_w, target_h),
        crop_mode=crop_mode,
        output_format=output_format,
        output_path=str(out_path.resolve()),
        dpi=_DPI,
        upscale_enabled=upscale,
        upscaled=was_upscaled,
        scale_factor=round(scale_factor, 6),
        background_color=bg_color,
        warnings=warnings,
    )
    return record, None


def _export_poster(
    poster_dir: Path,
    sizes: list[str],
    crop_mode: str,
    output_format: str,
    upscale: bool,
    bg_rgb: tuple[int, int, int],
    bg_color: str,
    exports_root: Path,
    overwrite: bool,
) -> PosterExportResult:
    """Export all requested sizes for one poster."""
    poster_id = poster_dir.name
    # Zero-pad single-digit ids for consistent output dir naming (poster_1 → poster_01)
    match = re.fullmatch(r"poster_(\d+)", poster_id)
    if match:
        padded = f"poster_{int(match.group(1)):02d}"
    else:
        padded = poster_id

    final_png = poster_dir / "final.png"
    poster_export_dir = exports_root / padded

    exports: list[ExportRecord] = []
    failed: list[dict] = []

    # Handle missing final.png
    if not final_png.exists():
        for size_key in sizes:
            failed.append({"size": size_key, "error": "final.png not found"})
        return PosterExportResult(
            poster_id=padded,
            source_path=str(final_png),
            source_width=0,
            source_height=0,
            source_sha256="",
            exports=[],
            failed=failed,
        )

    # Read source image (catch corrupt files)
    try:
        src_img = Image.open(final_png)
        src_img.load()  # force decode to catch corruption early
    except Exception as exc:
        for size_key in sizes:
            failed.append({"size": size_key, "error": f"corrupt image: {exc}"})
        return PosterExportResult(
            poster_id=padded,
            source_path=str(final_png.resolve()),
            source_width=0,
            source_height=0,
            source_sha256="",
            exports=[],
            failed=failed,
        )

    src_w, src_h = src_img.size
    src_sha256 = _sha256_of_file(final_png)

    for size_key in sizes:
        ps = PRINT_SIZES[size_key]
        size_out_dir = poster_export_dir / size_key

        record, err = _export_one_size(
            src_img=src_img,
            size_key=size_key,
            ps=ps,
            crop_mode=crop_mode,
            output_format=output_format,
            upscale=upscale,
            bg_rgb=bg_rgb,
            bg_color=bg_color,
            dest_dir=size_out_dir,
            overwrite=overwrite,
        )
        if record is not None:
            exports.append(record)
        else:
            failed.append(err)

    return PosterExportResult(
        poster_id=padded,
        source_path=str(final_png.resolve()),
        source_width=src_w,
        source_height=src_h,
        source_sha256=src_sha256,
        exports=exports,
        failed=failed,
    )


def _write_poster_metadata(
    poster_result: PosterExportResult,
    exports_root: Path,
) -> None:
    poster_export_dir = exports_root / poster_result.poster_id
    poster_export_dir.mkdir(parents=True, exist_ok=True)
    meta = asdict(poster_result)
    meta_path = poster_export_dir / "metadata.json"
    tmp = meta_path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, meta_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _write_manifest(
    result: ExportResult,
    exports_root: Path,
) -> None:
    poster_count = len(result.poster_results)
    export_count = sum(len(pr.exports) for pr in result.poster_results)
    failed_count = sum(len(pr.failed) for pr in result.poster_results)

    manifest = {
        "schema_version": "1.0",
        "run_dir": result.run_dir,
        "created_at": result.created_at,
        "crop_mode": result.crop_mode,
        "output_format": result.output_format,
        "upscale": result.upscale,
        "requested_sizes": result.requested_sizes,
        "poster_count": poster_count,
        "export_count": export_count,
        "failed_count": failed_count,
        "posters": [asdict(pr) for pr in result.poster_results],
        "warnings": result.warnings,
    }

    manifest_path = exports_root / "export_manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(manifest, indent=2))
        os.replace(tmp, manifest_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate_inputs(
    run_dir: Path,
    sizes: list[str],
    crop_mode: str,
    output_format: str,
    background_color: str,
) -> None:
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise ValueError(f"run_dir is not a directory: {run_dir}")

    images_dir = run_dir / "images"
    if not images_dir.exists() or not images_dir.is_dir():
        raise ValueError(f"images directory not found inside run_dir: {images_dir}")

    poster_dirs = _discover_posters(images_dir)
    if not poster_dirs:
        raise ValueError(
            f"No poster_XX directories found in {images_dir}. "
            "Run production first to generate images."
        )

    if not sizes:
        raise ValueError("sizes must not be empty; provide at least one size name.")

    if len(sizes) != len(set(sizes)):
        dupes = [s for s in sizes if sizes.count(s) > 1]
        raise ValueError(f"sizes contains duplicates: {sorted(set(dupes))}")

    unknown = [s for s in sizes if s not in PRINT_SIZES]
    if unknown:
        raise ValueError(
            f"Unknown size name(s): {unknown}. "
            f"Supported: {sorted(PRINT_SIZES.keys())}"
        )

    if crop_mode not in ("fit", "fill", "pad"):
        raise ValueError(
            f"crop_mode must be 'fit', 'fill', or 'pad'; got: {crop_mode!r}"
        )

    if output_format not in ("png", "jpg"):
        raise ValueError(
            f"output_format must be 'png' or 'jpg'; got: {output_format!r}"
        )

    _parse_hex_color(background_color)  # raises ValueError on bad input


# ── Public API ─────────────────────────────────────────────────────────────────

def export_prints(
    run_dir: "Path | str",
    *,
    sizes: "list[str] | None" = None,
    crop_mode: "Literal['fit', 'fill', 'pad']" = "fit",
    output_format: "Literal['png', 'jpg']" = "png",
    upscale: bool = False,
    background_color: str = "#FFFFFF",
    overwrite: bool = False,
) -> ExportResult:
    """
    Export poster images from a production run to standard print sizes at 300 DPI.

    Parameters
    ----------
    run_dir:
        Path to the production run directory (contains images/, manifest.json, etc.)
    sizes:
        List of size keys from PRINT_SIZES (e.g. ["2x3", "A4"]).
        If None, all 12 sizes are exported.
    crop_mode:
        "fit"  — fit artwork within target, pad remainder (no crop, no stretch).
        "fill" — cover target fully, center-crop excess (no stretch).
        "pad"  — identical rendering to "fit"; semantic emphasis on explicit canvas.
    output_format:
        "png" (default) or "jpg".
    upscale:
        If False (default), never enlarge artwork beyond its original pixel dimensions.
        If True, allow LANCZOS enlargement.
    background_color:
        Background/padding color as #RRGGBB hex. Default "#FFFFFF".
    overwrite:
        If False (default), skip existing output files and record them as failed.
        If True, replace existing files.

    Returns
    -------
    ExportResult with per-poster results and an overall manifest.
    """
    run_dir = Path(run_dir)

    if sizes is None:
        sizes = list(PRINT_SIZES.keys())

    _validate_inputs(run_dir, sizes, crop_mode, output_format, background_color)

    bg_rgb = _parse_hex_color(background_color)
    images_dir = run_dir / "images"
    exports_root = run_dir / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)

    poster_dirs = _discover_posters(images_dir)
    created_at = datetime.now(timezone.utc).isoformat()
    global_warnings: list[str] = []

    poster_results: list[PosterExportResult] = []
    for poster_dir in poster_dirs:
        pr = _export_poster(
            poster_dir=poster_dir,
            sizes=sizes,
            crop_mode=crop_mode,
            output_format=output_format,
            upscale=upscale,
            bg_rgb=bg_rgb,
            bg_color=background_color,
            exports_root=exports_root,
            overwrite=overwrite,
        )
        _write_poster_metadata(pr, exports_root)
        poster_results.append(pr)

    result = ExportResult(
        run_dir=str(run_dir.resolve()),
        created_at=created_at,
        crop_mode=crop_mode,
        output_format=output_format,
        upscale=upscale,
        requested_sizes=sizes,
        poster_results=poster_results,
        warnings=global_warnings,
    )

    _write_manifest(result, exports_root)
    return result
