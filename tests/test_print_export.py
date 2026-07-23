"""
Tests for Stage 11.1 — Print Export System (agent/print_export.py).

All image operations use real Pillow Image.new() — no mocking of Pillow internals.
No network calls. run_production() is never imported or called.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from agent.print_export import (
    PRINT_SIZES,
    ExportRecord,
    ExportResult,
    PosterExportResult,
    PrintSize,
    export_prints,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_run_dir(tmp_path: Path, poster_count: int = 1) -> Path:
    """Create a fake run directory with synthetic poster images."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    images_dir = run_dir / "images"
    images_dir.mkdir()
    for i in range(1, poster_count + 1):
        poster_dir = images_dir / f"poster_{i:02d}"
        poster_dir.mkdir()
        _make_final_png(poster_dir, width=1536, height=2304)
    return run_dir


def _make_final_png(
    poster_dir: Path,
    width: int = 1536,
    height: int = 2304,
    mode: str = "RGB",
    color=None,
) -> Path:
    """Create a synthetic final.png in poster_dir."""
    if color is None:
        color = (100, 150, 200) if mode == "RGB" else (100, 150, 200, 255)
    img = Image.new(mode, (width, height), color)
    path = poster_dir / "final.png"
    img.save(path, format="PNG")
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_output(
    run_dir: Path,
    poster_id: str,
    size: str,
    fmt: str = "png",
) -> Image.Image:
    ext = fmt
    p = run_dir / "exports" / poster_id / size / f"poster.{ext}"
    return Image.open(p)


# ── PRINT_SIZES registry ───────────────────────────────────────────────────────

class TestPrintSizes:
    @pytest.mark.parametrize("name,expected_w,expected_h", [
        ("2x3",   600,   900),
        ("3x4",   900,   1200),
        ("4x5",   1200,  1500),
        ("5x7",   1500,  2100),
        ("11x14", 3300,  4200),
        ("16x20", 4800,  6000),
        ("18x24", 5400,  7200),
        ("24x36", 7200,  10800),
        ("A5",    1748,  2480),
        ("A4",    2480,  3508),
        ("A3",    3508,  4961),
        ("A2",    4961,  7016),
    ])
    def test_all_12_sizes_pixel_dimensions(self, name, expected_w, expected_h):
        ps = PRINT_SIZES[name]
        assert ps.px_width == expected_w, f"{name} width mismatch"
        assert ps.px_height == expected_h, f"{name} height mismatch"

    def test_all_sizes_have_name_matching_key(self):
        for key, ps in PRINT_SIZES.items():
            assert ps.name == key

    def test_inch_sizes_have_inch_unit(self):
        inch_sizes = {"2x3", "3x4", "4x5", "5x7", "11x14", "16x20", "18x24", "24x36"}
        for k in inch_sizes:
            assert PRINT_SIZES[k].unit == "inch"

    def test_a_series_have_mm_unit(self):
        for k in ("A5", "A4", "A3", "A2"):
            assert PRINT_SIZES[k].unit == "mm"


# ── Fit mode ──────────────────────────────────────────────────────────────────

class TestFitMode:
    def test_fit_same_ratio_no_padding_needed(self, tmp_path):
        """2x3 source (600×900) → 2x3 target: no padding, exact fit."""
        run_dir = _make_run_dir(tmp_path)
        _make_final_png(run_dir / "images" / "poster_01", width=600, height=900)

        result = export_prints(run_dir, sizes=["2x3"], crop_mode="fit")
        assert not result.poster_results[0].failed

        img = _open_output(run_dir, "poster_01", "2x3")
        assert img.size == (600, 900)

    def test_fit_different_ratio_adds_padding(self, tmp_path):
        """Wide source on tall target → horizontal padding bands."""
        run_dir = _make_run_dir(tmp_path)
        # Wide image: 900×600; target 2x3: 600×900 → portrait
        # After fit, artwork occupies 600×400, padded to 600×900
        poster_dir = run_dir / "images" / "poster_01"
        _make_final_png(poster_dir, width=900, height=600, color=(255, 0, 0))

        result = export_prints(
            run_dir, sizes=["2x3"], crop_mode="fit", upscale=True,
            background_color="#FFFFFF"
        )
        assert not result.poster_results[0].failed

        img = _open_output(run_dir, "poster_01", "2x3")
        assert img.size == (600, 900)

        # Top-left corner should be white (background), not red (artwork)
        pixel = img.convert("RGB").getpixel((0, 0))
        assert pixel == (255, 255, 255), f"Expected white padding, got {pixel}"

    def test_fit_artwork_never_stretched(self, tmp_path):
        """Output image pixel counts must equal target size."""
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["4x5"], crop_mode="fit", upscale=True)
        img = _open_output(run_dir, "poster_01", "4x5")
        assert img.size == (1200, 1500)

    def test_fit_artwork_never_cropped(self, tmp_path):
        """For fit mode, artwork content must not be cropped out."""
        run_dir = _make_run_dir(tmp_path)
        # Red image: any red pixel in result confirms artwork is present
        _make_final_png(run_dir / "images" / "poster_01", color=(255, 0, 0))

        result = export_prints(
            run_dir, sizes=["4x5"], crop_mode="fit", upscale=True,
            background_color="#0000FF",
        )
        img = _open_output(run_dir, "poster_01", "4x5").convert("RGB")
        pixels = set()
        for x in range(0, img.width, 10):
            for y in range(0, img.height, 10):
                pixels.add(img.getpixel((x, y)))
        # Red artwork pixels must be present
        assert any(p[0] > 200 and p[1] < 50 and p[2] < 50 for p in pixels), \
            "Artwork (red) not found — it was cropped"


# ── Fill mode ─────────────────────────────────────────────────────────────────

class TestFillMode:
    def test_fill_canvas_fully_covered_when_upscale_enabled(self, tmp_path):
        """fill with upscale=True should produce exact target size."""
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["4x5"], crop_mode="fill", upscale=True)
        img = _open_output(run_dir, "poster_01", "4x5")
        assert img.size == (1200, 1500)

    def test_fill_no_stretching(self, tmp_path):
        """Output must be target dimensions."""
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["2x3"], crop_mode="fill", upscale=True)
        img = _open_output(run_dir, "poster_01", "2x3")
        assert img.size == (600, 900)

    def test_fill_center_crop_removes_edges(self, tmp_path):
        """Wide source on portrait target: center is kept, edges cropped."""
        run_dir = _make_run_dir(tmp_path)
        # Source 900×600 (wide); target 600×900 (portrait)
        # Fill would scale to cover 600×900:
        #   scale = max(600/900, 900/600) = max(0.667, 1.5) = 1.5
        #   new size = 1350×900, then crop to 600×900 from center
        _make_final_png(run_dir / "images" / "poster_01", width=900, height=600)
        result = export_prints(run_dir, sizes=["2x3"], crop_mode="fill", upscale=True)
        img = _open_output(run_dir, "poster_01", "2x3")
        assert img.size == (600, 900)


# ── Pad mode ─────────────────────────────────────────────────────────────────

class TestPadMode:
    def test_pad_artwork_centered_on_background(self, tmp_path):
        """Pad mode centers artwork on explicit background canvas."""
        run_dir = _make_run_dir(tmp_path)
        # Wide source → portrait target → background visible at top/bottom
        _make_final_png(
            run_dir / "images" / "poster_01", width=900, height=600, color=(0, 255, 0)
        )
        result = export_prints(
            run_dir, sizes=["2x3"], crop_mode="pad", upscale=True,
            background_color="#FF0000",
        )
        assert not result.poster_results[0].failed
        img = _open_output(run_dir, "poster_01", "2x3").convert("RGB")
        assert img.size == (600, 900)
        # Top pixel should be red background
        top_pixel = img.getpixel((img.width // 2, 0))
        assert top_pixel[0] > 200, f"Expected red background at top, got {top_pixel}"

    def test_pad_no_crop(self, tmp_path):
        """Pad mode must not crop the artwork."""
        run_dir = _make_run_dir(tmp_path)
        _make_final_png(
            run_dir / "images" / "poster_01", color=(0, 0, 255)
        )
        result = export_prints(run_dir, sizes=["4x5"], crop_mode="pad", upscale=True)
        img = _open_output(run_dir, "poster_01", "4x5")
        assert img.size == (1200, 1500)


# ── PNG output ────────────────────────────────────────────────────────────────

class TestPNGOutput:
    def test_png_file_exists_and_opens(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["2x3"], output_format="png")
        assert not result.poster_results[0].failed
        out_path = Path(result.poster_results[0].exports[0].output_path)
        assert out_path.exists()
        img = Image.open(out_path)
        assert img.format == "PNG"

    def test_png_dpi_metadata_is_300(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["2x3"], output_format="png")
        out_path = Path(result.poster_results[0].exports[0].output_path)
        img = Image.open(out_path)
        info = img.info
        dpi = info.get("dpi")
        assert dpi is not None, "No DPI metadata in PNG"
        assert round(dpi[0]) == 300, f"Expected 300 DPI, got {dpi[0]}"
        assert round(dpi[1]) == 300, f"Expected 300 DPI, got {dpi[1]}"


# ── JPG output ────────────────────────────────────────────────────────────────

class TestJPGOutput:
    def test_jpg_file_exists_and_opens(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["2x3"], output_format="jpg")
        assert not result.poster_results[0].failed
        out_path = Path(result.poster_results[0].exports[0].output_path)
        assert out_path.exists()
        img = Image.open(out_path)
        assert img.format == "JPEG"

    def test_jpg_mode_is_rgb(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["2x3"], output_format="jpg")
        out_path = Path(result.poster_results[0].exports[0].output_path)
        img = Image.open(out_path)
        assert img.mode == "RGB"

    def test_jpg_dpi_metadata_is_300(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["2x3"], output_format="jpg")
        out_path = Path(result.poster_results[0].exports[0].output_path)
        img = Image.open(out_path)
        info = img.info
        dpi = info.get("dpi")
        assert dpi is not None, "No DPI metadata in JPEG"
        assert dpi[0] == 300, f"Expected 300 DPI, got {dpi[0]}"
        assert dpi[1] == 300, f"Expected 300 DPI, got {dpi[1]}"


# ── Transparency ───────────────────────────────────────────────────────────────

class TestTransparency:
    def test_rgba_source_preserved_in_png(self, tmp_path):
        """RGBA source should remain RGBA in PNG output."""
        run_dir = _make_run_dir(tmp_path)
        _make_final_png(
            run_dir / "images" / "poster_01", mode="RGBA", color=(0, 128, 255, 200)
        )
        result = export_prints(run_dir, sizes=["2x3"], output_format="png")
        assert not result.poster_results[0].failed
        out_path = Path(result.poster_results[0].exports[0].output_path)
        img = Image.open(out_path)
        assert img.mode == "RGBA", f"Expected RGBA, got {img.mode}"

    def test_rgba_source_converted_to_rgb_for_jpg(self, tmp_path):
        """RGBA source must be converted to RGB for JPG output."""
        run_dir = _make_run_dir(tmp_path)
        _make_final_png(
            run_dir / "images" / "poster_01", mode="RGBA", color=(0, 128, 255, 128)
        )
        result = export_prints(run_dir, sizes=["2x3"], output_format="jpg")
        assert not result.poster_results[0].failed
        out_path = Path(result.poster_results[0].exports[0].output_path)
        img = Image.open(out_path)
        assert img.mode == "RGB"


# ── Source file integrity ─────────────────────────────────────────────────────

class TestSourceIntegrity:
    def test_source_file_unchanged_after_export(self, tmp_path):
        """Source final.png must be byte-for-byte identical after export."""
        run_dir = _make_run_dir(tmp_path)
        final_png = run_dir / "images" / "poster_01" / "final.png"
        before_bytes = final_png.read_bytes()

        export_prints(run_dir, sizes=["2x3", "A4"], output_format="png", upscale=True)

        after_bytes = final_png.read_bytes()
        assert before_bytes == after_bytes, "Source final.png was modified!"

    def test_source_sha256_recorded_correctly(self, tmp_path):
        """SHA256 in metadata must match actual source file hash."""
        run_dir = _make_run_dir(tmp_path)
        final_png = run_dir / "images" / "poster_01" / "final.png"
        expected_sha = _sha256(final_png)

        result = export_prints(run_dir, sizes=["2x3"])
        assert result.poster_results[0].source_sha256 == expected_sha


# ── Upscaling ─────────────────────────────────────────────────────────────────

class TestUpscaling:
    def test_upscale_false_artwork_does_not_exceed_source(self, tmp_path):
        """With upscale=False, the artwork rendered pixels never exceed source dims."""
        run_dir = _make_run_dir(tmp_path)
        # Source: 200×300 (tiny). Target 2x3: 600×900 — requires upscale
        _make_final_png(run_dir / "images" / "poster_01", width=200, height=300)

        result = export_prints(run_dir, sizes=["2x3"], upscale=False)
        assert not result.poster_results[0].failed

        rec = result.poster_results[0].exports[0]
        assert not rec.upscaled

        # The output file should be target-sized canvas (600×900)
        # but artwork within it should not be larger than 200×300
        img = _open_output(run_dir, "poster_01", "2x3")
        assert img.size == (600, 900)

    def test_upscale_true_artwork_can_be_enlarged(self, tmp_path):
        """With upscale=True, small source can be enlarged to fill target."""
        run_dir = _make_run_dir(tmp_path)
        # Source: 200×300. Target 2x3: 600×900 (3x upscale)
        _make_final_png(run_dir / "images" / "poster_01", width=200, height=300)

        result = export_prints(run_dir, sizes=["2x3"], upscale=True, crop_mode="fit")
        rec = result.poster_results[0].exports[0]
        assert rec.upscaled

        # Full canvas should be used (output equals target)
        img = _open_output(run_dir, "poster_01", "2x3")
        assert img.size == (600, 900)

    def test_upscale_warning_when_source_too_small(self, tmp_path):
        """When source is smaller than 300 DPI target and upscale=False, warn."""
        run_dir = _make_run_dir(tmp_path)
        # Source 200×300 is much smaller than 2x3 at 300 DPI (600×900)
        _make_final_png(run_dir / "images" / "poster_01", width=200, height=300)

        result = export_prints(run_dir, sizes=["2x3"], upscale=False)
        rec = result.poster_results[0].exports[0]
        assert rec.warnings, "Expected upscale warning but none present"
        assert any("300 DPI" in w or "upscale" in w.lower() for w in rec.warnings)

    def test_no_upscale_warning_when_source_large_enough(self, tmp_path):
        """Source bigger than or equal to target → no upscale warning."""
        run_dir = _make_run_dir(tmp_path)
        # Source 1536×2304 is bigger than 2x3 target (600×900)
        result = export_prints(run_dir, sizes=["2x3"], upscale=False)
        rec = result.poster_results[0].exports[0]
        assert not rec.warnings, f"Unexpected warnings: {rec.warnings}"


# ── Metadata ──────────────────────────────────────────────────────────────────

class TestMetadata:
    def test_metadata_json_structure(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["2x3", "A4"])

        meta_path = run_dir / "exports" / "poster_01" / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())

        assert meta["poster_id"] == "poster_01"
        assert "source_path" in meta
        assert "source_width" in meta
        assert "source_height" in meta
        assert "source_sha256" in meta
        assert isinstance(meta["exports"], list)
        assert isinstance(meta["failed"], list)
        assert len(meta["exports"]) == 2

    def test_metadata_required_export_fields(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["2x3"])

        meta_path = run_dir / "exports" / "poster_01" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        exp = meta["exports"][0]

        required = [
            "size", "physical_width", "physical_height", "unit",
            "target_width", "target_height", "target_ratio",
            "crop_mode", "output_format", "output_path", "dpi",
            "upscale_enabled", "upscaled", "scale_factor",
            "background_color", "warnings",
        ]
        for field in required:
            assert field in exp, f"Missing field: {field}"

    def test_export_manifest_structure(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, poster_count=2)
        result = export_prints(run_dir, sizes=["2x3", "A4"])

        manifest_path = run_dir / "exports" / "export_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())

        assert manifest["schema_version"] == "1.0"
        assert "run_dir" in manifest
        assert "created_at" in manifest
        assert manifest["poster_count"] == 2
        assert manifest["export_count"] == 4   # 2 posters × 2 sizes
        assert manifest["failed_count"] == 0
        assert isinstance(manifest["posters"], list)
        assert isinstance(manifest["warnings"], list)

    def test_export_manifest_counts_match(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, poster_count=3)
        result = export_prints(run_dir, sizes=["2x3", "4x5", "A4"])

        manifest = json.loads(
            (run_dir / "exports" / "export_manifest.json").read_text()
        )
        assert manifest["export_count"] == 9   # 3 × 3
        assert manifest["failed_count"] == 0


# ── Natural sort / discovery ──────────────────────────────────────────────────

class TestPosterDiscovery:
    def test_poster_natural_sort_order(self, tmp_path):
        """poster_10 must come after poster_9, not after poster_1 (lexicographic)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        images_dir = run_dir / "images"
        images_dir.mkdir()

        for i in [1, 2, 9, 10, 11]:
            d = images_dir / f"poster_{i:02d}"
            d.mkdir()
            _make_final_png(d)

        result = export_prints(run_dir, sizes=["2x3"])
        ids = [pr.poster_id for pr in result.poster_results]
        assert ids == ["poster_01", "poster_02", "poster_09", "poster_10", "poster_11"]

    def test_unrelated_dirs_ignored(self, tmp_path):
        """Directories not matching poster_\\d+ are ignored."""
        run_dir = _make_run_dir(tmp_path)
        # Add unrelated dirs/files
        (run_dir / "images" / "thumbnails").mkdir()
        (run_dir / "images" / "poster_abc").mkdir()
        (run_dir / "images" / "notes.txt").write_text("ignore me")

        result = export_prints(run_dir, sizes=["2x3"])
        ids = [pr.poster_id for pr in result.poster_results]
        assert ids == ["poster_01"]

    def test_unrelated_files_in_poster_dir_ignored(self, tmp_path):
        """Files other than final.png inside poster dirs are ignored."""
        run_dir = _make_run_dir(tmp_path)
        poster_dir = run_dir / "images" / "poster_01"
        (poster_dir / "attempt_1.png").write_bytes(b"not a real png")
        (poster_dir / "attempts.json").write_text("{}")

        result = export_prints(run_dir, sizes=["2x3"])
        assert not result.poster_results[0].failed


# ── Validation errors ─────────────────────────────────────────────────────────

class TestValidation:
    def test_missing_run_dir_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, ValueError)):
            export_prints(tmp_path / "nonexistent", sizes=["2x3"])

    def test_missing_images_dir_raises(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        with pytest.raises(ValueError, match="images"):
            export_prints(run_dir, sizes=["2x3"])

    def test_no_poster_dirs_raises(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "images").mkdir()
        with pytest.raises(ValueError):
            export_prints(run_dir, sizes=["2x3"])

    def test_unsupported_size_name_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        with pytest.raises(ValueError, match="Unknown size"):
            export_prints(run_dir, sizes=["99x99"])

    def test_invalid_crop_mode_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        with pytest.raises(ValueError, match="crop_mode"):
            export_prints(run_dir, sizes=["2x3"], crop_mode="stretch")  # type: ignore

    def test_invalid_output_format_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        with pytest.raises(ValueError, match="output_format"):
            export_prints(run_dir, sizes=["2x3"], output_format="tiff")  # type: ignore

    def test_invalid_background_color_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        with pytest.raises(ValueError):
            export_prints(run_dir, sizes=["2x3"], background_color="red")

    def test_invalid_background_color_short_hex_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        with pytest.raises(ValueError):
            export_prints(run_dir, sizes=["2x3"], background_color="#FFF")

    def test_duplicate_sizes_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        with pytest.raises(ValueError, match="duplicate"):
            export_prints(run_dir, sizes=["2x3", "A4", "2x3"])

    def test_empty_sizes_list_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        with pytest.raises(ValueError):
            export_prints(run_dir, sizes=[])

    def test_run_dir_as_file_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises((ValueError, FileNotFoundError)):
            export_prints(f, sizes=["2x3"])


# ── Missing / corrupt images ──────────────────────────────────────────────────

class TestMissingCorruptImages:
    def test_missing_final_png_goes_to_failed(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        images_dir = run_dir / "images"
        images_dir.mkdir()
        poster_dir = images_dir / "poster_01"
        poster_dir.mkdir()
        # No final.png created

        result = export_prints(run_dir, sizes=["2x3"])
        pr = result.poster_results[0]
        assert pr.failed
        assert any("final.png" in f["error"].lower() or "not found" in f["error"].lower()
                   for f in pr.failed)

    def test_corrupt_final_png_goes_to_failed_not_crash(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        images_dir = run_dir / "images"
        images_dir.mkdir()
        poster_dir = images_dir / "poster_01"
        poster_dir.mkdir()
        # Write garbage bytes as "PNG"
        (poster_dir / "final.png").write_bytes(b"\x00\x01\x02\x03corrupt")

        # Should not raise; should record in failed
        result = export_prints(run_dir, sizes=["2x3"])
        pr = result.poster_results[0]
        assert pr.failed
        assert any("corrupt" in f["error"].lower() or f["size"] == "2x3"
                   for f in pr.failed)


# ── Overwrite behavior ────────────────────────────────────────────────────────

class TestOverwrite:
    def test_overwrite_false_skips_existing_file(self, tmp_path):
        """With overwrite=False, existing output files are skipped (in failed)."""
        run_dir = _make_run_dir(tmp_path)
        # First export
        export_prints(run_dir, sizes=["2x3"], overwrite=False)

        # Record what was written
        out_path = run_dir / "exports" / "poster_01" / "2x3" / "poster.png"
        original_bytes = out_path.read_bytes()

        # Write a sentinel file in its place
        out_path.write_bytes(b"sentinel")

        # Second export with overwrite=False
        result2 = export_prints(run_dir, sizes=["2x3"], overwrite=False)
        pr = result2.poster_results[0]

        # Should be in failed (skipped)
        assert any(f["size"] == "2x3" for f in pr.failed)

        # File must not have been modified
        assert out_path.read_bytes() == b"sentinel", "Existing file was overwritten!"

    def test_overwrite_true_replaces_existing_file(self, tmp_path):
        """With overwrite=True, existing output files are replaced."""
        run_dir = _make_run_dir(tmp_path)
        # First export
        export_prints(run_dir, sizes=["2x3"], overwrite=True)

        out_path = run_dir / "exports" / "poster_01" / "2x3" / "poster.png"
        out_path.write_bytes(b"sentinel")

        # Second export with overwrite=True
        result2 = export_prints(run_dir, sizes=["2x3"], overwrite=True)
        pr = result2.poster_results[0]

        assert not pr.failed
        assert out_path.read_bytes() != b"sentinel", "File was not overwritten!"


# ── Output structure ──────────────────────────────────────────────────────────

class TestOutputStructure:
    def test_output_dirs_created(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        export_prints(run_dir, sizes=["2x3", "A4"])

        assert (run_dir / "exports").is_dir()
        assert (run_dir / "exports" / "poster_01").is_dir()
        assert (run_dir / "exports" / "poster_01" / "2x3").is_dir()
        assert (run_dir / "exports" / "poster_01" / "A4").is_dir()
        assert (run_dir / "exports" / "poster_01" / "2x3" / "poster.png").exists()
        assert (run_dir / "exports" / "poster_01" / "A4" / "poster.png").exists()

    def test_metadata_json_created_per_poster(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, poster_count=2)
        export_prints(run_dir, sizes=["2x3"])

        assert (run_dir / "exports" / "poster_01" / "metadata.json").exists()
        assert (run_dir / "exports" / "poster_02" / "metadata.json").exists()

    def test_export_manifest_created_at_root(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        export_prints(run_dir, sizes=["2x3"])
        assert (run_dir / "exports" / "export_manifest.json").exists()

    def test_export_record_output_path_is_absolute(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["2x3"])
        rec = result.poster_results[0].exports[0]
        assert Path(rec.output_path).is_absolute()


# ── ExportRecord fields ───────────────────────────────────────────────────────

class TestExportRecord:
    def test_export_record_target_ratio_string(self, tmp_path):
        """target_ratio should be expressed as 'W:H' in lowest terms."""
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["11x14"])
        rec = result.poster_results[0].exports[0]
        # 3300:4200 → gcd=300 → 11:14
        assert rec.target_ratio == "11:14"

    def test_export_record_dpi_is_300(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["2x3"])
        assert result.poster_results[0].exports[0].dpi == 300

    def test_export_record_crop_mode_recorded(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["2x3"], crop_mode="fill")
        assert result.poster_results[0].exports[0].crop_mode == "fill"

    def test_export_record_scale_factor_gt_zero(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = export_prints(run_dir, sizes=["2x3"])
        rec = result.poster_results[0].exports[0]
        assert rec.scale_factor > 0


# ── CLI tests ─────────────────────────────────────────────────────────────────

def _run_cli(args: list[str]) -> tuple[int, str, str]:
    """Run the CLI as a subprocess and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "-m", "scripts.export_prints"] + args
    # Alternatively, invoke the script directly via module
    script = str(Path(__file__).parent.parent / "scripts" / "export_prints.py")
    cmd = [sys.executable, script] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


class TestCLI:
    def test_cli_all_flag_runs_successfully(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        rc, stdout, stderr = _run_cli([str(run_dir), "--all"])
        assert rc == 0, f"CLI failed: {stderr}"

    def test_cli_sizes_flag_runs_with_specific_sizes(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        rc, stdout, stderr = _run_cli([str(run_dir), "--sizes", "2x3", "A4"])
        assert rc == 0, f"CLI failed: {stderr}"

    def test_cli_all_and_sizes_mutually_exclusive(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        rc, stdout, stderr = _run_cli([str(run_dir), "--all", "--sizes", "2x3"])
        assert rc != 0, "Expected non-zero exit for --all + --sizes"

    def test_cli_no_all_no_sizes_exits_nonzero(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        rc, stdout, stderr = _run_cli([str(run_dir)])
        assert rc != 0, "Expected non-zero exit when neither --all nor --sizes given"

    def test_cli_json_flag_outputs_valid_json(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        rc, stdout, stderr = _run_cli([str(run_dir), "--sizes", "2x3", "--json"])
        assert rc == 0, f"CLI failed: {stderr}"
        parsed = json.loads(stdout)
        assert "run_dir" in parsed
        assert "poster_results" in parsed

    def test_cli_format_jpg(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        rc, stdout, stderr = _run_cli(
            [str(run_dir), "--sizes", "2x3", "--format", "jpg"]
        )
        assert rc == 0, f"CLI failed: {stderr}"
        out = run_dir / "exports" / "poster_01" / "2x3" / "poster.jpg"
        assert out.exists()

    def test_cli_crop_fill(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        rc, stdout, stderr = _run_cli(
            [str(run_dir), "--sizes", "2x3", "--crop", "fill"]
        )
        assert rc == 0, f"CLI failed: {stderr}"


# ── No production orchestrator or network ─────────────────────────────────────

class TestNoForbiddenImports:
    def test_run_production_not_imported_in_print_export(self):
        """print_export must not import run_production or production_orchestrator."""
        import agent.print_export as pe
        import importlib
        import importlib.util

        src_file = Path(pe.__file__).read_text()
        assert "run_production" not in src_file
        assert "production_orchestrator" not in src_file

    def test_no_network_imports_in_print_export(self):
        """print_export must not import requests, httpx, openai, or anthropic."""
        import agent.print_export as pe
        src_file = Path(pe.__file__).read_text()
        for banned in ("import requests", "import httpx", "import openai", "import anthropic"):
            assert banned not in src_file, f"Banned import found: {banned}"

    def test_no_socket_calls_in_print_export(self, tmp_path, monkeypatch):
        """Importing and calling export_prints must not open any socket."""
        import socket

        def _no_connect(self, *args, **kwargs):
            raise AssertionError("export_prints must not make network connections!")

        monkeypatch.setattr(socket.socket, "connect", _no_connect)

        run_dir = _make_run_dir(tmp_path)
        # This must not raise due to socket connection
        export_prints(run_dir, sizes=["2x3"])

    def test_run_production_never_called(self, tmp_path, monkeypatch):
        """Calling export_prints must not invoke run_production in any way."""
        called = []

        def _fake_run_production(*args, **kwargs):
            called.append(True)

        # Patch it at the source if it's somehow imported
        import agent.production_orchestrator as orch
        monkeypatch.setattr(orch, "run_production", _fake_run_production)

        run_dir = _make_run_dir(tmp_path)
        export_prints(run_dir, sizes=["2x3"])

        assert not called, "run_production was called — it must NOT be!"


# ── Multiple posters ──────────────────────────────────────────────────────────

class TestMultiplePosters:
    def test_multiple_posters_all_exported(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, poster_count=3)
        result = export_prints(run_dir, sizes=["2x3", "A4"])
        assert len(result.poster_results) == 3
        for pr in result.poster_results:
            assert len(pr.exports) == 2
            assert not pr.failed

    def test_partial_failure_does_not_stop_other_posters(self, tmp_path):
        """A missing final.png for one poster should not abort the others."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        images_dir = run_dir / "images"
        images_dir.mkdir()

        # poster_01: valid
        p1 = images_dir / "poster_01"
        p1.mkdir()
        _make_final_png(p1)

        # poster_02: missing final.png
        p2 = images_dir / "poster_02"
        p2.mkdir()

        # poster_03: valid
        p3 = images_dir / "poster_03"
        p3.mkdir()
        _make_final_png(p3)

        result = export_prints(run_dir, sizes=["2x3"])
        assert len(result.poster_results) == 3
        assert len(result.poster_results[0].exports) == 1
        assert len(result.poster_results[1].failed) == 1
        assert len(result.poster_results[2].exports) == 1
