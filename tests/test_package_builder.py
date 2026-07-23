"""
Tests for Stage 11.2 / 11.3 — Package Builder & ZIP Export.

All tests use real file I/O (no mocking of stdlib).
No network calls. run_production() is never imported or called.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from agent.package_builder import (
    PackageManifest,
    PackageResult,
    build_package,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_png_bytes(width: int = 10, height: int = 10) -> bytes:
    """Return a minimal valid PNG file as bytes (using raw PNG format)."""
    # Use stdlib only — no Pillow dependency in package_builder tests
    import struct
    import zlib

    def png_chunk(name: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", crc)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = png_chunk(b"IHDR", ihdr_data)

    # Build image data: rows of RGB pixels
    raw_rows = b""
    for _ in range(height):
        raw_rows += b"\x00"  # filter byte
        raw_rows += b"\xFF\x00\x00" * width  # red pixels
    compressed = zlib.compress(raw_rows)
    idat = png_chunk(b"IDAT", compressed)
    iend = png_chunk(b"IEND", b"")

    return signature + ihdr + idat + iend


def _make_run_dir(tmp_path: Path) -> Path:
    """
    Create a minimal valid run directory with synthetic files.

    Structure:
        run_dir/
          images/poster_01/final.png   (real minimal PNG)
          images/poster_02/final.png
          exports/poster_01/A4/poster.png
          exports/poster_01/2x3/poster.png
          exports/poster_02/A4/poster.png
          exports/export_manifest.json
          mockups/poster_01/mockup_1.png
          listing/listing.json
          listing/description.txt
          manifest.json
          request.json
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    png_bytes = _make_png_bytes()

    # images/
    for i in (1, 2):
        p = run_dir / "images" / f"poster_{i:02d}"
        p.mkdir(parents=True)
        (p / "final.png").write_bytes(png_bytes)

    # exports/
    for poster in ("poster_01", "poster_02"):
        for size in ("A4",):
            d = run_dir / "exports" / poster / size
            d.mkdir(parents=True)
            (d / "poster.png").write_bytes(png_bytes)
    # also add 2x3 for poster_01
    d = run_dir / "exports" / "poster_01" / "2x3"
    d.mkdir(parents=True)
    (d / "poster.png").write_bytes(png_bytes)

    (run_dir / "exports" / "export_manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "poster_count": 2}), encoding="utf-8"
    )

    # mockups/
    (run_dir / "mockups" / "poster_01").mkdir(parents=True)
    (run_dir / "mockups" / "poster_01" / "mockup_1.png").write_bytes(png_bytes)

    # listing/
    (run_dir / "listing").mkdir()
    (run_dir / "listing" / "listing.json").write_text(
        json.dumps({"title": "My Collection"}), encoding="utf-8"
    )
    (run_dir / "listing" / "description.txt").write_text("Buy this art!", encoding="utf-8")

    # top-level metadata
    (run_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "query": "cozy art"}), encoding="utf-8"
    )
    (run_dir / "request.json").write_text(
        json.dumps({"query": "cozy art", "collection_size": 2}), encoding="utf-8"
    )

    return run_dir


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(pkg_path: str) -> dict:
    return json.loads((Path(pkg_path) / "package_manifest.json").read_text(encoding="utf-8"))


# ── Basic creation ─────────────────────────────────────────────────────────────

class TestBasicCreation:
    def test_package_dir_created(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert Path(result.package_path).is_dir()

    def test_package_naming_format(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        pkg_name = Path(result.package_path).name
        import re
        assert re.fullmatch(r"package_\d{8}_\d{6}", pkg_name), \
            f"Package name format wrong: {pkg_name}"

    def test_correct_subdirectory_structure(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        pkg = Path(result.package_path)
        assert (pkg / "Printable Files").is_dir()
        assert (pkg / "Preview Images").is_dir()
        assert (pkg / "Mockups").is_dir()
        assert (pkg / "Listing").is_dir()
        assert (pkg / "Metadata").is_dir()

    def test_readme_and_license_present(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        pkg = Path(result.package_path)
        assert (pkg / "README.txt").exists()
        assert (pkg / "LICENSE.txt").exists()

    def test_package_manifest_present(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert (Path(result.package_path) / "package_manifest.json").exists()

    def test_success_is_true(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert result.success is True

    def test_result_package_path_is_absolute(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert Path(result.package_path).is_absolute()


# ── Printable Files section ────────────────────────────────────────────────────

class TestPrintableFiles:
    def test_hierarchy_preserved(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        pkg = Path(result.package_path)
        assert (pkg / "Printable Files" / "poster_01" / "A4" / "poster.png").exists()
        assert (pkg / "Printable Files" / "poster_01" / "2x3" / "poster.png").exists()
        assert (pkg / "Printable Files" / "poster_02" / "A4" / "poster.png").exists()

    def test_all_export_files_copied(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert result.manifest.print_export_count == 3  # A4×2 + 2x3×1

    def test_source_exports_unchanged_after_copy(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        src = run_dir / "exports" / "poster_01" / "A4" / "poster.png"
        before = src.read_bytes()
        build_package(run_dir)
        assert src.read_bytes() == before

    def test_include_prints_false_skips_section(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, include_prints=False)
        assert not (Path(result.package_path) / "Printable Files").exists()


# ── Preview Images section ─────────────────────────────────────────────────────

class TestPreviewImages:
    def test_preview_images_renamed(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        pkg = Path(result.package_path)
        assert (pkg / "Preview Images" / "poster_01.png").exists()
        assert (pkg / "Preview Images" / "poster_02.png").exists()

    def test_source_images_unchanged(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        src = run_dir / "images" / "poster_01" / "final.png"
        before = src.read_bytes()
        build_package(run_dir)
        assert src.read_bytes() == before

    def test_no_final_dot_png_in_preview(self, tmp_path):
        """Preview images should be named poster_XX.png, not final.png."""
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        preview_dir = Path(result.package_path) / "Preview Images"
        names = [f.name for f in preview_dir.iterdir() if f.is_file()]
        assert "final.png" not in names


# ── Mockups section ────────────────────────────────────────────────────────────

class TestMockups:
    def test_mockup_files_copied(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        pkg = Path(result.package_path)
        assert (pkg / "Mockups" / "poster_01" / "mockup_1.png").exists()

    def test_missing_mockups_dir_warns_not_fail(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        # Remove mockups dir
        import shutil
        shutil.rmtree(run_dir / "mockups")
        result = build_package(run_dir, include_mockups=True)
        assert result.success
        assert any("mockup" in w.lower() for w in result.warnings)

    def test_include_mockups_false_skips_section(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, include_mockups=False)
        assert not (Path(result.package_path) / "Mockups").exists()

    def test_mockup_count_in_manifest(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert result.manifest.mockup_count == 1


# ── Listing section ────────────────────────────────────────────────────────────

class TestListing:
    def test_listing_files_copied(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        pkg = Path(result.package_path)
        assert (pkg / "Listing" / "listing.json").exists()
        assert (pkg / "Listing" / "description.txt").exists()

    def test_missing_listing_dir_raises_when_include_listing(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        import shutil
        shutil.rmtree(run_dir / "listing")
        with pytest.raises(ValueError, match="listing"):
            build_package(run_dir, include_listing=True)

    def test_include_listing_false_skips_section(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, include_listing=False)
        assert not (Path(result.package_path) / "Listing").exists()

    def test_listing_files_recorded_in_manifest(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert len(result.manifest.listing_files) >= 2


# ── Metadata section ───────────────────────────────────────────────────────────

class TestMetadata:
    def test_manifest_json_copied(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert (Path(result.package_path) / "Metadata" / "manifest.json").exists()

    def test_request_json_copied(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert (Path(result.package_path) / "Metadata" / "request.json").exists()

    def test_export_manifest_copied(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert (Path(result.package_path) / "Metadata" / "export_manifest.json").exists()

    def test_sha256_recorded_for_metadata_files(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        meta_files = result.manifest.metadata_files
        assert len(meta_files) > 0
        # Every metadata file path should appear in manifest.files
        paths_in_manifest = {f["path"] for f in result.manifest.files}
        for mf in meta_files:
            assert mf in paths_in_manifest

    def test_include_metadata_false_skips_section(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, include_metadata=False)
        assert not (Path(result.package_path) / "Metadata").exists()


# ── README.txt ─────────────────────────────────────────────────────────────────

class TestReadme:
    def test_readme_exists(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert (Path(result.package_path) / "README.txt").exists()

    def test_readme_contains_thank_you(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        content = (Path(result.package_path) / "README.txt").read_text(encoding="utf-8")
        assert "Thank you" in content

    def test_readme_contains_digital_download(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        content = (Path(result.package_path) / "README.txt").read_text(encoding="utf-8")
        assert "DIGITAL DOWNLOAD" in content

    def test_readme_contains_printing_recommendations(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        content = (Path(result.package_path) / "README.txt").read_text(encoding="utf-8")
        assert "PRINTING RECOMMENDATIONS" in content


# ── LICENSE.txt ────────────────────────────────────────────────────────────────

class TestLicense:
    def test_license_exists(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert (Path(result.package_path) / "LICENSE.txt").exists()

    def test_license_contains_commercial_use(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        content = (Path(result.package_path) / "LICENSE.txt").read_text(encoding="utf-8")
        assert "COMMERCIAL USE LICENSE" in content


# ── package_manifest.json ──────────────────────────────────────────────────────

class TestPackageManifest:
    def test_manifest_is_valid_json(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        manifest_path = Path(result.package_path) / "package_manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_manifest_schema_version(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        data = _read_manifest(result.package_path)
        assert data["schema_version"] == "1.0"

    def test_manifest_total_files_correct(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        data = _read_manifest(result.package_path)
        assert data["total_files"] > 0
        assert data["total_files"] == len(data["files"])

    def test_manifest_total_size_positive(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        data = _read_manifest(result.package_path)
        assert data["total_package_size_bytes"] > 0

    def test_manifest_sha256_for_every_file(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        data = _read_manifest(result.package_path)
        for entry in data["files"]:
            assert "sha256" in entry
            assert len(entry["sha256"]) == 64   # hex SHA256

    def test_manifest_zip_created_false_when_no_zip(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, create_zip=False)
        data = _read_manifest(result.package_path)
        assert data["zip_created"] is False
        assert data["zip_path"] is None

    def test_manifest_zip_created_true_when_zip(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, create_zip=True)
        data = _read_manifest(result.package_path)
        assert data["zip_created"] is True
        assert data["zip_path"] is not None

    def test_manifest_poster_count(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert result.manifest.poster_count == 2


# ── ZIP creation ───────────────────────────────────────────────────────────────

class TestZIPCreation:
    def test_zip_file_exists(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, create_zip=True)
        assert result.zip_path is not None
        assert Path(result.zip_path).exists()

    def test_zip_is_beside_package_folder(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, create_zip=True)
        pkg_parent = Path(result.package_path).parent
        zip_parent = Path(result.zip_path).parent
        assert pkg_parent == zip_parent

    def test_zip_contains_no_absolute_paths(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, create_zip=True)
        with zipfile.ZipFile(result.zip_path) as zf:
            for name in zf.namelist():
                assert not name.startswith("/"), f"Absolute path in ZIP: {name}"

    def test_zip_contains_all_package_files(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, create_zip=True)
        pkg = Path(result.package_path)
        # Gather all files in package
        pkg_files = {str(f.relative_to(pkg)) for f in pkg.rglob("*") if f.is_file()}
        with zipfile.ZipFile(result.zip_path) as zf:
            zip_files = set(zf.namelist())
        assert pkg_files == zip_files

    def test_zip_can_be_extracted(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, create_zip=True)
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        with zipfile.ZipFile(result.zip_path) as zf:
            zf.extractall(extract_dir)
        assert (extract_dir / "README.txt").exists()

    def test_zip_path_in_manifest(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, create_zip=True)
        assert result.manifest.zip_path == result.zip_path


# ── ZIP disabled ───────────────────────────────────────────────────────────────

class TestZIPDisabled:
    def test_no_zip_created_by_default(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, create_zip=False)
        assert result.zip_path is None
        # No zip files should be present
        packages_dir = Path(result.package_path).parent
        zip_files = list(packages_dir.glob("*.zip"))
        assert not zip_files


# ── SHA256 ─────────────────────────────────────────────────────────────────────

class TestSHA256:
    def test_sha256_matches_actual_file(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        # Pick any file in the manifest (skip package_manifest.json itself —
        # its hash in the files list is stale by the time it's fully written,
        # because the manifest includes self-reference entries).
        for entry in result.manifest.files:
            if entry["path"] == "package_manifest.json":
                continue
            full_path = Path(result.package_path) / entry["path"]
            if full_path.exists():
                actual = _sha256(full_path)
                assert actual == entry["sha256"], \
                    f"SHA256 mismatch for {entry['path']}"

    def test_different_content_different_sha256(self, tmp_path):
        run_dir1 = tmp_path / "run1"
        run_dir2 = tmp_path / "run2"
        run_dir1.mkdir()
        run_dir2.mkdir()

        # Different PNG content
        png_bytes_a = _make_png_bytes(width=10, height=10)
        png_bytes_b = _make_png_bytes(width=20, height=20)

        f1 = run_dir1 / "test.png"
        f2 = run_dir2 / "test.png"
        f1.write_bytes(png_bytes_a)
        f2.write_bytes(png_bytes_b)

        sha1 = _sha256(f1)
        sha2 = _sha256(f2)
        assert sha1 != sha2


# ── Overwrite protection ───────────────────────────────────────────────────────

class TestOverwrite:
    def test_overwrite_false_existing_package_raises(self, tmp_path, monkeypatch):
        """Patch the timestamp so two calls produce the same package name."""
        run_dir = _make_run_dir(tmp_path)

        # Patch datetime inside package_builder to return a fixed timestamp
        import agent.package_builder as pb
        from datetime import datetime, timezone

        class _FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        monkeypatch.setattr(pb, "datetime", _FixedDatetime)

        # First build creates the dir
        result1 = build_package(run_dir)
        # Second build with same timestamp → dir already exists → raises
        with pytest.raises(ValueError, match="already exists"):
            build_package(run_dir, overwrite=False)

    def test_overwrite_true_rebuilds(self, tmp_path, monkeypatch):
        """Patch timestamp so both calls produce the same name; overwrite=True succeeds."""
        run_dir = _make_run_dir(tmp_path)

        import agent.package_builder as pb
        from datetime import datetime, timezone

        class _FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        monkeypatch.setattr(pb, "datetime", _FixedDatetime)

        result1 = build_package(run_dir)
        result2 = build_package(run_dir, overwrite=True)
        assert result2.success

    def test_source_files_never_touched(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        src = run_dir / "images" / "poster_01" / "final.png"
        before = src.read_bytes()
        build_package(run_dir)
        assert src.read_bytes() == before


# ── Optional sections ──────────────────────────────────────────────────────────

class TestOptionalSections:
    def test_include_prints_false_no_printable_files_dir(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, include_prints=False)
        assert not (Path(result.package_path) / "Printable Files").exists()

    def test_include_mockups_false_no_mockups_dir(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, include_mockups=False)
        assert not (Path(result.package_path) / "Mockups").exists()

    def test_include_listing_false_no_listing_dir(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, include_listing=False)
        assert not (Path(result.package_path) / "Listing").exists()

    def test_include_metadata_false_no_metadata_dir(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir, include_metadata=False)
        assert not (Path(result.package_path) / "Metadata").exists()

    def test_all_false_still_creates_readme_license_manifest(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(
            run_dir,
            include_prints=False,
            include_mockups=False,
            include_listing=False,
            include_metadata=False,
        )
        pkg = Path(result.package_path)
        assert (pkg / "README.txt").exists()
        assert (pkg / "LICENSE.txt").exists()
        assert (pkg / "package_manifest.json").exists()


# ── Validation errors ──────────────────────────────────────────────────────────

class TestValidation:
    def test_missing_run_dir_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, ValueError)):
            build_package(tmp_path / "nonexistent")

    def test_run_dir_is_file_raises(self, tmp_path):
        f = tmp_path / "notadir.txt"
        f.write_text("hello")
        with pytest.raises((ValueError, FileNotFoundError)):
            build_package(f)

    def test_missing_images_dir_raises(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        with pytest.raises(ValueError, match="images"):
            build_package(run_dir)

    def test_no_poster_dirs_raises(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "images").mkdir()
        with pytest.raises(ValueError):
            build_package(run_dir)

    def test_missing_exports_when_include_prints_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        import shutil
        shutil.rmtree(run_dir / "exports")
        with pytest.raises(ValueError, match="exports"):
            build_package(run_dir, include_prints=True)

    def test_missing_listing_when_include_listing_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        import shutil
        shutil.rmtree(run_dir / "listing")
        with pytest.raises(ValueError, match="listing"):
            build_package(run_dir, include_listing=True)


# ── Poster count ───────────────────────────────────────────────────────────────

class TestPosterCount:
    def test_poster_count_correct(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert result.manifest.poster_count == 2

    def test_poster_natural_sort_10_after_9(self, tmp_path):
        """poster_10 must come after poster_09 in Preview Images."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        images_dir = run_dir / "images"
        images_dir.mkdir()
        png_bytes = _make_png_bytes()
        for i in (1, 2, 9, 10):
            p = images_dir / f"poster_{i:02d}"
            p.mkdir()
            (p / "final.png").write_bytes(png_bytes)

        # Create minimal exports and listing dirs
        exports_dir = run_dir / "exports"
        exports_dir.mkdir()
        d = exports_dir / "poster_01" / "A4"
        d.mkdir(parents=True)
        (d / "poster.png").write_bytes(png_bytes)
        (exports_dir / "export_manifest.json").write_text("{}", encoding="utf-8")
        (run_dir / "listing").mkdir()
        (run_dir / "listing" / "listing.json").write_text("{}", encoding="utf-8")
        (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (run_dir / "request.json").write_text("{}", encoding="utf-8")

        result = build_package(run_dir)
        preview_dir = Path(result.package_path) / "Preview Images"
        names = sorted([f.stem for f in preview_dir.iterdir() if f.is_file()])
        assert names == ["poster_01", "poster_02", "poster_09", "poster_10"]

    def test_unrelated_dirs_ignored(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        # Add unrelated dirs to images/
        (run_dir / "images" / "thumbnails").mkdir()
        (run_dir / "images" / "notes.txt").write_text("ignore")
        result = build_package(run_dir)
        assert result.manifest.poster_count == 2


# ── Source file integrity ──────────────────────────────────────────────────────

class TestSourceIntegrity:
    def test_source_bytes_unchanged_after_build(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        sources = {
            run_dir / "images" / "poster_01" / "final.png",
            run_dir / "exports" / "poster_01" / "A4" / "poster.png",
            run_dir / "listing" / "listing.json",
            run_dir / "manifest.json",
        }
        before = {p: p.read_bytes() for p in sources}
        build_package(run_dir)
        for p, data in before.items():
            assert p.read_bytes() == data, f"Source file modified: {p}"


# ── Atomic writes ──────────────────────────────────────────────────────────────

class TestAtomicWrites:
    def test_no_tmp_files_after_success(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        pkg = Path(result.package_path)
        tmp_files = list(pkg.rglob("*.tmp"))
        assert not tmp_files, f"Leftover .tmp files: {tmp_files}"

    def test_package_manifest_exists_after_build(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        result = build_package(run_dir)
        assert (Path(result.package_path) / "package_manifest.json").exists()


# ── CLI tests ──────────────────────────────────────────────────────────────────

def _run_cli(args: list[str]) -> tuple[int, str, str]:
    script = str(Path(__file__).parent.parent / "scripts" / "build_package.py")
    cmd = [sys.executable, script] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


class TestCLI:
    def test_cli_basic_build(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        rc, stdout, stderr = _run_cli([str(run_dir)])
        assert rc == 0, f"CLI failed:\nstdout: {stdout}\nstderr: {stderr}"

    def test_cli_zip_creates_zip(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        rc, stdout, stderr = _run_cli([str(run_dir), "--zip"])
        assert rc == 0, f"CLI failed: {stderr}"
        # A zip should be created
        packages_dir = run_dir / "packages"
        zips = list(packages_dir.glob("*.zip"))
        assert zips, "No ZIP file was created"

    def test_cli_no_mockups_skips_section(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        rc, stdout, stderr = _run_cli([str(run_dir), "--no-mockups"])
        assert rc == 0, f"CLI failed: {stderr}"
        # Find the package dir
        packages_dir = run_dir / "packages"
        pkg_dirs = [d for d in packages_dir.iterdir() if d.is_dir()]
        assert pkg_dirs
        assert not (pkg_dirs[0] / "Mockups").exists()

    def test_cli_json_output_valid(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        rc, stdout, stderr = _run_cli([str(run_dir), "--json"])
        assert rc == 0, f"CLI failed: {stderr}"
        data = json.loads(stdout)
        assert "package_path" in data
        assert data["success"] is True

    def test_cli_missing_run_dir_exits_nonzero(self, tmp_path):
        rc, stdout, stderr = _run_cli([str(tmp_path / "nonexistent")])
        assert rc != 0

    def test_cli_no_run_dir_exits_nonzero(self, tmp_path):
        rc, stdout, stderr = _run_cli([])
        assert rc != 0

    def test_cli_json_output_with_zip(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        rc, stdout, stderr = _run_cli([str(run_dir), "--zip", "--json"])
        assert rc == 0, f"CLI failed: {stderr}"
        data = json.loads(stdout)
        assert data["zip_path"] is not None


# ── No forbidden imports ───────────────────────────────────────────────────────

class TestNoForbiddenImports:
    def test_run_production_not_in_package_builder(self):
        import agent.package_builder as pb
        src = Path(pb.__file__).read_text(encoding="utf-8")
        assert "run_production" not in src
        assert "production_orchestrator" not in src

    def test_no_network_imports(self):
        import agent.package_builder as pb
        src = Path(pb.__file__).read_text(encoding="utf-8")
        for banned in ("import requests", "import httpx", "import openai", "import anthropic"):
            assert banned not in src, f"Banned import found: {banned}"

    def test_no_socket_calls(self, tmp_path, monkeypatch):
        import socket

        def _no_connect(self, *args, **kwargs):
            raise AssertionError("build_package must not make network connections!")

        monkeypatch.setattr(socket.socket, "connect", _no_connect)

        run_dir = _make_run_dir(tmp_path)
        build_package(run_dir)   # must not raise

    def test_run_production_never_called(self, tmp_path, monkeypatch):
        called = []

        def _fake_run_production(*args, **kwargs):
            called.append(True)

        import agent.production_orchestrator as orch
        monkeypatch.setattr(orch, "run_production", _fake_run_production)

        run_dir = _make_run_dir(tmp_path)
        build_package(run_dir)
        assert not called
