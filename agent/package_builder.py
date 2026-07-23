"""
Stage 11.2 / 11.3 — Package Builder & ZIP Export

Assembles a production run's outputs into a clean, customer-ready folder
structure and (optionally) a ZIP archive.

No network calls, no AI, no API calls — stdlib + shutil + zipfile only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class PackageFile:
    relative_path: str   # relative to package root
    source_path: str     # absolute original
    size_bytes: int
    sha256: str


@dataclass
class PackageSection:
    name: str            # e.g. "Printable Files"
    files: list[PackageFile]
    warnings: list[str]


@dataclass
class PackageManifest:
    schema_version: str  # "1.0"
    created_at: str      # ISO8601
    run_dir: str
    package_path: str
    zip_created: bool
    zip_path: str | None
    included_sections: list[str]
    poster_count: int
    print_export_count: int
    mockup_count: int
    listing_files: list[str]
    metadata_files: list[str]
    warnings: list[str]
    total_files: int
    total_package_size_bytes: int
    files: list[dict]    # sha256 records: {path, sha256, size_bytes}


@dataclass
class PackageResult:
    package_path: str    # absolute path to package dir
    zip_path: str | None
    manifest: PackageManifest
    warnings: list[str]
    success: bool


# ── Internal helpers ───────────────────────────────────────────────────────────

def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _natural_sort_key(name: str):
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


def _copy_file(src: Path, dest: Path) -> PackageFile:
    """Copy src to dest using shutil.copy2. Return PackageFile."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    size = dest.stat().st_size
    sha = _sha256_of_file(dest)
    return PackageFile(
        relative_path="",   # filled in by caller
        source_path=str(src.resolve()),
        size_bytes=size,
        sha256=sha,
    )


def _atomic_write_text(path: Path, content: str) -> None:
    """Write content to path atomically via .tmp + os.replace()."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write bytes to path atomically via .tmp + os.replace()."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_bytes(content)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


# ── Content generators ─────────────────────────────────────────────────────────

def _make_readme(included_sections: list[str], sizes: list[str], collection_name: str | None) -> str:
    name_line = ""
    if collection_name:
        name_line = f"\n{collection_name}\n"

    sizes_str = "\n".join(f"- {s}" for s in sizes) if sizes else "- (no print sizes exported)"

    sections_text = []
    if "Printable Files" in included_sections:
        sections_text.append("Printable Files/  - High-resolution print-ready artwork at 300 DPI")
    if "Preview Images" in included_sections:
        sections_text.append("Preview Images/   - Master artwork files for preview")
    if "Mockups" in included_sections:
        sections_text.append("Mockups/          - Room and frame preview images (if included)")
    if "Listing" in included_sections:
        sections_text.append("Listing/          - Product listing text and SEO keywords")
    if "Metadata" in included_sections:
        sections_text.append("Metadata/         - Technical metadata for this download")
    sections_block = "\n".join(sections_text) if sections_text else "(no sections included)"

    return f"""Thank you for your purchase!
============================={name_line}

INCLUDED FOLDERS
----------------
{sections_block}

SUPPORTED PRINT SIZES
----------------------
{sizes_str}

PRINTING RECOMMENDATIONS
-------------------------
Professional Printing:
- Upload the file from "Printable Files/" matching your desired print size
- Select "Actual size" or "100%" scaling — do not let the printer resize
- Use sRGB color profile
- Request matte or lustre finish for best results

Home Printing:
- Use the highest quality setting on your printer
- Use photo-quality paper for best results
- Select "Fit to page" only if your printer cannot handle the exact size

IMPORTANT — DIGITAL DOWNLOAD
------------------------------
This is a digital download only. No physical item will be shipped.

CONTACT
--------
If you have any questions about your purchase, please contact us through Etsy.
"""


_LICENSE_CONTENT = """\
COMMERCIAL USE LICENSE
========================

This digital artwork is licensed for personal and commercial use.

PERMITTED:
- Print for personal use
- Print for resale (printed physical copies)
- Use in personal projects

NOT PERMITTED:
- Resale of the digital files
- Redistribution of the digital files
- Use in AI training datasets

For licensing questions, contact us through Etsy.
"""


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate_run_dir(run_dir: Path, include_prints: bool, include_listing: bool) -> None:
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise ValueError(f"run_dir is not a directory: {run_dir}")

    images_dir = run_dir / "images"
    if not images_dir.exists() or not images_dir.is_dir():
        raise ValueError(f"images/ directory not found in run_dir: {images_dir}")

    poster_dirs = _discover_posters(images_dir)
    if not poster_dirs:
        raise ValueError(
            f"No poster_XX directories found in {images_dir}. "
            "Run production first to generate images."
        )

    # Check that at least one poster has final.png
    has_final = any((d / "final.png").exists() for d in poster_dirs)
    if not has_final:
        raise ValueError(
            f"No poster_XX/final.png found in {images_dir}. "
            "Run production first to generate images."
        )

    if include_prints:
        exports_dir = run_dir / "exports"
        if not exports_dir.exists() or not exports_dir.is_dir():
            raise ValueError(
                f"exports/ directory not found in run_dir: {exports_dir}. "
                "Run export_prints() first."
            )
        # Check at least one export file
        export_files = [
            f for f in exports_dir.rglob("poster.*")
            if f.is_file() and f.suffix in (".png", ".jpg")
        ]
        if not export_files:
            raise ValueError(
                f"No export files (poster.png / poster.jpg) found under {exports_dir}. "
                "Run export_prints() first."
            )

    if include_listing:
        listing_dir = run_dir / "listing"
        if not listing_dir.exists() or not listing_dir.is_dir():
            raise ValueError(
                f"listing/ directory not found in run_dir: {listing_dir}. "
                "Run listing generation first or pass include_listing=False."
            )
        listing_files = [f for f in listing_dir.iterdir() if f.is_file()]
        if not listing_files:
            raise ValueError(
                f"listing/ directory exists but contains no files: {listing_dir}."
            )


# ── Section builders ───────────────────────────────────────────────────────────

def _build_printable_files(
    run_dir: Path,
    pkg_dir: Path,
) -> PackageSection:
    """Copy exports/poster_XX/<size>/poster.{png,jpg} → Printable Files/poster_XX/<size>/poster.*"""
    section_dir = pkg_dir / "Printable Files"
    exports_dir = run_dir / "exports"
    files: list[PackageFile] = []
    warnings: list[str] = []

    export_files = [
        f for f in exports_dir.rglob("poster.*")
        if f.is_file() and f.suffix in (".png", ".jpg")
    ]
    export_files.sort(key=lambda p: str(p))

    for src in export_files:
        # src: exports/poster_01/A4/poster.png
        try:
            rel_to_exports = src.relative_to(exports_dir)
        except ValueError:
            continue
        dest_rel = Path("Printable Files") / rel_to_exports
        dest = pkg_dir / dest_rel
        pf = _copy_file(src, dest)
        pf.relative_path = str(dest_rel)
        files.append(pf)

    return PackageSection(name="Printable Files", files=files, warnings=warnings)


def _build_preview_images(
    run_dir: Path,
    pkg_dir: Path,
) -> PackageSection:
    """Copy images/poster_XX/final.png → Preview Images/poster_XX.png"""
    section_dir = pkg_dir / "Preview Images"
    section_dir.mkdir(parents=True, exist_ok=True)
    images_dir = run_dir / "images"
    files: list[PackageFile] = []
    warnings: list[str] = []

    poster_dirs = _discover_posters(images_dir)
    for poster_dir in poster_dirs:
        final_png = poster_dir / "final.png"
        if not final_png.exists():
            warnings.append(f"Missing final.png for {poster_dir.name} — skipped from Preview Images")
            continue
        # Zero-pad poster id for consistent output dir naming
        match = re.fullmatch(r"poster_(\d+)", poster_dir.name)
        padded = f"poster_{int(match.group(1)):02d}" if match else poster_dir.name
        dest_name = f"{padded}.png"
        dest_rel = Path("Preview Images") / dest_name
        dest = pkg_dir / dest_rel
        pf = _copy_file(final_png, dest)
        pf.relative_path = str(dest_rel)
        files.append(pf)

    return PackageSection(name="Preview Images", files=files, warnings=warnings)


def _build_mockups(
    run_dir: Path,
    pkg_dir: Path,
) -> PackageSection:
    """Copy mockups/ contents → Mockups/"""
    mockups_dir = run_dir / "mockups"
    files: list[PackageFile] = []
    warnings: list[str] = []

    if not mockups_dir.exists() or not mockups_dir.is_dir():
        warnings.append("mockups/ directory not found — Mockups section is empty")
        return PackageSection(name="Mockups", files=files, warnings=warnings)

    all_files = sorted(mockups_dir.rglob("*"))
    for src in all_files:
        if not src.is_file():
            continue
        rel = src.relative_to(mockups_dir)
        dest_rel = Path("Mockups") / rel
        dest = pkg_dir / dest_rel
        pf = _copy_file(src, dest)
        pf.relative_path = str(dest_rel)
        files.append(pf)

    return PackageSection(name="Mockups", files=files, warnings=warnings)


def _build_listing(
    run_dir: Path,
    pkg_dir: Path,
) -> PackageSection:
    """Copy listing/ contents → Listing/"""
    listing_dir = run_dir / "listing"
    files: list[PackageFile] = []
    warnings: list[str] = []

    if not listing_dir.exists() or not listing_dir.is_dir():
        warnings.append("listing/ directory not found — Listing section is empty")
        return PackageSection(name="Listing", files=files, warnings=warnings)

    all_files = sorted(listing_dir.rglob("*"))
    for src in all_files:
        if not src.is_file():
            continue
        rel = src.relative_to(listing_dir)
        dest_rel = Path("Listing") / rel
        dest = pkg_dir / dest_rel
        pf = _copy_file(src, dest)
        pf.relative_path = str(dest_rel)
        files.append(pf)

    return PackageSection(name="Listing", files=files, warnings=warnings)


def _build_metadata(
    run_dir: Path,
    pkg_dir: Path,
) -> PackageSection:
    """Copy manifest.json, request.json, export_manifest.json, and metadata.json files → Metadata/"""
    meta_dir = pkg_dir / "Metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    files: list[PackageFile] = []
    warnings: list[str] = []

    candidates = [
        run_dir / "manifest.json",
        run_dir / "request.json",
        run_dir / "exports" / "export_manifest.json",
    ]

    for src in candidates:
        if not src.exists():
            warnings.append(f"metadata file not found, skipped: {src.name}")
            continue
        dest_rel = Path("Metadata") / src.name
        dest = pkg_dir / dest_rel
        pf = _copy_file(src, dest)
        pf.relative_path = str(dest_rel)
        files.append(pf)

    # Any metadata.json files under exports/
    exports_dir = run_dir / "exports"
    if exports_dir.exists():
        for src in sorted(exports_dir.rglob("metadata.json")):
            if not src.is_file():
                continue
            try:
                rel = src.relative_to(exports_dir)
            except ValueError:
                continue
            dest_rel = Path("Metadata") / "exports" / rel
            dest = pkg_dir / dest_rel
            pf = _copy_file(src, dest)
            pf.relative_path = str(dest_rel)
            files.append(pf)

    return PackageSection(name="Metadata", files=files, warnings=warnings)


# ── ZIP builder ────────────────────────────────────────────────────────────────

def _build_zip(pkg_dir: Path, zip_path: Path) -> None:
    """Create a ZIP archive of pkg_dir at zip_path, storing relative paths only."""
    tmp_zip = zip_path.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fpath in sorted(pkg_dir.rglob("*")):
                if not fpath.is_file():
                    continue
                arcname = fpath.relative_to(pkg_dir)
                zf.write(fpath, arcname)
        os.replace(tmp_zip, zip_path)
    except Exception:
        if tmp_zip.exists():
            tmp_zip.unlink()
        raise


# ── Collection name helper ─────────────────────────────────────────────────────

def _read_collection_name(run_dir: Path) -> str | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Try common fields
        for key in ("collection_name", "name", "query", "run_name"):
            if key in data and isinstance(data[key], str) and data[key].strip():
                return data[key].strip()
    except Exception:
        pass
    return None


def _collect_print_sizes(run_dir: Path) -> list[str]:
    """Discover what print sizes are present in exports/."""
    exports_dir = run_dir / "exports"
    if not exports_dir.exists():
        return []
    sizes: set[str] = set()
    for p in exports_dir.rglob("poster.*"):
        if p.is_file() and p.suffix in (".png", ".jpg"):
            # parent is size dir
            sizes.add(p.parent.name)
    return sorted(sizes, key=_natural_sort_key)


# ── Public API ─────────────────────────────────────────────────────────────────

def build_package(
    run_dir: "Path | str",
    *,
    include_prints: bool = True,
    include_mockups: bool = True,
    include_listing: bool = True,
    include_metadata: bool = True,
    create_zip: bool = False,
    overwrite: bool = False,
) -> PackageResult:
    """
    Assemble a production run's outputs into a customer-ready package folder.

    Parameters
    ----------
    run_dir:
        Path to the production run directory.
    include_prints:
        Copy Printable Files (exports/) into the package.
    include_mockups:
        Copy Mockups into the package (warns but succeeds if missing).
    include_listing:
        Copy Listing files into the package.
    include_metadata:
        Copy Metadata (manifest.json, request.json, export_manifest.json) into package.
    create_zip:
        Create a ZIP archive beside the package folder.
    overwrite:
        If False (default), raises ValueError if the package dir already exists.
        If True, removes and recreates the package dir.

    Returns
    -------
    PackageResult
    """
    run_dir = Path(run_dir)

    # Validate
    _validate_run_dir(run_dir, include_prints=include_prints, include_listing=include_listing)

    # Create package directory name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pkg_name = f"package_{timestamp}"
    packages_dir = run_dir / "packages"
    packages_dir.mkdir(parents=True, exist_ok=True)
    pkg_dir = packages_dir / pkg_name
    zip_path = packages_dir / f"{pkg_name}.zip"

    # Overwrite check
    if pkg_dir.exists():
        if not overwrite:
            raise ValueError(
                f"Package directory already exists: {pkg_dir}. "
                "Use overwrite=True to replace it."
            )
        shutil.rmtree(pkg_dir)

    pkg_dir.mkdir(parents=True)

    all_warnings: list[str] = []
    included_sections: list[str] = []
    all_files: list[PackageFile] = []

    # Preview Images — always included (we validated images/ exists)
    preview_section = _build_preview_images(run_dir, pkg_dir)
    included_sections.append("Preview Images")
    all_files.extend(preview_section.files)
    all_warnings.extend(preview_section.warnings)

    # Printable Files
    if include_prints:
        prints_section = _build_printable_files(run_dir, pkg_dir)
        included_sections.append("Printable Files")
        all_files.extend(prints_section.files)
        all_warnings.extend(prints_section.warnings)
    else:
        prints_section = PackageSection(name="Printable Files", files=[], warnings=[])

    # Mockups
    if include_mockups:
        mockups_section = _build_mockups(run_dir, pkg_dir)
        included_sections.append("Mockups")
        all_files.extend(mockups_section.files)
        all_warnings.extend(mockups_section.warnings)
    else:
        mockups_section = PackageSection(name="Mockups", files=[], warnings=[])

    # Listing
    if include_listing:
        listing_section = _build_listing(run_dir, pkg_dir)
        included_sections.append("Listing")
        all_files.extend(listing_section.files)
        all_warnings.extend(listing_section.warnings)
    else:
        listing_section = PackageSection(name="Listing", files=[], warnings=[])

    # Metadata
    if include_metadata:
        metadata_section = _build_metadata(run_dir, pkg_dir)
        included_sections.append("Metadata")
        all_files.extend(metadata_section.files)
        all_warnings.extend(metadata_section.warnings)
    else:
        metadata_section = PackageSection(name="Metadata", files=[], warnings=[])

    # Poster count
    images_dir = run_dir / "images"
    poster_dirs = _discover_posters(images_dir)
    poster_count = len(poster_dirs)

    # Print export count
    print_export_count = len(prints_section.files)

    # Mockup count
    mockup_count = len(mockups_section.files)

    # Listing files
    listing_file_names = [pf.relative_path for pf in listing_section.files]

    # Metadata files
    metadata_file_names = [pf.relative_path for pf in metadata_section.files]

    # Collect print sizes for README
    sizes = _collect_print_sizes(run_dir)

    # Collect collection name
    collection_name = _read_collection_name(run_dir)

    # Write README.txt
    readme_content = _make_readme(included_sections, sizes, collection_name)
    readme_path = pkg_dir / "README.txt"
    _atomic_write_text(readme_path, readme_content)
    readme_sha = _sha256_of_file(readme_path)
    all_files.append(PackageFile(
        relative_path="README.txt",
        source_path=str(readme_path.resolve()),
        size_bytes=readme_path.stat().st_size,
        sha256=readme_sha,
    ))

    # Write LICENSE.txt
    license_path = pkg_dir / "LICENSE.txt"
    _atomic_write_text(license_path, _LICENSE_CONTENT)
    license_sha = _sha256_of_file(license_path)
    all_files.append(PackageFile(
        relative_path="LICENSE.txt",
        source_path=str(license_path.resolve()),
        size_bytes=license_path.stat().st_size,
        sha256=license_sha,
    ))

    # Build manifest
    total_size = sum(pf.size_bytes for pf in all_files)
    files_dicts = [
        {"path": pf.relative_path, "sha256": pf.sha256, "size_bytes": pf.size_bytes}
        for pf in all_files
    ]

    manifest = PackageManifest(
        schema_version="1.0",
        created_at=datetime.now(timezone.utc).isoformat(),
        run_dir=str(run_dir.resolve()),
        package_path=str(pkg_dir.resolve()),
        zip_created=False,  # updated below if zip created
        zip_path=None,
        included_sections=included_sections,
        poster_count=poster_count,
        print_export_count=print_export_count,
        mockup_count=mockup_count,
        listing_files=listing_file_names,
        metadata_files=metadata_file_names,
        warnings=all_warnings,
        total_files=len(all_files),
        total_package_size_bytes=total_size,
        files=files_dicts,
    )

    # Write package_manifest.json (before zip so it's included)
    manifest_path = pkg_dir / "package_manifest.json"
    _atomic_write_text(manifest_path, json.dumps(asdict(manifest), indent=2, ensure_ascii=False))

    # Update manifest to include itself
    manifest_sha = _sha256_of_file(manifest_path)
    manifest_size = manifest_path.stat().st_size
    manifest.files.append({
        "path": "package_manifest.json",
        "sha256": manifest_sha,
        "size_bytes": manifest_size,
    })
    manifest.total_files = len(manifest.files)
    manifest.total_package_size_bytes += manifest_size

    # Rewrite manifest with self-reference included
    _atomic_write_text(manifest_path, json.dumps(asdict(manifest), indent=2, ensure_ascii=False))

    # ZIP export
    actual_zip_path: str | None = None
    if create_zip:
        _build_zip(pkg_dir, zip_path)
        manifest.zip_created = True
        manifest.zip_path = str(zip_path.resolve())
        actual_zip_path = str(zip_path.resolve())
        # Update manifest on disk to reflect zip creation
        _atomic_write_text(manifest_path, json.dumps(asdict(manifest), indent=2, ensure_ascii=False))

    return PackageResult(
        package_path=str(pkg_dir.resolve()),
        zip_path=actual_zip_path,
        manifest=manifest,
        warnings=all_warnings,
        success=True,
    )
