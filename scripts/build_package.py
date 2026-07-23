#!/usr/bin/env python3
"""
Stage 11.2 / 11.3 — Package Builder CLI

Assembles a production run's outputs into a customer-ready folder and
(optionally) a ZIP archive.

Usage:
    python3 scripts/build_package.py outputs/run_xxx
    python3 scripts/build_package.py outputs/run_xxx --zip
    python3 scripts/build_package.py outputs/run_xxx --overwrite
    python3 scripts/build_package.py outputs/run_xxx --no-mockups
    python3 scripts/build_package.py outputs/run_xxx --no-listing
    python3 scripts/build_package.py outputs/run_xxx --no-prints
    python3 scripts/build_package.py outputs/run_xxx --no-metadata
    python3 scripts/build_package.py outputs/run_xxx --json
    python3 scripts/build_package.py outputs/run_xxx --zip --overwrite --json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# Allow running from repo root without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.package_builder import build_package, PackageResult


def _result_to_dict(result: PackageResult) -> dict:
    """Serialize PackageResult to a JSON-serializable dict."""
    return dataclasses.asdict(result)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a customer-ready package from a production run directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        help="Path to the production run directory (e.g. outputs/run_xxx)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        default=False,
        help="Create a ZIP archive beside the package folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Allow overwriting an existing package directory",
    )
    parser.add_argument(
        "--no-prints",
        action="store_true",
        default=False,
        help="Skip the Printable Files section",
    )
    parser.add_argument(
        "--no-mockups",
        action="store_true",
        default=False,
        help="Skip the Mockups section",
    )
    parser.add_argument(
        "--no-listing",
        action="store_true",
        default=False,
        help="Skip the Listing section",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        default=False,
        help="Skip the Metadata section",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Output machine-readable JSON to stdout instead of human-readable summary",
    )

    args = parser.parse_args()

    try:
        result = build_package(
            args.run_dir,
            include_prints=not args.no_prints,
            include_mockups=not args.no_mockups,
            include_listing=not args.no_listing,
            include_metadata=not args.no_metadata,
            create_zip=args.zip,
            overwrite=args.overwrite,
        )
    except (ValueError, FileNotFoundError) as exc:
        if args.json_output:
            print(json.dumps({"success": False, "error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if args.json_output:
            print(json.dumps({"success": False, "error": str(exc)}))
        else:
            print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(json.dumps(_result_to_dict(result), indent=2, ensure_ascii=False))
        return 0

    # Human-readable summary
    m = result.manifest
    print()
    print("Package built successfully!")
    print("=" * 50)
    print(f"Package path : {result.package_path}")
    if result.zip_path:
        print(f"ZIP path     : {result.zip_path}")
    print(f"Sections     : {', '.join(m.included_sections)}")
    print(f"Total files  : {m.total_files}")
    print(f"Total size   : {_human_size(m.total_package_size_bytes)}")
    print(f"Posters      : {m.poster_count}")
    if m.print_export_count:
        print(f"Print exports: {m.print_export_count}")
    if m.mockup_count:
        print(f"Mockups      : {m.mockup_count}")
    if result.warnings:
        print()
        print("Warnings:")
        for w in result.warnings:
            print(f"  ! {w}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
