#!/usr/bin/env python3
"""
Stage 11.1 — Print Export CLI

Export poster images from a production run to standard print sizes at 300 DPI.

Usage examples:
    python scripts/export_prints.py <run_dir> --all
    python scripts/export_prints.py <run_dir> --sizes 2x3 4x5 11x14
    python scripts/export_prints.py <run_dir> --sizes A4 A3 --format jpg
    python scripts/export_prints.py <run_dir> --sizes 2x3 --crop fill
    python scripts/export_prints.py <run_dir> --sizes 4x5 --crop pad --background "#F5F0E8"
    python scripts/export_prints.py <run_dir> --all --upscale
    python scripts/export_prints.py <run_dir> --all --overwrite
    python scripts/export_prints.py <run_dir> --all --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Add project root to sys.path so imports work regardless of CWD
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.print_export import PRINT_SIZES, export_prints


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_prints",
        description="Export poster images to standard print sizes at 300 DPI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "run_dir",
        help="Path to the production run directory.",
    )

    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--all",
        action="store_true",
        help="Export all supported print sizes.",
    )
    size_group.add_argument(
        "--sizes",
        nargs="+",
        metavar="SIZE",
        help=(
            "Specific size names to export "
            f"(supported: {', '.join(sorted(PRINT_SIZES.keys()))})."
        ),
    )

    parser.add_argument(
        "--crop",
        choices=["fit", "fill", "pad"],
        default="fit",
        dest="crop_mode",
        help=(
            "Crop mode: "
            "fit=fit artwork within target with padding (default); "
            "fill=cover target fully with center crop; "
            "pad=same as fit with explicit background canvas."
        ),
    )

    parser.add_argument(
        "--format",
        choices=["png", "jpg"],
        default="png",
        dest="output_format",
        help="Output image format (default: png).",
    )

    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Allow LANCZOS upscaling when source is smaller than target.",
    )

    parser.add_argument(
        "--background",
        default="#FFFFFF",
        dest="background_color",
        metavar="HEX",
        help="Background/padding color as #RRGGBB hex (default: #FFFFFF).",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing exported files.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output machine-readable JSON to stdout instead of human-readable summary.",
    )

    return parser


def _human_summary(result) -> str:
    lines = []
    lines.append(f"Export complete — {result.run_dir}")
    lines.append(f"  Created at : {result.created_at}")
    lines.append(f"  Crop mode  : {result.crop_mode}")
    lines.append(f"  Format     : {result.output_format}")
    lines.append(f"  Upscale    : {result.upscale}")
    lines.append(f"  Sizes      : {', '.join(result.requested_sizes)}")
    lines.append("")

    total_exports = 0
    total_failed = 0

    for pr in result.poster_results:
        n_ok = len(pr.exports)
        n_fail = len(pr.failed)
        total_exports += n_ok
        total_failed += n_fail
        status = "OK" if n_fail == 0 else f"{n_fail} FAILED"
        lines.append(f"  {pr.poster_id}: {n_ok} exports, {status}")
        for rec in pr.exports:
            lines.append(f"    [{rec.size}] {rec.output_path}")
            for w in rec.warnings:
                lines.append(f"      WARNING: {w}")
        for fail in pr.failed:
            lines.append(f"    [FAILED] {fail['size']}: {fail['error']}")

    lines.append("")
    lines.append(f"  Total: {total_exports} exported, {total_failed} failed")

    if result.warnings:
        lines.append("")
        lines.append("  Global warnings:")
        for w in result.warnings:
            lines.append(f"    {w}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Require --all or --sizes
    if not args.all and not args.sizes:
        parser.print_help()
        print(
            "\nerror: one of the following arguments is required: --all / --sizes",
            file=sys.stderr,
        )
        return 1

    sizes = None if args.all else args.sizes

    try:
        result = export_prints(
            args.run_dir,
            sizes=sizes,
            crop_mode=args.crop_mode,
            output_format=args.output_format,
            upscale=args.upscale,
            background_color=args.background_color,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        from dataclasses import asdict as _asdict
        print(json.dumps(_asdict(result), indent=2))
    else:
        print(_human_summary(result))

    # Non-zero exit if any exports failed
    total_failed = sum(len(pr.failed) for pr in result.poster_results)
    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
