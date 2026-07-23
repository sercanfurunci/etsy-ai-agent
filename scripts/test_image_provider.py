#!/usr/bin/env python3
"""
Manual CLI for testing image providers.

Usage examples:

  # Health check
  python3 scripts/test_image_provider.py --provider comfyui_sdxl --health
  python3 scripts/test_image_provider.py --provider comfyui_flux_schnell --health
  python3 scripts/test_image_provider.py --provider openai --health

  # Generate an image
  python3 scripts/test_image_provider.py \\
      --provider comfyui_sdxl \\
      --prompt "ukiyo-e crane over misty river" \\
      --output out.png

  # Generate with explicit seed
  python3 scripts/test_image_provider.py \\
      --provider comfyui_flux_schnell \\
      --prompt "ukiyo-e crane at sunrise" \\
      --output out.png --seed 42

  # Dump workflow JSON without any HTTP requests
  python3 scripts/test_image_provider.py \\
      --provider comfyui_sdxl \\
      --dump-workflow --prompt "test" --output /dev/null

  # Machine-readable JSON output
  python3 scripts/test_image_provider.py \\
      --provider comfyui_sdxl --health --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _json_out(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Test and debug image providers for etsy-ai-agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--provider",
        required=True,
        choices=["openai", "comfyui_sdxl", "comfyui_flux_schnell"],
        help="Which image provider to use.",
    )
    p.add_argument("--health", action="store_true", help="Run a health check only; no image generated.")
    p.add_argument("--prompt", help="Text prompt for image generation.")
    p.add_argument("--negative-prompt", default="", help="Negative prompt (SDXL only).")
    p.add_argument("--output", help="Output file path for the generated image.")
    p.add_argument("--width", type=int, help="Image width in pixels.")
    p.add_argument("--height", type=int, help="Image height in pixels.")
    p.add_argument("--seed", type=int, help="Random seed.")
    p.add_argument("--steps", type=int, help="Number of sampling steps.")
    p.add_argument("--cfg", type=float, help="CFG scale (SDXL).")
    p.add_argument("--guidance", type=float, help="Guidance scale (FLUX).")
    p.add_argument("--timeout", type=int, help="Override COMFYUI_TIMEOUT_SECONDS.")
    p.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print machine-readable JSON to stdout; log messages go to stderr.",
    )
    p.add_argument(
        "--dump-workflow",
        action="store_true",
        help=(
            "Print workflow JSON to stdout without making any HTTP requests. "
            "Requires --prompt. --output is ignored (use /dev/null to suppress)."
        ),
    )
    return p


def _build_overrides(args: argparse.Namespace) -> dict:
    """Build a config override dict from CLI flags."""
    overrides = {}
    if args.width:
        overrides["COMFYUI_SDXL_WIDTH"] = str(args.width)
        overrides["COMFYUI_FLUX_WIDTH"] = str(args.width)
    if args.height:
        overrides["COMFYUI_SDXL_HEIGHT"] = str(args.height)
        overrides["COMFYUI_FLUX_HEIGHT"] = str(args.height)
    if args.steps:
        overrides["COMFYUI_SDXL_STEPS"] = str(args.steps)
        overrides["COMFYUI_FLUX_STEPS"] = str(args.steps)
    if args.cfg:
        overrides["COMFYUI_SDXL_CFG"] = str(args.cfg)
    if args.guidance:
        overrides["COMFYUI_FLUX_GUIDANCE"] = str(args.guidance)
    if args.timeout:
        overrides["COMFYUI_TIMEOUT_SECONDS"] = str(args.timeout)
    return overrides


def _run_health(args: argparse.Namespace) -> int:
    if args.provider == "openai":
        # OpenAI doesn't have a health_check; just verify the key is present.
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            result = {"provider": "openai", "available": True, "message": "OPENAI_API_KEY is set."}
        else:
            result = {"provider": "openai", "available": False, "message": "OPENAI_API_KEY is not set."}
        if args.json_output:
            _json_out(result)
        else:
            status = "OK" if result["available"] else "FAIL"
            print(f"[{status}] {result['message']}")
        return 0 if result["available"] else 1

    from agent.providers.comfyui_provider import ComfyUISDXLProvider, ComfyUIFluxSchnellProvider

    overrides = _build_overrides(args)
    if args.provider == "comfyui_sdxl":
        provider = ComfyUISDXLProvider(config=overrides if overrides else None)
    else:
        provider = ComfyUIFluxSchnellProvider(config=overrides if overrides else None)

    if not args.json_output:
        _err(f"Checking health of {args.provider} ...")

    health = provider.health_check()

    if args.json_output:
        _json_out({
            "provider": health.provider,
            "available": health.available,
            "latency_ms": health.latency_ms,
            "message": health.message,
            "details": health.details,
        })
    else:
        status = "OK" if health.available else "FAIL"
        print(f"[{status}] {health.message}")
        if health.latency_ms is not None:
            print(f"      latency: {health.latency_ms:.1f} ms")

    return 0 if health.available else 1


def _run_dump_workflow(args: argparse.Namespace) -> int:
    if not args.prompt:
        _err("--dump-workflow requires --prompt")
        return 1

    import random as _random
    seed = args.seed if args.seed is not None else _random.randint(0, 2**32 - 1)
    overrides = _build_overrides(args)

    if args.provider == "openai":
        _err("openai provider does not have a ComfyUI workflow to dump.")
        return 1

    if args.provider == "comfyui_sdxl":
        from agent.providers.comfyui_provider import ComfyUISDXLProvider
        from agent.providers.comfyui_workflows import SDXL_OUTPUT_NODE_ID

        # Build with placeholder if checkpoint not set
        checkpoint = (
            overrides.get("COMFYUI_SDXL_CHECKPOINT")
            or os.environ.get("COMFYUI_SDXL_CHECKPOINT")
            or "placeholder.safetensors"
        )
        from agent.providers.comfyui_workflows import build_sdxl_workflow

        width = int(overrides.get("COMFYUI_SDXL_WIDTH") or os.environ.get("COMFYUI_SDXL_WIDTH") or 1024)
        height = int(overrides.get("COMFYUI_SDXL_HEIGHT") or os.environ.get("COMFYUI_SDXL_HEIGHT") or 1536)
        steps = int(overrides.get("COMFYUI_SDXL_STEPS") or os.environ.get("COMFYUI_SDXL_STEPS") or 30)
        cfg = float(overrides.get("COMFYUI_SDXL_CFG") or os.environ.get("COMFYUI_SDXL_CFG") or 7.0)
        vae = overrides.get("COMFYUI_SDXL_VAE") or os.environ.get("COMFYUI_SDXL_VAE") or None
        sampler = overrides.get("COMFYUI_SDXL_SAMPLER") or os.environ.get("COMFYUI_SDXL_SAMPLER") or "euler"
        scheduler = overrides.get("COMFYUI_SDXL_SCHEDULER") or os.environ.get("COMFYUI_SDXL_SCHEDULER") or "normal"

        workflow = build_sdxl_workflow(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt or "",
            checkpoint_name=checkpoint,
            vae_name=vae if vae else None,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
        )

    elif args.provider == "comfyui_flux_schnell":
        from agent.providers.comfyui_workflows import build_flux_schnell_workflow

        unet = overrides.get("COMFYUI_FLUX_UNET") or os.environ.get("COMFYUI_FLUX_UNET") or "flux.safetensors"
        clip_l = overrides.get("COMFYUI_FLUX_CLIP_L") or os.environ.get("COMFYUI_FLUX_CLIP_L") or "clip_l.safetensors"
        t5xxl = overrides.get("COMFYUI_FLUX_T5XXL") or os.environ.get("COMFYUI_FLUX_T5XXL") or "t5xxl.safetensors"
        vae = overrides.get("COMFYUI_FLUX_VAE") or os.environ.get("COMFYUI_FLUX_VAE") or "ae.safetensors"
        width = int(overrides.get("COMFYUI_FLUX_WIDTH") or os.environ.get("COMFYUI_FLUX_WIDTH") or 1024)
        height = int(overrides.get("COMFYUI_FLUX_HEIGHT") or os.environ.get("COMFYUI_FLUX_HEIGHT") or 1536)
        steps = int(overrides.get("COMFYUI_FLUX_STEPS") or os.environ.get("COMFYUI_FLUX_STEPS") or 4)
        guidance = float(overrides.get("COMFYUI_FLUX_GUIDANCE") or os.environ.get("COMFYUI_FLUX_GUIDANCE") or 3.5)

        workflow = build_flux_schnell_workflow(
            prompt=args.prompt,
            unet_name=unet,
            clip_l_name=clip_l,
            t5xxl_name=t5xxl,
            vae_name=vae,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            guidance=guidance,
        )
    else:
        _err(f"Unknown provider: {args.provider}")
        return 1

    print(json.dumps(workflow, indent=2, ensure_ascii=False))
    return 0


def _run_generate(args: argparse.Namespace) -> int:
    if not args.prompt:
        _err("Image generation requires --prompt")
        return 1
    if not args.output:
        _err("Image generation requires --output")
        return 1

    overrides = _build_overrides(args)

    if args.provider == "openai":
        from image.openai_provider import OpenAIImageProvider
        provider = OpenAIImageProvider()
    elif args.provider == "comfyui_sdxl":
        from agent.providers.comfyui_provider import ComfyUISDXLProvider
        provider = ComfyUISDXLProvider(config=overrides if overrides else None)
    elif args.provider == "comfyui_flux_schnell":
        from agent.providers.comfyui_provider import ComfyUIFluxSchnellProvider
        provider = ComfyUIFluxSchnellProvider(config=overrides if overrides else None)
    else:
        _err(f"Unknown provider: {args.provider}")
        return 1

    if not args.json_output:
        _err(f"Generating image with {args.provider} ...")

    usage_info: list[dict] = []
    t0 = time.monotonic()

    try:
        image_path = provider.generate(args.prompt, on_usage=usage_info.append)
    except Exception as exc:
        if args.json_output:
            _json_out({"success": False, "error": str(exc), "provider": args.provider})
        else:
            _err(f"Generation failed: {exc}")
        return 1

    elapsed = time.monotonic() - t0

    # Copy / validate output with Pillow, write atomically
    output_dest = Path(args.output)
    if str(output_dest) not in ("/dev/null", "nul"):
        try:
            from PIL import Image
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path)  # re-open after verify
        except ImportError:
            pass  # Pillow not installed — skip validation
        except Exception as exc:
            _err(f"Warning: image validation failed: {exc}")

        # Copy atomically
        src = Path(image_path)
        tmp = output_dest.with_suffix(".tmp")
        try:
            output_dest.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(src.read_bytes())
            os.replace(tmp, output_dest)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            if args.json_output:
                _json_out({"success": False, "error": f"Failed to write output: {exc}"})
            else:
                _err(f"Failed to write output: {exc}")
            return 1

    result = {
        "success": True,
        "provider": args.provider,
        "output": str(output_dest.resolve()) if str(output_dest) != "/dev/null" else image_path,
        "elapsed_seconds": round(elapsed, 2),
        "usage": usage_info[0] if usage_info else None,
    }

    if args.json_output:
        _json_out(result)
    else:
        print(f"[OK] Image saved to: {result['output']}")
        print(f"     Time: {elapsed:.1f}s")
        if usage_info:
            u = usage_info[0]
            print(f"     Size: {u.get('image_size', 'unknown')}")
            print(f"     API cost: ${u.get('api_cost', 0.0):.4f}")

    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.health:
        return _run_health(args)

    if args.dump_workflow:
        return _run_dump_workflow(args)

    return _run_generate(args)


if __name__ == "__main__":
    sys.exit(main())
