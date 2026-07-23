"""
Pure workflow builder functions for ComfyUI.

No network calls, no filesystem access, no model downloads.
All functions return JSON-serializable dicts suitable for POST /prompt.

FLUX Schnell note: FLUX does NOT use negative prompts. The negative_prompt
parameter is accepted in build_flux_schnell_workflow() for API uniformity
but is NOT wired into any conditioning node in the workflow.
"""

import math

# ── Output node IDs (SaveImage nodes) ─────────────────────────────────────────
SDXL_OUTPUT_NODE_ID = "9"
FLUX_OUTPUT_NODE_ID = "9"

# ── EmptyLatentImage class for FLUX (SD3 variant works for FLUX too) ──────────
_FLUX_LATENT_CLASS = "EmptySD3LatentImage"


def _validate_common(
    *,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> None:
    """Shared validation for both workflow types."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    for dim_name, dim_val in (("width", width), ("height", height)):
        if not isinstance(dim_val, int) or dim_val <= 0:
            raise ValueError(f"{dim_name} must be a positive integer, got {dim_val!r}")
        if dim_val % 8 != 0:
            raise ValueError(f"{dim_name} must be divisible by 8, got {dim_val}")
        if dim_val > 8192:
            raise ValueError(f"{dim_name} must not exceed 8192, got {dim_val}")

    if not isinstance(steps, int) or not (1 <= steps <= 150):
        raise ValueError(f"steps must be an integer between 1 and 150, got {steps!r}")

    if batch_size != 1:
        raise ValueError(f"batch_size must be 1 (got {batch_size}); multi-batch not yet supported")

    if not isinstance(seed, int) or not (0 <= seed <= 18446744073709551615):
        raise ValueError(
            f"seed must be an integer in [0, 18446744073709551615], got {seed!r}"
        )


def build_sdxl_workflow(
    *,
    prompt: str,
    negative_prompt: str,
    checkpoint_name: str,
    vae_name: str | None = None,
    width: int = 1024,
    height: int = 1536,
    seed: int,
    steps: int = 30,
    cfg: float = 7.0,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    batch_size: int = 1,
    filename_prefix: str = "etsy_agent_sdxl",
) -> dict:
    """
    Build an SDXL ComfyUI API workflow dict.

    Node layout:
        "1"  CheckpointLoaderSimple — loads checkpoint
        "2"  CLIPTextEncode        — positive conditioning
        "3"  CLIPTextEncode        — negative conditioning
        "4"  EmptyLatentImage      — blank latent canvas
        "5"  KSampler              — sampling step
        "6"  VAEDecode             — decode latent → pixels
        "7"  VAELoader             — external VAE (only when vae_name is given)
        "9"  SaveImage             — writes output file

    Returns a JSON-serializable dict.
    """
    _validate_common(
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
    )

    if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
        raise ValueError("checkpoint_name must be a non-empty string")

    if not math.isfinite(cfg):
        raise ValueError(f"cfg must be a finite float, got {cfg!r}")

    # ── Node 1: CheckpointLoaderSimple ────────────────────────────────────────
    node1 = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": checkpoint_name},
    }

    # ── Node 2: CLIPTextEncode (positive) ─────────────────────────────────────
    node2 = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": prompt,
            "clip": ["1", 1],  # port 1 of node 1 = CLIP output
        },
    }

    # ── Node 3: CLIPTextEncode (negative) ─────────────────────────────────────
    node3 = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": negative_prompt,
            "clip": ["1", 1],
        },
    }

    # ── Node 4: EmptyLatentImage ───────────────────────────────────────────────
    node4 = {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": width,
            "height": height,
            "batch_size": batch_size,
        },
    }

    # ── Node 5: KSampler ──────────────────────────────────────────────────────
    node5 = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],      # port 0 of node 1 = MODEL
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": 1.0,
        },
    }

    # ── Node 6: VAEDecode — vae source depends on vae_name ───────────────────
    if vae_name is None:
        # Use built-in VAE from checkpoint (port 2 of node 1)
        vae_source = ["1", 2]
        node6 = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": vae_source,
            },
        }
        workflow = {
            "1": node1,
            "2": node2,
            "3": node3,
            "4": node4,
            "5": node5,
            "6": node6,
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["6", 0],
                    "filename_prefix": filename_prefix,
                },
            },
        }
    else:
        # External VAE loader
        node7 = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        }
        node6 = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["7", 0],
            },
        }
        workflow = {
            "1": node1,
            "2": node2,
            "3": node3,
            "4": node4,
            "5": node5,
            "6": node6,
            "7": node7,
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["6", 0],
                    "filename_prefix": filename_prefix,
                },
            },
        }

    return workflow


def build_flux_schnell_workflow(
    *,
    prompt: str,
    unet_name: str,
    clip_l_name: str,
    t5xxl_name: str,
    vae_name: str,
    width: int = 1024,
    height: int = 1536,
    seed: int,
    steps: int = 4,
    guidance: float = 3.5,
    batch_size: int = 1,
    filename_prefix: str = "etsy_agent_flux",
) -> dict:
    """
    Build a FLUX.1 Schnell ComfyUI API workflow dict.

    IMPORTANT: FLUX Schnell does NOT use negative prompts. This function
    intentionally accepts no negative_prompt parameter because there is no
    negative conditioning node in the FLUX architecture.

    Node layout:
        "1"   UNETLoader            — loads diffusion U-Net (FLUX)
        "2"   DualCLIPLoader        — loads CLIP-L + T5-XXL text encoders
        "3"   VAELoader             — loads VAE decoder
        "4"   CLIPTextEncode        — encodes positive prompt
        "5"   EmptySD3LatentImage   — blank latent canvas (used for FLUX too)
        "6"   ModelSamplingFlux     — patches model with FLUX sigma schedule
        "7"   RandomNoise           — seeded noise source
        "8"   BasicGuider           — CFG-free guider (FLUX Schnell cfg=1)
        "9"   SaveImage             — writes output file
        "10"  KSamplerSelect        — selects euler sampler
        "11"  BasicScheduler        — simple scheduler + sigmas
        "12"  SamplerCustomAdvanced — runs sampling with custom noise/guider
        "13"  VAEDecode             — decode latent → pixels

    Returns a JSON-serializable dict.
    """
    _validate_common(
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
    )

    for name_label, name_val in (
        ("unet_name", unet_name),
        ("clip_l_name", clip_l_name),
        ("t5xxl_name", t5xxl_name),
        ("vae_name", vae_name),
    ):
        if not isinstance(name_val, str) or not name_val.strip():
            raise ValueError(f"{name_label} must be a non-empty string")

    if not math.isfinite(guidance):
        raise ValueError(f"guidance must be a finite float, got {guidance!r}")

    workflow = {
        # ── 1: UNETLoader ────────────────────────────────────────────────────
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet_name,
                "weight_dtype": "default",
            },
        },
        # ── 2: DualCLIPLoader ─────────────────────────────────────────────────
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": clip_l_name,
                "clip_name2": t5xxl_name,
                "type": "flux",
            },
        },
        # ── 3: VAELoader ──────────────────────────────────────────────────────
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        # ── 4: CLIPTextEncode (positive only — FLUX has no negative) ─────────
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["2", 0],
            },
        },
        # ── 5: EmptySD3LatentImage ────────────────────────────────────────────
        "5": {
            "class_type": _FLUX_LATENT_CLASS,
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": batch_size,
            },
        },
        # ── 6: ModelSamplingFlux ──────────────────────────────────────────────
        "6": {
            "class_type": "ModelSamplingFlux",
            "inputs": {
                "model": ["1", 0],
                "width": width,
                "height": height,
                "max_shift": 1.15,
                "base_shift": 0.5,
            },
        },
        # ── 7: RandomNoise ────────────────────────────────────────────────────
        "7": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        # ── 8: BasicGuider (CFG-free — guidance embedded via ModelSamplingFlux)
        "8": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["6", 0],
                "conditioning": ["4", 0],
            },
        },
        # ── 9: SaveImage ──────────────────────────────────────────────────────
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["13", 0],
                "filename_prefix": filename_prefix,
            },
        },
        # ── 10: KSamplerSelect ────────────────────────────────────────────────
        "10": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        # ── 11: BasicScheduler ────────────────────────────────────────────────
        "11": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["6", 0],
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        },
        # ── 12: SamplerCustomAdvanced ─────────────────────────────────────────
        "12": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["7", 0],
                "guider": ["8", 0],
                "sampler": ["10", 0],
                "sigmas": ["11", 0],
                "latent_image": ["5", 0],
            },
        },
        # ── 13: VAEDecode ─────────────────────────────────────────────────────
        "13": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["12", 0],   # "output" key from SamplerCustomAdvanced
                "vae": ["3", 0],
            },
        },
    }

    return workflow
