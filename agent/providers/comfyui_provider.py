"""
ComfyUI image providers — HTTP client for a locally running ComfyUI instance.

All HTTP calls use only Python stdlib (urllib.request / urllib.parse / urllib.error).
No requests, httpx, or aiohttp.

Security: only localhost URLs are accepted.  LAN/public addresses are rejected
at construction time to prevent accidental remote calls.

Usage:
    provider = ComfyUISDXLProvider()
    path = provider.generate("ukiyo-e crane over a misty river")

    provider = ComfyUIFluxSchnellProvider()
    health = provider.health_check()
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from image.base import ImageProvider
from agent.providers.comfyui_workflows import (
    SDXL_OUTPUT_NODE_ID,
    FLUX_OUTPUT_NODE_ID,
    build_sdxl_workflow,
    build_flux_schnell_workflow,
)

logger = logging.getLogger(__name__)

# ── Exception hierarchy ────────────────────────────────────────────────────────

class ComfyUIError(Exception):
    """Base class for all ComfyUI provider errors."""

class ComfyUIConfigurationError(ComfyUIError):
    """Missing or invalid configuration (env vars, model names, etc.)."""

class ComfyUIConnectionError(ComfyUIError):
    """Could not connect to ComfyUI (connection refused, DNS failure, etc.)."""

class ComfyUITimeoutError(ComfyUIError):
    """Generation timed out waiting for ComfyUI to complete the prompt."""

class ComfyUIExecutionError(ComfyUIError):
    """ComfyUI reported an execution error in the history entry."""

class ComfyUIResponseError(ComfyUIError):
    """Unexpected HTTP status or malformed JSON from ComfyUI."""

class ComfyUIImageValidationError(ComfyUIError):
    """Downloaded image bytes are empty or not a valid image."""


# ── ProviderHealth ─────────────────────────────────────────────────────────────

@dataclass
class ProviderHealth:
    provider: str
    available: bool
    latency_ms: float | None
    message: str
    details: dict = field(default_factory=dict)


# ── Local URL validation ───────────────────────────────────────────────────────

_LOCALHOST_HOSTNAMES = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _validate_local_url(url: str) -> str:
    """
    Raise ComfyUIConfigurationError for non-localhost URLs or non-http schemes.

    Allowed hosts:  127.0.0.1, localhost, ::1 / [::1]
    Allowed scheme: http only (https would require a TLS cert on localhost)
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        raise ComfyUIConfigurationError(f"Cannot parse COMFYUI_BASE_URL {url!r}: {exc}") from exc

    if parsed.scheme != "http":
        raise ComfyUIConfigurationError(
            f"COMFYUI_BASE_URL must use http:// scheme (got {parsed.scheme!r}). "
            "ComfyUI runs locally over plain HTTP."
        )

    hostname = parsed.hostname or ""
    if hostname not in _LOCALHOST_HOSTNAMES:
        raise ComfyUIConfigurationError(
            f"COMFYUI_BASE_URL hostname {hostname!r} is not a localhost address. "
            "Only 127.0.0.1, localhost, and ::1 are allowed to prevent "
            "accidental remote connections. Got URL: " + url
        )

    return url


# ── Base provider ──────────────────────────────────────────────────────────────

_OUTPUT_DIR = Path("output")
_DEFAULT_BASE_URL = "http://127.0.0.1:8188"
_DEFAULT_TIMEOUT = 600.0
_DEFAULT_POLL_INTERVAL = 1.0
_DEFAULT_CLIENT_ID = "etsy-agent"


class ComfyUIImageProvider(ImageProvider):
    """
    Shared HTTP lifecycle for all ComfyUI-based providers.
    Subclasses implement _build_workflow() and _provider_name().

    Config is read lazily (on first generate() / health_check() call)
    so that the class can be instantiated cheaply without side effects.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config = config  # optional override dict; else reads os.environ
        self._base_url: str | None = None        # set lazily
        self._timeout: float | None = None
        self._poll_interval: float | None = None
        self._client_id: str | None = None

    # ── Subclass interface ─────────────────────────────────────────────────────

    def _build_workflow(
        self, prompt: str, negative_prompt: str, seed: int
    ) -> tuple[dict, str]:
        """Return (workflow_dict, output_node_id). Subclasses must implement."""
        raise NotImplementedError

    def _provider_name(self) -> str:
        raise NotImplementedError

    # ── Config loading (lazy) ──────────────────────────────────────────────────

    def _env(self, key: str, default: str = "") -> str:
        if self._config and key in self._config:
            return self._config[key]
        return os.environ.get(key, default)

    def _load_base_config(self) -> None:
        """Load shared ComfyUI connection settings (called lazily)."""
        if self._base_url is not None:
            return  # already loaded

        raw_url = self._env("COMFYUI_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
        self._base_url = _validate_local_url(raw_url)

        try:
            self._timeout = float(self._env("COMFYUI_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT)))
        except ValueError:
            self._timeout = _DEFAULT_TIMEOUT

        try:
            self._poll_interval = float(
                self._env("COMFYUI_POLL_INTERVAL_SECONDS", str(_DEFAULT_POLL_INTERVAL))
            )
        except ValueError:
            self._poll_interval = _DEFAULT_POLL_INTERVAL

        self._client_id = self._env("COMFYUI_CLIENT_ID", _DEFAULT_CLIENT_ID)

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(self, prompt: str, on_usage: Callable | None = None) -> str:
        """
        Generate an image via ComfyUI and return the absolute path to the saved file.

        Steps:
          1. Build workflow (subclass)
          2. POST /prompt → prompt_id
          3. Poll GET /history/{prompt_id} until done
          4. Download image bytes from GET /view
          5. Validate image (Pillow)
          6. Save atomically to output/
          7. Call on_usage callback
        """
        self._load_base_config()
        self._validate_provider_config()

        seed = random.randint(0, 2**32 - 1)
        negative_prompt = ""
        workflow, output_node_id = self._build_workflow(prompt, negative_prompt, seed)

        prompt_id = self._submit_prompt(workflow)
        history_entry = self._poll_until_done(prompt_id)
        image_info = self._pick_output_image(history_entry, output_node_id)

        image_bytes = self._download_image(
            image_info["filename"],
            image_info.get("subfolder", ""),
            image_info.get("type", "output"),
        )
        width, height = self._validate_image_bytes(image_bytes)

        # Save atomically
        _OUTPUT_DIR.mkdir(exist_ok=True)
        slug = "".join(c if c.isalnum() else "_" for c in prompt[:40]).strip("_")
        dest = _OUTPUT_DIR / f"{slug}.png"
        tmp = dest.with_suffix(".tmp")
        try:
            tmp.write_bytes(image_bytes)
            tmp.replace(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        if on_usage is not None:
            on_usage({
                "provider": self._provider_name(),
                "model": self._provider_name(),
                "call_type": "image",
                "image_count": 1,
                "image_size": f"{width}x{height}",
                "api_cost": 0.0,
            })

        return str(dest.resolve())

    def health_check(self) -> ProviderHealth:
        """
        Call GET /system_stats to verify ComfyUI is running.
        Never queues a prompt — safe to call at any time.
        """
        self._load_base_config()
        url = f"{self._base_url}/system_stats"
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                latency_ms = (time.monotonic() - t0) * 1000
                raw = resp.read()
                try:
                    details = json.loads(raw)
                except Exception:
                    details = {}
                return ProviderHealth(
                    provider=self._provider_name(),
                    available=True,
                    latency_ms=round(latency_ms, 2),
                    message="ComfyUI is running",
                    details=details,
                )
        except urllib.error.URLError as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            reason = str(exc.reason) if hasattr(exc, "reason") else str(exc)
            return ProviderHealth(
                provider=self._provider_name(),
                available=False,
                latency_ms=round(latency_ms, 2),
                message=f"ComfyUI not reachable: {reason}",
                details={},
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            return ProviderHealth(
                provider=self._provider_name(),
                available=False,
                latency_ms=round(latency_ms, 2),
                message=f"Health check failed: {exc}",
                details={},
            )

    def _validate_provider_config(self) -> None:
        """Subclasses override to check model-specific env vars."""

    # ── Private HTTP helpers ───────────────────────────────────────────────────

    def _submit_prompt(self, workflow: dict) -> str:
        """POST /prompt with the workflow. Returns prompt_id string."""
        url = f"{self._base_url}/prompt"
        body = json.dumps({"prompt": workflow, "client_id": self._client_id}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raise ComfyUIResponseError(
                f"ComfyUI /prompt returned HTTP {exc.code}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            reason_str = str(reason)
            if "connection refused" in reason_str.lower() or "111" in reason_str:
                raise ComfyUIConnectionError(
                    f"Cannot connect to ComfyUI at {self._base_url} — is it running?"
                ) from exc
            raise ComfyUIConnectionError(
                f"Network error reaching ComfyUI: {reason_str}"
            ) from exc

        try:
            data = json.loads(raw)
        except Exception as exc:
            raise ComfyUIResponseError(
                f"ComfyUI /prompt returned non-JSON response: {raw[:200]!r}"
            ) from exc

        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyUIResponseError(
                f"ComfyUI /prompt response missing 'prompt_id'. Got: {data!r}"
            )
        return prompt_id

    def _poll_until_done(self, prompt_id: str) -> dict:
        """
        Poll GET /history/{prompt_id} until 'outputs' key appears or timeout.
        Returns the history entry dict for this prompt_id.
        """
        url = f"{self._base_url}/history/{urllib.parse.quote(prompt_id)}"
        deadline = time.monotonic() + self._timeout

        while True:
            if time.monotonic() >= deadline:
                raise ComfyUITimeoutError(
                    f"Timed out after {self._timeout}s waiting for ComfyUI "
                    f"to complete prompt_id={prompt_id!r}"
                )

            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read()
            except urllib.error.HTTPError as exc:
                raise ComfyUIResponseError(
                    f"ComfyUI /history returned HTTP {exc.code}: {exc.reason}"
                ) from exc
            except urllib.error.URLError as exc:
                reason_str = str(getattr(exc, "reason", exc))
                if "connection refused" in reason_str.lower() or "111" in reason_str:
                    raise ComfyUIConnectionError(
                        f"Lost connection to ComfyUI during polling: {reason_str}"
                    ) from exc
                raise ComfyUIConnectionError(
                    f"Network error polling ComfyUI history: {reason_str}"
                ) from exc

            try:
                history = json.loads(raw)
            except Exception as exc:
                raise ComfyUIResponseError(
                    f"ComfyUI /history returned non-JSON: {raw[:200]!r}"
                ) from exc

            entry = history.get(prompt_id)
            if entry and "outputs" in entry:
                # Check for execution errors
                status = entry.get("status", {})
                messages = status.get("messages", [])
                for msg_type, msg_data in messages:
                    if msg_type == "execution_error":
                        err_msg = msg_data.get("exception_message", "Unknown error")
                        raise ComfyUIExecutionError(
                            f"ComfyUI execution error for prompt {prompt_id!r}: {err_msg}"
                        )
                return entry

            time.sleep(self._poll_interval)

    def _download_image(self, filename: str, subfolder: str, type_: str) -> bytes:
        """Download image bytes from ComfyUI GET /view."""
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": type_,
        })
        url = f"{self._base_url}/view?{params}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            raise ComfyUIResponseError(
                f"ComfyUI /view returned HTTP {exc.code} for file {filename!r}"
            ) from exc
        except urllib.error.URLError as exc:
            reason_str = str(getattr(exc, "reason", exc))
            raise ComfyUIConnectionError(
                f"Network error downloading image {filename!r}: {reason_str}"
            ) from exc

    def _validate_image_bytes(self, data: bytes) -> tuple[int, int]:
        """
        Validate that bytes are a recognisable image.
        Returns (width, height).
        Raises ComfyUIImageValidationError on failure.
        """
        if not data:
            raise ComfyUIImageValidationError(
                "ComfyUI returned empty image data (0 bytes)."
            )

        try:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            img.verify()
            # Re-open after verify (PIL closes the file after verify)
            img = Image.open(io.BytesIO(data))
            return img.width, img.height
        except ImportError:
            # Pillow not installed — fall back to minimal PNG header check
            if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
                raise ComfyUIImageValidationError(
                    "Downloaded bytes do not appear to be a valid PNG image "
                    "(Pillow not installed; only PNG header check performed)."
                )
            # Extract width/height from IHDR chunk (bytes 16-24)
            import struct
            width = struct.unpack(">I", data[16:20])[0]
            height = struct.unpack(">I", data[20:24])[0]
            return width, height
        except Exception as exc:
            raise ComfyUIImageValidationError(
                f"Downloaded image bytes are not a valid image: {exc}"
            ) from exc

    def _pick_output_image(self, history_entry: dict, output_node_id: str) -> dict:
        """
        Extract the first output image info dict from a history entry.
        Warns if multiple images are present (picks index 0 deterministically).
        """
        outputs = history_entry.get("outputs", {})
        node_output = outputs.get(output_node_id, {})
        images = node_output.get("images", [])

        if not images:
            # Try to find any output node that has images
            for node_id, node_data in outputs.items():
                candidate = node_data.get("images", [])
                if candidate:
                    images = candidate
                    logger.warning(
                        "Output node %r had no images; using output from node %r instead.",
                        output_node_id,
                        node_id,
                    )
                    break

        if not images:
            raise ComfyUIResponseError(
                f"ComfyUI history has no output images for node {output_node_id!r}. "
                f"Available output keys: {list(outputs.keys())}"
            )

        if len(images) > 1:
            warnings.warn(
                f"ComfyUI returned {len(images)} output images; using index 0.",
                stacklevel=4,
            )

        return images[0]


# ── SDXL Provider ──────────────────────────────────────────────────────────────

class ComfyUISDXLProvider(ComfyUIImageProvider):
    """
    SDXL image generation via a locally running ComfyUI instance.

    Required env vars (when IMAGE_PROVIDER=comfyui_sdxl):
        COMFYUI_SDXL_CHECKPOINT  — safetensors filename in ComfyUI models/checkpoints/

    Optional env vars:
        COMFYUI_SDXL_VAE       — external VAE safetensors (empty = use checkpoint's built-in)
        COMFYUI_SDXL_WIDTH     — output width in pixels (default 1024, must be divisible by 8)
        COMFYUI_SDXL_HEIGHT    — output height in pixels (default 1536)
        COMFYUI_SDXL_STEPS     — sampling steps (default 30)
        COMFYUI_SDXL_CFG       — classifier-free guidance scale (default 7.0)
        COMFYUI_SDXL_SAMPLER   — sampler name (default euler)
        COMFYUI_SDXL_SCHEDULER — noise scheduler (default normal)
    """

    def _provider_name(self) -> str:
        return "comfyui_sdxl"

    def _validate_provider_config(self) -> None:
        checkpoint = self._env("COMFYUI_SDXL_CHECKPOINT", "")
        if not checkpoint.strip():
            raise ComfyUIConfigurationError(
                "COMFYUI_SDXL_CHECKPOINT is not set. "
                "Set it to your SDXL .safetensors filename in ComfyUI's checkpoints folder."
            )

    def _build_workflow(
        self, prompt: str, negative_prompt: str, seed: int
    ) -> tuple[dict, str]:
        checkpoint = self._env("COMFYUI_SDXL_CHECKPOINT", "")
        vae_name = self._env("COMFYUI_SDXL_VAE", "").strip() or None

        try:
            width = int(self._env("COMFYUI_SDXL_WIDTH", "1024"))
        except ValueError:
            width = 1024
        try:
            height = int(self._env("COMFYUI_SDXL_HEIGHT", "1536"))
        except ValueError:
            height = 1536
        try:
            steps = int(self._env("COMFYUI_SDXL_STEPS", "30"))
        except ValueError:
            steps = 30
        try:
            cfg = float(self._env("COMFYUI_SDXL_CFG", "7.0"))
        except ValueError:
            cfg = 7.0

        sampler = self._env("COMFYUI_SDXL_SAMPLER", "euler")
        scheduler = self._env("COMFYUI_SDXL_SCHEDULER", "normal")

        workflow = build_sdxl_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            checkpoint_name=checkpoint,
            vae_name=vae_name,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
        )
        return workflow, SDXL_OUTPUT_NODE_ID


# ── FLUX Schnell Provider ──────────────────────────────────────────────────────

class ComfyUIFluxSchnellProvider(ComfyUIImageProvider):
    """
    FLUX.1 Schnell image generation via a locally running ComfyUI instance.

    Required env vars (when IMAGE_PROVIDER=comfyui_flux_schnell):
        COMFYUI_FLUX_UNET   — FLUX U-Net safetensors filename
        COMFYUI_FLUX_CLIP_L — CLIP-L safetensors filename
        COMFYUI_FLUX_T5XXL  — T5-XXL safetensors filename
        COMFYUI_FLUX_VAE    — VAE (ae) safetensors filename

    Optional env vars:
        COMFYUI_FLUX_WIDTH    — output width (default 1024)
        COMFYUI_FLUX_HEIGHT   — output height (default 1536)
        COMFYUI_FLUX_STEPS    — diffusion steps (default 4 — Schnell is distilled)
        COMFYUI_FLUX_GUIDANCE — guidance scale (default 3.5)

    Note: FLUX Schnell does NOT use negative prompts.
    """

    def _provider_name(self) -> str:
        return "comfyui_flux_schnell"

    def _validate_provider_config(self) -> None:
        missing = []
        for var in ("COMFYUI_FLUX_UNET", "COMFYUI_FLUX_CLIP_L", "COMFYUI_FLUX_T5XXL", "COMFYUI_FLUX_VAE"):
            if not self._env(var, "").strip():
                missing.append(var)
        if missing:
            raise ComfyUIConfigurationError(
                f"Missing required FLUX model env vars: {', '.join(missing)}. "
                "Set these to the safetensors filenames in your ComfyUI models folder."
            )

    def _build_workflow(
        self, prompt: str, negative_prompt: str, seed: int
    ) -> tuple[dict, str]:
        # negative_prompt is intentionally ignored — FLUX Schnell has no negative conditioning
        unet_name = self._env("COMFYUI_FLUX_UNET", "")
        clip_l_name = self._env("COMFYUI_FLUX_CLIP_L", "")
        t5xxl_name = self._env("COMFYUI_FLUX_T5XXL", "")
        vae_name = self._env("COMFYUI_FLUX_VAE", "")

        try:
            width = int(self._env("COMFYUI_FLUX_WIDTH", "1024"))
        except ValueError:
            width = 1024
        try:
            height = int(self._env("COMFYUI_FLUX_HEIGHT", "1536"))
        except ValueError:
            height = 1536
        try:
            steps = int(self._env("COMFYUI_FLUX_STEPS", "4"))
        except ValueError:
            steps = 4
        try:
            guidance = float(self._env("COMFYUI_FLUX_GUIDANCE", "3.5"))
        except ValueError:
            guidance = 3.5

        workflow = build_flux_schnell_workflow(
            prompt=prompt,
            unet_name=unet_name,
            clip_l_name=clip_l_name,
            t5xxl_name=t5xxl_name,
            vae_name=vae_name,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            guidance=guidance,
        )
        return workflow, FLUX_OUTPUT_NODE_ID
