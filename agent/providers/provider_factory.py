"""
Factory for creating ImageProvider instances by name.

Usage:
    from agent.providers.provider_factory import create_image_provider

    provider = create_image_provider()              # reads IMAGE_PROVIDER env var
    provider = create_image_provider("openai")      # explicit
    provider = create_image_provider("comfyui_sdxl")
    provider = create_image_provider("comfyui_flux_schnell")

No provider instances are created at import time.
Config validation happens lazily inside each provider's generate() / health_check().
"""

from __future__ import annotations

import os

from image.base import ImageProvider

SUPPORTED_PROVIDERS = ("openai", "comfyui_sdxl", "comfyui_flux_schnell")


def create_image_provider(
    provider_name: str | None = None,
    *,
    config: dict | None = None,
) -> ImageProvider:
    """
    Return an ImageProvider for the given name.

    Resolution order:
      1. provider_name argument (if given)
      2. IMAGE_PROVIDER environment variable
      3. "openai" (default)

    Raises ValueError for unknown provider names.
    Does NOT validate provider-specific config at creation time.
    """
    if provider_name is None:
        provider_name = os.environ.get("IMAGE_PROVIDER", "openai") or "openai"

    provider_name = provider_name.strip().lower()

    if provider_name == "openai":
        from image.openai_provider import OpenAIImageProvider
        return OpenAIImageProvider()

    if provider_name == "comfyui_sdxl":
        from agent.providers.comfyui_provider import ComfyUISDXLProvider
        return ComfyUISDXLProvider(config=config)

    if provider_name == "comfyui_flux_schnell":
        from agent.providers.comfyui_provider import ComfyUIFluxSchnellProvider
        return ComfyUIFluxSchnellProvider(config=config)

    raise ValueError(
        f"Unknown image provider {provider_name!r}. "
        f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
    )
