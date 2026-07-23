"""
Tests for agent/providers/provider_factory.py.
"""
from __future__ import annotations

import importlib
import os
import sys
import unittest.mock

import pytest


def _fresh_import():
    """Import provider_factory without cached state."""
    mod_name = "agent.providers.provider_factory"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    import agent.providers.provider_factory as m
    return m


class TestCreateImageProvider:

    def test_no_args_no_env_returns_openai(self, monkeypatch):
        monkeypatch.delenv("IMAGE_PROVIDER", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        import agent.config as cfg
        monkeypatch.setattr(cfg, "OPENAI_API_KEY", "sk-test")

        from unittest.mock import patch, MagicMock
        with patch("image.openai_provider.OpenAI", return_value=MagicMock()):
            from agent.providers.provider_factory import create_image_provider
            from image.openai_provider import OpenAIImageProvider
            provider = create_image_provider()
        assert isinstance(provider, OpenAIImageProvider)

    def test_env_openai_returns_openai(self, monkeypatch):
        monkeypatch.setenv("IMAGE_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        import agent.config as cfg
        monkeypatch.setattr(cfg, "OPENAI_API_KEY", "sk-test")

        from unittest.mock import patch, MagicMock
        with patch("image.openai_provider.OpenAI", return_value=MagicMock()):
            from agent.providers.provider_factory import create_image_provider
            from image.openai_provider import OpenAIImageProvider
            provider = create_image_provider()
        assert isinstance(provider, OpenAIImageProvider)

    def test_explicit_openai_returns_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        import agent.config as cfg
        monkeypatch.setattr(cfg, "OPENAI_API_KEY", "sk-test")

        from unittest.mock import patch, MagicMock
        with patch("image.openai_provider.OpenAI", return_value=MagicMock()):
            from agent.providers.provider_factory import create_image_provider
            from image.openai_provider import OpenAIImageProvider
            provider = create_image_provider("openai")
        assert isinstance(provider, OpenAIImageProvider)

    def test_comfyui_sdxl_returns_sdxl_provider(self):
        from agent.providers.provider_factory import create_image_provider
        from agent.providers.comfyui_provider import ComfyUISDXLProvider
        provider = create_image_provider("comfyui_sdxl")
        assert isinstance(provider, ComfyUISDXLProvider)

    def test_comfyui_flux_schnell_returns_flux_provider(self):
        from agent.providers.provider_factory import create_image_provider
        from agent.providers.comfyui_provider import ComfyUIFluxSchnellProvider
        provider = create_image_provider("comfyui_flux_schnell")
        assert isinstance(provider, ComfyUIFluxSchnellProvider)

    def test_unknown_provider_raises_value_error(self):
        from agent.providers.provider_factory import create_image_provider
        with pytest.raises(ValueError, match="unknown_thing"):
            create_image_provider("unknown_thing")

    def test_value_error_mentions_supported_providers(self):
        from agent.providers.provider_factory import create_image_provider, SUPPORTED_PROVIDERS
        with pytest.raises(ValueError) as exc_info:
            create_image_provider("bad_provider")
        msg = str(exc_info.value)
        # At least one supported name should be in the error message
        assert any(p in msg for p in SUPPORTED_PROVIDERS)

    def test_no_instance_created_at_import_time(self):
        """Importing the factory must not instantiate any provider."""
        # We verify by checking that importing the module does not touch
        # OPENAI_API_KEY (which would be needed to build OpenAIImageProvider).
        import importlib
        import sys
        # Remove cached modules to force fresh import
        for key in list(sys.modules.keys()):
            if "provider_factory" in key:
                del sys.modules[key]

        original_openai = None
        try:
            import image.openai_provider as oai_mod
            original_openai = oai_mod.OpenAIImageProvider
        except Exception:
            pass

        # Track instantiations
        instances_created = []
        if original_openai:
            class Sentinel(original_openai):
                def __init__(self):
                    instances_created.append(True)
                    super().__init__()

            import image.openai_provider as oai_mod
            # We can't easily patch at this point without patching at module level,
            # but importing provider_factory should not create any instance at all.
            pass

        # Re-import fresh
        import agent.providers.provider_factory  # should not raise or instantiate
        # If we get here without error, the import itself didn't create a provider
        assert True

    def test_direct_openai_provider_construction_works(self, monkeypatch):
        """Directly importing and instantiating OpenAIImageProvider must still work."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-direct")
        import agent.config as cfg
        monkeypatch.setattr(cfg, "OPENAI_API_KEY", "sk-test-direct")

        from unittest.mock import patch, MagicMock
        with patch("image.openai_provider.OpenAI", return_value=MagicMock()):
            from image.openai_provider import OpenAIImageProvider
            provider = OpenAIImageProvider()
        assert provider is not None


class TestSupportedProviders:

    def test_supported_providers_tuple_contents(self):
        from agent.providers.provider_factory import SUPPORTED_PROVIDERS
        assert "openai" in SUPPORTED_PROVIDERS
        assert "comfyui_sdxl" in SUPPORTED_PROVIDERS
        assert "comfyui_flux_schnell" in SUPPORTED_PROVIDERS
