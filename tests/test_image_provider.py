"""
Unit tests for OpenAIImageProvider — no live API calls.
"""
import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from image.openai_provider import OpenAIImageProvider, MODEL


@pytest.fixture()
def provider(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import agent.config as cfg
    monkeypatch.setattr(cfg, "OPENAI_API_KEY", "sk-test")

    fake_client = MagicMock()
    fake_image_data = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8).decode()
    fake_response = MagicMock()
    fake_response.data = [MagicMock(b64_json=fake_image_data)]
    fake_client.images.generate.return_value = fake_response

    with patch("image.openai_provider.OpenAI", return_value=fake_client):
        p = OpenAIImageProvider()
        p._client = fake_client
        p._output_dir = tmp_path  # redirect output so we don't litter
        # patch OUTPUT_DIR at module level too
        import image.openai_provider as mod
        monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path)
        yield p, fake_client


def test_generate_requests_correct_size(provider):
    p, client = provider
    p.generate("ukiyo-e crane over river")
    call_kwargs = client.images.generate.call_args.kwargs
    assert call_kwargs["size"] == "1536x2304"


def test_generate_uses_correct_model(provider):
    p, client = provider
    p.generate("test prompt")
    call_kwargs = client.images.generate.call_args.kwargs
    assert call_kwargs["model"] == MODEL
    assert MODEL == "gpt-image-2"


def test_generate_returns_file_path_that_exists(provider, tmp_path):
    p, client = provider
    path = p.generate("test prompt")
    assert Path(path).exists()


def test_generate_does_not_pass_response_format(provider):
    p, client = provider
    p.generate("test prompt")
    call_kwargs = client.images.generate.call_args.kwargs
    assert "response_format" not in call_kwargs
