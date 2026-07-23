"""
Tests for agent/providers/comfyui_provider.py.

All HTTP calls are mocked — no real localhost connection required.
"""
from __future__ import annotations

import io
import json
import struct
import time
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch, call

import pytest

from agent.providers.comfyui_provider import (
    ComfyUIImageProvider,
    ComfyUISDXLProvider,
    ComfyUIFluxSchnellProvider,
    ComfyUIError,
    ComfyUIConfigurationError,
    ComfyUIConnectionError,
    ComfyUITimeoutError,
    ComfyUIExecutionError,
    ComfyUIResponseError,
    ComfyUIImageValidationError,
    ProviderHealth,
    _validate_local_url,
)
from agent.providers.comfyui_workflows import SDXL_OUTPUT_NODE_ID, FLUX_OUTPUT_NODE_ID


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fake_history(
    prompt_id: str,
    filename: str = "output.png",
    subfolder: str = "",
    type_: str = "output",
    output_node_id: str = SDXL_OUTPUT_NODE_ID,
) -> dict:
    """Build a minimal ComfyUI history response dict."""
    return {
        prompt_id: {
            "outputs": {
                output_node_id: {
                    "images": [
                        {"filename": filename, "subfolder": subfolder, "type": type_}
                    ]
                }
            },
            "status": {"completed": True, "messages": []},
        }
    }


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Build a minimal valid PNG byte string."""
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Fallback: craft a minimal valid PNG header + IHDR
        import zlib

        def chunk(name: bytes, data: bytes) -> bytes:
            c = name + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        # Minimal IDAT: a single white pixel scanline per row (simplified, may not render)
        raw_data = b""
        for _ in range(height):
            raw_data += b"\x00" + b"\xff\xff\xff" * width
        compressed = zlib.compress(raw_data)
        iend = b""
        return (
            sig
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", compressed)
            + chunk(b"IEND", iend)
        )


def _make_urlopen_response(body: bytes, status: int = 200) -> MagicMock:
    """Create a mock response context manager for urllib.request.urlopen."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.status = status
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_sdxl_provider(env: dict | None = None) -> ComfyUISDXLProvider:
    """Create an SDXLProvider with preset config."""
    config = {
        "COMFYUI_BASE_URL": "http://127.0.0.1:8188",
        "COMFYUI_TIMEOUT_SECONDS": "30",
        "COMFYUI_POLL_INTERVAL_SECONDS": "0.01",
        "COMFYUI_CLIENT_ID": "test-client",
        "COMFYUI_SDXL_CHECKPOINT": "sdxl.safetensors",
        "COMFYUI_SDXL_WIDTH": "512",
        "COMFYUI_SDXL_HEIGHT": "512",
        "COMFYUI_SDXL_STEPS": "1",
        "COMFYUI_SDXL_CFG": "7.0",
    }
    if env:
        config.update(env)
    return ComfyUISDXLProvider(config=config)


# ── Local URL validation ───────────────────────────────────────────────────────

class TestValidateLocalUrl:

    def test_accepts_127_0_0_1(self):
        url = "http://127.0.0.1:8188"
        assert _validate_local_url(url) == url

    def test_accepts_localhost(self):
        url = "http://localhost:8188"
        assert _validate_local_url(url) == url

    def test_accepts_ipv6_loopback(self):
        url = "http://[::1]:8188"
        assert _validate_local_url(url) == url

    def test_rejects_lan_ip(self):
        with pytest.raises(ComfyUIConfigurationError, match="localhost"):
            _validate_local_url("http://192.168.1.10:8188")

    def test_rejects_public_hostname(self):
        with pytest.raises(ComfyUIConfigurationError):
            _validate_local_url("http://example.com:8188")

    def test_rejects_https_scheme(self):
        with pytest.raises(ComfyUIConfigurationError, match="http://"):
            _validate_local_url("https://127.0.0.1:8188")

    def test_rejects_10_x_x_x(self):
        with pytest.raises(ComfyUIConfigurationError):
            _validate_local_url("http://10.0.0.1:8188")

    def test_rejects_172_16_x_x(self):
        with pytest.raises(ComfyUIConfigurationError):
            _validate_local_url("http://172.16.0.1:8188")


# ── HTTP lifecycle ─────────────────────────────────────────────────────────────

class TestSubmitPrompt:

    def test_successful_submit_returns_prompt_id(self):
        provider = _make_sdxl_provider()
        provider._load_base_config()

        prompt_id = "abc-123"
        body = json.dumps({"prompt_id": prompt_id}).encode()
        mock_resp = _make_urlopen_response(body)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = provider._submit_prompt({"1": {}})

        assert result == prompt_id

    def test_connection_refused_raises_connection_error(self):
        provider = _make_sdxl_provider()
        provider._load_base_config()

        err = urllib.error.URLError("Connection refused")
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(ComfyUIConnectionError):
                provider._submit_prompt({"1": {}})

    def test_http_500_raises_response_error(self):
        provider = _make_sdxl_provider()
        provider._load_base_config()

        http_err = urllib.error.HTTPError(
            url="http://127.0.0.1:8188/prompt",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            with pytest.raises(ComfyUIResponseError, match="500"):
                provider._submit_prompt({"1": {}})

    def test_invalid_json_raises_response_error(self):
        provider = _make_sdxl_provider()
        provider._load_base_config()

        mock_resp = _make_urlopen_response(b"not json")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(ComfyUIResponseError, match="non-JSON"):
                provider._submit_prompt({"1": {}})

    def test_missing_prompt_id_raises_response_error(self):
        provider = _make_sdxl_provider()
        provider._load_base_config()

        body = json.dumps({"something_else": "value"}).encode()
        mock_resp = _make_urlopen_response(body)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(ComfyUIResponseError, match="prompt_id"):
                provider._submit_prompt({"1": {}})


class TestPollUntilDone:

    def _provider(self):
        return _make_sdxl_provider()

    def test_poll_returns_history_entry_when_outputs_present(self):
        provider = self._provider()
        provider._load_base_config()

        pid = "test-pid"
        history = _fake_history(pid)
        mock_resp = _make_urlopen_response(json.dumps(history).encode())

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = provider._poll_until_done(pid)

        assert "outputs" in result

    def test_poll_waits_for_outputs(self):
        """First call returns empty dict, second has outputs."""
        provider = self._provider()
        provider._load_base_config()
        provider._poll_interval = 0.001

        pid = "test-pid"
        empty_resp = _make_urlopen_response(json.dumps({}).encode())
        full_resp = _make_urlopen_response(json.dumps(_fake_history(pid)).encode())

        responses = [empty_resp, full_resp]
        with patch("urllib.request.urlopen", side_effect=responses):
            result = provider._poll_until_done(pid)

        assert "outputs" in result

    def test_execution_error_in_history_raises_execution_error(self):
        provider = self._provider()
        provider._load_base_config()

        pid = "err-pid"
        history = {
            pid: {
                "outputs": {"9": {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]}},
                "status": {
                    "completed": False,
                    "messages": [
                        ("execution_error", {"exception_message": "CUDA out of memory"}),
                    ],
                },
            }
        }
        mock_resp = _make_urlopen_response(json.dumps(history).encode())
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(ComfyUIExecutionError, match="CUDA out of memory"):
                provider._poll_until_done(pid)

    def test_timeout_raises_timeout_error(self):
        provider = self._provider()
        provider._load_base_config()
        provider._timeout = 0.01  # very short timeout
        provider._poll_interval = 0.001

        pid = "timeout-pid"
        empty_resp = _make_urlopen_response(json.dumps({}).encode())

        with patch("urllib.request.urlopen", return_value=empty_resp):
            with patch("time.monotonic") as mock_mono:
                # First call: before deadline. All subsequent: past deadline.
                mock_mono.side_effect = [0.0, 0.0, 100.0]
                with pytest.raises(ComfyUITimeoutError):
                    provider._poll_until_done(pid)


class TestPickOutputImage:

    def test_picks_index_0_with_single_image(self):
        provider = _make_sdxl_provider()
        history_entry = _fake_history("pid1")[
            "pid1"
        ]
        img = provider._pick_output_image(history_entry, SDXL_OUTPUT_NODE_ID)
        assert img["filename"] == "output.png"

    def test_picks_index_0_with_multiple_images(self):
        provider = _make_sdxl_provider()
        history_entry = {
            "outputs": {
                SDXL_OUTPUT_NODE_ID: {
                    "images": [
                        {"filename": "first.png", "subfolder": "", "type": "output"},
                        {"filename": "second.png", "subfolder": "", "type": "output"},
                    ]
                }
            },
            "status": {"completed": True, "messages": []},
        }
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            img = provider._pick_output_image(history_entry, SDXL_OUTPUT_NODE_ID)
        assert img["filename"] == "first.png"
        assert any("2" in str(warning.message) for warning in w)


class TestImageValidation:

    def test_empty_bytes_raises_validation_error(self):
        provider = _make_sdxl_provider()
        with pytest.raises(ComfyUIImageValidationError, match="empty"):
            provider._validate_image_bytes(b"")

    def test_corrupt_bytes_raises_validation_error(self):
        provider = _make_sdxl_provider()
        with pytest.raises(ComfyUIImageValidationError):
            provider._validate_image_bytes(b"not an image at all")

    def test_valid_png_returns_dimensions(self):
        provider = _make_sdxl_provider()
        png = _make_png_bytes(64, 64)
        w, h = provider._validate_image_bytes(png)
        assert w == 64
        assert h == 64


class TestHealthCheck:

    def test_success_returns_available_true(self):
        provider = _make_sdxl_provider()
        stats = {"system": {"os": "posix"}}
        mock_resp = _make_urlopen_response(json.dumps(stats).encode())

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = provider.health_check()

        assert result.available is True
        assert result.latency_ms is not None
        assert result.provider == "comfyui_sdxl"

    def test_connection_error_returns_available_false(self):
        provider = _make_sdxl_provider()
        err = urllib.error.URLError("Connection refused")

        with patch("urllib.request.urlopen", side_effect=err):
            result = provider.health_check()

        assert result.available is False
        assert "not reachable" in result.message.lower() or "comfyui" in result.message.lower()

    def test_health_check_calls_system_stats_not_prompt(self):
        provider = _make_sdxl_provider()
        mock_resp = _make_urlopen_response(json.dumps({}).encode())
        captured_urls = []

        def fake_urlopen(req, **kwargs):
            captured_urls.append(req.full_url if hasattr(req, "full_url") else str(req))
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            provider.health_check()

        assert len(captured_urls) == 1
        assert "system_stats" in captured_urls[0]
        assert "prompt" not in captured_urls[0]


# ── on_usage callback ──────────────────────────────────────────────────────────

class TestOnUsageCallback:

    def test_called_with_zero_api_cost(self, tmp_path):
        provider = _make_sdxl_provider()

        pid = "usage-pid"
        png_bytes = _make_png_bytes(512, 512)

        submit_resp = _make_urlopen_response(json.dumps({"prompt_id": pid}).encode())
        history_resp = _make_urlopen_response(json.dumps(_fake_history(pid)).encode())
        image_resp = _make_urlopen_response(png_bytes)

        usage_calls = []

        with patch("urllib.request.urlopen", side_effect=[submit_resp, history_resp, image_resp]):
            with patch("agent.providers.comfyui_provider._OUTPUT_DIR", tmp_path):
                provider.generate("test prompt", on_usage=usage_calls.append)

        assert len(usage_calls) == 1
        usage = usage_calls[0]
        assert usage["api_cost"] == 0.0

    def test_called_with_correct_provider_name(self, tmp_path):
        provider = _make_sdxl_provider()

        pid = "prov-pid"
        png_bytes = _make_png_bytes(512, 512)

        submit_resp = _make_urlopen_response(json.dumps({"prompt_id": pid}).encode())
        history_resp = _make_urlopen_response(json.dumps(_fake_history(pid)).encode())
        image_resp = _make_urlopen_response(png_bytes)

        usage_calls = []

        with patch("urllib.request.urlopen", side_effect=[submit_resp, history_resp, image_resp]):
            with patch("agent.providers.comfyui_provider._OUTPUT_DIR", tmp_path):
                provider.generate("test prompt", on_usage=usage_calls.append)

        assert usage_calls[0]["provider"] == "comfyui_sdxl"

    def test_called_with_correct_image_size_string(self, tmp_path):
        provider = _make_sdxl_provider()

        pid = "size-pid"
        png_bytes = _make_png_bytes(512, 512)

        submit_resp = _make_urlopen_response(json.dumps({"prompt_id": pid}).encode())
        history_resp = _make_urlopen_response(json.dumps(_fake_history(pid)).encode())
        image_resp = _make_urlopen_response(png_bytes)

        usage_calls = []

        with patch("urllib.request.urlopen", side_effect=[submit_resp, history_resp, image_resp]):
            with patch("agent.providers.comfyui_provider._OUTPUT_DIR", tmp_path):
                provider.generate("test prompt", on_usage=usage_calls.append)

        size_str = usage_calls[0]["image_size"]
        assert "x" in size_str
        w_str, h_str = size_str.split("x")
        assert int(w_str) > 0
        assert int(h_str) > 0


# ── Output file ────────────────────────────────────────────────────────────────

class TestOutputFile:

    def test_returns_absolute_path(self, tmp_path):
        provider = _make_sdxl_provider()

        pid = "abs-pid"
        png_bytes = _make_png_bytes(64, 64)

        submit_resp = _make_urlopen_response(json.dumps({"prompt_id": pid}).encode())
        history_resp = _make_urlopen_response(json.dumps(_fake_history(pid)).encode())
        image_resp = _make_urlopen_response(png_bytes)

        with patch("urllib.request.urlopen", side_effect=[submit_resp, history_resp, image_resp]):
            with patch("agent.providers.comfyui_provider._OUTPUT_DIR", tmp_path):
                path = provider.generate("test prompt")

        from pathlib import Path
        assert Path(path).is_absolute()

    def test_written_to_output_directory(self, tmp_path):
        provider = _make_sdxl_provider()

        pid = "dir-pid"
        png_bytes = _make_png_bytes(64, 64)

        submit_resp = _make_urlopen_response(json.dumps({"prompt_id": pid}).encode())
        history_resp = _make_urlopen_response(json.dumps(_fake_history(pid)).encode())
        image_resp = _make_urlopen_response(png_bytes)

        with patch("urllib.request.urlopen", side_effect=[submit_resp, history_resp, image_resp]):
            with patch("agent.providers.comfyui_provider._OUTPUT_DIR", tmp_path):
                path = provider.generate("test prompt")

        from pathlib import Path
        assert Path(path).parent.resolve() == tmp_path.resolve()

    def test_no_tmp_file_after_success(self, tmp_path):
        provider = _make_sdxl_provider()

        pid = "tmp-pid"
        png_bytes = _make_png_bytes(64, 64)

        submit_resp = _make_urlopen_response(json.dumps({"prompt_id": pid}).encode())
        history_resp = _make_urlopen_response(json.dumps(_fake_history(pid)).encode())
        image_resp = _make_urlopen_response(png_bytes)

        with patch("urllib.request.urlopen", side_effect=[submit_resp, history_resp, image_resp]):
            with patch("agent.providers.comfyui_provider._OUTPUT_DIR", tmp_path):
                provider.generate("test prompt")

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Found leftover .tmp files: {tmp_files}"


# ── SDXL provider config ───────────────────────────────────────────────────────

class TestSDXLProviderConfig:

    def test_reads_checkpoint_from_env(self):
        provider = ComfyUISDXLProvider(config={
            "COMFYUI_BASE_URL": "http://127.0.0.1:8188",
            "COMFYUI_SDXL_CHECKPOINT": "my_model.safetensors",
        })
        provider._load_base_config()
        # Should not raise
        provider._validate_provider_config()

    def test_raises_config_error_if_checkpoint_missing(self):
        provider = ComfyUISDXLProvider(config={
            "COMFYUI_BASE_URL": "http://127.0.0.1:8188",
            "COMFYUI_SDXL_CHECKPOINT": "",
        })
        provider._load_base_config()
        with pytest.raises(ComfyUIConfigurationError, match="COMFYUI_SDXL_CHECKPOINT"):
            provider._validate_provider_config()

    def test_config_not_validated_at_init(self):
        """Instantiation must not fail even with empty config."""
        # Should NOT raise even though no checkpoint is set
        provider = ComfyUISDXLProvider(config={})
        assert provider is not None


# ── FLUX provider config ───────────────────────────────────────────────────────

class TestFluxProviderConfig:

    def _full_config(self) -> dict:
        return {
            "COMFYUI_BASE_URL": "http://127.0.0.1:8188",
            "COMFYUI_FLUX_UNET": "flux.safetensors",
            "COMFYUI_FLUX_CLIP_L": "clip_l.safetensors",
            "COMFYUI_FLUX_T5XXL": "t5xxl_fp16.safetensors",
            "COMFYUI_FLUX_VAE": "ae.safetensors",
        }

    def test_raises_config_error_if_unet_missing(self):
        cfg = self._full_config()
        cfg["COMFYUI_FLUX_UNET"] = ""
        provider = ComfyUIFluxSchnellProvider(config=cfg)
        provider._load_base_config()
        with pytest.raises(ComfyUIConfigurationError, match="COMFYUI_FLUX_UNET"):
            provider._validate_provider_config()

    def test_raises_config_error_if_clip_l_missing(self):
        cfg = self._full_config()
        cfg["COMFYUI_FLUX_CLIP_L"] = ""
        provider = ComfyUIFluxSchnellProvider(config=cfg)
        provider._load_base_config()
        with pytest.raises(ComfyUIConfigurationError, match="COMFYUI_FLUX_CLIP_L"):
            provider._validate_provider_config()

    def test_raises_config_error_if_t5xxl_missing(self):
        cfg = self._full_config()
        cfg["COMFYUI_FLUX_T5XXL"] = ""
        provider = ComfyUIFluxSchnellProvider(config=cfg)
        provider._load_base_config()
        with pytest.raises(ComfyUIConfigurationError, match="COMFYUI_FLUX_T5XXL"):
            provider._validate_provider_config()

    def test_raises_config_error_if_vae_missing(self):
        cfg = self._full_config()
        cfg["COMFYUI_FLUX_VAE"] = ""
        provider = ComfyUIFluxSchnellProvider(config=cfg)
        provider._load_base_config()
        with pytest.raises(ComfyUIConfigurationError, match="COMFYUI_FLUX_VAE"):
            provider._validate_provider_config()

    def test_config_not_validated_at_init(self):
        """Instantiation with empty config must not raise."""
        provider = ComfyUIFluxSchnellProvider(config={})
        assert provider is not None
