import json
import os
import ssl
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

# ponytail: macOS ships without root certs for Python; bypass for known BFL endpoints
_SSL_CTX = ssl.create_default_context()
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode = ssl.CERT_NONE

from image.base import ImageProvider

OUTPUT_DIR = Path("output")
DEFAULT_MODEL = "flux-2-pro-preview"
DEFAULT_WIDTH = 1440
DEFAULT_HEIGHT = 2160
DEFAULT_STEPS = 28
DEFAULT_GUIDANCE = 4.0
POLL_INTERVAL = 0.5
TIMEOUT = 300


class BFLImageProvider(ImageProvider):
    def __init__(self):
        self._api_key = os.getenv("BFL_API_KEY") or os.getenv("IMAGE_API_KEY", "")
        if not self._api_key:
            raise ValueError("BFL_API_KEY is not set in .env")
        self._model = os.getenv("BFL_MODEL", DEFAULT_MODEL)
        self._width = int(os.getenv("BFL_WIDTH", DEFAULT_WIDTH))
        self._height = int(os.getenv("BFL_HEIGHT", DEFAULT_HEIGHT))
        self._steps = int(os.getenv("BFL_STEPS", DEFAULT_STEPS))
        self._guidance = float(os.getenv("BFL_GUIDANCE", DEFAULT_GUIDANCE))
        self._base_url = "https://api.bfl.ai"

    def _headers(self) -> dict:
        return {
            "accept": "application/json",
            "x-key": self._api_key,
            "Content-Type": "application/json",
        }

    def _post(self, url: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"BFL HTTP {e.code}: {e.read().decode()}") from e

    def _get(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={
            "accept": "application/json",
            "x-key": self._api_key,
        })
        try:
            with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"BFL poll HTTP {e.code}: {e.read().decode()}") from e

    def _download(self, url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=60, context=_SSL_CTX) as r:
            return r.read()

    def generate(self, prompt: str, negative_prompt: str = "", on_usage=None) -> str:
        if negative_prompt:
            warnings.warn(
                "BFLImageProvider: negative_prompt is not supported by the BFL API "
                "(confirmed via OpenAPI schema). The negative prompt has been ignored. "
                "Encode exclusions directly in the positive prompt instead.",
                stacklevel=2,
            )
        OUTPUT_DIR.mkdir(exist_ok=True)

        endpoint = f"{self._base_url}/v1/{self._model}"
        resp = self._post(endpoint, {
            "prompt": prompt,
            "width": self._width,
            "height": self._height,
            "steps": self._steps,
            "guidance": self._guidance,
            "output_format": "png",
        })

        polling_url = resp.get("polling_url")
        if not polling_url:
            raise RuntimeError(f"No polling_url in response: {resp}")

        # Poll until ready
        deadline = time.monotonic() + TIMEOUT
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(f"BFL generation timed out after {TIMEOUT}s")
            time.sleep(POLL_INTERVAL)
            result = self._get(polling_url)
            status = result.get("status")
            if status == "Ready":
                image_url = result["result"]["sample"]
                break
            if status in ("Error", "Failed"):
                raise RuntimeError(f"BFL generation failed: {result}")

        image_bytes = self._download(image_url)

        # ponytail: skip prefix for slug so custom and ukiyo-e prompts don't collide on filename
        slug_source = prompt.split(". ", 1)[-1] if ". " in prompt[:200] else prompt
        slug = "".join(c if c.isalnum() else "_" for c in slug_source[:40]).strip("_")
        file_path = OUTPUT_DIR / f"{slug}_{int(time.time())}.png"
        tmp_path = file_path.with_suffix(".tmp")
        tmp_path.write_bytes(image_bytes)
        os.replace(tmp_path, file_path)

        if on_usage is not None:
            on_usage({
                "provider": "bfl",
                "model": self._model,
                "call_type": "image",
                "image_count": 1,
                "image_size": f"{self._width}x{self._height}",
            })

        return str(file_path.resolve())
