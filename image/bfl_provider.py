import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from image.base import ImageProvider

OUTPUT_DIR = Path("output")
DEFAULT_MODEL = "flux-2-pro-preview"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1536
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
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"BFL HTTP {e.code}: {e.read().decode()}") from e

    def _get(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={
            "accept": "application/json",
            "x-key": self._api_key,
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"BFL poll HTTP {e.code}: {e.read().decode()}") from e

    def _download(self, url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()

    def generate(self, prompt: str, on_usage=None) -> str:
        OUTPUT_DIR.mkdir(exist_ok=True)

        endpoint = f"{self._base_url}/v1/{self._model}"
        resp = self._post(endpoint, {
            "prompt": prompt,
            "width": self._width,
            "height": self._height,
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

        slug = "".join(c if c.isalnum() else "_" for c in prompt[:40]).strip("_")
        file_path = OUTPUT_DIR / f"{slug}.png"
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
