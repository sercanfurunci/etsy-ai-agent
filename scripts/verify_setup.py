#!/usr/bin/env python3
"""
Verify the local environment is correctly configured.
No live API calls are made.
"""
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on the path regardless of where this script is run from
sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = "[PASS]"
FAIL = "[FAIL]"
failures = []


def ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")


def fail(msg: str) -> None:
    print(f"  {FAIL}  {msg}")
    failures.append(msg)


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


# ── Python version ─────────────────────────────────────────────────────────────
section("Python version")
ver = sys.version_info
print(f"  Python {sys.version}")
if ver >= (3, 11):
    ok(f"Python {ver.major}.{ver.minor} meets minimum requirement (3.11+)")
else:
    fail(f"Python {ver.major}.{ver.minor} is below the minimum requirement (3.11+)")

# ── Environment variables ──────────────────────────────────────────────────────
section("Environment variables")

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
    ok("python-dotenv loaded .env")
except ImportError:
    fail("python-dotenv not installed — run: pip install -r requirements.txt")

required_vars = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
for var in required_vars:
    val = os.getenv(var, "")
    if val:
        ok(f"{var} is set (value hidden)")
    else:
        fail(f"{var} is not set — add it to your .env file")

optional_vars = ["IMAGE_PROVIDER", "IMAGE_API_KEY"]
for var in optional_vars:
    val = os.getenv(var, "")
    status = "set" if val else "not set (optional)"
    print(f"  [INFO] {var}: {status}")

# ── Directory creation ─────────────────────────────────────────────────────────
section("Directory creation")
with tempfile.TemporaryDirectory() as td:
    test_path = Path(td) / "outputs" / "test_run"
    try:
        test_path.mkdir(parents=True)
        ok(f"Can create nested output directories")
    except OSError as e:
        fail(f"Cannot create directories: {e}")

# ── Module imports ─────────────────────────────────────────────────────────────
section("Module imports")

modules = [
    ("anthropic", "anthropic"),
    ("openai", "openai"),
    ("ddgs", "ddgs"),
    ("agent.config", "agent.config"),
    ("agent.analyzer", "agent.analyzer"),
    ("agent.prompt_optimizer", "agent.prompt_optimizer"),
    ("agent.collection_generator", "agent.collection_generator"),
    ("agent.vision_critic", "agent.vision_critic"),
    ("agent.retry_generator", "agent.retry_generator"),
    ("agent.mockup_generator", "agent.mockup_generator"),
    ("agent.listing_generator", "agent.listing_generator"),
    ("agent.production_orchestrator", "agent.production_orchestrator"),
    ("research.web_provider", "research.web_provider"),
    ("research.mock_provider", "research.mock_provider"),
    ("image.openai_provider", "image.openai_provider"),
]

for label, mod in modules:
    try:
        __import__(mod)
        ok(f"import {label}")
    except ImportError as e:
        fail(f"import {label}: {e}")
    except Exception as e:
        # e.g. config loading failures — treat as warning
        print(f"  [WARN] import {label}: {e}")

# ── Summary ────────────────────────────────────────────────────────────────────
section("Summary")
if failures:
    print(f"\n  {len(failures)} issue(s) found:\n")
    for f in failures:
        print(f"    - {f}")
    sys.exit(1)
else:
    print("\n  All checks passed. Ready to run.")
    sys.exit(0)
