# Migration Guide

Exact steps to move this project to a new computer.

---

## 1. Clone the repository

```bash
git clone https://github.com/sercanfurunci/etsy-ai-agent.git
cd etsy-ai-agent
```

## 2. Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 4. Copy your `.env` file

`.env` is intentionally excluded from Git (it contains secrets).
Copy it manually from your previous machine, or create a new one:

```bash
cp .env.example .env
# then open .env and fill in your real API keys
```

Required keys:
- `ANTHROPIC_API_KEY` — from https://console.anthropic.com/
- `OPENAI_API_KEY` — from https://platform.openai.com/api-keys

## 5. Run setup verification

```bash
python3 scripts/verify_setup.py
```

Expected output: `All checks passed. Ready to run.`

If any check fails, the output will tell you exactly what to fix.

## 6. Run unit tests

```bash
python3 -m pytest -q
```

Expected: 45 tests passing (27 collection batching + 18 orchestrator).
No API calls are made during unit tests.

## 7. Run a small integration test

This calls live APIs and costs money — use it to confirm end-to-end connectivity:

```bash
python3 test_production.py
# Enter a short query (e.g. "cozy cafe wall art")
# Choose collection size: 3
# Press Enter for all other options
```

## 8. Generated output folders

The following directories are created at runtime and are **not stored in Git**:

| Directory | Contents |
|-----------|----------|
| `outputs/` | Production runs — manifests, images, listings |
| `output/` | Standalone image generation test outputs |

These grow with each run. Back them up separately if needed.

---

## Common issues

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'ddgs'` | `pip install -r requirements.txt` |
| `ANTHROPIC_API_KEY is not set` | Add key to `.env`, run `source .venv/bin/activate` |
| `python3: command not found` (Windows) | Use `python` instead of `python3` |
| `pytest: command not found` | Activate venv first: `source .venv/bin/activate` |
| `JSONDecodeError` during collection generation | Retry — Claude occasionally returns truncated output; the agent has a built-in retry |
