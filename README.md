# etsy-ai-agent

Local Python CLI that researches Etsy trends and generates complete printable wall art product collections — from concept through images, mockup plans, and Etsy listings.

Targets niches like ukiyo-e woodblock, botanical illustration, vintage travel posters, and cozy illustration styles (inspired by shops like TheWorldGallery and NeuralPrint).

---

## Pipeline stages

| # | Stage | Module | Description |
|---|-------|--------|-------------|
| 1 | Research | `research/web_provider.py` | DuckDuckGo search for Etsy trend data |
| 2 | Concept generation | `agent/analyzer.py` | Claude extracts 2 poster concepts from research |
| 3 | Concept selection | `agent/production_orchestrator.py` | Manual or automatic (first concept) |
| 4 | Prompt optimization | `agent/prompt_optimizer.py` | Claude refines image + negative prompts |
| 5 | Collection generation | `agent/collection_generator.py` | 3–8 unique posters via batched Claude calls |
| 6 | Image generation | `image/openai_provider.py` | gpt-image-2 generates each poster |
| 7 | Vision critique | `agent/vision_critic.py` | Claude Vision scores each generated image |
| 8 | Retry planning | `agent/retry_generator.py` | Claude plans prompt revisions if score is low |
| 9 | Mockup planning | `agent/mockup_generator.py` | Compositing-mode mockup specifications |
| 10 | Listing generation | `agent/listing_generator.py` | Full Etsy listing: title, description, tags, SEO |
| 11 | Finalize | `agent/production_orchestrator.py` | Write manifest, mark complete |

---

## Architecture

```
etsy-ai-agent/
├── agent/                  # Pipeline logic (Claude calls)
│   ├── config.py           # Env vars
│   ├── claude_client.py    # Anthropic API wrapper
│   ├── analyzer.py         # Research → poster concepts
│   ├── prompt_optimizer.py # Concept → optimized prompts
│   ├── collection_generator.py  # Batched collection (Bible + Posters)
│   ├── vision_critic.py    # Image quality scoring
│   ├── retry_generator.py  # Prompt revision planning
│   ├── mockup_generator.py # Mockup specifications
│   ├── listing_generator.py # Etsy listing copy
│   └── production_orchestrator.py  # Orchestrates all stages
├── research/               # Research provider abstraction
│   ├── base.py             # ResearchProvider ABC
│   ├── web_provider.py     # DuckDuckGo (live)
│   └── mock_provider.py    # Offline mock data
├── image/                  # Image provider abstraction
│   ├── base.py             # ImageProvider ABC
│   ├── openai_provider.py  # gpt-image-2 (live)
│   └── mock_provider.py    # Offline mock
├── tests/                  # Pytest unit tests (no API calls)
│   ├── test_collection_batching.py
│   └── test_production_orchestrator.py
├── scripts/
│   └── verify_setup.py     # Environment health check
├── test_production.py      # Interactive CLI entry point
└── outputs/                # Generated per run (gitignored)
```

---

## Requirements

- Python 3.11+
- Anthropic API key (Claude)
- OpenAI API key (gpt-image-2 image generation)

---

## Installation

### macOS / Linux

```bash
git clone https://github.com/sercanfurunci/etsy-ai-agent.git
cd etsy-ai-agent

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Windows PowerShell

```powershell
git clone https://github.com/sercanfurunci/etsy-ai-agent.git
cd etsy-ai-agent

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## Environment setup

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required variables:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
IMAGE_PROVIDER=openai
```

Run the setup checker:

```bash
python3 scripts/verify_setup.py
```

---

## Running unit tests

```bash
python3 -m pytest -q
```

No API keys needed. All network calls are replaced with fakes.

---

## Running integration scripts

These make live API calls and cost money.

```bash
# Full production run (interactive)
python3 test_production.py

# Research only
python3 test_web_provider.py

# Collection generation only
python3 test_collection.py

# Listing generation only
python3 test_listing.py
```

---

## Output directory structure

Each production run creates:

```
outputs/
└── 20260722_143000_cozy-cafe-wall-art/
    ├── manifest.json               # Stage lifecycle + status
    ├── request.json                # Original ProductionRequest
    ├── research/
    │   └── result.json
    ├── concepts/
    │   ├── concepts.json
    │   └── selected_concept.json
    ├── prompts/
    │   └── optimized_prompt.json
    ├── collection/
    │   └── collection_plan.json
    ├── images/
    │   └── poster_01/
    │       ├── original.png
    │       ├── final.png           # Copy of last accepted attempt
    │       ├── vision_report_1.json
    │       ├── retry_plan_1.json
    │       └── attempts.json
    ├── mockups/
    │   └── mockup_plan.json
    └── listing/
        └── listing_plan.json
```

---

## Continuing development on another computer

See [MIGRATION.md](MIGRATION.md) for exact step-by-step instructions.

---

## Common setup errors

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'ddgs'` | `pip install -r requirements.txt` |
| `ANTHROPIC_API_KEY is not set in .env` | Add key to `.env` |
| `ValueError: query must not be empty` | Enter a non-empty query when prompted |
| `_TruncatedResponseError` | Transient Claude token limit issue; retry the run |

---

## Security

**Never commit `.env`.** It is excluded in `.gitignore`. If you accidentally stage it:

```bash
git rm --cached .env
```

Your API keys should never appear in any tracked file.

---

## Current limitations

- Image generation uses 1024×1024 square output (gpt-image-2 default); final print files require upscaling
- Vision critique uses Claude's base vision — not fine-tuned for print quality
- No resume capability: failed runs restart from stage 1
- No Etsy API integration; listings are generated as JSON, not uploaded
- Collection size is capped at 8 posters per run

## Planned next stages

- Stage 10.2: Resume interrupted production runs
- Stage 17: Upscaling / print-ready export (2:3 ratio, 300 DPI equivalent)
- Stage 18: Etsy API draft listing upload
- Stage 19: A/B variation runner
