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

## Job queue

Batch multiple production requests and process them sequentially:

```bash
# Create a queue
python3 scripts/run_queue.py create queues/my_batch

# Add jobs (no API calls)
python3 scripts/run_queue.py add queues/my_batch --query "Vintage Japanese nature" --collection-size 4
python3 scripts/run_queue.py add queues/my_batch --query "Cozy frog wizard" --collection-size 4

# Run all pending jobs
python3 scripts/run_queue.py run queues/my_batch

# Resume after interruption
python3 scripts/run_queue.py resume queues/my_batch

# Inspect status
python3 scripts/run_queue.py list queues/my_batch

# Cancel a pending job
python3 scripts/run_queue.py cancel queues/my_batch job_002

# Remove a stale lock (after a crash)
python3 scripts/run_queue.py unlock queues/my_batch --force
```

Queue state is persisted to `queues/my_batch/queue.json` after every transition.
Production outputs remain under the configured `--output-root` (not inside the queue directory).
Only one queue runner may execute at a time — a `.queue.lock` file prevents accidental parallel launches.

---

---

## Print Export (Stage 11.1)

Export poster images to standard print sizes at 300 DPI. This step is **optional and manual** — run it after a production run to get print-ready files.

### CLI examples

```bash
# Export all supported sizes
python3 scripts/export_prints.py outputs/my_run --all

# Export specific sizes
python3 scripts/export_prints.py outputs/my_run --sizes 2x3 4x5 11x14

# ISO A-series in JPG
python3 scripts/export_prints.py outputs/my_run --sizes A4 A3 --format jpg

# Fill crop mode (center-crop to fill canvas)
python3 scripts/export_prints.py outputs/my_run --sizes 2x3 --crop fill

# Custom background color with pad mode
python3 scripts/export_prints.py outputs/my_run --sizes 4x5 --crop pad --background "#F5F0E8"

# Enable upscaling for small sources
python3 scripts/export_prints.py outputs/my_run --all --upscale

# Overwrite existing exports
python3 scripts/export_prints.py outputs/my_run --all --overwrite

# Machine-readable JSON output
python3 scripts/export_prints.py outputs/my_run --all --json
```

### Python API

```python
from agent.print_export import export_prints

result = export_prints(
    "outputs/my_run",
    sizes=["2x3", "A4", "11x14"],   # None = all 12 sizes
    crop_mode="fit",                  # "fit" | "fill" | "pad"
    output_format="png",              # "png" | "jpg"
    upscale=False,
    background_color="#FFFFFF",
    overwrite=False,
)
```

### Output layout

```
outputs/my_run/
  exports/
    poster_01/
      2x3/
        poster.png
      A4/
        poster.png
      metadata.json     ← per-poster export records + SHA256
    poster_02/
      ...
    export_manifest.json  ← counts, settings, all poster records
```

### Supported sizes (300 DPI)

| Name  | Physical size    | Pixels         |
|-------|-----------------|----------------|
| 2x3   | 2×3 in          | 600×900        |
| 3x4   | 3×4 in          | 900×1200       |
| 4x5   | 4×5 in          | 1200×1500      |
| 5x7   | 5×7 in          | 1500×2100      |
| 11x14 | 11×14 in        | 3300×4200      |
| 16x20 | 16×20 in        | 4800×6000      |
| 18x24 | 18×24 in        | 5400×7200      |
| 24x36 | 24×36 in        | 7200×10800     |
| A5    | 148×210 mm      | 1748×2480      |
| A4    | 210×297 mm      | 2480×3508      |
| A3    | 297×420 mm      | 3508×4961      |
| A2    | 420×594 mm      | 4961×7016      |

### Crop modes

| Mode | Behavior |
|------|----------|
| `fit` | Fit artwork within target, add padding if ratios differ. Never crops, never stretches. |
| `fill` | Scale to cover target fully, center-crop the excess. Never stretches. |
| `pad` | Identical rendering to `fit`; semantically explicit about placing artwork on a colored canvas. |

### Upscaling behavior

- `upscale=False` (default): artwork is never enlarged beyond its source pixel dimensions. The output canvas is still the full target size — artwork is centered and the background fills the rest. A warning is recorded in `ExportRecord.warnings` when the source is smaller than the 300 DPI target.
- `upscale=True`: LANCZOS enlargement is used. `ExportRecord.upscaled` is set to `True` and `scale_factor` records the multiplier.

### DPI note

DPI is embedded in file metadata only (PNG pHYs chunk / JPEG APP0). No new detail or resolution is created — upscaling via LANCZOS interpolates pixels from the source image.

### Overwrite behavior

- `overwrite=False` (default): if an output file already exists, it is skipped and recorded in `PosterExportResult.failed` with the reason `"file exists, use overwrite=True"`. Source files are never touched.
- `overwrite=True`: existing output files are replaced atomically (write to `.tmp`, then `os.replace()`).

---

## Package Builder (Stage 11.2–11.3)

Assembles a production run's outputs into a clean, customer-ready folder and (optionally) a ZIP archive. This step is **optional and manual** — run it after `export_prints` to create a distributable download package.

### CLI examples

```bash
# Basic package (no ZIP)
python3 scripts/build_package.py outputs/my_run

# With ZIP archive
python3 scripts/build_package.py outputs/my_run --zip

# Allow overwriting an existing package
python3 scripts/build_package.py outputs/my_run --overwrite

# Skip individual sections
python3 scripts/build_package.py outputs/my_run --no-mockups
python3 scripts/build_package.py outputs/my_run --no-listing
python3 scripts/build_package.py outputs/my_run --no-prints
python3 scripts/build_package.py outputs/my_run --no-metadata

# Machine-readable JSON output
python3 scripts/build_package.py outputs/my_run --json

# Combine flags
python3 scripts/build_package.py outputs/my_run --zip --overwrite --json
```

### Python API

```python
from agent.package_builder import build_package

result = build_package(
    "outputs/my_run",
    include_prints=True,    # copy Printable Files (exports/)
    include_mockups=True,   # copy Mockups section
    include_listing=True,   # copy Listing files
    include_metadata=True,  # copy manifest.json, request.json, etc.
    create_zip=False,       # create ZIP archive beside package folder
    overwrite=False,        # raise ValueError if package dir already exists
)

print(result.package_path)      # absolute path to package directory
print(result.zip_path)          # absolute path to ZIP (or None)
print(result.manifest.total_files)
print(result.warnings)          # list of non-fatal warnings
```

### Output folder layout

```
outputs/my_run/
  packages/
    package_YYYYMMDD_HHMMSS/
        Printable Files/
            poster_01/
                2x3/
                    poster.png
                A4/
                    poster.png
            poster_02/
                ...
        Preview Images/
            poster_01.png
            poster_02.png
        Mockups/
            poster_01/
                mockup_1.png
        Listing/
            listing.json
            description.txt
        Metadata/
            manifest.json
            request.json
            export_manifest.json
            exports/
                poster_01/
                    metadata.json
        README.txt
        LICENSE.txt
        package_manifest.json
    package_YYYYMMDD_HHMMSS.zip   ← only when --zip
```

### What each folder contains

| Folder | Contents |
|--------|----------|
| `Printable Files/` | High-resolution print-ready exports at 300 DPI (from `exports/`) |
| `Preview Images/` | Master final.png renamed to `poster_XX.png` (from `images/`) |
| `Mockups/` | Room and frame preview images (from `mockups/`) |
| `Listing/` | Product listing text, description, and SEO tags (from `listing/`) |
| `Metadata/` | Technical metadata: manifest.json, request.json, export records |
| `README.txt` | Customer instructions and printing recommendations |
| `LICENSE.txt` | Commercial use license |
| `package_manifest.json` | Machine-readable build record with SHA256 for every file |

### ZIP export

Pass `create_zip=True` (or `--zip` in CLI). The ZIP is written atomically (`.tmp` then `os.replace()`). It stores paths relative to the package folder root — no absolute paths. The archive extracts cleanly to a single folder.

### Overwrite behavior

- `overwrite=False` (default): raises `ValueError` if a package directory with the same timestamp name already exists. Source files are never touched.
- `overwrite=True`: removes and recreates the package directory.
- Source files (`images/`, `exports/`, `listing/`, etc.) are **never modified, moved, or deleted**.

### Known limitations

- Requires `export_prints()` to have been run first when `include_prints=True` (exports/ must exist).
- Requires listing generation to have been run when `include_listing=True` (listing/ must exist).
- The `package_manifest.json` SHA256 entry for itself may not reflect the final written file (self-referential limitation).

---

## Provider Selection (Stage 15.1)

Set `IMAGE_PROVIDER` in `.env` to choose how images are generated:

| Value | Description |
|-------|-------------|
| `openai` (default) | gpt-image-2 via OpenAI API — requires `OPENAI_API_KEY` |
| `comfyui_sdxl` | SDXL via a locally running ComfyUI instance — zero API cost |
| `comfyui_flux_schnell` | FLUX.1 Schnell via a locally running ComfyUI instance — zero API cost |

```bash
# .env
IMAGE_PROVIDER=comfyui_sdxl
COMFYUI_SDXL_CHECKPOINT=juggernautXL.safetensors
```

### Local ComfyUI setup

ComfyUI must be installed and running separately on the same computer. This repository does NOT install ComfyUI or download model weights.

1. Install ComfyUI: https://github.com/comfyanonymous/ComfyUI
2. Download model weights into ComfyUI's `models/` folder
3. Start ComfyUI: `python main.py`
4. Set `IMAGE_PROVIDER` and model filenames in `.env`

Test the connection:

```bash
python3 scripts/test_image_provider.py --provider comfyui_sdxl --health
python3 scripts/test_image_provider.py --provider comfyui_flux_schnell --health
```

Generate a test image:

```bash
python3 scripts/test_image_provider.py \
    --provider comfyui_sdxl \
    --prompt "ukiyo-e crane over misty river" \
    --output test_out.png
```

Dump a workflow without HTTP (ComfyUI can be off):

```bash
python3 scripts/test_image_provider.py \
    --provider comfyui_sdxl --dump-workflow \
    --prompt "test" --output /dev/null
```

**Required env vars for SDXL:**
- `COMFYUI_SDXL_CHECKPOINT` — safetensors filename in ComfyUI's `models/checkpoints/`

**Required env vars for FLUX Schnell:**
- `COMFYUI_FLUX_UNET`, `COMFYUI_FLUX_CLIP_L`, `COMFYUI_FLUX_T5XXL`, `COMFYUI_FLUX_VAE`

See `.env.example` for all optional tuning variables.

---

## Current limitations

- Vision critique uses Claude's base vision — not fine-tuned for print quality
- Interrupted runs can be resumed via `resume_production(run_dir)` or `resume_queue(queue_dir)` — completed stages are not re-executed
- No Etsy API integration; listings are generated as JSON, not uploaded
- Collection size is capped at 8 posters per run
- Queue processes one job at a time (no parallel workers)
- ComfyUI providers only accept localhost connections (127.0.0.1 / localhost / ::1)

## Planned next stages

- Stage 18: Etsy API draft listing upload
- Stage 19: A/B variation runner
