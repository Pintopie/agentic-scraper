# Frontier Dental Agentic Scraper

This repo crawls e-commerce product catalog pages, extracts structured product data using a 5-agent AI pipeline, validates it with Pydantic and exports JSON for future querying.

---

## Architecture Overview

```
config.yaml + .env
       │
┌──────▼────────────────────────────────────────────┐
│               OrchestratorAgent                   │
│  loads config → wires agents → runs categories    │
└──┬────────────┬───────────────┬───────────────────┘
   │            │               │
   ▼            ▼               ▼
Navigator   Classifier      Extractor ──► Validator
  │                             │              │
Playwright             ┌────────┴────────┐  SQLite
+ XHR                  │  Tier 1: XHR    │  dedup
intercept              │  Tier 2: DOM    │  + JSON
+ pagination           │  Tier 3: LLM   │  export
                       │  + LLM enrich  │
                       └─────────────────┘
```

| Agent | LLM use? | Responsibility |
| --- | --- | --- |
| OrchestratorAgent | No | Loads config, wires agents, runs category crawls, exports JSON output. |
| NavigatorAgent | No | Opens pages with Playwright, captures XHR/fetch JSON, discovers product/category links, records visited/failed URLs. |
| PageClassifierAgent | Yes (ambiguous pages) | Classifies pages as `category_listing`, `product_detail`, `irrelevant`, or `unknown` using heuristics first, LLM for ambiguous pages. |
| ExtractorAgent | Yes (sparse DOM + enrichment) | 3-tier extraction: intercepted JSON → DOM selectors → LLM fallback. After tier-1 extraction, calls LLM to fill missing structured fields (`unit_pack_size`, `specifications`, `category_hierarchy`) from description text. |
| ValidatorAgent | No | Validates `Product`, deduplicates by `sku + product_url`, writes per-category and merged JSON to `output/`. |


## Why This Approach

The target site is JavaScript-rendered, so static HTTP requests are insufficient. 

I use Playwright to provide a real browser with network interception, allowing us to capture clean JSON API responses (tier 1) before even parsing the DOM.

LLM agents are used selectively:
- **PageClassifier**: only invoked when heuristics can't determine page type, keeping classification cost near zero for known URL patterns.
- **ExtractorAgent tier 3**: invoked when DOM selector coverage falls below the configured threshold (`llm_fallback_threshold`).
- **ExtractorAgent enrichment**: invoked after every tier-1 (api_intercept) product that is missing `unit_pack_size`, `specifications`, or `category_hierarchy` but has a `description`. The LLM parses the description text to fill these structured fields. Logs `llm_enrichment_start` and `llm_enrichment_complete` with `enriched_fields` to make LLM decisions fully traceable.

The normal path is deterministic and efficient (XHR + CSS selectors). LLM is reserved for gaps and ambiguity to avoid calling it on every page, which would be costly and unnecessary.

The `allowed_domains` set is derived at runtime from the category URLs in `config.yaml`.

---

## Setup

```bash
cd frontier-dental-scraper
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
# edit .env: set NVIDIA_API_KEY (or NIM_API_KEY) to enable the LLM path
```

---

## Run

**Bare Python:**

```bash
python main.py --config config.yaml
```

**Docker:**

```bash
docker compose up --build scraper
```

Output files appear in `output/`. The SQLite checkpoint (`output/checkpoint.db`) persists across runs via the volume mount.
The committed `output/*.json` files are sample reviewer artifacts; local or Docker runs overwrite them with fresh crawl output.

**Resume / fresh run:**

```bash
# skip already-visited URLs (default when checkpoint exists)
python main.py --config config.yaml

# reset checkpoint and output for configured categories, start over
python main.py --config config.yaml --fresh
```

---

## Sample Output Schema

```python
class Product(BaseModel):
    product_name: str
    brand: Optional[str]
    sku: Optional[str]
    category_hierarchy: List[str]   # filled by LLM enrichment when absent from API
    category: str
    product_url: str
    price: Optional[str]
    unit_pack_size: Optional[str]   # filled by LLM enrichment when absent from API
    availability: Optional[str]
    description: Optional[str]
    specifications: Dict[str, str]  # filled by LLM enrichment when absent from API
    image_urls: List[str]
    alternative_products: List[str]
    extraction_method: Literal["api_intercept", "dom_selector", "llm_fallback"]
    scraped_at: datetime
```

**`extraction_method` as a data quality signal:**

| Value | Meaning |
| --- | --- |
| `api_intercept` | Clean JSON captured from XHR (most reliable) |
| `dom_selector` | CSS selectors on rendered HTML (normal fallback) |
| `llm_fallback` | LLM extracted from raw HTML (signals selector drift) |

---

## Sample Output

Committed sample datasets:
- `output/products.json`
- `output/gloves.json`
- `output/sutures.json`

```json
{
  "product_name": "Alasta Pro",
  "brand": "Safco Dental",
  "sku": "DRCDK",
  "category_hierarchy": ["Gloves", "Nitrile Exam Gloves"],
  "category": "gloves",
  "product_url": "https://www.safcodental.com/product/alasta-pro",
  "price": "23.49",
  "unit_pack_size": "200/box",
  "availability": "In Stock",
  "description": "Each box contains 200 premium nitrile gloves...",
  "specifications": {
    "Material": "Nitrile",
    "Certifications": "Fentanyl-tested, Chemo-approved"
  },
  "image_urls": ["https://www.safcodental.com/media/catalog/product/d/r/drcdk.jpg"],
  "alternative_products": [],
  "extraction_method": "api_intercept",
  "scraped_at": "2026-04-29T17:00:00+00:00"
}
```

---

## Exception/Failure Handling

- **Retries:** tenacity wraps page loads and LLM calls — 3 attempts, exponential backoff (2s → 4s → 8s), configurable via `retry.max_attempts` and `retry.base_delay`.
- **Failed URLs:** logged with full context and stored in `failed_urls` table. The crawl continues; one bad page never stops the run.
- **Resumability:** re-running skips already-visited URLs via the `visited_urls` SQLite table. Safe to Ctrl+C and restart at any time.
- **LLM circuit breaker:** if any LLM call raises an unrecoverable error, the agent sets `_llm_disabled_reason` and bypasses all subsequent LLM calls for that run, falling back to heuristics only.
- **idempotent output:** ValidatorAgent deduplicates products by `sku + product_url` before writing JSON, ensuring that re-runs don't create duplicates in the output files.

---

## Limitations

- Categories are seeded in `config.yaml` rather than auto-discovered from the Safco homepage.
- Prices may require login and can be `null`.
- The LLM enrichment step fires only when the description is non-empty. Products with no description won't have LLM-filled structured fields.
- Selector drift reduces DOM extraction quality; the `selector_hints` field in LLM fallback output provides repair suggestions.
- NVIDIA NIM free tier has rate limits — tune `request_delay` and `concurrency` in `config.yaml`. (**Note**: Can use any LLM provider that supports the required API calls; the code is modular. I use NVIDIA NIM for its generous free tier and strong performance on structured extraction tasks.)
- Playwright Chromium makes the Docker image larger (~1.5 GB) than a static HTTP scraper would be.

---

## Room for production hardening and scaling:

| Stage | Change |
| --- | --- |
| Current | One Docker container, SQLite checkpoint, local JSON. |
| Scale v1 | One container per category; SQLite → PostgreSQL for concurrent writes. |
| Scale v2 | Redis URL queue; independent Navigator and Extractor workers. |
| Scale v3 | Kubernetes workers with Browserless or a shared Playwright browser pool. |
| Orchestration | Airflow or Prefect for scheduling, DAG visibility, and alerting. |

For future scaling: the modular agent design allows us to independently scale the Navigator (which is I/O bound) and Extractor (which is CPU/LLM bound). We could run multiple Navigators feeding URLs into a Redis queue, with a pool of Extractor workers consuming from that queue. For LLM calls, we can implement a circuit breaker pattern to disable LLM fallback if we hit rate limits or errors, ensuring the crawl continues with heuristics only.


