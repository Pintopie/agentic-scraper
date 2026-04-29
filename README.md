# Frontier Dental Agentic Scraper

Crawls Safco Dental Supply category pages for Sutures & Surgical Products and Dental Exam Gloves, extracts product data, validates it with Pydantic, checkpoints progress in SQLite, and exports JSON.

## Architecture

```text
main.py -> OrchestratorAgent
  -> NavigatorAgent: Playwright crawl, pagination/link discovery, XHR interception
  -> PageClassifierAgent: deterministic page typing, optional NVIDIA NIM fallback 
  -> ExtractorAgent: API JSON -> listing cards -> detail DOM selectors -> optional LLM fallback
  -> ValidatorAgent: Pydantic validation, SQLite dedup/checkpoint, JSON export
```

| Agent | Responsibility |
| --- | --- |
| OrchestratorAgent | Loads config, wires dependencies, runs category crawls, exports output. |
| NavigatorAgent | Opens Safco pages with Playwright, captures XHR/fetch JSON, discovers product/category links, and records visited/failed URLs. |
| PageClassifierAgent | Classifies pages as listing, detail, irrelevant, or unknown. Uses heuristics first and NVIDIA NIM via LangChain for ambiguous pages when configured. |
| ExtractorAgent | Extracts products from intercepted JSON, category listing cards, rendered detail selectors, or LLM fallback when selector coverage is low. |
| ValidatorAgent | Validates `Product`, deduplicates by `sku + product_url`, and writes `output/products.json`, `output/gloves.json`, and `output/sutures.json`. |

## Why This Approach

Safco is JavaScript-rendered, so static requests are not enough. So I use Playwright as it gives a real browser and network interception. AI agents are used for classifying ambiguous rendered pages and recovering fields/selectors when the DOM changes. The normal path stays deterministic and cheap through XHR parsing and CSS selectors.

## Setup

```bash
cd frontier-dental-scraper
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
```

Set `NVIDIA_API_KEY` or `NIM_API_KEY` in `.env` to enable the NVIDIA NIM prototype LLM path. Keep `NIM_BASE_URL=https://integrate.api.nvidia.com/v1`; LangChain appends `/chat/completions` internally. Production can swap this OpenAI-compatible client to OpenAI, Claude, Perplexity, or another provider.

## Run

Bare Python:

```bash
python main.py --config config.yaml
```

Docker:

```bash
docker compose up --build scraper
```

Optional SQLite viewer:

```bash
docker compose --profile debug up viewer
```

For a clean rerun that clears checkpoint rows and stale JSON output for the configured categories:

```bash
python main.py --config config.yaml --fresh
```

## Product Schema

```python
class Product(BaseModel):
    product_name: str
    brand: Optional[str]
    sku: Optional[str]
    category_hierarchy: list[str]
    category: str
    product_url: str
    price: Optional[str]
    unit_pack_size: Optional[str]
    availability: Optional[str]
    description: Optional[str]
    specifications: dict[str, str]
    image_urls: list[str]
    alternative_products: list[str]
    extraction_method: Literal["api_intercept", "dom_selector", "llm_fallback"]
    scraped_at: datetime
```

## Failure Handling

Page loads and LLM calls use tenacity retries with exponential backoff from `config.yaml`. Each successful page is stored in `visited_urls`; failed pages are stored in `failed_urls`; products are inserted with `INSERT OR IGNORE`. Re-running the same config resumes from `checkpoint.db` and skips completed URLs.

## Limitations

- Prices may require login and can be `null`.
- Selector drift can reduce DOM extraction quality; LLM fallback logs selector hints for repair.
- NVIDIA NIM free tier can rate-limit; tune `request_delay` and `concurrency`. (Will be retuned to OpenAI or another provider for production.)
- Playwright Chromium makes the Docker image larger than a static HTTP scraper.
- JSON exports are flushed after each category so partial progress is visible during long runs.

## Scaling Path

| Stage | Change |
| --- | --- |
| Prototype | One Docker container, SQLite checkpoint, local JSON output. |
| Scale v1 | One container per category; move SQLite to PostgreSQL for concurrent writes. |
| Scale v2 | Redis URL queue; independent Navigator and Extractor workers. |
| Scale v3 | Kubernetes workers with Browserless or a shared browser pool. |
| Orchestration | Airflow or Prefect for scheduling, visibility, and retries. |
| Observability | Ship JSON logs to Datadog/Grafana and alert on high failure or LLM fallback rates. |

## Data Quality Monitoring

Track the distribution of `extraction_method`. A high `llm_fallback` rate means Safco's DOM or API payloads likely changed. Also monitor Pydantic validation errors and failed URL counts in `checkpoint.db`.
