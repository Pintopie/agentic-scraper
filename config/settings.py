from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class CategoryConfig(BaseModel):
    name: str
    url: str


class ScraperConfig(BaseModel):
    concurrency: int = Field(default=2, ge=1)
    request_delay: float = Field(default=1.5, ge=0)
    max_pages: int = Field(default=20, ge=1)
    max_products_per_category: int = Field(default=10, ge=1)
    timeout_ms: int = Field(default=30000, ge=1000)
    headless: bool = True


class ExtractorConfig(BaseModel):
    llm_fallback_threshold: float = Field(default=0.5, ge=0, le=1)


class RetryConfig(BaseModel):
    max_attempts: int = Field(default=3, ge=1)
    base_delay: float = Field(default=2.0, ge=0)


class OutputConfig(BaseModel):
    dir: Path = Path("output")
    checkpoint_db: Path = Path("checkpoint.db")


class LLMConfig(BaseModel):
    api_key: Optional[str] = None
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "meta/llama-4-maverick-17b-128e-instruct"
    temperature: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


class Settings(BaseModel):
    categories: List[CategoryConfig]
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    project_root: Path

    @property
    def allowed_domains(self) -> Set[str]:
        """All netloc variants (with and without www) derived from category URLs."""
        domains: Set[str] = set()
        for cat in self.categories:
            netloc = urlparse(cat.url).netloc
            if netloc:
                domains.add(netloc)
                bare = netloc.removeprefix("www.")
                domains.add(bare)
                domains.add(f"www.{bare}")
        return domains


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _normalize_openai_base_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return url
    # LangChain expects the base URL, not the fully qualified /chat/completions endpoint.
    return url.removesuffix("/chat/completions").rstrip("/")


def load_settings(config_path: str) -> Settings:
    load_dotenv()
    config_file = Path(config_path).expanduser().resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    # Environment variables override YAML so secrets and model targets stay out of source control.
    raw.setdefault("llm", {})
    raw["llm"].update(
        {
            "api_key": (
                os.getenv("NVIDIA_API_KEY")
                or os.getenv("NIM_API_KEY")
                or raw["llm"].get("api_key")
            ),
            "base_url": _normalize_openai_base_url(
                os.getenv("NIM_BASE_URL") or raw["llm"].get("base_url")
            ),
            "model": os.getenv("NIM_MODEL") or raw["llm"].get("model"),
        }
    )
    raw["project_root"] = config_file.parent

    settings = Settings.model_validate(raw)
    settings.output.dir = _resolve_path(settings.output.dir, settings.project_root)
    settings.output.checkpoint_db = _resolve_path(
        settings.output.checkpoint_db, settings.project_root
    )
    return settings
