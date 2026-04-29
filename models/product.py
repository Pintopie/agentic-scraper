from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


ExtractionMethod = Literal["api_intercept", "dom_selector", "llm_fallback"]


class Product(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    # Keep the schema explicit so validation failures surface missing fields early.
    product_name: str
    brand: Optional[str] = None
    sku: Optional[str] = None
    category_hierarchy: List[str] = Field(default_factory=list)
    category: str
    product_url: str
    price: Optional[str] = None
    unit_pack_size: Optional[str] = None
    availability: Optional[str] = None
    description: Optional[str] = None
    specifications: Dict[str, str] = Field(default_factory=dict)
    image_urls: List[str] = Field(default_factory=list)
    alternative_products: List[str] = Field(default_factory=list)
    extraction_method: ExtractionMethod
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("product_name", "category", "product_url")
    @classmethod
    def required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required field cannot be empty")
        return value

    @field_validator("image_urls", "alternative_products")
    @classmethod
    def dedupe_urls(cls, values: List[str]) -> List[str]:
        # Preserve order while dropping duplicate links from noisy pages.
        seen = set()
        deduped: List[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    def dedup_key(self) -> str:
        # SKU can be missing, so pair it with URL to keep the store stable and idempotent.
        sku_part = (self.sku or "").strip().lower()
        return f"{sku_part}|{self.product_url.strip().lower()}"
