from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.extractor import ExtractorAgent
from config.settings import ExtractorConfig, LLMConfig
from models.product import Product
from utils.retry import RetryManager


@pytest.fixture
def extractor():
    return ExtractorAgent(
        ExtractorConfig(llm_fallback_threshold=0.5),
        LLMConfig(api_key=None),
        RetryManager(max_attempts=1, base_delay=0),
    )


def test_extract_from_api_payload(sample_product_json, extractor):
    products = extractor.extract_from_api_payload(
        sample_product_json,
        "https://www.safcodental.com/catalog/gloves",
        "gloves",
    )

    assert len(products) == 1
    product = products[0]
    assert product.product_name == "Premium Nitrile Exam Gloves"
    assert product.sku == "CRN-9004-S"
    assert product.category == "gloves"
    assert product.extraction_method == "api_intercept"


def test_extract_from_api_payload_rejects_category_shell(extractor):
    products = extractor.extract_from_api_payload(
        {
            "name": "Dental Exam Gloves",
            "price": "31.49",
            "description": "Category copy",
        },
        "https://www.safcodental.com/catalog/gloves",
        "gloves",
    )

    assert products == []


def test_extract_from_api_payload_rejects_navigation_entry_without_sku(extractor):
    # Site-menu API responses look like products but lack item numbers — they must be rejected.
    products = extractor.extract_from_api_payload(
        [
            {"name": "Endo Files Subcategories", "url": "/endodontics/endo-files"},
            {"name": "Gloves Products", "url": "/catalog/gloves"},
        ],
        "https://www.safcodental.com/catalog/gloves",
        "gloves",
    )

    assert products == []


def test_extract_from_api_payload_handles_jsonld_nested_brand(extractor):
    # schema.org JSON-LD encodes brand as {"@type": "Brand", "name": "..."} — must produce str.
    products = extractor.extract_from_api_payload(
        {
            "@type": "Product",
            "@id": "https://www.safcodental.com/5-0-polysorb-suture.html",
            "name": "5-0 POLYSORB Coated Suture",
            "sku": "L1622",
            "brand": {"@type": "Brand", "name": "Covidien"},
            "offers": {"price": "29.99", "availability": "InStock"},
            "image": "https://www.safcodental.com/media/catalog/product/s/u/suture.jpg",
        },
        "https://www.safcodental.com/catalog/sutures-surgical-products",
        "sutures",
    )

    assert len(products) == 1
    assert products[0].brand == "Covidien"
    assert products[0].sku == "L1622"
    assert products[0].extraction_method == "api_intercept"


def test_extract_from_listing_html(sample_listing_html, extractor):
    products = extractor.extract_from_listing_html(
        sample_listing_html,
        "https://www.safcodental.com/catalog/gloves",
        "gloves",
    )

    names = {product.product_name for product in products}

    assert "Aquasoft Nitrile Exam Gloves" in names
    assert "MaxTouch Latex Exam Gloves" in names
    assert "JSON-LD Surgical Suture" in names
    assert all("/catalog/" not in product.product_url for product in products)
    assert {product.extraction_method for product in products} == {
        "api_intercept",
        "dom_selector",
    }


@pytest.mark.asyncio
async def test_extract_from_html(sample_product_html, extractor):
    product = await extractor.extract_from_html(
        sample_product_html,
        "https://www.safcodental.com/product/gloves/crn-9004-s",
        "gloves",
    )

    assert product is not None
    assert product.product_name == "Premium Nitrile Exam Gloves"
    assert product.brand == "Cranberry"
    assert product.sku == "CRN-9004-S"
    assert product.price == "$14.99"
    assert product.image_urls == ["https://www.safcodental.com/images/crn9004s.jpg"]
    assert product.extraction_method == "dom_selector"


@pytest.mark.asyncio
async def test_extract_from_html_skips_catalog_url(sample_product_html, extractor):
    product = await extractor.extract_from_html(
        sample_product_html,
        "https://www.safcodental.com/catalog/gloves",
        "gloves",
    )

    assert product is None


@pytest.mark.asyncio
async def test_enrich_with_llm_fills_missing_fields():
    # LLM enrichment fires when unit_pack_size/specifications/category_hierarchy are absent
    # but a description is present — verifies the agentic enrichment path.
    extractor = ExtractorAgent(
        ExtractorConfig(llm_fallback_threshold=0.5),
        LLMConfig(api_key="test-key"),
        RetryManager(max_attempts=1, base_delay=0),
    )
    product = Product(
        product_name="Alasta Pro Nitrile Gloves",
        brand="Safco Dental",
        sku="DRCDK",
        category="gloves",
        product_url="https://www.safcodental.com/product/alasta-pro",
        description="Each box contains 200 premium nitrile gloves. Fentanyl-tested and chemo-approved.",
        extraction_method="api_intercept",
        scraped_at=datetime.now(timezone.utc),
    )
    mock_response = MagicMock()
    mock_response.content = (
        '{"unit_pack_size": "200/box", '
        '"specifications": {"Material": "Nitrile", "Certifications": "Fentanyl-tested"}, '
        '"category_hierarchy": ["Gloves", "Nitrile Exam Gloves"]}'
    )
    with patch.object(extractor, "_llm", create=True, new=MagicMock()):
        extractor._llm.ainvoke = AsyncMock(return_value=mock_response)
        enriched = await extractor.enrich_with_llm(product)

    assert enriched.unit_pack_size == "200/box"
    assert enriched.specifications == {"Material": "Nitrile", "Certifications": "Fentanyl-tested"}
    assert enriched.category_hierarchy == ["Gloves", "Nitrile Exam Gloves"]
    assert enriched.extraction_method == "api_intercept"  # method unchanged after enrichment


@pytest.mark.asyncio
async def test_enrich_with_llm_skips_when_no_description():
    extractor = ExtractorAgent(
        ExtractorConfig(llm_fallback_threshold=0.5),
        LLMConfig(api_key="test-key"),
        RetryManager(max_attempts=1, base_delay=0),
    )
    product = Product(
        product_name="Mystery Glove",
        sku="MG-001",
        category="gloves",
        product_url="https://www.safcodental.com/product/mystery-glove",
        extraction_method="api_intercept",
        scraped_at=datetime.now(timezone.utc),
    )
    with patch.object(extractor, "_llm", create=True, new=MagicMock()) as mock_llm:
        mock_llm.ainvoke = AsyncMock()
        enriched = await extractor.enrich_with_llm(product)
        mock_llm.ainvoke.assert_not_called()

    assert enriched is product  # same object, no changes


@pytest.mark.asyncio
async def test_enrich_with_llm_skips_when_llm_disabled(extractor):
    # extractor fixture has api_key=None so LLM is disabled
    product = Product(
        product_name="Glove A",
        sku="G-A",
        category="gloves",
        product_url="https://www.safcodental.com/product/glove-a",
        description="Box of 100 nitrile gloves.",
        extraction_method="api_intercept",
        scraped_at=datetime.now(timezone.utc),
    )
    enriched = await extractor.enrich_with_llm(product)

    assert enriched is product  # no-op pass-through
