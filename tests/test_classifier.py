from __future__ import annotations

import pytest

from agents.classifier import PageClassifierAgent
from config.settings import LLMConfig
from utils.retry import RetryManager


@pytest.fixture
def classifier():
    return PageClassifierAgent(LLMConfig(api_key=None), RetryManager(max_attempts=1, base_delay=0))


@pytest.mark.asyncio
async def test_catalog_url_stays_listing_even_with_product_markers(classifier):
    result = await classifier.classify(
        "https://www.safcodental.com/catalog/gloves",
        "<html><h1>Dental Exam Gloves</h1><button>Add to Cart</button></html>",
    )

    assert result.page_type == "category_listing"
    assert result.should_follow is True


@pytest.mark.asyncio
async def test_non_catalog_category_page_stays_listing(classifier):
    result = await classifier.classify(
        "https://www.safcodental.com/endodontics/gates-glidden-drills-peaso-reamers",
        """
        <html>
          <div class="products-grid">
            <a class="product-item-link">NTI Peeso Reamers</a>
            <button>Add to Cart</button>
          </div>
        </html>
        """,
    )

    assert result.page_type == "category_listing"


@pytest.mark.asyncio
async def test_select_product_links_filters_site_chrome(classifier):
    links = await classifier.select_product_links(
        page_url="https://www.safcodental.com/catalog/gloves",
        html="<html></html>",
        category="gloves",
        candidates=[
            {
                "href": "https://www.safcodental.com/catalog/gloves",
                "text": "Gloves",
                "class": "nav-link",
            },
            {
                "href": "https://www.safcodental.com/aquasoft-nitrile-exam-gloves.html",
                "text": "Aquasoft Nitrile Exam Gloves",
                "class": "product-item-link",
            },
            {
                "href": "https://www.safcodental.com/customer/account/login",
                "text": "Sign In",
                "class": "",
            },
        ],
    )

    assert links == ["https://www.safcodental.com/aquasoft-nitrile-exam-gloves.html"]
