from __future__ import annotations

from datetime import datetime, timezone

from models.product import Product
from utils.checkpoint import CheckpointStore


def _product(url: str = "https://example.com/product/1") -> Product:
    return Product(
        product_name="Premium Nitrile Exam Gloves",
        sku="CRN-9004-S",
        category="gloves",
        product_url=url,
        extraction_method="dom_selector",
        scraped_at=datetime.now(timezone.utc),
    )


def test_checkpoint_dedupes_products(tmp_db):
    store = CheckpointStore(tmp_db)
    try:
        assert store.insert_product(_product()) is True
        assert store.insert_product(_product()) is False
        assert len(store.list_products("gloves")) == 1
    finally:
        store.close()


def test_checkpoint_tracks_visited_urls(tmp_db):
    store = CheckpointStore(tmp_db)
    try:
        url = "https://www.safcodental.com/catalog/gloves"
        assert store.has_visited(url) is False
        store.mark_visited(url, "gloves")
        assert store.has_visited(url) is True
    finally:
        store.close()
