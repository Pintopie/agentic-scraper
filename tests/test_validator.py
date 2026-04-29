from __future__ import annotations

from datetime import datetime, timezone

from agents.validator import ValidatorAgent
from models.product import Product
from utils.checkpoint import CheckpointStore


def test_validator_rejects_category_shell(tmp_db, tmp_path):
    store = CheckpointStore(tmp_db)
    validator = ValidatorAgent(store, tmp_path)
    try:
        inserted = validator.validate_and_store(
            Product(
                product_name="Dental Exam Gloves",
                category="gloves",
                product_url="https://www.safcodental.com/catalog/gloves",
                extraction_method="dom_selector",
                scraped_at=datetime.now(timezone.utc),
            )
        )

        assert inserted is False
        assert store.list_products("gloves") == []
    finally:
        store.close()


def test_validator_accepts_detail_without_sku_or_brand(tmp_db, tmp_path):
    store = CheckpointStore(tmp_db)
    validator = ValidatorAgent(store, tmp_path)
    try:
        inserted = validator.validate_and_store(
            Product(
                product_name="NTI Peeso Reamers",
                category="sutures",
                product_url="https://www.safcodental.com/nti-peeso-reamers.html",
                price="35.99",
                extraction_method="dom_selector",
                scraped_at=datetime.now(timezone.utc),
            )
        )

        assert inserted is True
        assert len(store.list_products("sutures")) == 1
    finally:
        store.close()


def test_validator_clears_output_files(tmp_db, tmp_path):
    store = CheckpointStore(tmp_db)
    validator = ValidatorAgent(store, tmp_path)
    try:
        (tmp_path / "products.json").write_text("[]", encoding="utf-8")
        (tmp_path / "gloves.json").write_text("[]", encoding="utf-8")

        validator.clear_output_files(["gloves"])

        assert not (tmp_path / "products.json").exists()
        assert not (tmp_path / "gloves.json").exists()
    finally:
        store.close()
