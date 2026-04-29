from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import urlparse

from models.product import Product
from utils.checkpoint import CheckpointStore
from utils.logger import get_logger


class ValidatorAgent:
    def __init__(self, checkpoint: CheckpointStore, output_dir: Path) -> None:
        self.checkpoint = checkpoint
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("validator")

    def validate_and_store(self, product_data: Product) -> bool:
        # Validation happens before the checkpoint write so only schema-clean products persist.
        product = (
            product_data
            if isinstance(product_data, Product)
            else Product.model_validate(product_data)
        )
        if not self._is_exportable_product(product.model_dump(mode="json")):
            self.logger.info(
                "product_rejected_non_detail",
                product_name=product.product_name,
                sku=product.sku,
                category=product.category,
                product_url=product.product_url,
            )
            return False
        inserted = self.checkpoint.insert_product(product)
        self.logger.info(
            "product_validated",
            product_name=product.product_name,
            sku=product.sku,
            category=product.category,
            extraction_method=product.extraction_method,
            inserted=inserted,
        )
        return inserted

    def validate_many(self, products: Iterable[Product]) -> int:
        inserted = 0
        scanned = 0
        for product in products:
            scanned += 1
            if self.validate_and_store(product):
                inserted += 1
        self.logger.info("batch_validation_complete", scanned=scanned, inserted=inserted)
        return inserted

    def export_json(self, categories: List[str]) -> Dict[str, int]:
        # Export from SQLite, not from in-memory crawl state, so interrupted runs remain resumable.
        counts: Dict[str, int] = {}
        all_products = self._exportable_products(self.checkpoint.list_products())
        self._write_json(self.output_dir / "products.json", all_products)
        counts["products"] = len(all_products)

        for category in categories:
            products = self._exportable_products(self.checkpoint.list_products(category))
            self._write_json(self.output_dir / f"{category}.json", products)
            counts[category] = len(products)

        self.logger.info("products_exported", output_dir=str(self.output_dir), counts=counts)
        return counts

    def clear_output_files(self, categories: List[str]) -> None:
        # Fresh runs remove stale JSON so the filesystem mirrors the checkpoint state exactly.
        paths = [self.output_dir / "products.json"]
        paths.extend(self.output_dir / f"{category}.json" for category in categories)
        removed: List[str] = []
        for path in paths:
            if path.exists():
                path.unlink()
                removed.append(str(path))
        if removed:
            self.logger.info("output_files_cleared", removed=removed)

    @staticmethod
    def _write_json(path: Path, payload: List[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def _exportable_products(self, products: List[dict]) -> List[dict]:
        exportable = [product for product in products if self._is_exportable_product(product)]
        rejected = len(products) - len(exportable)
        if rejected:
            self.logger.info("products_rejected_from_export", rejected=rejected)
        return exportable

    @staticmethod
    def _is_exportable_product(product: dict) -> bool:
        product_url = str(product.get("product_url") or "")
        path = urlparse(product_url).path.lower()
        if "/catalog/" in path:
            return False
        name = str(product.get("product_name") or "").strip()
        if not name:
            return False
        return True
