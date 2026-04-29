from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional

from models.product import Product
from utils.logger import get_logger


class CheckpointStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.logger = get_logger("checkpoint")
        self._init_schema()
        self.logger.info("checkpoint_ready", db_path=str(self.db_path))

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS visited_urls (
                    url TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    visited_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS failed_urls (
                    url TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    error TEXT NOT NULL,
                    failed_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS products (
                    dedup_key TEXT PRIMARY KEY,
                    sku TEXT,
                    product_url TEXT NOT NULL,
                    category TEXT NOT NULL,
                    product_name TEXT NOT NULL,
                    extraction_method TEXT NOT NULL,
                    product_json TEXT NOT NULL,
                    scraped_at TEXT NOT NULL
                );
                """
            )

    def close(self) -> None:
        self._conn.close()

    def clear_categories(self, categories: List[str]) -> None:
        if not categories:
            return
        placeholders = ",".join("?" for _ in categories)
        with self._lock, self._conn:
            # Fresh runs only clear the categories the user asked to rescrape.
            self._conn.execute(
                f"DELETE FROM visited_urls WHERE category IN ({placeholders})",
                categories,
            )
            self._conn.execute(
                f"DELETE FROM failed_urls WHERE category IN ({placeholders})",
                categories,
            )
            self._conn.execute(
                f"DELETE FROM products WHERE category IN ({placeholders})",
                categories,
            )
        self.logger.info("checkpoint_categories_cleared", categories=categories)

    def has_visited(self, url: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM visited_urls WHERE url = ? LIMIT 1", (url,)
        ).fetchone()
        return row is not None

    def mark_visited(self, url: str, category: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO visited_urls (url, category) VALUES (?, ?)",
                (url, category),
            )
        self.logger.info("checkpoint_marked_visited", url=url, category=category)

    def mark_failed(self, url: str, category: str, error: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO failed_urls (url, category, error)
                VALUES (?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    category = excluded.category,
                    error = excluded.error,
                    failed_at = CURRENT_TIMESTAMP
                """,
                (url, category, error[:2000]),
            )
        self.logger.info(
            "checkpoint_marked_failed",
            url=url,
            category=category,
            error=error[:500],
        )

    def insert_product(self, product: Product) -> bool:
        # SQLite enforces idempotency for the same SKU/url pair through the dedup key.
        payload = product.model_dump(mode="json")
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO products (
                    dedup_key, sku, product_url, category, product_name,
                    extraction_method, product_json, scraped_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product.dedup_key(),
                    product.sku,
                    product.product_url,
                    product.category,
                    product.product_name,
                    product.extraction_method,
                    json.dumps(payload, ensure_ascii=False),
                    payload["scraped_at"],
                ),
            )
            inserted = cursor.rowcount == 1
        self.logger.info(
            "checkpoint_product_recorded",
            product_name=product.product_name,
            sku=product.sku,
            category=product.category,
            inserted=inserted,
            extraction_method=product.extraction_method,
        )
        return inserted

    def list_products(self, category: Optional[str] = None) -> List[Dict]:
        # Export reads from the checkpoint store so the JSON output is always derived from stored state.
        if category:
            rows = self._conn.execute(
                "SELECT product_json FROM products WHERE category = ? ORDER BY product_name",
                (category,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT product_json FROM products ORDER BY category, product_name"
            ).fetchall()
        self.logger.info(
            "checkpoint_products_loaded",
            category=category,
            count=len(rows),
        )
        return [json.loads(row["product_json"]) for row in rows]
