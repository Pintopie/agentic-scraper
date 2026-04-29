from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import ExtractorConfig, LLMConfig
from models.product import Product
from utils.logger import get_logger
from utils.retry import RetryManager


class ExtractorAgent:
    def __init__(
        self,
        extractor_config: ExtractorConfig,
        llm_config: LLMConfig,
        retry_manager: RetryManager,
    ) -> None:
        self.extractor_config = extractor_config
        self.llm_config = llm_config
        self.retry_manager = retry_manager
        self.logger = get_logger("extractor")
        self._llm: Optional[ChatOpenAI] = None
        self._llm_disabled_reason: Optional[str] = None

    @property
    def llm(self) -> Optional[ChatOpenAI]:
        if self._llm_disabled_reason or not self.llm_config.enabled:
            return None
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
            )
        return self._llm

    def extract_from_api_payload(
        self, payload: Any, page_url: str, category: str
    ) -> List[Product]:
        # Tier 1 extraction: pull clean product objects directly from intercepted JSON.
        self.logger.info(
            "api_payload_scan_start",
            page_url=page_url,
            category=category,
            payload_type=type(payload).__name__,
        )
        products: List[Product] = []
        for candidate in self._product_candidates(payload):
            self.logger.info(
                "api_candidate_detected",
                page_url=page_url,
                category=category,
                candidate_keys=sorted(list(candidate.keys()))[:20],
            )
            normalized = self._normalize_api_product(candidate, page_url, category)
            if normalized:
                try:
                    products.append(Product.model_validate(normalized))
                    self.logger.info(
                        "api_candidate_validated",
                        page_url=page_url,
                        category=category,
                        product_name=normalized.get("product_name"),
                        sku=normalized.get("sku"),
                    )
                except Exception as exc:
                    self.logger.debug(
                        "api_product_validation_failed",
                        url=page_url,
                        error=str(exc),
                        candidate_keys=list(candidate.keys()),
                    )
        return products

    def extract_from_listing_html(
        self, html: str, page_url: str, category: str
    ) -> List[Product]:
        # Tier 1.5: Safco listing pages often expose enough card data to store products without visiting detail pages.
        soup = BeautifulSoup(html, "lxml")
        products: List[Product] = []
        seen = set()

        for payload in self._listing_script_payloads(soup):
            for product in self.extract_from_api_payload(payload, page_url, category):
                if product.dedup_key() not in seen:
                    seen.add(product.dedup_key())
                    products.append(product)

        for data in self._listing_dom_products(soup, page_url, category):
            try:
                product = Product.model_validate(data)
            except Exception as exc:
                self.logger.debug(
                    "listing_product_validation_failed",
                    page_url=page_url,
                    category=category,
                    error=str(exc),
                )
                continue
            if product.dedup_key() not in seen:
                seen.add(product.dedup_key())
                products.append(product)

        self.logger.info(
            "listing_html_extraction_complete",
            page_url=page_url,
            category=category,
            extracted=len(products),
        )
        return products

    async def extract_from_html(
        self, html: str, product_url: str, category: str
    ) -> Optional[Product]:
        if self._is_category_url(product_url):
            # Listing URLs should never be forced through the detail-page extractor.
            self.logger.info(
                "dom_extraction_skipped_listing",
                url=product_url,
                category=category,
            )
            return None

        self.logger.info(
            "dom_extraction_start",
            url=product_url,
            category=category,
            html_length=len(html),
        )
        dom_data = self._extract_dom_data(html, product_url, category)
        completeness = self._field_completeness(dom_data)
        self.logger.info(
            "dom_extraction_snapshot",
            url=product_url,
            category=category,
            completeness=completeness,
            product_name=dom_data.get("product_name"),
            sku=dom_data.get("sku"),
            price=dom_data.get("price"),
        )

        if completeness >= self.extractor_config.llm_fallback_threshold:
            # When the DOM is already rich enough, validate it directly instead of spending tokens.
            try:
                product = Product.model_validate(dom_data)
                self.logger.info(
                    "dom_extraction_validated",
                    url=product_url,
                    category=category,
                    extraction_method=product.extraction_method,
                )
                return product
            except Exception as exc:
                self.logger.warning("dom_product_validation_failed", url=product_url, error=str(exc))

        if self.llm:
            # LLM fallback is reserved for sparse or irregular pages where selectors stop being reliable.
            self.logger.info(
                "llm_fallback_start",
                url=product_url,
                category=category,
                completeness=completeness,
            )
            llm_product = await self._extract_with_llm(html, product_url, category, dom_data)
            if llm_product:
                self.logger.info(
                    "llm_fallback_complete",
                    url=product_url,
                    category=category,
                    extraction_method=llm_product.extraction_method,
                )
                return llm_product

        if dom_data.get("product_name"):
            try:
                product = Product.model_validate(dom_data)
                self.logger.info(
                    "partial_dom_validation_used",
                    url=product_url,
                    category=category,
                )
                return product
            except Exception as exc:
                self.logger.warning("partial_dom_validation_failed", url=product_url, error=str(exc))
        self.logger.info(
            "extraction_no_product",
            url=product_url,
            category=category,
            completeness=completeness,
        )
        return None

    def _normalize_api_product(
        self, item: Dict[str, Any], page_url: str, category: str
    ) -> Optional[Dict[str, Any]]:
        # Normalize multiple vendor field names into one stable product schema.
        name = self._first_value(
            item,
            [
                "productName",
                "product_name",
                "name",
                "title",
                "descriptionShort",
                "shortDescription",
            ],
        )
        if not name:
            return None

        # Navigation/category JSON entries never have SKUs. Real dental supply products always do.
        # Rejecting no-SKU items prevents site-menu API payloads from appearing as products.
        sku = self._first_value(item, ["sku", "itemNumber", "itemNo", "code"])
        if not sku:
            return None

        url = self._first_value(
            item,
            [
                "productUrl",
                "product_url",
                "canonical_url",
                "url",
                "href",
                "link",
                "@id",
            ],
        )
        if not url:
            url_key = self._first_value(item, ["url_key", "urlKey"])
            url_suffix = self._first_value(item, ["url_suffix", "urlSuffix"]) or ""
            if url_key:
                url = f"/{str(url_key).lstrip('/')}{url_suffix}"

        images = self._coerce_list(
            self._first_value(
                item,
                [
                    "imageUrls",
                    "images",
                    "media_gallery",
                    "mediaGallery",
                    "imageUrl",
                    "image",
                    "small_image",
                    "thumbnail",
                ],
            )
        )
        offers = item.get("offers") if isinstance(item.get("offers"), dict) else {}
        price = (
            self._format_price(
                self._first_value(item, ["price", "priceFormatted", "displayPrice"])
            )
            or self._format_price(offers.get("price"))
            or self._price_from_range(item.get("price_range") or item.get("priceRange"))
        )
        availability = self._normalize_availability(
            self._first_value(item, ["availability", "stockStatus", "stock_status", "inStock"])
            or offers.get("availability")
        )
        product_url = urljoin(page_url, str(url)).split("#")[0] if url else page_url.split("#")[0]
        if self._is_category_shell_candidate(product_url, page_url, item):
            # Reject category shells so the store only contains actual product records.
            self.logger.info(
                "api_candidate_rejected_category_shell",
                page_url=page_url,
                category=category,
                product_name=name,
            )
            return None

        return {
            "product_name": str(name),
            "brand": self._coerce_string(self._first_value(item, ["brand", "manufacturer", "brandName"])),
            "sku": sku,
            "category_hierarchy": self._coerce_list(
                self._first_value(item, ["categoryHierarchy", "breadcrumbs", "categories"])
            ),
            "category": category,
            "product_url": product_url,
            "price": price,
            "unit_pack_size": self._first_value(
                item, ["unitPackSize", "unit_size", "unitSize", "packSize"]
            ),
            "availability": availability,
            "description": self._first_value(
                item, ["description", "longDescription", "summary", "body"]
            ),
            "specifications": self._coerce_specs(
                self._first_value(
                    item,
                    ["specifications", "attributes", "custom_attributes", "features"],
                )
            ),
            "image_urls": self._filter_image_urls(
                [urljoin(page_url, str(image)) for image in images]
            ),
            "alternative_products": self._coerce_list(
                self._first_value(item, ["alternativeProducts", "relatedProducts"])
            ),
            "extraction_method": "api_intercept",
            "scraped_at": datetime.now(timezone.utc),
        }

    def _extract_dom_data(self, html: str, product_url: str, category: str) -> Dict[str, Any]:
        # Detail pages can change shape, so we use a set of broad selectors and keep the output sparse.
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(" ", strip=True)
        product_name = self._select_text(
            soup,
            [
                "h1.product-title",
                ".product-title h1",
                "[data-testid*='product-name']",
                "[class*='product-name']",
                "[class*='ProductName']",
                "h1",
                "meta[property='og:title']",
            ],
        )
        product_name = self._clean_meta_title(product_name)

        sku = self._select_text(
            soup,
            [
                ".sku-value",
                "[class*='sku']",
                "[data-testid*='sku']",
                "[itemprop='sku']",
            ],
        ) or self._regex_value(text, r"(?:Item\s*#|Item\s*No\.?|SKU)\s*:?\s*([A-Z0-9._-]+)")

        return {
            "product_name": product_name,
            "brand": self._select_text(
                soup,
                [
                    ".brand-name",
                    "[class*='brand']",
                    "[itemprop='brand']",
                    "[data-testid*='brand']",
                ],
            ),
            "sku": sku,
            "category_hierarchy": self._breadcrumbs(soup),
            "category": category,
            "product_url": product_url,
            "price": self._select_text(
                soup,
                [
                    ".price",
                    "[class*='price']",
                    "[data-testid*='price']",
                    "[itemprop='price']",
                ],
            ) or self._regex_value(text, r"(\$\s?\d+(?:\.\d{2})?)"),
            "unit_pack_size": self._select_text(
                soup,
                [
                    ".unit-size",
                    "[class*='unit']",
                    "[class*='pack']",
                    "[data-testid*='pack']",
                ],
            ),
            "availability": self._select_text(
                soup,
                [
                    ".availability",
                    "[class*='availability']",
                    "[class*='stock']",
                    "[data-testid*='availability']",
                ],
            ),
            "description": self._select_text(
                soup,
                [
                    ".product-description",
                    "[class*='description']",
                    "[data-testid*='description']",
                    "[itemprop='description']",
                    "meta[name='description']",
                ],
            ),
            "specifications": self._specifications(soup),
            "image_urls": self._image_urls(soup, product_url),
            "alternative_products": self._alternative_products(soup, product_url),
            "extraction_method": "dom_selector",
            "scraped_at": datetime.now(timezone.utc),
        }

    def _listing_dom_products(
        self, soup: BeautifulSoup, page_url: str, category: str
    ) -> Iterable[Dict[str, Any]]:
        # Listing cards are turned into Product-shaped dicts so validation/export stay uniform.
        selectors = [
            "[class*='product-item']",
            "[class*='product-card']",
            "[data-product-sku]",
            "[data-product-id]",
            "li.item.product",
        ]
        seen_nodes = set()
        for card in soup.select(",".join(selectors)):
            node_id = id(card)
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            text = card.get_text(" ", strip=True)
            link = self._listing_product_link(card, page_url)
            name = self._listing_product_name(card)
            if not name or not link:
                continue
            if self._is_category_url(link):
                continue
            sku = (
                card.get("data-product-sku")
                or card.get("data-sku")
                or self._regex_value(text, r"(?:Item\s*#|Item\s*No\.?|SKU)\s*:?\s*([A-Z0-9._-]+)")
            )
            yield {
                "product_name": self._clean_meta_title(name),
                "brand": self._select_text_from_node(
                    card, [".brand-name", "[class*='brand']", "[itemprop='brand']"]
                ),
                "sku": sku,
                "category_hierarchy": self._breadcrumbs(soup),
                "category": category,
                "product_url": link,
                "price": self._select_text_from_node(
                    card, [".price", "[class*='price']", "[itemprop='price']"]
                ) or self._regex_value(text, r"(\$\s?\d+(?:\.\d{2})?)"),
                "unit_pack_size": self._select_text_from_node(
                    card, ["[class*='unit']", "[class*='pack']"]
                ),
                "availability": self._select_text_from_node(
                    card, ["[class*='availability']", "[class*='stock']"]
                ),
                "description": self._select_text_from_node(
                    card, ["[class*='description']", "[itemprop='description']"]
                ),
                "specifications": {},
                "image_urls": self._filter_image_urls(
                    [
                        urljoin(page_url, src)
                        for src in self._card_image_sources(card)
                    ]
                ),
                "alternative_products": [],
                "extraction_method": "dom_selector",
                "scraped_at": datetime.now(timezone.utc),
            }

    @staticmethod
    def _listing_script_payloads(soup: BeautifulSoup) -> Iterable[Any]:
        # Some listings embed JSON-LD or application/json blobs with product data.
        for script in soup.select("script[type='application/ld+json'], script[type='application/json']"):
            text = script.string or script.get_text("", strip=True)
            if not text:
                continue
            try:
                yield json.loads(text)
            except Exception:
                continue

    @staticmethod
    def _listing_product_link(card, page_url: str) -> Optional[str]:
        # Prefer explicit product links inside the card; fall back to any safe anchor if needed.
        for selector in [
            "a.product-item-link[href]",
            "a[class*='product'][href]",
            "a[itemprop='url'][href]",
            "a[href]",
        ]:
            node = card.select_one(selector)
            if not node:
                continue
            href = node.get("href")
            if not href:
                continue
            url = urljoin(page_url, href).split("#")[0]
            lowered = url.lower()
            if any(
                blocked in lowered
                for blocked in ["/cart", "/login", "/customer", "/checkout", "/media", "/static"]
            ):
                continue
            return url
        return None

    @staticmethod
    def _listing_product_name(card) -> Optional[str]:
        # Product names can appear as anchors, headings, or itemprop metadata depending on the template.
        for selector in [
            ".product-item-link",
            "[class*='product-name']",
            "[itemprop='name']",
            "h2",
            "h3",
            "a[href]",
        ]:
            node = card.select_one(selector)
            if not node:
                continue
            value = (
                node.get("content")
                or node.get("title")
                or node.get("aria-label")
                or node.get_text(" ", strip=True)
            )
            if value:
                return re.sub(r"\s+", " ", value).strip()
        return None

    @staticmethod
    def _select_text_from_node(node, selectors: List[str]) -> Optional[str]:
        # Reuse the same selector logic for each listing card so the extraction stays compact.
        for selector in selectors:
            selected = node.select_one(selector)
            if not selected:
                continue
            value = (
                selected.get("content")
                if selected.name == "meta"
                else selected.get_text(" ", strip=True)
            )
            if value:
                return re.sub(r"\s+", " ", value).strip()
        return None

    @staticmethod
    def _card_image_sources(card) -> List[str]:
        sources: List[str] = []
        for image in card.select("img"):
            source = image.get("src") or image.get("data-src") or image.get("data-lazy-src")
            if source:
                sources.append(source)
        return sources

    async def _extract_with_llm(
        self, html: str, product_url: str, category: str, dom_data: Dict[str, Any]
    ) -> Optional[Product]:
        @self.retry_manager.async_retry()
        async def invoke() -> Optional[Product]:
            prompt = (
                "Extract one dental product from this rendered HTML. Return strict JSON "
                "matching these keys: product_name, brand, sku, category_hierarchy, "
                "price, unit_pack_size, availability, description, specifications, "
                "image_urls, alternative_products, selector_hints. Use null for unknown "
                "scalar values. Do not invent data."
            )
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"URL: {product_url}\nHTML:\n{html[:12000]}"),
                ]
            )
            payload = self._parse_json(response.content)
            selector_hints = payload.pop("selector_hints", None)
            if selector_hints:
                self.logger.info(
                    "llm_selector_hints",
                    url=product_url,
                    selector_hints=selector_hints,
                )
            merged = {**dom_data, **{k: v for k, v in payload.items() if v not in (None, "", [])}}
            merged.update(
                {
                    "category": category,
                    "product_url": product_url,
                    "extraction_method": "llm_fallback",
                    "scraped_at": datetime.now(timezone.utc),
                }
            )
            return Product.model_validate(merged)

        try:
            return await invoke()
        except Exception as exc:
            self.logger.warning("llm_extraction_failed", url=product_url, error=str(exc))
            self._llm_disabled_reason = str(exc)
            return None

    def _product_candidates(self, payload: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(payload, dict):
            if self._looks_like_product(payload):
                yield payload
            for value in payload.values():
                yield from self._product_candidates(value)
        elif isinstance(payload, list):
            for value in payload:
                yield from self._product_candidates(value)

    @staticmethod
    def _looks_like_product(item: Dict[str, Any]) -> bool:
        keys = {str(key).lower() for key in item.keys()}
        has_name = bool(
            keys
            & {
                "productname",
                "product_name",
                "name",
                "title",
                "descriptionshort",
                "shortdescription",
            }
        )
        has_product_signal = bool(
            keys
            & {
                "sku",
                "itemnumber",
                "itemno",
                "price",
                "price_range",
                "producturl",
                "canonical_url",
                "url_key",
                "@id",
                "imageurl",
                "small_image",
                "availability",
                "offers",
            }
        )
        return has_name and has_product_signal

    @classmethod
    def _is_category_shell_candidate(
        cls, product_url: str, page_url: str, item: Dict[str, Any]
    ) -> bool:
        if not cls._is_category_url(product_url):
            return False
        sku = cls._first_value(item, ["sku", "itemNumber", "itemNo", "code"])
        explicit_url = cls._first_value(
            item,
            ["productUrl", "product_url", "canonical_url", "url", "href", "link", "url_key"],
        )
        return not sku or not explicit_url or product_url.rstrip("/") == page_url.rstrip("/")

    @staticmethod
    def _is_category_url(url: str) -> bool:
        return "/catalog/" in urlparse(url).path.lower()

    @staticmethod
    def _first_value(item: Dict[str, Any], keys: List[str]) -> Any:
        lowered = {str(key).lower(): value for key, value in item.items()}
        for key in keys:
            value = lowered.get(key.lower())
            if value not in (None, "", [], {}):
                return value
        return None

    @staticmethod
    def _coerce_string(value: Any) -> Optional[str]:
        # schema.org fields like brand arrive as {"@type": "Brand", "name": "..."} — extract name.
        if value is None or value == "":
            return None
        if isinstance(value, dict):
            extracted = value.get("name") or value.get("label") or value.get("value")
            return str(extracted).strip() if extracted else None
        return str(value).strip() or None

    _SCHEMA_ORG_AVAILABILITY: Dict[str, str] = {
        "instock": "In Stock",
        "outofstock": "Out of Stock",
        "preorder": "Pre-Order",
        "discontinued": "Discontinued",
        "limitedavailability": "Limited Availability",
        "onlineonly": "Online Only",
        "soldout": "Sold Out",
    }

    @classmethod
    def _normalize_availability(cls, value: Any) -> Optional[str]:
        # schema.org availability arrives as a URI; map to a human-readable label.
        if not value:
            return None
        text = str(value)
        # Extract the trailing word from schema.org URIs like "https://schema.org/InStock".
        slug = text.rstrip("/").rsplit("/", 1)[-1].lower()
        return cls._SCHEMA_ORG_AVAILABILITY.get(slug, text)

    @staticmethod
    def _coerce_list(value: Any) -> List[str]:
        if value in (None, "", [], {}):
            return []
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, dict):
                    result.append(
                        str(
                            item.get("url")
                            or item.get("file")
                            or item.get("name")
                            or item.get("label")
                            or item
                        )
                    )
                else:
                    result.append(str(item))
            return result
        if isinstance(value, dict):
            direct = (
                value.get("url")
                or value.get("file")
                or value.get("name")
                or value.get("label")
                or value.get("value")
            )
            return [str(direct)] if direct else []
        return [str(value)]

    @staticmethod
    def _coerce_specs(value: Any) -> Dict[str, str]:
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items() if v not in (None, "")}
        if isinstance(value, list):
            specs: Dict[str, str] = {}
            for item in value:
                if isinstance(item, dict):
                    key = (
                        item.get("name")
                        or item.get("label")
                        or item.get("key")
                        or item.get("attribute_code")
                    )
                    val = item.get("value") or item.get("text")
                    if key and val:
                        specs[str(key)] = str(val)
            return specs
        return {}

    @staticmethod
    def _format_price(value: Any) -> Optional[str]:
        if value in (None, "", [], {}):
            return None
        if isinstance(value, dict):
            amount = value.get("value") or value.get("amount")
            currency = value.get("currency") or value.get("currency_code")
            if amount is not None:
                return f"{currency + ' ' if currency else ''}{amount}"
            return None
        return str(value)

    @classmethod
    def _price_from_range(cls, value: Any) -> Optional[str]:
        if not isinstance(value, dict):
            return None
        for path in [
            ("minimum_price", "final_price"),
            ("minimumPrice", "finalPrice"),
            ("regular_price",),
        ]:
            node: Any = value
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            formatted = cls._format_price(node)
            if formatted:
                return formatted
        return None

    @staticmethod
    def _select_text(soup: BeautifulSoup, selectors: List[str]) -> Optional[str]:
        for selector in selectors:
            node = soup.select_one(selector)
            if not node:
                continue
            value = node.get("content") if node.name == "meta" else node.get_text(" ", strip=True)
            if value:
                return re.sub(r"\s+", " ", value).strip()
        return None

    @staticmethod
    def _clean_meta_title(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return re.sub(r"\s*\|\s*Safco.*$", "", value).strip()

    @staticmethod
    def _regex_value(text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _breadcrumbs(soup: BeautifulSoup) -> List[str]:
        crumbs = []
        for node in soup.select(
            ".breadcrumb a, .breadcrumbs a, nav[aria-label*='breadcrumb' i] a"
        ):
            text = node.get_text(" ", strip=True)
            if text and text.lower() not in {"home", "catalog"}:
                crumbs.append(text)
        return crumbs

    @staticmethod
    def _specifications(soup: BeautifulSoup) -> Dict[str, str]:
        specs: Dict[str, str] = {}
        for row in soup.select("table tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.select("th, td")]
            if len(cells) >= 2 and cells[0] and cells[1]:
                specs[cells[0].rstrip(":")] = cells[1]
        for node in soup.select("dl"):
            terms = [term.get_text(" ", strip=True) for term in node.select("dt")]
            definitions = [dd.get_text(" ", strip=True) for dd in node.select("dd")]
            for key, value in zip(terms, definitions):
                if key and value:
                    specs[key.rstrip(":")] = value
        return specs

    @staticmethod
    def _image_urls(soup: BeautifulSoup, base_url: str) -> List[str]:
        urls = []
        for node in soup.select("img"):
            source = node.get("src") or node.get("data-src") or node.get("data-lazy-src")
            if not source:
                continue
            alt = (node.get("alt") or "").lower()
            classes = " ".join(node.get("class", [])).lower()
            if "logo" in alt or "logo" in classes:
                continue
            urls.append(urljoin(base_url, source))
        return ExtractorAgent._filter_image_urls(urls)

    @staticmethod
    def _filter_image_urls(urls: List[str]) -> List[str]:
        filtered: List[str] = []
        seen = set()
        for url in urls:
            if not url or ExtractorAgent._is_noisy_image_url(url):
                continue
            if url not in seen:
                seen.add(url)
                filtered.append(url)
        return filtered

    @staticmethod
    def _is_noisy_image_url(url: str) -> bool:
        parsed = urlparse(url)
        lowered = url.lower()
        path = parsed.path.lower()
        if parsed.netloc and parsed.netloc not in {"www.safcodental.com", "safcodental.com"}:
            return True
        noisy_markers = [
            "facebook.com/tr",
            "pinterest.com",
            "bing.com/action",
            "payment",
            "visa",
            "mastercard",
            "social_icon",
            "phone-call",
            "message-circle",
            "printer",
            "mail-01",
            "truck-01",
            "download.png",
            "iconsearch",
            "carticon",
            "shoppingcart",
            "icon-cart",
            "icon-remove",
            "pencil-line",
            "cms/",
            "wysiwyg/cms",
        ]
        if any(marker in lowered for marker in noisy_markers):
            return True
        if path.startswith("/static/"):
            return True
        if path.startswith("/media/wysiwyg/") or path.startswith("/media/.renditions/"):
            return True
        return False

    @staticmethod
    def _alternative_products(soup: BeautifulSoup, base_url: str) -> List[str]:
        urls = []
        for node in soup.select("a[href]"):
            text = node.get_text(" ", strip=True).lower()
            href = node.get("href")
            if href and any(term in text for term in ["related", "alternative", "similar"]):
                urls.append(urljoin(base_url, href))
        return urls

    @staticmethod
    def _field_completeness(data: Dict[str, Any]) -> float:
        optional_fields = [
            "product_name",
            "brand",
            "sku",
            "price",
            "unit_pack_size",
            "availability",
            "description",
            "image_urls",
        ]
        populated = sum(1 for field in optional_fields if data.get(field))
        return populated / len(optional_fields)

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group(0) if match else text)
