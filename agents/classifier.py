from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import LLMConfig
from utils.logger import get_logger
from utils.retry import RetryManager


PageType = Literal["category_listing", "product_detail", "unknown", "irrelevant"]


@dataclass
class ClassificationResult:
    page_type: PageType
    reason: str
    should_follow: bool = False


class PageClassifierAgent:
    def __init__(
        self,
        llm_config: LLMConfig,
        retry_manager: RetryManager,
        allowed_domains: Optional[Set[str]] = None,
    ) -> None:
        self.llm_config = llm_config
        self.retry_manager = retry_manager
        self._allowed_domains = allowed_domains
        self.logger = get_logger("classifier")
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

    async def classify(self, url: str, html: str) -> ClassificationResult:
        # Heuristics decide the common cases; the LLM is reserved for ambiguous pages.
        self.logger.info(
            "page_classification_start",
            url=url,
            html_length=len(html),
            llm_available=bool(self.llm),
        )
        heuristic = self._heuristic_classify(url, html)
        self.logger.info(
            "page_classification_heuristic",
            url=url,
            page_type=heuristic.page_type,
            reason=heuristic.reason,
            should_follow=heuristic.should_follow,
        )
        if heuristic.page_type != "unknown":
            return heuristic

        if not self.llm:
            self.logger.info("page_classification_no_llm", url=url)
            return heuristic

        try:
            result = await self._classify_with_llm(url, html[:2000])
            self.logger.info(
                "llm_page_classified",
                url=url,
                page_type=result.page_type,
                reason=result.reason,
                should_follow=result.should_follow,
            )
            return result
        except Exception as exc:
            self.logger.warning("llm_classification_failed", url=url, error=str(exc))
            self._llm_disabled_reason = str(exc)
            return heuristic

    def _heuristic_classify(self, url: str, html: str) -> ClassificationResult:
        lowered_url = url.lower()
        lowered_html = html.lower()

        if self._allowed_domains and not any(d in lowered_url for d in self._allowed_domains):
            return ClassificationResult("irrelevant", "off-domain URL")

        # Catalog URLs are known category listings even if the DOM contains product-like text.
        if "/catalog/" in lowered_url:
            return ClassificationResult(
                "category_listing",
                "catalog URL is a listing page",
                True,
            )

        product_markers = [
            "add to cart",
            "item #",
            "item#",
            "sku",
            "product-details",
            "product detail",
            "availability",
        ]
        listing_markers = [
            "page_type=category",
            '"page_type":"category"',
            "product-item",
            "products-grid",
            "products list",
            "pagination",
            "load more",
            "product-grid",
            "product-list",
            "sort by",
            "filter",
            "category",
        ]

        if lowered_url.endswith(".html") and any(
            marker in lowered_html for marker in product_markers
        ):
            return ClassificationResult("product_detail", "product detail URL and markers")

        if any(marker in lowered_html for marker in listing_markers):
            return ClassificationResult("category_listing", "listing DOM markers", True)

        if re.search(r"/(product|item|p)/", lowered_url) and any(
            marker in lowered_html for marker in product_markers
        ):
            return ClassificationResult("product_detail", "product URL plus detail markers")

        if any(marker in lowered_html for marker in product_markers) and (
            "<h1" in lowered_html or "product-title" in lowered_html
        ):
            return ClassificationResult("product_detail", "detail DOM markers")

        return ClassificationResult(
            "unknown",
            "no strong deterministic page markers",
            should_follow=(
                not self._allowed_domains
                or any(d in lowered_url for d in self._allowed_domains)
            ),
        )

    async def select_product_links(
        self,
        page_url: str,
        html: str,
        candidates: List[Dict[str, str]],
        category: str,
    ) -> List[str]:
        # Prefer deterministic link scoring first so the crawl stays cheap and reproducible.
        heuristic_links = self._heuristic_product_links(candidates)
        self.logger.info(
            "product_link_selection_heuristic",
            url=page_url,
            category=category,
            candidates=len(candidates),
            selected=len(heuristic_links),
        )
        if not self.llm or not candidates:
            return heuristic_links

        try:
            links = await self._select_links_with_llm(page_url, html[:3000], candidates, category)
            self.logger.info(
                "product_link_selection_llm",
                url=page_url,
                category=category,
                selected=len(links),
            )
            return links or heuristic_links
        except Exception as exc:
            self.logger.warning(
                "llm_link_selection_failed",
                url=page_url,
                category=category,
                error=str(exc),
            )
            self._llm_disabled_reason = str(exc)
            return heuristic_links

    def _heuristic_product_links(self, candidates: List[Dict[str, str]]) -> List[str]:
        links: List[str] = []
        seen = set()
        for candidate in candidates:
            href = candidate.get("href", "")
            text = candidate.get("text", "")
            class_name = candidate.get("class", "")
            if not self._looks_like_product_link(href, text, class_name, self._allowed_domains):
                continue
            if href not in seen:
                seen.add(href)
                links.append(href)
        return links

    @staticmethod
    def _looks_like_product_link(
        href: str,
        text: str,
        class_name: str = "",
        allowed_domains: Optional[Set[str]] = None,
    ) -> bool:
        parsed = urlparse(href)
        path = parsed.path.lower()
        lowered_text = text.lower().strip()
        lowered_class = class_name.lower()
        if not href or (allowed_domains and parsed.netloc not in allowed_domains):
            return False
        if any(
            blocked in path
            for blocked in [
                "/catalog/",
                "/customer/",
                "/checkout/",
                "/cart",
                "/login",
                "/search",
                "/media/",
                "/static/",
            ]
        ):
            return False
        if lowered_text in {"", "home", "shop", "view", "show more", "learn more"}:
            return False
        if "product-item" in lowered_class or "product" in lowered_class:
            return True
        # Safco uses multiple product URL patterns, not just a single /product/ slug.
        if path.endswith(".html"):
            return True
        if path.count("/") >= 2 and not any(
            blocked in path
            for blocked in [
                "/customer/",
                "/checkout/",
                "/cart",
                "/login",
                "/search",
                "/media/",
                "/static/",
            ]
        ):
            return True
        if "-" in path.rsplit("/", 1)[-1] and len(lowered_text) >= 6:
            return True
        if re.search(r"\b(glove|suture|needle|surgical|scalpel|foam|latex|nitrile)\b", lowered_text):
            return True
        return False

    async def _select_links_with_llm(
        self,
        page_url: str,
        snippet: str,
        candidates: List[Dict[str, str]],
        category: str,
    ) -> List[str]:
        @self.retry_manager.async_retry()
        async def invoke() -> List[str]:
            # The prompt is intentionally strict: only return detail URLs, never categories or chrome.
            prompt = (
                "You are selecting product detail URLs from a product category page. "
                "Return strict JSON with key product_urls as an array. Choose only links "
                "that likely open individual product detail pages, not categories, account, "
                "cart, marketing, image, or tracking links."
            )
            compact_candidates = candidates[:80]
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(
                        content=(
                            f"Category: {category}\nPage URL: {page_url}\n"
                            f"HTML snippet:\n{snippet}\n"
                            f"Candidates:\n{json.dumps(compact_candidates)}"
                        )
                    ),
                ]
            )
            payload = self._parse_json(response.content)
            urls = payload.get("product_urls") or []
            if not isinstance(urls, list):
                return []
            allowed = {candidate.get("href") for candidate in candidates}
            return [url for url in urls if isinstance(url, str) and url in allowed]

        return await invoke()

    async def _classify_with_llm(self, url: str, snippet: str) -> ClassificationResult:
        @self.retry_manager.async_retry()
        async def invoke() -> ClassificationResult:
            prompt = (
                "Classify this e-commerce page. Return strict JSON with keys "
                "page_type, reason, should_follow. page_type must be one of "
                "category_listing, product_detail, unknown, irrelevant."
            )
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"URL: {url}\nHTML snippet:\n{snippet}"),
                ]
            )
            payload = self._parse_json(response.content)
            page_type = payload.get("page_type", "unknown")
            if page_type not in {
                "category_listing",
                "product_detail",
                "unknown",
                "irrelevant",
            }:
                page_type = "unknown"
            return ClassificationResult(
                page_type=page_type,
                reason=str(payload.get("reason") or "LLM classification"),
                should_follow=bool(payload.get("should_follow", False)),
            )

        return await invoke()

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group(0) if match else text)
