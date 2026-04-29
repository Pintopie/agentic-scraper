from __future__ import annotations

import asyncio
import re
from collections import deque
from typing import Any, Deque, Dict, List, Set
from urllib.parse import urljoin, urlparse

from playwright.async_api import (
    Browser,
    Page,
    Request,
    Response,
    Route,
    TimeoutError as PlaywrightTimeoutError,
)

from agents.classifier import PageClassifierAgent
from agents.extractor import ExtractorAgent
from agents.validator import ValidatorAgent
from config.settings import ScraperConfig
from utils.checkpoint import CheckpointStore
from utils.logger import get_logger
from utils.retry import RetryManager


class NavigatorAgent:
    def __init__(
        self,
        scraper_config: ScraperConfig,
        retry_manager: RetryManager,
        checkpoint: CheckpointStore,
        classifier: PageClassifierAgent,
        extractor: ExtractorAgent,
        validator: ValidatorAgent,
    ) -> None:
        self.config = scraper_config
        self.retry_manager = retry_manager
        self.checkpoint = checkpoint
        self.classifier = classifier
        self.extractor = extractor
        self.validator = validator
        self.logger = get_logger("navigator")
        self._semaphore = asyncio.Semaphore(scraper_config.concurrency)

    async def crawl_category(self, browser: Browser, category: str, start_url: str) -> int:
        # The queue is intentionally shallow at first; discovered links expand it as the crawl proceeds.
        queue: Deque[str] = deque([start_url])
        queued: Set[str] = {start_url}
        pages_seen = 0
        products_inserted = 0

        self.logger.info(
            "category_queue_initialized",
            category=category,
            start_url=start_url,
            queued=len(queued),
            max_pages=self.config.max_pages,
            max_products=self.config.max_products_per_category,
        )

        while (
            queue
            and pages_seen < self.config.max_pages
            and products_inserted < self.config.max_products_per_category
        ):
            url = queue.popleft()
            self.logger.info(
                "page_dequeued",
                category=category,
                url=url,
                remaining_queue=len(queue),
                pages_seen=pages_seen,
            )
            if self.checkpoint.has_visited(url) and self._can_skip_visited_url(url):
                # Category pages are allowed to re-run; non-category pages can be skipped after success.
                self.logger.info("url_skipped_checkpoint", url=url, category=category)
                continue

            try:
                result = await self._crawl_page(browser, category, url)
                pages_seen += 1
                products_inserted += result["products_inserted"]
                for link in result["links"]:
                    if link not in queued and self._is_allowed_link(link, start_url):
                        queued.add(link)
                        queue.append(link)
                await asyncio.sleep(self.config.request_delay)
            except Exception as exc:
                self.checkpoint.mark_failed(url, category, str(exc))
                self.logger.error("page_failed", url=url, category=category, error=str(exc))

        self.logger.info(
            "category_crawl_finished",
            category=category,
            pages_seen=pages_seen,
            queued=len(queue),
            products_inserted=products_inserted,
            product_cap=self.config.max_products_per_category,
        )
        return products_inserted

    async def _crawl_page(self, browser: Browser, category: str, url: str) -> Dict[str, Any]:
        async with self._semaphore:
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            )
            await context.route("**/*", self._route_request)
            page = await context.new_page()
            page.set_default_timeout(self.config.timeout_ms)
            intercepted_products = []
            self.logger.info(
                "page_context_created",
                category=category,
                url=url,
                timeout_ms=self.config.timeout_ms,
            )

            async def handle_response(response: Response) -> None:
                products = await self._extract_response_products(response, page.url, category)
                if products:
                    intercepted_products.extend(products)
                    self.logger.info(
                        "api_products_intercepted",
                        url=page.url,
                        response_url=response.url,
                        count=len(products),
                    )

            page.on(
                "response",
                lambda response: asyncio.create_task(handle_response(response)),
            )

            try:
                self.logger.info("page_load_starting", category=category, url=url)
                await self._goto(page, url)
                await self._settle_spa(page)
                html = await page.content()
                current_url = page.url
                classification = await self.classifier.classify(current_url, html)
                self.logger.info(
                    "page_classified",
                    category=category,
                    url=current_url,
                    page_type=classification.page_type,
                    reason=classification.reason,
                    should_follow=classification.should_follow,
                    html_length=len(html),
                )

                inserted = self.validator.validate_many(intercepted_products)
                links: List[str] = []
                if intercepted_products:
                    self.logger.info(
                        "intercepted_products_ready",
                        category=category,
                        url=current_url,
                        count=len(intercepted_products),
                    )

                if classification.page_type == "product_detail":
                    # Detail pages are the normal extraction path after navigation.
                    self.logger.info(
                        "detail_extraction_start",
                        category=category,
                        url=current_url,
                    )
                    product = await self.extractor.extract_from_html(html, current_url, category)
                    if product and self.validator.validate_and_store(product):
                        inserted += 1
                    self.logger.info(
                        "detail_extraction_complete",
                        category=category,
                        url=current_url,
                        extracted=bool(product),
                    )
                elif classification.page_type == "category_listing":
                    # Listing pages can yield products directly, even before following detail links.
                    listing_products = self.extractor.extract_from_listing_html(
                        html, current_url, category
                    )
                    inserted += self.validator.validate_many(listing_products)
                    if listing_products:
                        self.logger.info(
                            "listing_products_extracted",
                            category=category,
                            url=current_url,
                            count=len(listing_products),
                            inserted=inserted,
                        )
                    self.logger.info(
                        "listing_link_discovery_start",
                        category=category,
                        url=current_url,
                    )
                    links = await self._discover_links(page, current_url, category, html)
                    more_links = await self._click_load_more_and_collect(
                        page, current_url, category
                    )
                    links.extend(more_links)
                    self.logger.info(
                        "listing_link_discovery_complete",
                        category=category,
                        url=current_url,
                        discovered=len(links),
                    )
                elif classification.should_follow:
                    self.logger.info(
                        "unknown_page_following_links",
                        category=category,
                        url=current_url,
                    )
                    links = await self._discover_links(page, current_url, category, html)

                self.checkpoint.mark_visited(current_url, category)
                self.logger.info(
                    "page_crawled",
                    url=current_url,
                    category=category,
                    page_type=classification.page_type,
                    links=len(links),
                    products_inserted=inserted,
                )
                return {"links": list(dict.fromkeys(links)), "products_inserted": inserted}
            finally:
                await context.close()

    async def _extract_response_products(
        self, response: Response, page_url: str, category: str
    ) -> List:
        # Only inspect likely JSON payloads from the target domain to avoid parsing tracker noise.
        request_type = response.request.resource_type
        content_type = response.headers.get("content-type", "")
        if request_type not in {"xhr", "fetch"} and "json" not in content_type:
            return []
        if not self._is_candidate_json_response(response.url):
            return []
        self.logger.info(
            "candidate_json_response_seen",
            page_url=page_url,
            response_url=response.url,
            resource_type=request_type,
            content_type=content_type,
            category=category,
        )
        try:
            payload = await response.json()
        except Exception:
            self.logger.debug(
                "response_json_parse_skipped",
                page_url=page_url,
                response_url=response.url,
                category=category,
            )
            return []
        products = self.extractor.extract_from_api_payload(payload, page_url, category)
        self.logger.info(
            "response_json_parsed",
            page_url=page_url,
            response_url=response.url,
            category=category,
            extracted_count=len(products),
        )
        return products

    async def _goto(self, page: Page, url: str) -> None:
        @self.retry_manager.async_retry()
        async def load() -> None:
            self.logger.info("page_goto_attempt", url=url)
            await page.goto(url, wait_until="domcontentloaded", timeout=self.config.timeout_ms)

        await load()

    async def _settle_spa(self, page: Page) -> None:
        try:
            self.logger.info("spa_settle_waiting", url=page.url)
            await page.wait_for_load_state("networkidle", timeout=8000)
        except PlaywrightTimeoutError:
            pass
        await page.mouse.wheel(0, 1200)
        await page.wait_for_timeout(1000)
        self.logger.info("spa_settle_complete", url=page.url)

    async def _discover_links(
        self,
        page: Page,
        base_url: str,
        category: str,
        html: str,
    ) -> List[str]:
        # Scrape all anchors, then progressively narrow them to likely product detail links.
        raw_candidates = await page.eval_on_selector_all(
            "a[href]",
            """nodes => nodes.map(a => ({
                href: a.href,
                text: (a.innerText || a.getAttribute('aria-label') || '').trim(),
                class: a.className || '',
                ariaLabel: a.getAttribute('aria-label') || '',
                title: a.getAttribute('title') || '',
                itemprop: a.getAttribute('itemprop') || ''
            }))""",
        )
        self.logger.info(
            "link_candidates_scanned",
            url=base_url,
            candidates=len(raw_candidates),
        )
        candidates: List[Dict[str, str]] = []
        for link in raw_candidates:
            href = link.get("href")
            if not href:
                continue
            normalized = urljoin(base_url, href).split("#")[0]
            candidates.append(
                {
                    "href": normalized,
                    "text": link.get("text") or link.get("ariaLabel") or link.get("title") or "",
                    "class": str(link.get("class") or ""),
                    "itemprop": str(link.get("itemprop") or ""),
                }
            )
        candidate_pool = list(candidates)
        candidates = self._candidate_links_for_category(candidate_pool, category)
        self.logger.info(
            "link_candidates_category_filtered",
            url=base_url,
            category=category,
            candidates=len(candidates),
        )
        if not candidates:
            # If category-specific scoring is too strict, fall back to broader link retention.
            candidates = self._fallback_candidates_for_category(candidate_pool)
            self.logger.info(
                "link_candidates_category_fallback",
                url=base_url,
                category=category,
                candidates=len(candidates),
            )
        if not candidates:
            # Last resort: keep any non-chrome anchors so the LLM can still score them.
            candidates = self._fallback_candidates_for_category(raw_candidates)
            self.logger.info(
                "link_candidates_raw_fallback",
                url=base_url,
                category=category,
                candidates=len(candidates),
            )
        links = await self.classifier.select_product_links(
            page_url=base_url,
            html=html,
            candidates=candidates,
            category=category,
        )
        self.logger.info(
            "link_candidates_filtered",
            url=base_url,
            selected=len(links),
        )
        return list(dict.fromkeys(links))

    async def _click_load_more_and_collect(
        self, page: Page, base_url: str, category: str
    ) -> List[str]:
        collected: List[str] = []
        button_patterns = [
            re.compile("load more", re.I),
            re.compile("show more", re.I),
            re.compile("next", re.I),
        ]
        for pattern in button_patterns:
            locator = page.get_by_role("button", name=pattern)
            if await locator.count() == 0:
                locator = page.get_by_role("link", name=pattern)
            if await locator.count() == 0:
                continue
            try:
                self.logger.info("load_more_attempt", url=base_url, pattern=pattern.pattern)
                await locator.first.click(timeout=3000)
                await self._settle_spa(page)
                html = await page.content()
                collected.extend(await self._discover_links(page, base_url, category, html))
                self.logger.info(
                    "load_more_success",
                    url=base_url,
                    pattern=pattern.pattern,
                    collected=len(collected),
                )
            except Exception as exc:
                self.logger.debug("load_more_failed", url=base_url, error=str(exc))
            break
        return collected

    async def _route_request(self, route: Route, request: Request) -> None:
        # Abort asset noise so the browser spends time on HTML/JSON, not images and fonts.
        if request.resource_type in {"image", "font", "media"}:
            await route.abort()
            return
        await route.continue_()

    @staticmethod
    def _is_candidate_json_response(url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc not in {"www.safcodental.com", "safcodental.com"}:
            return False
        lowered = url.lower()
        blocked_markers = [
            "google",
            "facebook",
            "pinterest",
            "yotpo",
            "newrelic",
            "clarity",
            "fullstory",
            "userguiding",
            "adobedc",
            "shop.pe",
            "contentsquare",
            "hubspot",
        ]
        return not any(marker in lowered for marker in blocked_markers)

    @staticmethod
    def _candidate_links_for_category(
        candidates: List[Dict[str, str]], category: str
    ) -> List[Dict[str, str]]:
        category_terms = {
            "sutures": [
                "suture",
                "surgical",
                "surgi",
                "scalpel",
                "blade",
                "needle",
                "packing",
                "periodontal",
                "hemost",
                "foam",
                "dressing",
            ],
            "gloves": [
                "glove",
                "nitrile",
                "latex",
                "vinyl",
                "chloroprene",
                "powder",
                "exam",
                "surgical",
            ],
        }.get(category, [])
        filtered: List[Dict[str, str]] = []
        for candidate in candidates:
            href = candidate.get("href", "")
            text = candidate.get("text", "")
            class_name = candidate.get("class", "")
            itemprop = candidate.get("itemprop", "")
            if NavigatorAgent._is_blocked_candidate(href, text):
                continue
            lowered_href = href.lower()
            lowered_text = text.lower()
            lowered_class = class_name.lower()
            path = urlparse(lowered_href).path
            slug = path.rsplit("/", 1)[-1]
            depth = len([part for part in path.split("/") if part])
            class_signal = any(
                term in lowered_class
                for term in ["product", "item", "card", "tile", "listing", "grid", "sku", "price"]
            )
            href_signal = lowered_href.endswith(".html") or (
                depth >= 2 and (len(slug) >= 8 or "-" in slug or len(path) >= 18)
            )
            text_signal = len(lowered_text) >= 12 and len(lowered_text.split()) >= 2
            category_signal = any(
                term in lowered_href or term in lowered_text for term in category_terms
            )
            itemprop_signal = str(itemprop).lower() in {"name", "url", "itemurl"}
            if class_signal or href_signal or (itemprop_signal and (href_signal or text_signal)) or (
                category_signal and (class_signal or href_signal or text_signal)
            ) or (text_signal and depth >= 2):
                filtered.append(candidate)
        return filtered

    @staticmethod
    def _fallback_candidates_for_category(
        candidates: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        fallback: List[Dict[str, str]] = []
        for candidate in candidates:
            href = candidate.get("href", "")
            text = candidate.get("text", "")
            if NavigatorAgent._is_blocked_candidate(href, text):
                continue
            fallback.append(candidate)
        return fallback

    @staticmethod
    def _is_blocked_candidate(href: str, text: str) -> bool:
        lowered_href = href.lower()
        lowered_text = text.lower().strip()
        if any(
            blocked in lowered_href
            for blocked in [
                "/cart",
                "/login",
                "/account",
                "/customer",
                "/checkout",
                "/media",
                "/static",
                "/wishlist",
                "/compare",
                "/search",
            ]
        ):
            return True
        if lowered_text in {
            "",
            "home",
            "shop",
            "view",
            "show more",
            "load more",
            "learn more",
            "read more",
            "next",
            "previous",
            "menu",
            "search",
            "account",
            "cart",
            "checkout",
            "filter",
            "sort",
        }:
            return True
        return False

    @staticmethod
    def _is_allowed_link(url: str, start_url: str) -> bool:
        parsed = urlparse(url)
        start = urlparse(start_url)
        if parsed.netloc != start.netloc:
            return False
        if any(
            blocked in parsed.path.lower()
            for blocked in [
                "/cart",
                "/login",
                "/account",
                "/customer",
                "/checkout",
                "/media",
                "/static",
                "/catalog/",
            ]
        ):
            return False
        return True

    @staticmethod
    def _looks_product_or_catalog_link(url: str, text: str) -> bool:
        lowered = url.lower()
        if any(skip in lowered for skip in ["mailto:", "tel:", "/cart", "/login"]):
            return False
        if any(part in lowered for part in ["/catalog/", "/product/", "/item/", "/p/"]):
            return True
        productish_text = bool(re.search(r"\b(glove|suture|needle|surgical|exam)\b", text))
        return "safcodental.com" in lowered and productish_text

    @staticmethod
    def _can_skip_visited_url(url: str) -> bool:
        return "/catalog/" not in urlparse(url).path.lower()
