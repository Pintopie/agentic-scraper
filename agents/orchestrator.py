from __future__ import annotations

from agents.classifier import PageClassifierAgent
from agents.extractor import ExtractorAgent
from agents.navigator import NavigatorAgent
from agents.validator import ValidatorAgent
from config.settings import Settings
from utils.browser import BrowserSession
from utils.checkpoint import CheckpointStore
from utils.logger import get_logger
from utils.retry import RetryManager


class OrchestratorAgent:
    def __init__(self, settings: Settings, fresh: bool = False) -> None:
        self.settings = settings
        self.fresh = fresh
        self.logger = get_logger("orchestrator")
        self.retry_manager = RetryManager(
            max_attempts=settings.retry.max_attempts,
            base_delay=settings.retry.base_delay,
        )
        self.checkpoint = CheckpointStore(settings.output.checkpoint_db)
        self.classifier = PageClassifierAgent(settings.llm, self.retry_manager)
        self.extractor = ExtractorAgent(
            settings.extractor, settings.llm, self.retry_manager
        )
        self.validator = ValidatorAgent(self.checkpoint, settings.output.dir)
        self.navigator = NavigatorAgent(
            settings.scraper,
            self.retry_manager,
            self.checkpoint,
            self.classifier,
            self.extractor,
            self.validator,
        )

    async def run(self) -> None:
        if self.fresh:
            # Fresh mode only resets the configured categories so historical data stays available for others.
            categories = [category.name for category in self.settings.categories]
            self.checkpoint.clear_categories(categories)
            self.validator.clear_output_files(categories)
            self.logger.info("fresh_checkpoint_cleared", categories=categories)

        self.logger.info(
            "scraper_started",
            categories=[category.name for category in self.settings.categories],
            llm_enabled=self.settings.llm.enabled,
            checkpoint_db=str(self.settings.output.checkpoint_db),
        )
        try:
            async with BrowserSession(
                headless=self.settings.scraper.headless,
                timeout_ms=self.settings.scraper.timeout_ms,
            ) as browser:
                completed_categories: list[str] = []
                for category in self.settings.categories:
                    # Run categories sequentially so each pass can reuse the same checkpoint/output semantics.
                    self.logger.info(
                        "category_run_started",
                        category=category.name,
                        url=category.url,
                    )
                    await self.navigator.crawl_category(
                        browser, category.name, category.url
                    )
                    completed_categories.append(category.name)
                    counts = self.validator.export_json(completed_categories)
                    self.logger.info(
                        "category_export_complete",
                        category=category.name,
                        counts=counts,
                    )
                    self.logger.info(
                        "category_run_finished",
                        category=category.name,
                        url=category.url,
                    )
            self.logger.info("scraper_finished")
        finally:
            self.checkpoint.close()
