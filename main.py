from __future__ import annotations

import argparse
import asyncio

from agents.orchestrator import OrchestratorAgent
from config.settings import load_settings
from utils.logger import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frontier Dental agentic scraper")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear checkpoint/output state for the configured categories before crawling",
    )
    return parser.parse_args()


async def async_main() -> None:
    # Keep the entrypoint thin: parse config, configure logging, then hand off to the orchestrator.
    args = parse_args()
    configure_logging()
    settings = load_settings(args.config)
    orchestrator = OrchestratorAgent(settings, fresh=args.fresh)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(async_main())
