from __future__ import annotations

from typing import Callable, TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

F = TypeVar("F", bound=Callable)


class RetryManager:
    def __init__(self, max_attempts: int = 3, base_delay: float = 2.0) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    def async_retry(self) -> Callable[[F], F]:
        # Wrap network and LLM calls with the same retry policy to keep failure handling uniform.
        return retry(
            reraise=True,
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=self.base_delay, min=self.base_delay, max=8),
            retry=retry_if_exception_type(Exception),
        )
