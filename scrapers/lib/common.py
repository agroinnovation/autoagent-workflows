"""
Common utilities for scrapers.
"""

import asyncio
import random
import json
from datetime import datetime
from typing import Any


def output_result(success: bool, data: Any = None, error: str = None, source_url: str = None):
    """Print standardized JSON result to stdout."""
    result = {
        "success": success,
        "scraped_at": datetime.utcnow().isoformat() + "Z",
    }
    if source_url:
        result["source_url"] = source_url
    if success:
        result["data"] = data
    else:
        result["error"] = error
    print(json.dumps(result, indent=2))


async def rate_limit_delay(min_seconds: float = 240, max_seconds: float = 600):
    """
    Wait between requests to simulate human pace.
    Default: 4-10 minutes (suitable for government sites).
    """
    delay = random.uniform(min_seconds, max_seconds)
    await asyncio.sleep(delay)


async def retry_with_backoff(coro_func, max_attempts: int = 3, base_delay: float = 5.0):
    """
    Retry an async function with exponential backoff.

    Usage:
        result = await retry_with_backoff(lambda: scrape_page(url))
    """
    last_error = None
    for attempt in range(max_attempts):
        try:
            return await coro_func()
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
    raise last_error
