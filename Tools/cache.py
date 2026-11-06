"""Optional five-minute Redis cache for read-only tool results."""

import hashlib
import json
import logging
import os
from collections.abc import Callable

from redis import Redis
from redis.exceptions import RedisError


logger = logging.getLogger(__name__)
CACHE_TTL_SECONDS = 300
DEFAULT_CACHEABLE_TOOLS = "search_uploaded_knowledge"


def cache_enabled(tool_name: str) -> bool:
    configured = os.getenv("CACHEABLE_TOOLS", DEFAULT_CACHEABLE_TOOLS)
    return tool_name in {name.strip() for name in configured.split(",") if name.strip()}


def cached_tool_result(tool_name: str, arguments: dict, compute: Callable[[], str]) -> str:
    """Cache successful text results; Redis failures leave the tool usable."""
    redis_url = os.getenv("REDIS_URL")
    if not redis_url or not cache_enabled(tool_name):
        return compute()

    payload = json.dumps(
        {"tool": tool_name, "args": arguments},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    key = f"tool-result:v1:{tool_name}:{digest}"
    client = Redis.from_url(redis_url, decode_responses=True)
    try:
        try:
            cached = client.get(key)
        except RedisError:
            logger.warning("Redis tool cache read failed", exc_info=True)
            return compute()
        if cached is not None:
            return cached

        result = compute()
        if isinstance(result, str):
            try:
                client.setex(key, CACHE_TTL_SECONDS, result)
            except RedisError:
                logger.warning("Redis tool cache write failed", exc_info=True)
        return result
    finally:
        client.close()


def invalidate_tool_results(redis_url: str | None, tool_name: str) -> None:
    """Discard cached search results when a new document is indexed."""
    if not redis_url:
        return
    client = Redis.from_url(redis_url, decode_responses=True)
    try:
        try:
            for key in client.scan_iter(match=f"tool-result:v1:{tool_name}:*"):
                client.unlink(key)
        except RedisError:
            logger.warning("Redis tool cache invalidation failed", exc_info=True)
    finally:
        client.close()
