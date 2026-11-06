"""Short-lived cache for read-only LangChain tool results."""

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Iterable

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import RedisError


logger = logging.getLogger(__name__)


def invalidate_tool_results(redis_url: str | None, tool_name: str) -> None:
    """Remove cached results for one read-only tool after its source data changes."""
    if not redis_url:
        return
    client = Redis.from_url(redis_url, decode_responses=True)
    try:
        for key in client.scan_iter(match=f"tool-result:v1:{tool_name}:*"):
            client.unlink(key)
    except RedisError:
        logger.warning("Redis tool cache invalidation failed", exc_info=True)
    finally:
        client.close()


class ToolResultCacheMiddleware(AgentMiddleware):
    """Cache successful text results for explicitly selected tools."""

    def __init__(
        self,
        redis_url: str | None,
        cacheable_tools: Iterable[str],
        ttl_seconds: int = 300,
        excluded_tools: Iterable[str] = (),
    ) -> None:
        super().__init__()
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.cacheable_tools = set(cacheable_tools) - set(excluded_tools)
        self.ttl_seconds = ttl_seconds
        self.redis = Redis.from_url(redis_url, decode_responses=True) if redis_url else None
        self.async_redis = (
            AsyncRedis.from_url(redis_url, decode_responses=True) if redis_url else None
        )

    def _cache_key(self, request: ToolCallRequest) -> str | None:
        tool_name = request.tool_call["name"]
        if tool_name not in self.cacheable_tools or self.redis is None:
            return None
        try:
            payload = json.dumps(
                {"tool": tool_name, "args": request.tool_call["args"]},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except (TypeError, ValueError):
            return None
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"tool-result:v1:{tool_name}:{digest}"

    @staticmethod
    def _cached_message(request: ToolCallRequest, content: str) -> ToolMessage:
        return ToolMessage(
            content=content,
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    @staticmethod
    def _cacheable_content(result: ToolMessage | Command) -> str | None:
        if (
            isinstance(result, ToolMessage)
            and result.status == "success"
            and result.artifact is None
            and isinstance(result.content, str)
        ):
            return result.content
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        key = self._cache_key(request)
        if key is None:
            return handler(request)
        try:
            cached = self.redis.get(key)
            if cached is not None:
                return self._cached_message(request, cached)
        except RedisError:
            logger.warning("Redis tool cache read failed", exc_info=True)

        result = handler(request)
        content = self._cacheable_content(result)
        if content is not None:
            try:
                self.redis.setex(key, self.ttl_seconds, content)
            except RedisError:
                logger.warning("Redis tool cache write failed", exc_info=True)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        key = self._cache_key(request)
        if key is None:
            return await handler(request)
        try:
            cached = await self.async_redis.get(key)
            if cached is not None:
                return self._cached_message(request, cached)
        except RedisError:
            logger.warning("Redis tool cache read failed", exc_info=True)

        result = await handler(request)
        content = self._cacheable_content(result)
        if content is not None:
            try:
                await self.async_redis.setex(key, self.ttl_seconds, content)
            except RedisError:
                logger.warning("Redis tool cache write failed", exc_info=True)
        return result
