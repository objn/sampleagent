import unittest
from types import SimpleNamespace

from langchain.messages import ToolMessage

from Models.redis import ToolResultCacheMiddleware


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.writes = []

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value
        self.writes.append((key, ttl, value))


class FakeAsyncRedis:
    def __init__(self):
        self.values = {}
        self.writes = []

    async def get(self, key):
        return self.values.get(key)

    async def setex(self, key, ttl, value):
        self.values[key] = value
        self.writes.append((key, ttl, value))


def request(name, args, call_id):
    return SimpleNamespace(tool_call={"name": name, "args": args, "id": call_id})


class ToolCacheTests(unittest.TestCase):
    def test_successful_result_is_cached_for_five_minutes(self):
        cache = ToolResultCacheMiddleware("redis://localhost:2126/1", {"lookup"})
        cache.redis = FakeRedis()
        calls = []

        def handler(req):
            calls.append(req)
            return ToolMessage(content="found", tool_call_id=req.tool_call["id"])

        first = cache.wrap_tool_call(request("lookup", {"q": "x"}, "call-1"), handler)
        second = cache.wrap_tool_call(request("lookup", {"q": "x"}, "call-2"), handler)

        self.assertEqual(first.content, "found")
        self.assertEqual(second.content, "found")
        self.assertEqual(second.tool_call_id, "call-2")
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.redis.writes[0][1], 300)

    def test_side_effecting_tool_is_never_cached(self):
        cache = ToolResultCacheMiddleware(
            "redis://localhost:2126/1", {"send_email"}, excluded_tools={"send_email"}
        )
        cache.redis = FakeRedis()
        calls = []

        def handler(req):
            calls.append(req)
            return ToolMessage(content="sent", tool_call_id=req.tool_call["id"])

        req = request("send_email", {"to": "a@example.com"}, "call-1")
        cache.wrap_tool_call(req, handler)
        cache.wrap_tool_call(req, handler)

        self.assertEqual(len(calls), 2)
        self.assertEqual(cache.redis.writes, [])

    def test_failed_result_is_not_cached(self):
        cache = ToolResultCacheMiddleware("redis://localhost:2126/1", {"lookup"})
        cache.redis = FakeRedis()

        def handler(req):
            return ToolMessage(
                content="failed", status="error", tool_call_id=req.tool_call["id"]
            )

        cache.wrap_tool_call(request("lookup", {}, "call-1"), handler)
        self.assertEqual(cache.redis.writes, [])


class AsyncToolCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_call_uses_same_five_minute_ttl(self):
        cache = ToolResultCacheMiddleware("redis://localhost:2126/1", {"lookup"})
        cache.async_redis = FakeAsyncRedis()
        calls = []

        async def handler(req):
            calls.append(req)
            return ToolMessage(content="found", tool_call_id=req.tool_call["id"])

        await cache.awrap_tool_call(request("lookup", {"q": "x"}, "call-1"), handler)
        cached = await cache.awrap_tool_call(
            request("lookup", {"q": "x"}, "call-2"), handler
        )

        self.assertEqual(cached.tool_call_id, "call-2")
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.async_redis.writes[0][1], 300)
