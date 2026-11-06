import fnmatch
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from redis.exceptions import ConnectionError as RedisConnectionError

from Agents.sample import TicketState
from Tools.cache import CACHE_TTL_SECONDS, invalidate_tool_results
from Tools.customer_history import check_customer_history
from Tools.knowledge_base import search_uploaded_knowledge


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.writes = []
        self.closed = 0

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value
        self.writes.append((key, ttl, value))

    def scan_iter(self, match):
        return (key for key in list(self.values) if fnmatch.fnmatch(key, match))

    def unlink(self, key):
        self.values.pop(key, None)

    def close(self):
        self.closed += 1


def history_graph():
    builder = StateGraph(TicketState)
    builder.add_node("history", ToolNode([check_customer_history]))
    builder.add_edge(START, "history")
    builder.add_edge("history", END)
    return builder.compile()


def history_state(user_id, message_id):
    return {
        "messages": [
            HumanMessage(content="Current ticket", id=message_id),
            AIMessage(content="", tool_calls=[{
                "name": "check_customer_history", "args": {}, "id": "call-1"
            }]),
        ],
        "user_id": user_id,
        "session_id": "current-session",
    }


class ToolCacheTests(unittest.TestCase):
    def test_knowledge_tool_uses_five_minute_cache_when_selected(self):
        redis = FakeRedis()
        matches = [{
            "filename": "faq.txt", "category": "general",
            "page_number": None, "content": "Reset your password",
        }]
        with (
            patch.dict("os.environ", {
                "REDIS_URL": "redis://localhost:2126/1",
                "CACHEABLE_TOOLS": "search_uploaded_knowledge",
            }),
            patch("Tools.cache.Redis.from_url", return_value=redis),
            patch("Tools.knowledge_base.search_knowledge", return_value=matches) as search,
        ):
            first = search_uploaded_knowledge.invoke({"query": "password"})
            second = search_uploaded_knowledge.invoke({"query": "password"})
        self.assertEqual(first, second)
        self.assertIn("Reset your password", second)
        search.assert_called_once_with("password", category=None)
        self.assertEqual(len(redis.writes), 1)
        self.assertEqual(redis.writes[0][1], CACHE_TTL_SECONDS)
        self.assertEqual(CACHE_TTL_SECONDS, 300)
        self.assertEqual(redis.closed, 2)

    def test_each_tool_can_be_disabled_independently(self):
        with (
            patch.dict("os.environ", {
                "REDIS_URL": "redis://localhost:2126/1",
                "CACHEABLE_TOOLS": "check_customer_history",
            }),
            patch("Tools.cache.Redis.from_url") as create_redis,
            patch("Tools.knowledge_base.search_knowledge", return_value=[]) as search,
        ):
            search_uploaded_knowledge.invoke({"query": "missing"})
            search_uploaded_knowledge.invoke({"query": "missing"})
        self.assertEqual(search.call_count, 2)
        create_redis.assert_not_called()

    def test_empty_selection_disables_all_caching(self):
        with (
            patch.dict("os.environ", {
                "REDIS_URL": "redis://localhost:2126/1",
                "CACHEABLE_TOOLS": "",
            }),
            patch("Tools.cache.Redis.from_url") as create_redis,
            patch("Tools.knowledge_base.search_knowledge", return_value=[]) as search,
        ):
            search_uploaded_knowledge.invoke({"query": "missing"})
            search_uploaded_knowledge.invoke({"query": "missing"})
        self.assertEqual(search.call_count, 2)
        create_redis.assert_not_called()

    def test_history_cache_is_scoped_to_customer_and_message(self):
        redis = FakeRedis()
        graph = history_graph()
        with (
            patch.dict("os.environ", {
                "REDIS_URL": "redis://localhost:2126/1",
                "CACHEABLE_TOOLS": "check_customer_history",
            }),
            patch("Tools.cache.Redis.from_url", return_value=redis),
            patch("Tools.customer_history.recent_customer_messages", side_effect=
                  lambda user_id, **_kwargs: [{"id": 1, "content": f"Earlier {user_id}"}]) as recent,
        ):
            first = graph.invoke(history_state("customer-a", "msg-1"))
            repeated = graph.invoke(history_state("customer-a", "msg-1"))
            next_ticket = graph.invoke(history_state("customer-a", "msg-2"))
            other_customer = graph.invoke(history_state("customer-b", "msg-1"))
        self.assertEqual(recent.call_count, 3)
        self.assertEqual(len(redis.writes), 3)
        self.assertEqual(first["messages"][-1].content, repeated["messages"][-1].content)
        self.assertIn("customer-a", next_ticket["messages"][-1].content)
        self.assertIn("customer-b", other_customer["messages"][-1].content)

    def test_history_without_message_id_bypasses_cache(self):
        redis = FakeRedis()
        state = {
            "messages": [HumanMessage(content="Question")],
            "user_id": "customer-a", "session_id": "session-a",
        }
        with (
            patch.dict("os.environ", {
                "REDIS_URL": "redis://localhost:2126/1",
                "CACHEABLE_TOOLS": "check_customer_history",
            }),
            patch("Tools.cache.Redis.from_url", return_value=redis) as create_redis,
            patch("Tools.customer_history.recent_customer_messages", return_value=[]) as recent,
        ):
            check_customer_history.func(state)
            check_customer_history.func(state)
        self.assertEqual(recent.call_count, 2)
        create_redis.assert_not_called()

    def test_tool_error_is_not_cached(self):
        redis = FakeRedis()
        with (
            patch.dict("os.environ", {
                "REDIS_URL": "redis://localhost:2126/1",
                "CACHEABLE_TOOLS": "search_uploaded_knowledge",
            }),
            patch("Tools.cache.Redis.from_url", return_value=redis),
            patch("Tools.knowledge_base.search_knowledge", side_effect=ValueError("failed")),
        ):
            with self.assertRaisesRegex(ValueError, "failed"):
                search_uploaded_knowledge.invoke({"query": "bad"})
        self.assertEqual(redis.writes, [])
        self.assertEqual(redis.closed, 1)

    def test_redis_failure_falls_back_to_tool(self):
        redis = FakeRedis()
        redis.get = lambda _key: (_ for _ in ()).throw(RedisConnectionError("offline"))
        with (
            patch.dict("os.environ", {
                "REDIS_URL": "redis://localhost:2126/1",
                "CACHEABLE_TOOLS": "search_uploaded_knowledge",
            }),
            patch("Tools.cache.Redis.from_url", return_value=redis),
            patch("Tools.knowledge_base.search_knowledge", return_value=[]) as search,
        ):
            with self.assertLogs("Tools.cache", level="WARNING"):
                result = search_uploaded_knowledge.invoke({"query": "missing"})
        self.assertIn("No uploaded documents", result)
        search.assert_called_once()
        self.assertEqual(redis.closed, 1)

    def test_new_document_invalidates_only_knowledge_cache(self):
        redis = FakeRedis()
        redis.values = {
            "tool-result:v1:search_uploaded_knowledge:abc": "old answer",
            "tool-result:v1:check_customer_history:def": "old history",
        }
        with patch("Tools.cache.Redis.from_url", return_value=redis):
            invalidate_tool_results("redis://localhost:2126/1", "search_uploaded_knowledge")
        self.assertEqual(list(redis.values), ["tool-result:v1:check_customer_history:def"])
        self.assertEqual(redis.closed, 1)


class AsyncToolCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_tool_node_uses_same_cache(self):
        builder = StateGraph(TicketState)
        builder.add_node("knowledge", ToolNode([search_uploaded_knowledge]))
        builder.add_edge(START, "knowledge")
        builder.add_edge("knowledge", END)
        graph = builder.compile()

        def state(call_id):
            return {"messages": [
                HumanMessage(content="How do I reset my password?"),
                AIMessage(content="", tool_calls=[{
                    "name": "search_uploaded_knowledge",
                    "args": {"query": "password"}, "id": call_id,
                }]),
            ]}

        redis = FakeRedis()
        with (
            patch.dict("os.environ", {
                "REDIS_URL": "redis://localhost:2126/1",
                "CACHEABLE_TOOLS": "search_uploaded_knowledge",
            }),
            patch("Tools.cache.Redis.from_url", return_value=redis),
            patch("Tools.knowledge_base.search_knowledge", return_value=[]) as search,
        ):
            first = await graph.ainvoke(state("call-1"))
            second = await graph.ainvoke(state("call-2"))
        search.assert_called_once()
        self.assertEqual(first["messages"][-1].content, second["messages"][-1].content)
        self.assertEqual(redis.writes[0][1], 300)


if __name__ == "__main__":
    unittest.main()
