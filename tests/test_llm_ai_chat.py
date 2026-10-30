from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.ai_chat_service import AIChatService
from core.searcher import SearchResult


class UnavailableLLM:
    is_available = False


class ScriptedLLM:
    """Replays fixed chat chunks, like a provider that answers or fails."""

    is_available = True

    def __init__(self, *chunks):
        self.chunks = chunks

    async def chat(self, **_kwargs):
        for chunk in self.chunks:
            yield chunk


class AIChatServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_current_user_message_is_not_duplicated(self):
        messages = AIChatService._prepare_messages(
            "find rain",
            [
                {"role": "user", "content": "older"},
                {"role": "assistant", "content": "response"},
                {"role": "user", "content": "find rain"},
            ],
            limit=10,
        )
        self.assertEqual(messages[-1], {"role": "user", "content": "find rain"})
        self.assertEqual(
            sum(item == {"role": "user", "content": "find rain"} for item in messages),
            1,
        )
        self.assertIn({"role": "assistant", "content": "response"}, messages)

    def test_searchers_are_cached_by_project(self):
        created = []

        def factory(*, project_id, **_kwargs):
            value = object()
            created.append((project_id, value))
            return value

        service = AIChatService(project_id="p1")
        manager = mock.Mock()
        manager.get_index_manifest.return_value = None
        with (
            mock.patch(
                "core.ai_chat_service.get_optimized_searcher_sync",
                side_effect=factory,
            ),
            mock.patch("core.database.get_db_manager", return_value=manager),
        ):
            first = service.get_searcher("p1")
            again = service.get_searcher("p1")
            second = service.get_searcher("p2")
        self.assertIs(first, again)
        self.assertIsNot(first, second)
        self.assertEqual([item[0] for item in created], ["p1", "p2"])

    async def test_unavailable_llm_falls_back_to_raw_project_search(self):
        service = AIChatService(project_id="project-a")
        service._llm_client = UnavailableLLM()
        with mock.patch.object(service, "_search", return_value=[]) as search:
            chunks = [chunk async for chunk in service.chat("暴雨", top_k=5)]
        search.assert_awaited_once_with(
            "暴雨", 5, 0.1, project_id="project-a"
        )
        searching = next(chunk for chunk in chunks if chunk["type"] == "searching")
        results = next(chunk for chunk in chunks if chunk["type"] == "results")
        warning = next(chunk for chunk in chunks if chunk["type"] == "warning")
        self.assertEqual(searching["query"], "暴雨")
        self.assertEqual(results["project_id"], "project-a")
        self.assertIn("没有可用的对话模型配置", warning["content"])
        self.assertTrue(results["fallback"])

    async def test_failed_model_call_is_reported_before_raw_search(self):
        service = AIChatService(project_id="project-a")
        service._llm_client = ScriptedLLM(
            {"type": "error", "message": "LLM 服务暂时不可用 (HTTP 503)"}
        )
        with mock.patch.object(service, "_search", return_value=[]) as search:
            chunks = [chunk async for chunk in service.chat("手雷爆炸", top_k=5)]
        search.assert_awaited_once_with("手雷爆炸", 5, 0.1, project_id="project-a")
        types = [chunk["type"] for chunk in chunks]
        self.assertLess(types.index("warning"), types.index("results"))
        warning = chunks[types.index("warning")]
        self.assertIn("模型调用失败", warning["content"])
        self.assertIn("HTTP 503", warning["content"])
        results = chunks[types.index("results")]
        self.assertTrue(results["fallback"])
        self.assertTrue(results["summary"].startswith("（按原文直接搜索）"))

    async def test_unparseable_model_reply_is_reported_with_an_excerpt(self):
        service = AIChatService(project_id="project-a")
        service._llm_client = ScriptedLLM(
            {"type": "content", "content": "Sure! Grenade explosions are loud."}
        )
        with mock.patch.object(service, "_search", return_value=[]):
            chunks = [chunk async for chunk in service.chat("手雷爆炸")]
        warning = next(chunk for chunk in chunks if chunk["type"] == "warning")
        self.assertIn("不是要求的 JSON", warning["content"])
        self.assertIn("Grenade explosions are loud", warning["content"])

    async def test_understood_request_searches_model_keywords_without_warning(self):
        service = AIChatService(project_id="project-a")
        service._llm_client = ScriptedLLM(
            {"type": "content", "content": '{"type":"search","keywords":["grenade explosion"]}'}
        )
        with mock.patch.object(service, "_search", return_value=[]) as search:
            chunks = [chunk async for chunk in service.chat("手雷爆炸", top_k=5)]
        search.assert_awaited_once_with("grenade explosion", 5, 0.1, project_id="project-a")
        self.assertNotIn("warning", [chunk["type"] for chunk in chunks])
        results = next(chunk for chunk in chunks if chunk["type"] == "results")
        self.assertFalse(results["fallback"])

    async def test_search_failure_is_an_error_not_an_empty_result(self):
        service = AIChatService(project_id="project-a")
        service._llm_client = UnavailableLLM()
        searcher = mock.Mock()
        searcher.search_async = mock.AsyncMock(side_effect=RuntimeError("index offline"))
        with mock.patch.object(service, "get_searcher", return_value=searcher):
            chunks = [chunk async for chunk in service.chat("rain")]
        types = [chunk["type"] for chunk in chunks]
        self.assertNotIn("results", types)
        error = next(chunk for chunk in chunks if chunk["type"] == "error")
        self.assertIn("index offline", error["content"])

    async def test_results_expose_stable_file_id_for_transcoded_playback(self):
        service = AIChatService(project_id="project-a")
        service._llm_client = UnavailableLLM()
        result = SearchResult(
            file_path="C:/Audio/impact.wma",
            filename="impact.wma",
            similarity=0.9,
            duration=1.0,
            format="wma",
            metadata={"file_id": "stable-wma-id"},
        )
        with mock.patch.object(service, "_search", return_value=[result]):
            chunks = [chunk async for chunk in service.chat("impact")]

        payload = next(chunk for chunk in chunks if chunk["type"] == "results")
        self.assertEqual(payload["results"][0]["file_id"], "stable-wma-id")
        self.assertEqual(
            payload["results"][0]["metadata"]["file_id"], "stable-wma-id"
        )

    def test_json_extractor_accepts_fenced_and_prefixed_object(self):
        value = AIChatService._extract_json(
            'analysis first\n```json\n{"type":"search","keywords":["rain"]}\n```'
        )
        self.assertEqual(value["keywords"], ["rain"])


if __name__ == "__main__":
    unittest.main()
