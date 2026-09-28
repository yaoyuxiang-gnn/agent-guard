"""Tests for the LangGraph / LangChain callback handler.

The payloads are SimpleNamespace fakes shaped exactly like LangChain's
``LLMResult``: a ``llm_output`` mapping plus ``generations`` of message chunks.
No langchain-core needed — the handler falls back to a duck-typed class.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace as NS

from agentguard import BudgetExceeded, Guard, GuardConfigError, LoopDetected
from agentguard.integrations.langgraph import guard_langgraph


def llm_result(llm_output: dict | None = None, generations: list | None = None) -> NS:
    return NS(llm_output=llm_output or {}, generations=generations or [])


def generation(input_tokens: int, output_tokens: int, model: str | None = None) -> NS:
    metadata = {"model_name": model} if model else {}
    return NS(
        message=NS(
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            response_metadata=metadata,
        )
    )


class RecordingTests(unittest.TestCase):
    def test_openai_style_token_usage_is_recorded(self) -> None:
        guard = Guard()
        handler = guard_langgraph(guard)
        handler.on_llm_end(
            llm_result(
                llm_output={
                    "model_name": "gpt-4o",
                    "token_usage": {"prompt_tokens": 1_000, "completion_tokens": 50},
                }
            )
        )
        self.assertEqual(guard.calls, 1)
        self.assertAlmostEqual(guard.spent_usd, 0.0025 + 0.0005, places=9)

    def test_usage_metadata_on_generations_is_recorded(self) -> None:
        guard = Guard()
        handler = guard_langgraph(guard)
        handler.on_llm_end(llm_result(generations=[[generation(300, 42, model="claude-sonnet-4")]]))
        self.assertEqual(guard.calls, 1)
        record = guard.tracker.records[0]
        self.assertEqual(record.model, "claude-sonnet-4")
        self.assertEqual(record.usage.output_tokens, 42)

    def test_multiple_generations_are_summed(self) -> None:
        guard = Guard()
        handler = guard_langgraph(guard)
        handler.on_llm_end(
            llm_result(
                generations=[
                    [
                        generation(100, 10, model="gpt-4o-mini"),
                        generation(100, 10, model="gpt-4o-mini"),
                    ]
                ]
            )
        )
        self.assertEqual(guard.usage.input_tokens, 200)
        self.assertEqual(guard.usage.output_tokens, 20)
        self.assertGreater(guard.spent_usd, 0)

    def test_a_result_without_usage_warns_like_a_raw_response(self) -> None:
        guard = Guard(on_unknown_model="ignore")
        handler = guard_langgraph(guard)
        with self.assertWarns(RuntimeWarning):
            handler.on_llm_end(llm_result())
        self.assertEqual(guard.calls, 1)  # counted, costed at $0, warned

    def test_handler_exposes_the_guard(self) -> None:
        guard = Guard()
        self.assertIs(guard_langgraph(guard).guard, guard)


class SafetyTests(unittest.TestCase):
    def test_budget_trip_propagates_out_of_the_callback(self) -> None:
        handler = guard_langgraph(max_usd=0.0001)
        with self.assertRaises(BudgetExceeded):
            handler.on_llm_end(
                llm_result(
                    llm_output={
                        "model_name": "gpt-4o",
                        "token_usage": {"prompt_tokens": 10_000, "completion_tokens": 0},
                    }
                )
            )

    def test_raise_error_is_set_so_langchain_re_raises(self) -> None:
        # Without this the callback manager would log a trip and keep going.
        self.assertTrue(guard_langgraph(Guard()).raise_error)

    def test_tool_loops_trip_the_loop_detector(self) -> None:
        handler = guard_langgraph(Guard())
        for _ in range(2):
            handler.on_tool_start({"name": "search"}, '{"q": "agent guard"}')
        with self.assertRaises(LoopDetected):
            handler.on_tool_start({"name": "search"}, '{"q": "agent guard"}')

    def test_distinct_tool_calls_do_not_trip(self) -> None:
        handler = guard_langgraph(Guard())
        payloads = [
            '{"city": "paris", "units": "metric", "lang": "fr"}',
            '{"path": "/tmp/report.pdf", "mode": "read-only"}',
            '{"sql": "select count(*) from runs where ok"}',
            "[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]",
            '{"url": "https://example.com/docs", "depth": 3}',
        ]
        for index, payload in enumerate(payloads):
            handler.on_tool_start({"name": f"tool_{index}"}, payload)
        self.assertIsNone(handler.guard.tripped)

    def test_no_arguments_is_an_explicit_error(self) -> None:
        with self.assertRaises(GuardConfigError):
            guard_langgraph()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
