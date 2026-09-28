"""Tests for the client adapters.

The adapters are pure duck-typing, so the fakes here are deliberately shaped like
real SDK clients (nested resources ending in a ``create`` method) without pulling
in any provider package.
"""

from __future__ import annotations

import unittest
from typing import Any

from agent_guard import BudgetExceeded, Guard, GuardConfigError
from agent_guard.adapters import GuardedClient, guard_client
from agent_guard.adapters.anthropic import guard_anthropic
from agent_guard.adapters.openai import guard_openai


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "model": kwargs.get("model", "gpt-4o"),
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
        }

    def list(self) -> list[str]:
        return ["a", "b"]


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "model": "claude-sonnet-4",
            "usage": {"input_tokens": 2000, "output_tokens": 400},
        }


class FakeClient:
    def __init__(self) -> None:
        self.chat = FakeChat()
        self.messages = FakeMessages()
        self.api_key = "sk-secret"
        self.timeout = 30


class ProxyBehaviourTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeClient()
        self.guard = Guard(max_usd=100.0)
        self.proxy = GuardedClient(self.client, self.guard)

    def test_records_calls_made_through_nested_resources(self) -> None:
        self.proxy.chat.completions.create(model="gpt-4o")
        self.assertEqual(self.guard.calls, 1)
        # The fake reports 1,000 in / 100 out at gpt-4o list prices.
        self.assertAlmostEqual(
            self.guard.spent_usd, (1000 * 2.5 + 100 * 10.0) / 1_000_000
        )

    def test_returns_the_underlying_response_untouched(self) -> None:
        response = self.proxy.chat.completions.create(model="gpt-4o")
        self.assertEqual(response["model"], "gpt-4o")
        self.assertIn("usage", response)

    def test_calls_the_real_client(self) -> None:
        self.proxy.chat.completions.create(model="gpt-4o")
        self.assertEqual(len(self.client.chat.completions.calls), 1)

    def test_methods_outside_record_on_are_not_wrapped(self) -> None:
        self.assertEqual(self.proxy.chat.completions.list(), ["a", "b"])
        self.assertEqual(self.guard.calls, 0)

    def test_plain_attributes_pass_through(self) -> None:
        self.assertEqual(self.proxy.api_key, "sk-secret")
        self.assertEqual(self.proxy.timeout, 30)

    def test_attribute_assignment_reaches_the_wrapped_client(self) -> None:
        self.proxy.timeout = 99
        self.assertEqual(self.client.timeout, 99)

    def test_missing_attributes_raise_attribute_error(self) -> None:
        with self.assertRaises(AttributeError):
            _ = self.proxy.does_not_exist

    def test_guard_property(self) -> None:
        self.assertIs(self.proxy.guard, self.guard)

    def test_target_property(self) -> None:
        self.assertIs(self.proxy.target, self.client)

    def test_repr(self) -> None:
        self.assertEqual(repr(self.proxy), "GuardedClient(FakeClient)")

    def test_dir_includes_wrapped_attributes(self) -> None:
        self.assertIn("api_key", dir(self.proxy))

    def test_depth_limit_stops_proxying(self) -> None:
        shallow = GuardedClient(self.client, self.guard, depth=0)
        shallow.chat.completions.create(model="gpt-4o")
        self.assertEqual(self.guard.calls, 0)

    def test_budget_trips_through_the_adapter(self) -> None:
        guard = Guard(max_usd=0.0001)
        proxy = GuardedClient(FakeClient(), guard)
        with self.assertRaises(BudgetExceeded):
            proxy.chat.completions.create(model="gpt-4o")

    def test_context_manager_delegates(self) -> None:
        class Contextual:
            def __init__(self) -> None:
                self.entered = False
                self.exited = False

            def __enter__(self) -> "Contextual":
                self.entered = True
                return self

            def __exit__(self, *exc: Any) -> bool:
                self.exited = True
                return False

        target = Contextual()
        with GuardedClient(target, self.guard) as proxy:
            self.assertIsInstance(proxy, GuardedClient)
        self.assertTrue(target.entered)
        self.assertTrue(target.exited)


class GuardClientFactoryTests(unittest.TestCase):
    def test_requires_a_guard(self) -> None:
        with self.assertRaises(GuardConfigError):
            guard_client(FakeClient(), None)  # type: ignore[arg-type]

    def test_generic_factory_records(self) -> None:
        guard = Guard(max_usd=100.0)
        proxy = guard_client(FakeClient(), guard)
        proxy.chat.completions.create(model="gpt-4o")
        self.assertEqual(guard.calls, 1)


class OpenAIPresetTests(unittest.TestCase):
    def test_accepts_an_explicit_guard(self) -> None:
        guard = Guard(max_usd=100.0)
        proxy = guard_openai(FakeClient(), guard)
        proxy.chat.completions.create(model="gpt-4o")
        self.assertEqual(guard.calls, 1)

    def test_creates_a_guard_from_keyword_arguments(self) -> None:
        proxy = guard_openai(FakeClient(), max_usd=100.0, max_steps=5, name="openai")
        proxy.chat.completions.create(model="gpt-4o")
        self.assertEqual(proxy.guard.calls, 1)
        self.assertEqual(proxy.guard.name, "openai")

    def test_rejects_guard_and_keyword_arguments_together(self) -> None:
        with self.assertRaises(GuardConfigError):
            guard_openai(FakeClient(), Guard(), max_usd=1.0)

    def test_requires_some_guard_configuration(self) -> None:
        with self.assertRaises(GuardConfigError):
            guard_openai(FakeClient())


class AnthropicPresetTests(unittest.TestCase):
    def test_records_anthropic_style_usage(self) -> None:
        guard = Guard(max_usd=100.0)
        proxy = guard_anthropic(FakeClient(), guard)
        proxy.messages.create(model="claude-sonnet-4")
        self.assertEqual(guard.calls, 1)
        self.assertEqual(guard.usage.total_tokens, 2400)
        self.assertAlmostEqual(guard.spent_usd, (2000 * 3.0 + 400 * 15.0) / 1_000_000)

    def test_creates_a_guard_from_keyword_arguments(self) -> None:
        proxy = guard_anthropic(FakeClient(), max_usd=50.0)
        self.assertEqual(proxy.guard.name, None)
        self.assertAlmostEqual(proxy.guard.remaining_usd or 0.0, 50.0)


class PreflightTests(unittest.TestCase):
    def test_preflight_refuses_and_skips_the_call(self) -> None:
        client = FakeClient()
        guard = Guard(max_usd=0.001)
        proxy = guard_openai(client, guard, preflight=True)

        with self.assertRaises(BudgetExceeded):
            proxy.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=1000,
            )
        # The refused call must never reach the provider.
        self.assertEqual(len(client.chat.completions.calls), 0)

    def test_preflight_allows_an_affordable_call(self) -> None:
        client = FakeClient()
        guard = Guard(max_usd=100.0)
        proxy = guard_openai(client, guard, preflight=True)

        proxy.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1000,
        )
        self.assertEqual(len(client.chat.completions.calls), 1)
        self.assertEqual(guard.calls, 1)

    def test_preflight_is_skipped_without_a_usable_model_or_payload(self) -> None:
        client = FakeClient()
        guard = Guard(max_usd=0.001)
        proxy = guard_openai(client, guard, preflight=True)

        # No `messages` and no `model`, so there is nothing to estimate. The call
        # must still go out, and be caught afterwards by the ordinary post-hoc
        # budget check rather than being blocked before it was issued.
        with self.assertRaises(BudgetExceeded):
            proxy.chat.completions.create(model="gpt-4o")
        self.assertEqual(len(client.chat.completions.calls), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
