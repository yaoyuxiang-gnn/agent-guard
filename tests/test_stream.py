"""Tests for GuardedStream: recording streaming responses once, on drain."""

from __future__ import annotations

import asyncio
import unittest
import warnings
from types import SimpleNamespace as NS

from agentguard import Guard
from agentguard.adapters import GuardedStream
from agentguard.adapters.anthropic import guard_anthropic
from agentguard.adapters.openai import guard_openai


def openai_chunks(*, with_usage: bool = True) -> list[NS]:
    """The shape of OpenAI's stream: usage only on the final chunk, if asked."""
    chunks = [
        NS(model="gpt-4o", choices=[NS(delta=NS(content="Hello"))], usage=None),
        NS(model="gpt-4o", choices=[NS(delta=NS(content=" world"))], usage=None),
    ]
    if with_usage:
        chunks.append(
            NS(
                model="gpt-4o",
                choices=[],
                usage=NS(prompt_tokens=1_000, completion_tokens=50),
            )
        )
    return chunks


def anthropic_events() -> list[NS]:
    """The shape of Anthropic's raw stream: input on start, cumulative output."""
    return [
        NS(
            type="message_start",
            message=NS(
                model="claude-sonnet-4",
                usage=NS(input_tokens=2_102, output_tokens=1, cache_read_input_tokens=0),
            ),
        ),
        NS(type="content_block_delta", delta=NS(text="...")),
        NS(type="message_delta", usage=NS(output_tokens=64)),
        NS(type="message_delta", usage=NS(output_tokens=112)),
        NS(type="message_stop"),
    ]


class OpenAIStyleTests(unittest.TestCase):
    def test_full_iteration_records_exactly_once(self) -> None:
        guard = Guard()
        stream = GuardedStream(iter(openai_chunks()), guard)
        self.assertEqual(len(list(stream)), 3)
        self.assertEqual(guard.calls, 1)
        # 1,000 in at $2.5/1M + 50 out at $10/1M
        self.assertAlmostEqual(guard.spent_usd, 0.0025 + 0.0005, places=9)
        self.assertEqual(guard.usage.input_tokens, 1_000)
        self.assertEqual(guard.usage.output_tokens, 50)

    def test_model_comes_from_the_chunks(self) -> None:
        guard = Guard()
        list(GuardedStream(iter(openai_chunks()), guard))
        record = guard.tracker.records[0]
        self.assertEqual(record.model, "gpt-4o")

    def test_model_hint_is_used_when_chunks_carry_none(self) -> None:
        guard = Guard()
        chunks = [NS(usage=NS(prompt_tokens=10, completion_tokens=2))]
        list(GuardedStream(iter(chunks), guard, model_hint="gpt-4o-mini"))
        self.assertEqual(guard.tracker.records[0].model, "gpt-4o-mini")

    def test_stream_without_usage_warns_and_records_nothing(self) -> None:
        guard = Guard()
        stream = GuardedStream(iter(openai_chunks(with_usage=False)), guard)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            list(stream)
        self.assertEqual(guard.calls, 0)
        self.assertTrue(any("include_usage" in str(w.message) for w in caught))

    def test_breaking_out_early_records_what_was_seen(self) -> None:
        guard = Guard()
        stream = GuardedStream(iter(openai_chunks()), guard)
        iterator = iter(stream)
        next(iterator)  # only the first chunk: no usage yet
        with self.assertWarns(RuntimeWarning):
            iterator.close()  # abandoning the stream finalises it deterministically
        self.assertEqual(guard.calls, 0)  # warned, nothing invented

    def test_close_records_once_and_is_idempotent(self) -> None:
        guard = Guard()
        stream = GuardedStream(iter(openai_chunks()), guard)
        for _ in stream:
            pass
        stream.close()
        stream._finish()
        self.assertEqual(guard.calls, 1)

    def test_dict_chunks_work_too(self) -> None:
        guard = Guard()
        chunks = [{"model": "gpt-4o", "usage": {"prompt_tokens": 100, "completion_tokens": 5}}]
        list(GuardedStream(iter(chunks), guard))
        self.assertEqual(guard.usage.total_tokens, 105)


class AnthropicStyleTests(unittest.TestCase):
    def test_message_events_accumulate_with_max_merge(self) -> None:
        guard = Guard()
        list(GuardedStream(iter(anthropic_events()), guard))
        self.assertEqual(guard.calls, 1)
        record = guard.tracker.records[0]
        self.assertEqual(record.model, "claude-sonnet-4")
        self.assertEqual(record.usage.input_tokens, 2_102)
        self.assertEqual(record.usage.output_tokens, 112)

    def test_manager_style_stream_finalises_on_exit(self) -> None:
        """messages.stream() shape: context manager + get_final_message()."""
        final = NS(
            model="claude-sonnet-4",
            usage=NS(input_tokens=300, output_tokens=42),
        )

        class MessageStream:
            def __iter__(self):
                return iter([NS(type="content_block_delta", delta=NS(text="hi"))])

            def get_final_message(self):
                return final

        class Manager:
            def __enter__(self):
                return MessageStream()

            def __exit__(self, *exc):
                return False

        guard = Guard()
        with GuardedStream(Manager(), guard) as stream:
            for _ in stream:
                pass  # deltas carry no usage; the final message does
        self.assertEqual(guard.calls, 1)
        self.assertEqual(guard.usage.output_tokens, 42)

    def test_user_called_get_final_message_is_not_double_counted(self) -> None:
        final = NS(model="claude-sonnet-4", usage=NS(input_tokens=10, output_tokens=5))

        class MessageStream:
            def __iter__(self):
                return iter([NS(type="message_start", message=final)])

            def get_final_message(self):
                return final

        class Manager:
            def __enter__(self):
                return MessageStream()

            def __exit__(self, *exc):
                return False

        guard = Guard()
        with GuardedStream(Manager(), guard) as stream:
            list(stream)
            stream.get_final_message()
        self.assertEqual(guard.calls, 1)


class _FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()


class _AsyncCompletions:
    async def create(self, **kwargs):
        if kwargs.get("stream"):
            return _FakeAsyncStream(openai_chunks())
        return NS(model="gpt-4o", usage=NS(prompt_tokens=100, completion_tokens=10))


class _AsyncClient:
    def __init__(self):
        self.chat = NS(completions=_AsyncCompletions())


class AsyncTests(unittest.TestCase):
    def test_async_call_is_recorded_after_await(self) -> None:
        guard = Guard()

        async def run() -> None:
            client = guard_openai(_AsyncClient(), guard)
            await client.chat.completions.create(model="gpt-4o")

        asyncio.run(run())
        self.assertEqual(guard.calls, 1)
        self.assertEqual(guard.usage.total_tokens, 110)

    def test_async_stream_is_recorded_on_drain(self) -> None:
        guard = Guard()

        async def run() -> int:
            client = guard_openai(_AsyncClient(), guard)
            stream = await client.chat.completions.create(model="gpt-4o", stream=True)
            seen = 0
            async for _ in stream:
                seen += 1
            return seen

        self.assertEqual(asyncio.run(run()), 3)
        self.assertEqual(guard.calls, 1)
        self.assertEqual(guard.usage.input_tokens, 1_000)


class WrappedClientTests(unittest.TestCase):
    def test_openai_create_stream_true_returns_a_guarded_stream(self) -> None:
        class Completions:
            def create(self, **kwargs):
                return iter(openai_chunks())

        client = guard_openai(NS(chat=NS(completions=Completions())), max_usd=1.0)
        result = client.chat.completions.create(model="gpt-4o", stream=True)
        self.assertIsInstance(result, GuardedStream)
        list(result)
        self.assertEqual(client.guard.calls, 1)

    def test_non_streaming_call_is_unaffected(self) -> None:
        class Completions:
            def create(self, **kwargs):
                return NS(model="gpt-4o", usage=NS(prompt_tokens=10, completion_tokens=2))

        client = guard_openai(NS(chat=NS(completions=Completions())), max_usd=1.0)
        client.chat.completions.create(model="gpt-4o")
        self.assertEqual(client.guard.usage.total_tokens, 12)

    def test_anthropic_messages_stream_is_wrapped(self) -> None:
        events = anthropic_events()

        class MessageStream:
            def __iter__(self):
                return iter(events)

        class Manager:
            def __enter__(self):
                return MessageStream()

            def __exit__(self, *exc):
                return False

        class Messages:
            def stream(self, **kwargs):
                return Manager()

        client = guard_anthropic(NS(messages=Messages()), max_usd=1.0)
        with client.messages.stream(model="claude-sonnet-4") as stream:
            for _ in stream:
                pass
        self.assertEqual(client.guard.calls, 1)
        self.assertEqual(client.guard.usage.output_tokens, 112)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
