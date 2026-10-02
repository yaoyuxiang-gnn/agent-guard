"""Tests for the opt-in price snapshot: fetching, checksums, and merge order.

Distinct from ``tests/test_snapshot.py``, which covers *checkpointing* a run
(``Guard.snapshot``). This module covers the downloaded price catalogue.

Nothing here touches the network. The one function that could is given a fake
opener, so the download path is exercised — status handling, the size cap,
malformed bodies — without a request leaving the process.
"""

from __future__ import annotations

import json
import unittest
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agentguard import Guard, PriceTable
from agentguard.config import config_paths, load_config
from agentguard.exceptions import GuardConfigError
from agentguard.snapshot import (
    DEFAULT_SNAPSHOT_URL,
    SNAPSHOT_VERSION,
    PriceSnapshot,
    fetch_snapshot,
    load_snapshot,
    parse_snapshot_payload,
    read_snapshot_file,
    remove_snapshot,
    snapshot_path,
    write_snapshot_file,
)


def catalogue(*entries: tuple[str, str, str]) -> dict[str, object]:
    """A catalogue document from ``(id, prompt_rate, completion_rate)`` triples."""
    return {
        "data": [
            {"id": identifier, "pricing": {"prompt": prompt, "completion": completion}}
            for identifier, prompt, completion in entries
        ]
    }


def environment(folder: str) -> dict[str, str]:
    """An env mapping that puts the user config directory inside ``folder``.

    ``AGENTGUARD_CONFIG`` is pinned to empty rather than left unset: an empty value
    means "discover as usual", and pinning it stops a test that sets it to ``none``
    from leaking into every test that runs after it in the same process. The
    platform-specific roots are both set so the path resolves the same way wherever
    the suite runs.
    """
    return {
        "APPDATA": folder,
        "LOCALAPPDATA": folder,
        "XDG_CONFIG_HOME": folder,
        "AGENTGUARD_CONFIG": "",
    }


class ParsePayloadTests(unittest.TestCase):
    def test_per_token_strings_become_per_million_rates(self) -> None:
        snapshot = parse_snapshot_payload(catalogue(("openai/gpt-4o", "0.0000025", "0.00001")))
        price = snapshot.models["gpt-4o"]
        self.assertEqual(price.input_per_1m, 2.5)
        self.assertEqual(price.output_per_1m, 10.0)

    def test_the_conversion_is_exact_not_binary_approximate(self) -> None:
        # 0.0000002 * 1e6 in float arithmetic is 0.19999999999999998, which is a
        # rate nobody can check against the page it came from.
        snapshot = parse_snapshot_payload(
            {
                "data": [
                    {
                        "id": "anthropic/claude-x",
                        "pricing": {
                            "prompt": "0.000002",
                            "completion": "0.00001",
                            "input_cache_read": "0.0000002",
                        },
                    }
                ]
            }
        )
        self.assertEqual(snapshot.models["claude-x"].cached_input_per_1m, 0.2)

    def test_gateway_namespacing_is_stripped(self) -> None:
        snapshot = parse_snapshot_payload(
            catalogue(
                ("anthropic/claude-sonnet-4.5", "0.000003", "0.000015"),
                ("openai/gpt-4o", "0.0000025", "0.00001"),
            )
        )
        self.assertEqual(sorted(snapshot.models), ["claude-sonnet-4.5", "gpt-4o"])

    def test_a_negative_rate_is_a_sentinel_and_the_entry_is_skipped(self) -> None:
        # OpenRouter publishes "-1" for a router it cannot price. Reading that as a
        # rate would bill its calls at a *negative* cost.
        snapshot = parse_snapshot_payload(
            catalogue(("openai/gpt-4o", "0.0000025", "0.00001"), ("x/router", "-1", "-1"))
        )
        self.assertNotIn("router", snapshot.models)
        self.assertEqual(snapshot.skipped, 1)

    def test_a_batch_variant_never_overrides_the_standard_rate(self) -> None:
        # Regression: variants normalize to the same model name, and keeping the
        # cheapest meant a real catalogue priced most models at the *batch* rate —
        # half list. An agent billed at list would then have a cap firing at twice
        # the spend it thought it was tracking.
        snapshot = parse_snapshot_payload(
            {
                "data": [
                    {
                        "id": "anthropic/claude-sonnet-5.5:batch",
                        "pricing": {"prompt": "0.000001", "completion": "0.000005"},
                    },
                    {
                        "id": "anthropic/claude-sonnet-5.5",
                        "pricing": {"prompt": "0.000002", "completion": "0.00001"},
                    },
                ]
            }
        )
        self.assertEqual(snapshot.models["claude-sonnet-5.5"].input_per_1m, 2.0)
        self.assertEqual(snapshot.skipped, 1)

    def test_a_free_variant_does_not_make_a_paid_model_free(self) -> None:
        snapshot = parse_snapshot_payload(
            {
                "data": [
                    {"id": "x/model:free", "pricing": {"prompt": "0", "completion": "0"}},
                    {
                        "id": "x/model",
                        "pricing": {"prompt": "0.000003", "completion": "0.000009"},
                    },
                ]
            }
        )
        self.assertEqual(snapshot.models["model"].input_per_1m, 3.0)

    def test_a_lone_free_variant_is_unpriced_rather_than_free(self) -> None:
        # The conservative direction: "this call cost nothing" is a claim only the
        # invoice can settle, so the model reports as unpriced instead. A second,
        # ordinary entry keeps the "nothing usable at all" refusal out of the way.
        snapshot = parse_snapshot_payload(
            {
                "data": [
                    {"id": "x/model:free", "pricing": {"prompt": "0", "completion": "0"}},
                    {"id": "x/other", "pricing": {"prompt": "0.000001", "completion": "0.000001"}},
                ]
            }
        )
        self.assertNotIn("model", snapshot.models)
        self.assertEqual(snapshot.models_count, 1)

    def test_two_ids_for_one_model_keep_the_higher_rate(self) -> None:
        # For a cap, assuming the higher rate fires early rather than late.
        snapshot = parse_snapshot_payload(
            catalogue(
                ("us.anthropic.claude-sonnet-4", "0.000004", "0.00002"),
                ("anthropic/claude-sonnet-4", "0.000003", "0.000015"),
            )
        )
        self.assertEqual(snapshot.models["claude-sonnet-4"].input_per_1m, 4.0)

    def test_free_models_are_kept_at_zero(self) -> None:
        snapshot = parse_snapshot_payload(catalogue(("x/free-model", "0", "0")))
        self.assertEqual(snapshot.models["free-model"].input_per_1m, 0.0)

    def test_an_entry_without_usable_pricing_is_skipped(self) -> None:
        # One usable entry alongside the unusable ones, so the skip count is what
        # the assertion is about rather than the "nothing usable" refusal.
        snapshot = parse_snapshot_payload(
            {
                "data": [
                    {"id": "a/b"},
                    {"id": "c/d", "pricing": "not an object"},
                    {
                        "id": "openai/gpt-4o",
                        "pricing": {"prompt": "0.000002", "completion": "0.00001"},
                    },
                ]
            }
        )
        self.assertEqual(snapshot.models_count, 1)
        self.assertEqual(snapshot.skipped, 2)

    def test_a_catalogue_with_nothing_usable_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError) as ctx:
            parse_snapshot_payload({"data": []})
        self.assertIn(DEFAULT_SNAPSHOT_URL, str(ctx.exception))

    def test_a_non_object_payload_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError):
            parse_snapshot_payload(["not", "a", "mapping"])

    def test_a_non_numeric_rate_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError):
            parse_snapshot_payload(
                {"data": [{"id": "a/b", "pricing": {"prompt": "cheap", "completion": "1"}}]}
            )

    def test_this_modules_own_document_shape_can_be_reimported(self) -> None:
        original = parse_snapshot_payload(catalogue(("openai/gpt-4o", "0.000002", "0.00001")))
        reimported = parse_snapshot_payload(original.as_document())
        self.assertEqual(reimported.models["gpt-4o"], original.models["gpt-4o"])


class ChecksumTests(unittest.TestCase):
    def test_a_snapshot_survives_a_write_read_cycle(self) -> None:
        # The regression this guards: the checksum hashed the in-memory mapping on
        # one side and the decoded JSON on the other, so every file failed its own
        # integrity check the moment it was read back.
        snapshot = parse_snapshot_payload(
            catalogue(("openai/gpt-4o", "0.000002", "0.00001"), ("x/y", "0.000001", "0.000003"))
        )
        with TemporaryDirectory() as folder:
            path = write_snapshot_file(Path(folder) / "snap.json", snapshot)
            loaded = read_snapshot_file(path)
        self.assertEqual(loaded.models, snapshot.models)
        self.assertEqual(loaded.checksum, snapshot.checksum)

    def test_an_edited_snapshot_is_refused(self) -> None:
        snapshot = parse_snapshot_payload(catalogue(("openai/gpt-4o", "0.000002", "0.00001")))
        with TemporaryDirectory() as folder:
            path = write_snapshot_file(Path(folder) / "snap.json", snapshot)
            document = json.loads(path.read_text(encoding="utf-8"))
            document["models"]["gpt-4o"]["input"] = 0.0000001
            document["models"]["gpt-4o"]["output"] = 0.0000001
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(GuardConfigError) as ctx:
                read_snapshot_file(path)
        self.assertIn("checksum", str(ctx.exception))
        self.assertIn("pricing --update", str(ctx.exception))

    def test_a_wrong_checksum_is_refused(self) -> None:
        snapshot = parse_snapshot_payload(catalogue(("openai/gpt-4o", "0.000002", "0.00001")))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "snap.json"
            document = snapshot.as_document()
            document["checksum"] = "0" * 64
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(GuardConfigError):
                read_snapshot_file(path)

    def test_a_newer_snapshot_version_is_refused(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "snap.json"
            document = parse_snapshot_payload(
                catalogue(("openai/gpt-4o", "0.000002", "0.00001"))
            ).as_document()
            document["version"] = SNAPSHOT_VERSION + 1
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(GuardConfigError) as ctx:
                read_snapshot_file(path)
        self.assertIn("format version", str(ctx.exception))

    def test_a_missing_snapshot_is_an_error_for_the_reader(self) -> None:
        with TemporaryDirectory() as folder, self.assertRaises(GuardConfigError):
            read_snapshot_file(Path(folder) / "nothing.json")

    def test_a_bom_is_tolerated(self) -> None:
        # A JSON file touched by a Windows editor usually gains a BOM, and
        # json.loads rejects one outright with an inscrutable column-1 error.
        snapshot = parse_snapshot_payload(catalogue(("openai/gpt-4o", "0.000002", "0.00001")))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "snap.json"
            path.write_text(json.dumps(snapshot.as_document()), encoding="utf-8-sig")
            self.assertEqual(read_snapshot_file(path).models_count, 1)

    def test_models_mapping_is_read_only(self) -> None:
        # A parsed snapshot is memoised and shared between every guard built from
        # it, so one caller mutating "its" mapping would reprice all the others.
        snapshot = parse_snapshot_payload(catalogue(("openai/gpt-4o", "0.000002", "0.00001")))
        with self.assertRaises(TypeError):
            snapshot.models["injected"] = snapshot.models["gpt-4o"]  # type: ignore[index]

    def test_an_entry_with_no_rates_is_refused(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "snap.json"
            path.write_text(
                json.dumps({"version": SNAPSHOT_VERSION, "checksum": "x", "models": {"m": {}}}),
                encoding="utf-8",
            )
            with self.assertRaises(GuardConfigError):
                read_snapshot_file(path)


class MergeOrderTests(unittest.TestCase):
    """The snapshot is a floor, never a ceiling: your own prices always win."""

    def setUp(self) -> None:
        self._folder = TemporaryDirectory()
        self.folder = Path(self._folder.name)
        self.env = environment(str(self.folder))
        self.snapshot_file = snapshot_path(env=self.env)

    def tearDown(self) -> None:
        self._folder.cleanup()

    def write_snapshot(self, *entries: tuple[str, str, str]) -> None:
        write_snapshot_file(self.snapshot_file, parse_snapshot_payload(catalogue(*entries)))

    def write_user_config(self, models: dict[str, object]) -> None:
        target = self.folder / "agentguard" / "pricing.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"version": 1, "models": models}), encoding="utf-8")

    def test_the_snapshot_reprices_a_bundled_model(self) -> None:
        self.write_snapshot(("openai/gpt-4o", "0.000001", "0.000002"))
        table = PriceTable.from_config(load_config(env=self.env))
        self.assertEqual(table.resolve_price("gpt-4o").input_per_1m, 1.0)

    def test_the_snapshot_adds_a_model_the_bundle_never_had(self) -> None:
        self.write_snapshot(("acme/brand-new", "0.000003", "0.000012"))
        table = PriceTable.from_config(load_config(env=self.env))
        self.assertEqual(table.resolve_price("brand-new").output_per_1m, 12.0)

    def test_a_users_own_price_beats_the_snapshot(self) -> None:
        self.write_snapshot(("openai/gpt-4o", "0.000001", "0.000002"))
        self.write_user_config({"gpt-4o": [9.0, 9.0]})
        table = PriceTable.from_config(load_config(env=self.env))
        self.assertEqual(table.resolve_price("gpt-4o").input_per_1m, 9.0)

    def test_the_snapshot_is_loaded_before_the_user_config(self) -> None:
        self.write_snapshot(("openai/gpt-4o", "0.000001", "0.000002"))
        self.write_user_config({"gpt-4o": [9.0, 9.0]})
        names = [path.name for path in config_paths(env=self.env)]
        self.assertEqual(names, ["pricing-snapshot.json", "pricing.json"])

    def test_both_sources_are_reported(self) -> None:
        self.write_snapshot(("openai/gpt-4o", "0.000001", "0.000002"))
        self.write_user_config({"gpt-4o": [9.0, 9.0]})
        self.assertEqual(len(load_config(env=self.env).sources), 2)

    def test_use_snapshot_false_skips_it(self) -> None:
        self.write_snapshot(("openai/gpt-4o", "0.000001", "0.000002"))
        config = load_config(env=self.env, use_snapshot=False)
        self.assertEqual(PriceTable.from_config(config).resolve_price("gpt-4o").input_per_1m, 2.5)

    def test_guard_use_snapshot_false_skips_it(self) -> None:
        self.write_snapshot(("openai/gpt-4o", "0.000001", "0.000002"))
        with mock.patch.dict("os.environ", self.env, clear=False):
            guard = Guard(use_config=True, use_snapshot=False)
        self.assertEqual(guard.price_table.resolve_price("gpt-4o").input_per_1m, 2.5)

    def test_guard_picks_the_snapshot_up_by_default(self) -> None:
        self.write_snapshot(("openai/gpt-4o", "0.000001", "0.000002"))
        with mock.patch.dict("os.environ", self.env, clear=False):
            guard = Guard(use_config=True, on_unknown_model="ignore")
        self.assertEqual(guard.price_table.resolve_price("gpt-4o").input_per_1m, 1.0)
        self.assertIn("pricing-snapshot.json", guard.report().pricing_sources[0])

    def test_an_explicit_config_path_replaces_discovery(self) -> None:
        self.write_snapshot(("openai/gpt-4o", "0.000001", "0.000002"))
        explicit = self.folder / "elsewhere.json"
        explicit.write_text(json.dumps({"version": 1, "models": {}}), encoding="utf-8")
        env = dict(self.env, AGENTGUARD_CONFIG=str(explicit))
        self.assertEqual(config_paths(env=env), (explicit,))

    def test_config_disabled_disables_the_snapshot_too(self) -> None:
        self.write_snapshot(("openai/gpt-4o", "0.000001", "0.000002"))
        env = dict(self.env, AGENTGUARD_CONFIG="none")
        self.assertEqual(config_paths(env=env), ())
        self.assertTrue(load_config(env=env).is_empty)

    def test_a_corrupt_snapshot_raises_rather_than_being_ignored(self) -> None:
        # Silently falling back would leave the user believing prices were refreshed.
        self.write_snapshot(("openai/gpt-4o", "0.000001", "0.000002"))
        document = json.loads(self.snapshot_file.read_text(encoding="utf-8"))
        document["models"]["gpt-4o"]["input"] = 0.001
        self.snapshot_file.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(GuardConfigError):
            load_config(env=self.env)

    def test_an_edit_is_picked_up_by_the_next_load(self) -> None:
        self.write_snapshot(("openai/gpt-4o", "0.000001", "0.000002"))
        self.assertEqual(
            PriceTable.from_config(load_config(env=self.env)).resolve_price("gpt-4o").input_per_1m,
            1.0,
        )
        self.write_snapshot(("openai/gpt-4o", "0.000007", "0.000007"))
        self.assertEqual(
            PriceTable.from_config(load_config(env=self.env)).resolve_price("gpt-4o").input_per_1m,
            7.0,
        )

    def test_load_snapshot_returns_none_when_there_is_no_file(self) -> None:
        self.assertIsNone(load_snapshot(env=self.env))

    def test_remove_snapshot_deletes_it(self) -> None:
        self.write_snapshot(("openai/gpt-4o", "0.000001", "0.000002"))
        self.assertIsNotNone(remove_snapshot(env=self.env))
        self.assertIsNone(load_snapshot(env=self.env))
        self.assertIsNone(remove_snapshot(env=self.env))


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self, size: int = -1) -> bytes:
        return self._body if size < 0 else self._body[:size]

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class FetchTests(unittest.TestCase):
    """The download path, with no request leaving the process."""

    def test_a_good_response_becomes_a_snapshot(self) -> None:
        body = json.dumps(catalogue(("openai/gpt-4o", "0.000002", "0.00001"))).encode()
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(body)) as opener:
            snapshot = fetch_snapshot("https://example.test/models")
        self.assertEqual(snapshot.models_count, 1)
        self.assertEqual(snapshot.url, "https://example.test/models")
        request = opener.call_args[0][0]
        self.assertEqual(request.get_header("User-agent"), "agentguard-pricing-update")

    def test_a_non_http_scheme_is_refused_before_any_request(self) -> None:
        # file:// would turn --url into a local file read.
        with (
            mock.patch("urllib.request.urlopen") as opener,
            self.assertRaises(GuardConfigError) as ctx,
        ):
            fetch_snapshot("file:///etc/passwd")
        self.assertIn("http", str(ctx.exception))
        opener.assert_not_called()

    def test_a_relative_path_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError):
            fetch_snapshot("models.json")

    def test_an_http_error_names_the_status(self) -> None:
        failure = urllib.error.HTTPError(
            "https://example.test",
            404,
            "Not Found",
            {},
            None,  # type: ignore[arg-type]
        )
        with (
            mock.patch("urllib.request.urlopen", side_effect=failure),
            self.assertRaises(GuardConfigError) as ctx,
        ):
            fetch_snapshot("https://example.test/models")
        self.assertIn("404", str(ctx.exception))

    def test_a_network_failure_is_reported_with_the_url(self) -> None:
        with (
            mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no such host")),
            self.assertRaises(GuardConfigError) as ctx,
        ):
            fetch_snapshot("https://nowhere.test/models")
        self.assertIn("nowhere.test", str(ctx.exception))

    def test_an_oversized_body_is_refused(self) -> None:
        body = b"[" + b" " * 5000 + b"]"
        with (
            mock.patch("urllib.request.urlopen", return_value=FakeResponse(body)),
            self.assertRaises(GuardConfigError) as ctx,
        ):
            fetch_snapshot("https://example.test/models", max_bytes=1024)
        self.assertIn("larger than", str(ctx.exception))

    def test_a_non_json_body_is_refused(self) -> None:
        with (
            mock.patch("urllib.request.urlopen", return_value=FakeResponse(b"<html>x</html>")),
            self.assertRaises(GuardConfigError) as ctx,
        ):
            fetch_snapshot("https://example.test/models")
        self.assertIn("not JSON", str(ctx.exception))

    def test_an_unexpected_status_is_refused(self) -> None:
        body = json.dumps(catalogue(("openai/gpt-4o", "0.000002", "0.00001"))).encode()
        with (
            mock.patch("urllib.request.urlopen", return_value=FakeResponse(body, status=204)),
            self.assertRaises(GuardConfigError) as ctx,
        ):
            fetch_snapshot("https://example.test/models")
        self.assertIn("204", str(ctx.exception))

    def test_a_page_that_is_not_a_catalogue_is_refused(self) -> None:
        with (
            mock.patch(
                "urllib.request.urlopen",
                return_value=FakeResponse(json.dumps({"items": []}).encode()),
            ),
            self.assertRaises(GuardConfigError),
        ):
            fetch_snapshot("https://example.test/models")


class SnapshotDataclassTests(unittest.TestCase):
    def test_describe_mentions_the_count_and_date(self) -> None:
        snapshot = PriceSnapshot(
            models={},
            checksum="x",
            url="https://example.test",
            retrieved_at="2026-09-30T12:00:00+00:00",
        )
        self.assertEqual(snapshot.describe(), "0 models, fetched 2026-09-30")

    def test_verify_accepts_a_snapshot_it_built(self) -> None:
        snapshot = parse_snapshot_payload(catalogue(("openai/gpt-4o", "0.000002", "0.00001")))
        self.assertIsNone(snapshot.verify())

    def test_verify_rejects_a_tampered_one(self) -> None:
        snapshot = PriceSnapshot(
            models={},
            checksum="not-the-right-digest",
            url="https://example.test",
            retrieved_at="2026-09-30T12:00:00+00:00",
        )
        with self.assertRaises(GuardConfigError):
            snapshot.verify()


class MalformedInputTests(unittest.TestCase):
    """Every one of these is a refusal, because a half-read price table has holes."""

    def test_a_non_numeric_rate_in_a_catalogue_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError) as ctx:
            parse_snapshot_payload(
                {"data": [{"id": "a/b", "pricing": {"prompt": "cheap", "completion": "1"}}]}
            )
        self.assertIn("prompt", str(ctx.exception))

    def test_a_boolean_rate_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError):
            parse_snapshot_payload(
                {"data": [{"id": "a/b", "pricing": {"prompt": True, "completion": "1"}}]}
            )

    def test_a_non_finite_rate_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError):
            parse_snapshot_payload(
                {"data": [{"id": "a/b", "pricing": {"prompt": "Infinity", "completion": "1"}}]}
            )

    def test_a_numeric_json_rate_is_accepted(self) -> None:
        # Not every catalogue quotes rates as strings.
        snapshot = parse_snapshot_payload(
            {"data": [{"id": "a/b", "pricing": {"prompt": 0.000002, "completion": 0.00001}}]}
        )
        self.assertEqual(snapshot.models["b"].input_per_1m, 2.0)

    def test_an_empty_rate_string_is_skipped(self) -> None:
        snapshot = parse_snapshot_payload(
            {
                "data": [
                    {"id": "a/blank", "pricing": {"prompt": "  ", "completion": "1"}},
                    {"id": "a/ok", "pricing": {"prompt": "0.000001", "completion": "0.000001"}},
                ]
            }
        )
        self.assertEqual(sorted(snapshot.models), ["ok"])

    def test_an_entry_that_is_not_an_object_is_skipped(self) -> None:
        snapshot = parse_snapshot_payload(
            {
                "data": [
                    "not an object",
                    {"id": "a/ok", "pricing": {"prompt": "0.000001", "completion": "0.000001"}},
                ]
            }
        )
        self.assertEqual(snapshot.skipped, 1)

    def test_an_entry_with_a_blank_id_is_skipped(self) -> None:
        snapshot = parse_snapshot_payload(
            {
                "data": [
                    {"id": "   ", "pricing": {"prompt": "0.000001", "completion": "0.000001"}},
                    {"id": "a/ok", "pricing": {"prompt": "0.000001", "completion": "0.000001"}},
                ]
            }
        )
        self.assertEqual(snapshot.skipped, 1)

    def test_a_models_key_that_is_not_an_object_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError):
            parse_snapshot_payload({"models": "nope"})

    def test_a_snapshot_entry_that_is_not_an_object_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError) as ctx:
            parse_snapshot_payload({"models": {"m": "cheap"}})
        self.assertIn("not an object", str(ctx.exception))

    def test_a_snapshot_entry_with_an_empty_name_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError):
            parse_snapshot_payload({"models": {"  ": {"input": 1.0, "output": 2.0}}})

    def test_a_snapshot_entry_with_a_boolean_rate_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError):
            parse_snapshot_payload({"models": {"m": {"input": True, "output": 2.0}}})

    def test_a_snapshot_entry_missing_a_rate_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError) as ctx:
            parse_snapshot_payload({"models": {"m": {"input": 1.0}}})
        self.assertIn("output", str(ctx.exception))

    def test_the_on_disk_shape_round_trips_through_parse(self) -> None:
        # Covers the `models`-shaped branch rather than the catalogue branch.
        original = parse_snapshot_payload(catalogue(("openai/gpt-4o", "0.000002", "0.00001")))
        self.assertEqual(parse_snapshot_payload(original.as_document()).models, original.models)


class ReadFailureTests(unittest.TestCase):
    def test_a_missing_snapshot_is_reported_by_load(self) -> None:
        with TemporaryDirectory() as folder:
            env = environment(folder)
            with self.assertRaises(GuardConfigError) as ctx:
                # read_snapshot_file is the strict reader; load_snapshot tolerates
                # absence, so this asserts the strict one still reports it.
                read_snapshot_file(snapshot_path(env=env))
            self.assertIn("no price snapshot", str(ctx.exception))

    def test_a_snapshot_that_is_not_json_is_refused(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "snap.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(GuardConfigError) as ctx:
                read_snapshot_file(path)
            self.assertIn("not valid JSON", str(ctx.exception))

    def test_a_snapshot_that_is_not_an_object_is_refused(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "snap.json"
            path.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaises(GuardConfigError) as ctx:
                read_snapshot_file(path)
            self.assertIn("JSON object", str(ctx.exception))

    def test_a_snapshot_without_a_models_object_is_refused(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "snap.json"
            path.write_text(json.dumps({"version": SNAPSHOT_VERSION}), encoding="utf-8")
            with self.assertRaises(GuardConfigError) as ctx:
                read_snapshot_file(path)
            self.assertIn("'models'", str(ctx.exception))

    def test_load_snapshot_reports_a_corrupt_file_rather_than_ignoring_it(self) -> None:
        with TemporaryDirectory() as folder:
            env = environment(folder)
            path = snapshot_path(env=env)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(GuardConfigError):
                load_snapshot(env=env)


class UrlGuardTests(unittest.TestCase):
    def test_a_url_without_a_host_is_refused(self) -> None:
        with self.assertRaises(GuardConfigError):
            fetch_snapshot("https://")

    def test_a_missing_file_body_says_so(self) -> None:
        with (
            mock.patch("urllib.request.urlopen", side_effect=OSError("connection reset")),
            self.assertRaises(GuardConfigError) as ctx,
        ):
            fetch_snapshot("https://example.test/models")
        self.assertIn("could not read", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
