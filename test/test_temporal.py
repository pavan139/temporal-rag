from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from temporal_rag.retrieval import (
    EventStore,
    MAX_EVENTS,
    Query,
    ReferenceAnswerer,
    TemporalError,
    TemporalRetriever,
    timestamp,
    validate_answer,
)
from temporal_rag.data import at, evaluation_cases, generate_events, load_events
from temporal_rag.evaluation import evaluate
from temporal_rag.__main__ import main


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.events = generate_events()
        self.engine = TemporalRetriever(self.events)

    def query(self, as_of, known_at="2025-04-15", entity="atlas", attribute="owner"):
        return Query(entity, attribute, at(as_of), at(known_at))

    def ids(self, query):
        return [event.id for event in self.engine.retrieve(query)]

    def test_future_announcements_do_not_change_history(self):
        self.assertEqual(self.ids(self.query("2025-01-15")), ["a1"])
        self.assertEqual(self.ids(self.query("2025-04-15")), ["a4"])

    def test_knowledge_cutoff_preserves_original_fact(self):
        self.assertEqual(self.ids(self.query("2025-02-15", "2025-02-15")), ["a2"])
        self.assertEqual(self.ids(self.query("2025-02-15", "2025-03-15")), ["a3"])

    def test_correction_chain_retires_all_ancestors(self):
        self.assertEqual(self.ids(self.query("2025-02-15")), ["a5"])

    def test_late_arrival_has_two_time_dimensions(self):
        self.assertEqual(self.ids(self.query("2025-03-15", "2025-03-15", "beacon")), ["b1"])
        self.assertEqual(self.ids(self.query("2025-03-15", "2025-04-15", "beacon")), ["b2"])
        self.assertEqual(self.ids(self.query("2025-02-15", "2025-04-15", "beacon")), ["b1"])

    def test_equal_boundaries_are_inclusive(self):
        self.assertEqual(self.ids(self.query("2025-02-01", "2025-02-01")), ["a2"])

    def test_timezone_normalization_changes_boundary(self):
        query = Query("atlas", "owner", "2025-02-01T00:00:00+02:00", at("2025-02-15"))
        self.assertEqual(self.ids(query), ["a1"])

    def test_conflict_is_preserved_and_answer_abstains(self):
        query = self.query("2025-02-15", entity="cobalt")
        evidence = self.engine.retrieve(query)
        self.assertEqual({event.id for event in evidence}, {"c2", "c3"})
        answer = ReferenceAnswerer().answer(query, evidence)
        self.assertEqual(answer["status"], "abstained")
        self.assertEqual(answer["reason"], "contradictory_context")
        self.assertIsNone(answer["value"])

    def test_missing_evidence_abstains(self):
        for query in (self.query("2024-12-31"), self.query("2025-04-15", entity="missing")):
            answer = ReferenceAnswerer().answer(query, self.engine.retrieve(query))
            self.assertEqual(answer["reason"], "missing_context")
            self.assertEqual(answer["citations"], [])

    def test_untrusted_document_cannot_override_fact(self):
        answer = self.engine.query("delta", "owner", at("2025-04-15"), at("2025-04-15"))
        self.assertEqual(answer["value"], "Jules")
        self.assertEqual(answer["citations"], ["d1"])

    def test_same_value_multiple_sources_are_not_conflict(self):
        original = next(event for event in self.events if event.id == "a1")
        second = replace(original, id="confirmation", source_uri="synthetic://confirmation")
        query = self.query("2025-01-15", "2025-01-15")
        evidence = TemporalRetriever([original, second]).retrieve(query)
        answer = ReferenceAnswerer().answer(query, evidence)
        self.assertEqual(answer["value"], "Ada")
        self.assertEqual(set(answer["citations"]), {"a1", "confirmation"})

    def test_ingestion_order_does_not_change_retrieval(self):
        for case in evaluation_cases():
            query = Query(**case["query"])
            other = TemporalRetriever(generate_events(seed=42)).retrieve(query)
            self.assertEqual(self.ids(query), [event.id for event in other])

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(TemporalError, "unique"):
            EventStore([self.events[0], self.events[0]])

    def test_invalid_correction_target_and_time_rejected(self):
        original = next(event for event in self.events if event.id == "a1")
        for correction in (
            replace(original, id="bad", corrects="missing"),
            replace(original, id="bad", corrects="a1"),
            replace(original, id="bad", corrects="a1", attribute="status", recorded_at=at("2025-02-01")),
        ):
            with self.assertRaises(TemporalError):
                EventStore([original, correction])

    def test_divergent_corrections_remain_contradictory(self):
        initial = next(event for event in self.events if event.id == "a1")
        left = replace(initial, id="left", corrects="a1", value="Left", recorded_at=at("2025-02-01"))
        right = replace(initial, id="right", corrects="a1", value="Right", recorded_at=at("2025-02-02"))
        evidence = TemporalRetriever([initial, left, right]).retrieve(self.query("2025-01-15"))
        self.assertEqual({event.id for event in evidence}, {"left", "right"})

    def test_invalid_and_naive_timestamps_rejected(self):
        for value in ("2025-01-01", "garbage", 5, "x" * 100):
            with self.assertRaises(TemporalError):
                timestamp(value)

    def test_oversized_query_and_events_rejected(self):
        with self.assertRaises(TemporalError):
            Query("atlas", "owner", at("2025-01-01"), at("2025-01-01"), "x" * 501)
        with self.assertRaisesRegex(TemporalError, "5000"):
            EventStore(self.events[0] for _ in range(MAX_EVENTS + 1))

    def test_invalid_event_field_type_rejected(self):
        with self.assertRaises(TemporalError):
            replace(self.events[0], value=123)

    def test_fabricated_citation_and_value_rejected(self):
        query = self.query("2025-01-15", "2025-01-15")
        evidence = self.engine.retrieve(query)
        answer = ReferenceAnswerer().answer(query, evidence)
        for bad in (
            {**answer, "citations": ["invented"]},
            {**answer, "value": "Mallory"},
            {**answer, "citations": ["a1", "a1"]},
        ):
            with self.assertRaises(TemporalError):
                validate_answer(bad, evidence)

    def test_validator_cannot_cherry_pick_conflicting_evidence(self):
        evidence = self.engine.retrieve(self.query("2025-02-15", entity="cobalt"))
        with self.assertRaises(TemporalError):
            validate_answer({"status": "answered", "value": "Hugo", "citations": ["c2"]}, evidence)

    def test_wrong_entity_context_rejected(self):
        with self.assertRaises(TemporalError):
            ReferenceAnswerer().answer(
                self.query("2025-01-15", entity="other"), self.engine.retrieve(self.query("2025-01-15"))
            )

    def test_import_invalid_schema_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.json"
            for content in ("not json", "{}", '[{"bad": "schema"}]'):
                path.write_text(content)
                with self.assertRaises(TemporalError):
                    load_events(path)
            path.write_text(json.dumps([event.to_dict() for event in self.events]))
            self.assertEqual(len(load_events(path)), len(self.events))
            path.write_bytes(b"\xff\xfe")
            with self.assertRaises(TemporalError):
                load_events(path)

    def test_cli_writes_actual_json_and_returns_stable_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "nested" / "demo.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["demo", "--output", str(output)]), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["results"]["timestamp_aware"]["value"], "Ada")
            self.assertEqual(result["results"]["lexical_latest"]["status"], "rejected_context")
            self.assertEqual(result["results"]["lexical_latest"]["reason"], "future_effective_context")
            invalid = Path(temp) / "invalid.json"
            invalid.write_text("not JSON")
            error = io.StringIO()
            with redirect_stderr(error):
                self.assertEqual(main(["demo", "--events", str(invalid)]), 2)
            self.assertEqual(json.loads(error.getvalue())["error"], "invalid_event")

    def test_reference_eval_has_separate_metrics_and_real_baseline_failures(self):
        report = evaluate()
        aware = report["systems"]["timestamp_aware"]
        baseline = report["systems"]["lexical_latest"]
        self.assertGreaterEqual(report["case_count"], 12)
        self.assertEqual(aware["answer"]["correct"], report["case_count"])
        self.assertLess(baseline["answer"]["accuracy"], aware["answer"]["accuracy"])
        self.assertEqual(len(report["dataset_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
