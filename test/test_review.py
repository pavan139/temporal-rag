"""Behavioral regressions from the independent temporal-boundary review."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from temporal_rag.__main__ import main
from temporal_rag.retrieval import Query, ReferenceAnswerer, TemporalError, TemporalRetriever, timestamp
from temporal_rag.data import at, generate_events, load_events
from temporal_rag.evaluation import evaluate
from temporal_rag.local_hf import build_prompt


class TemporalReviewTests(unittest.TestCase):
    def setUp(self):
        self.events = generate_events()
        self.query = Query("atlas", "owner", at("2025-01-15"), at("2025-01-15"))
        self.original = next(event for event in self.events if event.id == "a1")

    def test_answer_boundary_rejects_future_recording(self):
        unpublished = replace(self.original, id="unpublished", recorded_at=at("2025-01-16"))
        with self.assertRaises(TemporalError) as error:
            ReferenceAnswerer().answer(self.query, [unpublished])
        self.assertEqual(error.exception.code, "future_knowledge")

    def test_answer_boundary_rejects_future_effective_fact(self):
        announced = replace(self.original, id="announced", effective_at=at("2025-02-01"))
        with self.assertRaises(TemporalError) as error:
            ReferenceAnswerer().answer(self.query, [announced])
        self.assertEqual(error.exception.code, "future_effective_context")

    def test_generation_boundary_rejects_future_context_before_model_call(self):
        future = replace(self.original, recorded_at=at("2025-04-15"))
        with self.assertRaises(TemporalError):
            build_prompt(self.query, [future])

    def test_duplicate_or_unvalidated_evidence_is_rejected(self):
        for evidence in ([self.original, self.original], [self.original.to_dict()]):
            with self.subTest(evidence=evidence), self.assertRaises(TemporalError):
                ReferenceAnswerer().answer(self.query, evidence)

    def test_utc_overflow_has_stable_temporal_error(self):
        for value in ("0001-01-01T00:00:00+14:00", "9999-12-31T23:00:00-14:00"):
            with self.subTest(value=value), self.assertRaises(TemporalError) as error:
                timestamp(value)
            self.assertEqual(error.exception.code, "invalid_timestamp")

    def test_retrieval_explanation_hides_unpublished_event_ids(self):
        audit = TemporalRetriever(self.events).explain(self.query)
        self.assertEqual(audit["selected_ids"], ["a1"])
        self.assertEqual([row["event_id"] for row in audit["decisions"]], ["a1"])
        self.assertNotIn("a3", json.dumps(audit))

    def test_retrieval_explanation_classifies_corrections_and_future_announcements(self):
        query = Query("atlas", "owner", at("2025-03-15"), at("2025-04-15"))
        audit = TemporalRetriever(self.events).explain(query)
        decisions = {row["event_id"]: row["decision"] for row in audit["decisions"]}
        self.assertEqual(audit["selected_ids"], ["a5"])
        self.assertEqual(decisions["a1"], "older_effective_boundary")
        self.assertEqual(decisions["a2"], "superseded_by_visible_correction")
        self.assertEqual(decisions["a3"], "superseded_by_visible_correction")
        self.assertEqual(decisions["a4"], "not_yet_effective")

    def test_import_rejects_duplicate_json_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            encoded = json.dumps(self.original.to_dict())
            path.write_text("[" + encoded.replace('"id": "a1"', '"id": "a1", "id": "replacement"') + "]")
            with self.assertRaises(TemporalError):
                load_events(path)

    def test_explicit_query_cli_accepts_custom_entity_and_explanation(self):
        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "custom.json"
            events.write_text(json.dumps([replace(self.original, entity="my-object").to_dict()]))
            stream = io.StringIO()
            with redirect_stdout(stream):
                status = main(
                    [
                        "query",
                        "--events",
                        str(events),
                        "--entity",
                        "my-object",
                        "--attribute",
                        "owner",
                        "--as-of",
                        self.query.as_of,
                        "--known-at",
                        self.query.known_at,
                        "--explain",
                    ]
                )
            self.assertEqual(status, 0)
            result = json.loads(stream.getvalue())
            self.assertEqual(result["query_result"]["value"], "Ada")
            self.assertEqual(result["retrieval_audit"]["selected_ids"], ["a1"])

    def test_evaluation_records_context_rejections_and_slice_denominators(self):
        report = evaluate()
        baseline = report["systems"]["lexical_latest"]
        self.assertGreater(baseline["answer"]["rejected_context_count"], 0)
        self.assertEqual(sum(row["case_count"] for row in baseline["slices"].values()), report["case_count"])
        self.assertEqual(report["systems"]["timestamp_aware"]["answer"]["rejected_context_count"], 0)


if __name__ == "__main__":
    unittest.main()
