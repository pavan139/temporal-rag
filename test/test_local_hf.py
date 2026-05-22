from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from temporal_rag import Query, TemporalError, TemporalRetriever
from temporal_rag.__main__ import main
from temporal_rag.data import at, generate_events
from temporal_rag.local_hf import LocalHFAnswerer, build_prompt, evaluate_local_hf, parse_grounded_output


class LocalHFTests(unittest.TestCase):
    def setUp(self):
        self.query = Query("atlas", "owner", at("2025-01-15"), at("2025-04-15"))
        self.evidence = TemporalRetriever(generate_events()).retrieve(self.query)

    def test_supported_json_is_accepted(self):
        result = parse_grounded_output('{"status":"answered","value":"Ada","citations":["a1"]}', self.evidence)
        self.assertEqual(result["value"], "Ada")

    def test_malformed_markdown_or_extra_fields_are_rejected(self):
        samples = [
            "not JSON",
            '```json\n{"status":"abstained","value":null,"citations":[]}\n```',
            '{"status":"abstained","value":null,"citations":[],"explanation":"extra"}',
            '{"status":"answered","status":"abstained","value":null,"citations":[]}',
            "[]",
        ]
        for sample in samples:
            with self.subTest(sample=sample), self.assertRaises(TemporalError):
                parse_grounded_output(sample, self.evidence)

    def test_unauthorized_citation_ids_are_rejected(self):
        for citations in (["a2"], ["invented"], [7], ["a1", "a1"]):
            with self.subTest(citations=citations), self.assertRaises(TemporalError):
                parse_grounded_output(
                    json.dumps({"status": "answered", "value": "Ada", "citations": citations}), self.evidence
                )

    def test_unsupported_or_wrong_type_values_are_rejected(self):
        for value in ("Mallory", 1, {}, None):
            with self.subTest(value=value), self.assertRaises(TemporalError):
                parse_grounded_output(
                    json.dumps({"status": "answered", "value": value, "citations": ["a1"]}), self.evidence
                )

    def test_conflicting_or_missing_evidence_cannot_support_answer(self):
        conflict_query = Query("cobalt", "owner", at("2025-02-15"), at("2025-04-15"))
        conflict = TemporalRetriever(generate_events()).retrieve(conflict_query)
        with self.assertRaises(TemporalError):
            parse_grounded_output('{"status":"answered","value":"Hugo","citations":["c2"]}', conflict)
        with self.assertRaises(TemporalError):
            parse_grounded_output('{"status":"answered","value":"Ada","citations":[]}', [])

    def test_abstention_requires_null_value(self):
        self.assertEqual(
            parse_grounded_output('{"status":"abstained","value":null,"citations":[]}', [])["status"], "abstained"
        )
        with self.assertRaises(TemporalError):
            parse_grounded_output('{"status":"abstained","value":"Ada","citations":[]}', self.evidence)

    def test_missing_checkpoint_does_not_import_optional_libraries(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch(
                "temporal_rag.local_hf.importlib.import_module", side_effect=AssertionError("No ML imports allowed")
            ):
                with self.assertRaisesRegex(TemporalError, "existing local checkpoint"):
                    LocalHFAnswerer(Path(temp) / "missing")
                report = evaluate_local_hf(Path(temp) / "missing")
            self.assertEqual(report["status"], "not_run")
            self.assertFalse(report["executed"])
            self.assertEqual(report["model_calls"], 0)
            self.assertEqual(report["case_count"], 19)

    def test_valid_config_initialization_is_lazy_and_budgets_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "config.json").write_text('{"model_type":"test_fixture"}')
            with patch(
                "temporal_rag.local_hf.importlib.import_module", side_effect=AssertionError("No ML imports allowed")
            ):
                answerer = LocalHFAnswerer(temp)
                self.assertFalse(answerer.provenance["loaded"])
                self.assertEqual(answerer.provenance["device"], "cpu")
                for kwargs in ({"max_input_tokens": 2049}, {"max_new_tokens": 257}, {"max_context_chars": 0}):
                    with self.subTest(kwargs=kwargs), self.assertRaises(TemporalError):
                        LocalHFAnswerer(temp, **kwargs)

    def test_policy_abstention_never_loads_model(self):
        conflict_query = Query("cobalt", "owner", at("2025-02-15"), at("2025-04-15"))
        conflict = TemporalRetriever(generate_events()).retrieve(conflict_query)
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "config.json").write_text("{}")
            answerer = LocalHFAnswerer(temp)
            with patch(
                "temporal_rag.local_hf.importlib.import_module", side_effect=AssertionError("No ML imports allowed")
            ):
                for query, evidence in ((self.query, []), (conflict_query, conflict)):
                    result = answerer.answer(query, evidence)
                    self.assertEqual(result["status"], "abstained")
                    self.assertEqual(result["generation_status"], "policy_abstention")
            self.assertEqual(answerer.calls, 0)

    def test_prompt_contains_only_query_and_supplied_evidence(self):
        prompt = build_prompt(self.query, self.evidence)
        self.assertIn('"id": "a1"', prompt)
        self.assertIn('"value": "Ada"', prompt)
        self.assertNotIn('"id": "a2"', prompt)
        self.assertNotIn("expected_value", prompt)
        self.assertNotIn("expected_sources", prompt)
        with self.assertRaises(TemporalError):
            build_prompt(self.query, self.evidence, max_context_chars=5)

    def test_invalid_output_is_bounded(self):
        with self.assertRaises(TemporalError):
            parse_grounded_output("x" * 8193, self.evidence)

    def test_cli_records_not_run_without_model_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp, "hf.json")
            with redirect_stdout(io.StringIO()):
                result = main(["hf-evaluate", "--checkpoint", str(Path(temp) / "missing"), "--output", str(output)])
            self.assertEqual(result, 2)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "not_run")
            self.assertEqual(report["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
