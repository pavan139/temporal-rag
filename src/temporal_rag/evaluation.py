"""Offline, synthetic-only metrics with a dataset hash and failure records."""

from __future__ import annotations

import hashlib
import json

from .retrieval import (
    LexicalLatestBaseline,
    Query,
    ReferenceAnswerer,
    TemporalError,
    TemporalRetriever,
    validate_answer,
)
from .data import evaluation_cases, generate_events


def evaluate() -> dict:
    events, cases = generate_events(), evaluation_cases()
    corpus = [event.to_dict() for event in sorted(events, key=lambda event: event.id)]
    dataset_hash = hashlib.sha256(json.dumps({"events": corpus, "cases": cases}, sort_keys=True).encode()).hexdigest()
    report = {
        "dataset": "synthetic_temporal_events_v1",
        "dataset_sha256": dataset_hash,
        "event_count": len(events),
        "case_count": len(cases),
        "answerer": ReferenceAnswerer.name,
        "live_model_calls": 0,
        "systems": {},
    }
    for retriever in (LexicalLatestBaseline(events), TemporalRetriever(events)):
        rows = []
        for case in cases:
            query = Query(**case["query"])
            evidence = retriever.retrieve(query)
            context_error = None
            try:
                result = ReferenceAnswerer().answer(query, evidence)
                validate_answer(result, evidence)
            except TemporalError as exc:
                context_error = {"code": exc.code, "message": str(exc)}
                result = {
                    "answerer": ReferenceAnswerer.name,
                    "status": "rejected_context",
                    "value": None,
                    "citations": [],
                    "reason": exc.code,
                }
            expected, actual = set(case["expected_sources"]), {event.id for event in evidence}
            rows.append(
                {
                    "case_id": case["id"],
                    "tag": case["tag"],
                    "retrieval_exact": actual == expected,
                    "retrieval_recall": len(expected & actual) / len(expected) if expected else None,
                    "answer_correct": result["status"] == case["expected_status"]
                    and result["value"] == case["expected_value"],
                    "citation_ids_valid": context_error is None,
                    "context_valid": context_error is None,
                    "context_error": context_error,
                    "expected_sources": sorted(expected),
                    "retrieved_sources": sorted(actual),
                    "expected_value": case["expected_value"],
                    "result": result,
                }
            )
        relevant_rows = [row for row in rows if row["retrieval_recall"] is not None]
        abstention_cases = [row for row, case in zip(rows, cases) if case["expected_status"] == "abstained"]
        answered_rows = [row for row in rows if row["context_valid"]]
        slices = {}
        for tag in sorted({row["tag"] for row in rows}):
            selected = [row for row in rows if row["tag"] == tag]
            slices[tag] = {
                "case_count": len(selected),
                "answer_correct": sum(row["answer_correct"] for row in selected),
                "retrieval_exact": sum(row["retrieval_exact"] for row in selected),
                "context_rejections": sum(not row["context_valid"] for row in selected),
            }
        report["systems"][retriever.name] = {
            "retrieval": {
                "exact_source_sets": sum(row["retrieval_exact"] for row in rows),
                "case_count": len(rows),
                "exact_source_set_accuracy": sum(row["retrieval_exact"] for row in rows) / len(rows),
                "macro_recall": sum(row["retrieval_recall"] for row in relevant_rows) / len(relevant_rows),
                "nonempty_relevance_case_count": len(relevant_rows),
            },
            "answer": {
                "correct": sum(row["answer_correct"] for row in rows),
                "case_count": len(rows),
                "accuracy": sum(row["answer_correct"] for row in rows) / len(rows),
                "citation_membership_accuracy": 1.0 if answered_rows else None,
                "citation_membership_case_count": len(answered_rows),
                "rejected_context_count": len(rows) - len(answered_rows),
                "correct_abstentions": sum(row["answer_correct"] for row in abstention_cases),
                "expected_abstention_count": len(abstention_cases),
            },
            "slices": slices,
            "failures": [row["case_id"] for row in rows if not row["retrieval_exact"] or not row["answer_correct"]],
            "cases": rows,
        }
    return report
