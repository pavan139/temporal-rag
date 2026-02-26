from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sqlite3
from pathlib import Path
import sys

from .retrieval import LexicalLatestBaseline, Query, ReferenceAnswerer, TemporalError, TemporalRetriever
from .data import at, generate_events, load_events
from .evaluation import evaluate


def emit(payload: dict, output: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Temporal retrieval with explicit deterministic reference answers")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("demo", "evaluate"):
        child = sub.add_parser(command)
        child.add_argument("--output", help="Optional JSON report path")
        if command == "demo":
            child.add_argument("--events", help="Optional JSON event array instead of synthetic fixture")
    hf = sub.add_parser("hf-evaluate", help="Evaluate actual generation from an existing local Hugging Face checkpoint")
    hf.add_argument("--checkpoint", required=True, help="Existing local causal-LM checkpoint directory; never a hub ID")
    hf.add_argument("--output", help="Optional JSON report path")
    query_parser = sub.add_parser("query", help="Query imported or synthetic events at explicit bitemporal cutoffs")
    query_parser.add_argument("--entity", required=True)
    query_parser.add_argument("--attribute", required=True)
    query_parser.add_argument("--as-of", required=True)
    query_parser.add_argument("--known-at", required=True)
    query_parser.add_argument("--events")
    query_parser.add_argument("--text", default="")
    query_parser.add_argument("--explain", action="store_true")
    query_parser.add_argument("--output")
    storage_import = sub.add_parser(
        "store-import", help="Atomically append up to 1000 events to SQLite with version CAS"
    )
    storage_import.add_argument("--database", required=True)
    storage_import.add_argument("--events", required=True)
    storage_import.add_argument("--expected-version", type=int, required=True)
    storage_import.add_argument("--output")
    storage_query = sub.add_parser("store-query", help="Query a consistent SQLite WAL snapshot")
    storage_query.add_argument("--database", required=True)
    for field in ("entity", "attribute", "as-of", "known-at"):
        storage_query.add_argument("--" + field, required=True)
    storage_query.add_argument("--version", type=int)
    storage_query.add_argument("--output")
    storage_benchmark = sub.add_parser("store-benchmark", help="Compare indexed SQL to an independent scan oracle")
    storage_benchmark.add_argument("--entities", type=int, default=128)
    storage_benchmark.add_argument("--versions", type=int, default=64)
    storage_benchmark.add_argument("--queries", type=int, default=128)
    storage_benchmark.add_argument("--seed", type=int, default=7)
    storage_benchmark.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "store-benchmark":
            from .storage_benchmark import benchmark

            report = benchmark(entities=args.entities, versions=args.versions, queries=args.queries, seed=args.seed)
        elif args.command in ("store-import", "store-query"):
            from .sqlite_store import SQLiteTemporalStore

            if args.command == "store-query" and not Path(args.database).is_file():
                raise TemporalError("missing_store", "Query requires an existing database")
            with SQLiteTemporalStore(args.database) as store:
                if args.command == "store-import":
                    report = {
                        "storage": "sqlite_wal",
                        "append": asdict(
                            store.append(load_events(args.events), expected_version=args.expected_version)
                        ),
                    }
                else:
                    query = Query(args.entity, args.attribute, args.as_of, args.known_at)
                    with store.snapshot(version=args.version) as snapshot:
                        report = {
                            "storage": "sqlite_wal",
                            "result": snapshot.answer(query),
                            "query_plan": snapshot.query_plan(query),
                        }
        elif args.command == "hf-evaluate":
            from .local_hf import evaluate_local_hf

            report = evaluate_local_hf(args.checkpoint)
        elif args.command == "query":
            events = load_events(args.events) if args.events else generate_events()
            query = Query(args.entity, args.attribute, args.as_of, args.known_at, args.text)
            engine = TemporalRetriever(events)
            report = {
                "mode": "offline_reference",
                "query_result": ReferenceAnswerer().answer(query, engine.retrieve(query)),
                "live_model_calls": 0,
            }
            if args.explain:
                report["retrieval_audit"] = engine.explain(query)
        elif args.command == "evaluate":
            report = evaluate()
        else:
            events = load_events(args.events) if args.events else generate_events()
            query = Query(
                "atlas", "owner", at("2025-01-15"), at("2025-04-15"), "Who was the owner of atlas on January 15?"
            )
            results = {}
            for engine in (LexicalLatestBaseline(events), TemporalRetriever(events)):
                try:
                    results[engine.name] = ReferenceAnswerer().answer(query, engine.retrieve(query))
                except TemporalError as exc:
                    results[engine.name] = {
                        "status": "rejected_context",
                        "value": None,
                        "citations": [],
                        "reason": exc.code,
                        "error": str(exc),
                    }
            report = {
                "mode": "offline_synthetic_reference" if not args.events else "offline_imported_reference",
                "live_model_calls": 0,
                "query": {
                    "entity": query.entity,
                    "attribute": query.attribute,
                    "as_of": query.as_of,
                    "known_at": query.known_at,
                },
                "results": results,
            }
        emit(report, args.output)
        return (
            2
            if report.get("status") in ("not_run", "completed_with_failures") or report.get("mismatch_count", 0)
            else 0
        )
    except (TemporalError, OSError, ValueError, sqlite3.Error) as exc:
        print(json.dumps({"error": getattr(exc, "code", "io_error"), "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
