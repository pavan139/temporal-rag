"""Seeded indexed-versus-scan experiments, with real timings and explicit work units."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import platform
from pathlib import Path
import random
import sqlite3
import statistics
import tempfile
import time

from .retrieval import Event, Query, timestamp
from .sqlite_store import MAX_BATCH, SQLiteTemporalStore


def seeded_history(entities: int, versions: int, seed: int) -> list[Event]:
    if type(entities) is not int or not 1 <= entities <= 1000 or type(versions) is not int or not 2 <= versions <= 1000:
        raise ValueError("entities must be 1..1000 and versions 2..1000")
    if entities * versions > 100_000:
        raise ValueError("benchmark history budget is 100000 base events")
    rng = random.Random(seed)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    events = []
    for entity_index in range(entities):
        entity = f"entity-{entity_index:04d}"
        for boundary in range(versions):
            valid = base + timedelta(days=boundary * 3)
            recorded = valid + timedelta(days=rng.randint(-2, 8))
            identifier = f"{entity}-{boundary:04d}"
            original = Event(
                identifier,
                entity,
                "owner",
                f"owner-{boundary % 11}",
                valid.isoformat(),
                recorded.isoformat(),
                f"{entity} owner change {boundary}",
                f"synthetic://history/{identifier}",
            )
            events.append(original)
            if rng.random() < 0.3:
                recorded += timedelta(days=rng.randint(1, 12))
                correction = Event(
                    identifier + "-c1",
                    entity,
                    "owner",
                    f"corrected-{boundary % 7}",
                    valid.isoformat(),
                    recorded.isoformat(),
                    "Published correction",
                    f"synthetic://history/{identifier}-c1",
                    original.id,
                )
                events.append(correction)
                if rng.random() < 0.25:
                    events.append(
                        Event(
                            identifier + "-c2",
                            entity,
                            "owner",
                            f"final-{boundary % 5}",
                            valid.isoformat(),
                            (recorded + timedelta(days=1)).isoformat(),
                            "Second correction",
                            f"synthetic://history/{identifier}-c2",
                            correction.id,
                        )
                    )
                if rng.random() < 0.08:
                    events.append(
                        Event(
                            identifier + "-branch",
                            entity,
                            "owner",
                            "disputed",
                            valid.isoformat(),
                            (recorded + timedelta(days=2)).isoformat(),
                            "Independent correction branch",
                            f"synthetic://history/{identifier}-branch",
                            original.id,
                        )
                    )
    return sorted(events, key=lambda event: (timestamp(event.recorded_at), event.id))


def independent_scan(events: list[Event], query: Query) -> list[str]:
    """Independent Python oracle; does not call SQL or TemporalRetriever."""
    known, valid = timestamp(query.known_at), timestamp(query.as_of)
    selected = {}
    for event in events:
        if event.entity == query.entity and event.attribute == query.attribute:
            if timestamp(event.recorded_at) <= known and timestamp(event.effective_at) <= valid:
                selected[event.id] = event
    retired = {event.corrects for event in selected.values() if event.corrects is not None}
    boundaries = {}
    for identifier, event in selected.items():
        if identifier not in retired:
            boundaries.setdefault(timestamp(event.effective_at), []).append(identifier)
    return sorted(boundaries[max(boundaries)]) if boundaries else []


def _summary(values):
    ordered = sorted(values)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


def benchmark(*, entities: int = 128, versions: int = 64, queries: int = 128, seed: int = 7) -> dict:
    if type(queries) is not int or not 1 <= queries <= 1000 or type(seed) is not int:
        raise ValueError("queries must be 1..1000 and seed must be an integer")
    events = seeded_history(entities, versions, seed)
    rng = random.Random(seed ^ 8171)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    requests = [
        Query(
            f"entity-{rng.randrange(entities):04d}",
            "owner",
            (base + timedelta(days=rng.randrange(versions * 3 + 20))).isoformat(),
            (base + timedelta(days=rng.randrange(versions * 3 + 20))).isoformat(),
        )
        for _ in range(queries)
    ]
    dataset_hash = hashlib.sha256(
        json.dumps([event.to_dict() for event in events], sort_keys=True).encode()
    ).hexdigest()
    indexed_seconds, scan_seconds, work, failures, updates, cases = [], [], [], [], [], []
    with tempfile.TemporaryDirectory() as directory:
        with SQLiteTemporalStore(Path(directory) / "memory.sqlite") as store:
            version = 0
            for start in range(0, len(events), MAX_BATCH):
                update = store.append(events[start : start + MAX_BATCH], expected_version=version)
                version = update.version
                updates.append(asdict(update))
            with store.snapshot() as snapshot:
                plan = snapshot.query_plan(requests[0])
                for index, query in enumerate(requests):
                    got = [event.id for event in snapshot.retrieve(query)]
                    indexed_seconds.append(snapshot.last_metrics["query_seconds"])
                    work.append(snapshot.last_metrics["sqlite_vm_steps_estimate"])
                    started = time.perf_counter()
                    expected = independent_scan(events, query)
                    scan_seconds.append(time.perf_counter() - started)
                    cases.append(
                        {
                            "query_index": index,
                            "entity": query.entity,
                            "as_of": query.as_of,
                            "known_at": query.known_at,
                            "oracle_ids": expected,
                            "indexed_ids": got,
                            "indexed_metrics": dict(snapshot.last_metrics),
                            "scan_seconds": scan_seconds[-1],
                            "scan_events_examined": len(events),
                        }
                    )
                    if got != expected:
                        failures.append({"query": index, "expected": expected, "actual": got})
            page_count = store.connection.execute("PRAGMA page_count").fetchone()[0]
            page_size = store.connection.execute("PRAGMA page_size").fetchone()[0]
    return {
        "experiment": "sqlite_wal_indexed_bitemporal_vs_independent_scan",
        "dataset_sha256": dataset_hash,
        "seed": seed,
        "event_count": len(events),
        "entity_count": entities,
        "base_versions_per_entity": versions,
        "query_count": queries,
        "mismatch_count": len(failures),
        "failures": failures,
        "sqlite_version": sqlite3.sqlite_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "indexed_query_seconds": _summary(indexed_seconds),
        "independent_scan_seconds": _summary(scan_seconds),
        "scan_events_examined": len(events) * queries,
        "sqlite_vm_steps_estimate": _summary(work),
        "sqlite_vm_step_granularity": 100,
        "query_plan": plan,
        "append_transactions": updates,
        "cases": cases,
        "append_total_seconds": sum(update["transaction_seconds"] for update in updates),
        "database_page_bytes": page_count * page_size,
        "limitations": [
            "VM instructions, Python events examined, and disk page reads are different work units; no row-scan equivalence is claimed.",
            "Timings are one local warm-cache run, including progress-handler overhead; no serving SLA or hardware-independent speedup is claimed.",
            "The independent scan is a correctness oracle, not an optimized in-memory temporal index.",
            "Source records are synthetic and trusted metadata; no learned model was evaluated.",
        ],
    }
