from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from temporal_rag.retrieval import Query, TemporalError
from temporal_rag.data import at, evaluation_cases, generate_events
from temporal_rag.sqlite_store import SQLiteTemporalStore, utc_microseconds
from temporal_rag.storage_benchmark import benchmark, independent_scan, seeded_history


class PersistentStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "memory.sqlite"
        self.store = SQLiteTemporalStore(self.path)
        self.events = generate_events()

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_existing_hand_labeled_cases_match_persistent_queries(self):
        result = self.store.append(self.events, expected_version=0)
        self.assertEqual(result.version, 1)
        for case in evaluation_cases():
            with self.subTest(case=case["id"]):
                query = Query(**case["query"])
                answer = self.store.query(query)
                self.assertEqual(answer["status"], case["expected_status"])
                self.assertEqual(answer["value"], case["expected_value"])
                self.assertEqual(answer["citations"], sorted(case["expected_sources"]))

    def test_cas_stale_writer_does_not_append_or_advance_version(self):
        # Use a standalone original event, not a shuffled correction.
        first = next(event for event in self.events if event.id == "a1")
        self.store.append([first], expected_version=0)
        with self.assertRaises(TemporalError) as error:
            self.store.append([replace(first, id="second")], expected_version=0)
        self.assertEqual(error.exception.code, "stale_version")
        self.assertEqual(self.store.version, 1)
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)

    def test_pinned_wal_snapshot_remains_consistent_during_writer_commit(self):
        original = next(event for event in self.events if event.id == "a2")
        corrected = next(event for event in self.events if event.id == "a3")
        self.store.append([original], expected_version=0)
        query = Query("atlas", "owner", at("2025-02-15"), at("2025-04-15"))
        with self.store.snapshot() as snapshot:
            self.assertEqual(snapshot.answer(query)["value"], "Ben")
            with SQLiteTemporalStore(self.path) as writer:
                writer.append([corrected], expected_version=1)
            self.assertEqual(snapshot.answer(query)["value"], "Ben")
            self.assertEqual(snapshot.version, 1)
        self.assertEqual(self.store.query(query)["value"], "Cora")
        self.assertEqual(self.store.query(query, version=1)["value"], "Ben")
        with self.assertRaises(TemporalError):
            snapshot.retrieve(query)

    def test_correction_chain_batch_order_is_topological_and_provenance_is_complete(self):
        selected = [next(event for event in self.events if event.id == key) for key in ("a5", "a3", "a2")]
        self.store.append(selected, expected_version=0)
        answer = self.store.query(Query("atlas", "owner", at("2025-03-15"), at("2025-04-15")))
        self.assertEqual(answer["citations"], ["a5"])
        self.assertEqual([event["id"] for event in answer["correction_chains"]["a5"]], ["a2", "a3", "a5"])

    def test_invalid_correction_rolls_back_entire_batch(self):
        original = next(event for event in self.events if event.id == "a1")
        self.store.append([original], expected_version=0)
        invalid = replace(
            original, id="bad", recorded_at=at("2025-02-01"), effective_at=at("2025-02-01"), corrects=original.id
        )
        with self.assertRaises(TemporalError):
            self.store.append([replace(original, id="otherwise-valid"), invalid], expected_version=1)
        self.assertEqual(self.store.version, 1)
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)

    def test_events_are_immutable_even_through_raw_sql(self):
        original = next(event for event in self.events if event.id == "a1")
        self.store.append([original], expected_version=0)
        for statement in ("UPDATE events SET value='tampered'", "DELETE FROM events"):
            with self.subTest(statement=statement), self.assertRaises(sqlite3.IntegrityError):
                self.store.connection.execute(statement)

    def test_concurrent_cas_writers_have_exactly_one_winner(self):
        barrier = threading.Barrier(4)
        original = next(event for event in self.events if event.id == "a1")
        results = []
        lock = threading.Lock()

        def write(index):
            with SQLiteTemporalStore(self.path) as writer:
                barrier.wait(timeout=10)
                try:
                    writer.append([replace(original, id=f"writer-{index}")], expected_version=0)
                    status = "committed"
                except TemporalError as error:
                    status = error.code
                with lock:
                    results.append(status)

        threads = [threading.Thread(target=write, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(results.count("committed"), 1)
        self.assertEqual(results.count("stale_version"), 3)
        self.assertEqual(self.store.version, 1)

    def test_explicit_microsecond_time_boundaries_use_integer_arithmetic(self):
        self.assertEqual(
            utc_microseconds("2025-01-01T01:00:00.000001+01:00"), utc_microseconds("2025-01-01T00:00:00.000001Z")
        )
        self.assertEqual(utc_microseconds("1970-01-01T00:00:00.000001Z"), 1)
        self.assertEqual(utc_microseconds("1969-12-31T23:59:59.999999Z"), -1)

    def test_announced_future_fact_and_exact_recorded_boundary(self):
        self.store.append(self.events, expected_version=0)
        query = Query("atlas", "owner", at("2025-04-01"), at("2025-03-20"))
        self.assertEqual(self.store.query(query)["value"], "Dina")
        before_publication = replace(query, known_at="2025-03-19T23:59:59.999999Z")
        self.assertEqual(self.store.query(before_publication)["value"], "Cora")

    def test_vm_budget_interrupts_query_without_destroying_read_snapshot(self):
        self.store.append(self.events, expected_version=0)
        query = Query("atlas", "owner", at("2025-03-15"), at("2025-04-15"))
        with self.store.snapshot() as snapshot:
            with self.assertRaises(TemporalError) as error:
                snapshot.retrieve(query, max_vm_steps=100)
            self.assertEqual(error.exception.code, "query_budget")
            self.assertEqual([event.id for event in snapshot.retrieve(query)], ["a5"])

    def test_provenance_rejects_forged_or_future_event(self):
        self.store.append(self.events, expected_version=0)
        query = Query("atlas", "owner", at("2025-02-15"), at("2025-03-15"))
        with self.store.snapshot() as snapshot:
            actual = snapshot.retrieve(query)[0]
            with self.assertRaises(TemporalError):
                snapshot.correction_chain(replace(actual, value="forged"), query)
            with self.assertRaises(TemporalError):
                snapshot.correction_chain(next(event for event in self.events if event.id == "a5"), query)

    def test_index_plan_and_independent_large_history_oracle(self):
        events = seeded_history(24, 30, 7)
        self.store.append(events[:1000], expected_version=0)
        if len(events) > 1000:
            self.store.append(events[1000:], expected_version=1)
        query = Query("entity-0012", "owner", "2020-02-15T00:00:00Z", "2020-03-15T00:00:00Z")
        with self.store.snapshot() as snapshot:
            self.assertEqual([event.id for event in snapshot.retrieve(query)], independent_scan(events, query))
            self.assertTrue(any("USING INDEX ix_events_key" in line for line in snapshot.query_plan(query)))

    def test_small_reproducible_storage_benchmark_records_work_and_failures(self):
        report = benchmark(entities=8, versions=16, queries=20, seed=3)
        self.assertEqual(report["mismatch_count"], 0)
        self.assertEqual(report["scan_events_examined"], report["event_count"] * 20)
        self.assertGreater(report["indexed_query_seconds"]["median"], 0)
        self.assertTrue(any("ix_events" in line for line in report["query_plan"]))


if __name__ == "__main__":
    unittest.main()
