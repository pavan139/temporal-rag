"""Persistent indexed bitemporal memory with transactional compare-and-swap writes.

An append transaction owns a monotonically increasing commit version. Valid time
and knowledge time remain independent event attributes; commit version pins a
physical/logical read snapshot rather than replacing either timestamp.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import time
from typing import Iterable

from .retrieval import Event, Query, ReferenceAnswerer, TemporalError, timestamp, validate_context

MAX_BATCH = 1000
MAX_STORED_EVENTS = 1_000_000
SCHEMA_VERSION = 1
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_COLUMNS = "id,entity,attribute,value,effective_at,recorded_at,text,source_uri,corrects"
_SELECT = """
WITH eligible AS MATERIALIZED (
 SELECT * FROM events
 WHERE entity=? AND attribute=? AND effective_us<=? AND recorded_us<=? AND commit_version<=?
), active AS MATERIALIZED (
 SELECT e.* FROM eligible e
 WHERE NOT EXISTS (
   SELECT 1 FROM events c
   WHERE c.corrects=e.id AND c.recorded_us<=? AND c.commit_version<=?
 )
)
SELECT id,entity,attribute,value,effective_at,recorded_at,text,source_uri,corrects
FROM active WHERE effective_us=(SELECT MAX(effective_us) FROM active)
ORDER BY id
"""
_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
 singleton INTEGER PRIMARY KEY CHECK(singleton=1),
 version INTEGER NOT NULL CHECK(version>=0), event_count INTEGER NOT NULL CHECK(event_count>=0)
);
INSERT OR IGNORE INTO metadata(singleton,version,event_count) VALUES(1,0,0);
CREATE TABLE IF NOT EXISTS events (
 sequence INTEGER PRIMARY KEY,
 id TEXT NOT NULL UNIQUE, entity TEXT NOT NULL, attribute TEXT NOT NULL, value TEXT NOT NULL,
 effective_at TEXT NOT NULL, recorded_at TEXT NOT NULL,
 effective_us INTEGER NOT NULL, recorded_us INTEGER NOT NULL,
 text TEXT NOT NULL, source_uri TEXT NOT NULL,
 corrects TEXT REFERENCES events(id), commit_version INTEGER NOT NULL CHECK(commit_version>0)
);
CREATE INDEX IF NOT EXISTS ix_events_key_valid ON events(entity,attribute,effective_us DESC,recorded_us,commit_version);
CREATE INDEX IF NOT EXISTS ix_events_key_known ON events(entity,attribute,recorded_us,commit_version);
CREATE INDEX IF NOT EXISTS ix_events_corrections ON events(corrects,recorded_us,commit_version);
CREATE TRIGGER IF NOT EXISTS immutable_events_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT,'events_are_immutable'); END;
CREATE TRIGGER IF NOT EXISTS immutable_events_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT,'events_are_immutable'); END;
"""


def utc_microseconds(value: str | datetime) -> int:
    delta = timestamp(value) - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def _event(row: sqlite3.Row) -> Event:
    return Event(**{column: row[column] for column in _COLUMNS.split(",")})


@dataclass(frozen=True)
class AppendResult:
    version: int
    events_inserted: int
    correction_target_lookups: int
    transaction_seconds: float


class SQLiteTemporalStore:
    """One connection per instance/thread; independent instances share WAL state.

    Use application-owned local paths. SQL parameters bind all event/query data.
    Database-file authorization and source authenticity are outside this layer.
    """

    def __init__(self, path: str | Path, *, busy_timeout_seconds: float = 5.0):
        if sqlite3.sqlite_version_info < (3, 35, 0):
            raise TemporalError("sqlite_version", "Persistent queries require SQLite 3.35 or newer")
        if type(busy_timeout_seconds) not in (float, int) or not 0 < busy_timeout_seconds <= 30:
            raise TemporalError("invalid_budget", "Busy timeout must be in (0,30] seconds")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=busy_timeout_seconds, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self._closed = False
        try:
            self.connection.execute("PRAGMA foreign_keys=ON")
            journal = self.connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if journal.lower() != "wal":
                raise TemporalError("wal_unavailable", "Persistent store requires SQLite WAL mode")
            self.connection.execute("PRAGMA synchronous=FULL")
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise TemporalError("schema_version", "Unsupported database schema version")
            self.connection.executescript(_SCHEMA)
            self.connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        except BaseException:
            self.connection.close()
            self._closed = True
            raise

    def _check_open(self):
        if self._closed:
            raise TemporalError("store_closed", "Store connection is closed")

    def close(self):
        if not self._closed:
            self.connection.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def version(self) -> int:
        self._check_open()
        return self.connection.execute("SELECT version FROM metadata WHERE singleton=1").fetchone()[0]

    def append(self, events: Iterable[Event], *, expected_version: int) -> AppendResult:
        self._check_open()
        if type(expected_version) is not int or expected_version < 0:
            raise TemporalError("invalid_version", "Expected version must be a nonnegative integer")
        batch = []
        for event in events:
            if len(batch) == MAX_BATCH:
                raise TemporalError("batch_limit", f"A transaction accepts at most {MAX_BATCH} events")
            if not isinstance(event, Event):
                raise TemporalError("invalid_event", "Append requires validated Event instances")
            batch.append(event)
        if not batch or len({event.id for event in batch}) != len(batch):
            raise TemporalError("invalid_batch", "Append requires nonempty unique event IDs")
        if self.connection.in_transaction:
            raise TemporalError("transaction_active", "Cannot append through a pinned read snapshot")
        started = time.perf_counter()
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            metadata = self.connection.execute("SELECT version,event_count FROM metadata WHERE singleton=1").fetchone()
            if metadata["version"] != expected_version:
                raise TemporalError(
                    "stale_version", f"Expected version {expected_version}; actual {metadata['version']}"
                )
            if metadata["event_count"] + len(batch) > MAX_STORED_EVENTS:
                raise TemporalError("event_limit", "Persistent event limit exceeded")
            new_version = expected_version + 1
            incoming = {event.id: event for event in batch}
            target_cache = {}
            lookups = 0
            for event in batch:
                if event.corrects:
                    target = incoming.get(event.corrects)
                    if target is None:
                        if event.corrects not in target_cache:
                            row = self.connection.execute(
                                f"SELECT {_COLUMNS} FROM events WHERE id=?", (event.corrects,)
                            ).fetchone()
                            target_cache[event.corrects] = _event(row) if row is not None else None
                            lookups += 1
                        target = target_cache[event.corrects]
                    if target is None:
                        raise TemporalError("invalid_correction", "Correction target does not exist")
                    if (target.entity, target.attribute, utc_microseconds(target.effective_at)) != (
                        event.entity,
                        event.attribute,
                        utc_microseconds(event.effective_at),
                    ) or utc_microseconds(event.recorded_at) <= utc_microseconds(target.recorded_at):
                        raise TemporalError(
                            "invalid_correction", "Correction must amend the same boundary and be recorded later"
                        )
            # Recording time is a topological order because each correction must
            # strictly follow its target; foreign keys also validate the insert.
            for event in sorted(batch, key=lambda item: (utc_microseconds(item.recorded_at), item.id)):
                self.connection.execute(
                    "INSERT INTO events(id,entity,attribute,value,effective_at,recorded_at,effective_us,recorded_us,text,source_uri,corrects,commit_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        event.id,
                        event.entity,
                        event.attribute,
                        event.value,
                        event.effective_at,
                        event.recorded_at,
                        utc_microseconds(event.effective_at),
                        utc_microseconds(event.recorded_at),
                        event.text,
                        event.source_uri,
                        event.corrects,
                        new_version,
                    ),
                )
            changed = self.connection.execute(
                "UPDATE metadata SET version=?,event_count=event_count+? WHERE singleton=1 AND version=?",
                (new_version, len(batch), expected_version),
            ).rowcount
            if changed != 1:
                raise TemporalError("stale_version", "Compare-and-swap update failed")
            self.connection.execute("COMMIT")
            return AppendResult(new_version, len(batch), lookups, time.perf_counter() - started)
        except BaseException as error:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            if isinstance(error, sqlite3.IntegrityError):
                raise TemporalError(
                    "append_integrity", "Append violates an ID, reference, or immutable-event constraint"
                ) from error
            if isinstance(error, sqlite3.OperationalError):
                raise TemporalError(
                    "store_busy", "SQLite could not acquire or complete the write transaction"
                ) from error
            raise

    @contextmanager
    def snapshot(self, *, version: int | None = None):
        self._check_open()
        if self.connection.in_transaction:
            raise TemporalError("transaction_active", "Nested read snapshots are not supported")
        self.connection.execute("BEGIN")
        snapshot = None
        try:
            latest = self.connection.execute("SELECT version FROM metadata WHERE singleton=1").fetchone()[0]
            selected = latest if version is None else version
            if type(selected) is not int or not 0 <= selected <= latest:
                raise TemporalError("invalid_version", "Snapshot version must be an existing commit version")
            snapshot = SQLiteSnapshot(self.connection, selected)
            yield snapshot
        finally:
            if snapshot is not None:
                snapshot.active = False
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")

    def retrieve(self, query: Query) -> list[Event]:
        with self.snapshot() as snapshot:
            return snapshot.retrieve(query)

    def query(self, query: Query, *, version: int | None = None) -> dict:
        with self.snapshot(version=version) as snapshot:
            return snapshot.answer(query)


class SQLiteSnapshot:
    def __init__(self, connection: sqlite3.Connection, version: int):
        self.connection, self.version, self.active = connection, version, True
        self.last_metrics: dict = {}

    def _check(self):
        if not self.active:
            raise TemporalError("snapshot_closed", "Snapshot cannot be used after its context exits")

    def _parameters(self, query: Query):
        return (
            query.entity,
            query.attribute,
            utc_microseconds(query.as_of),
            utc_microseconds(query.known_at),
            self.version,
            utc_microseconds(query.known_at),
            self.version,
        )

    def retrieve(self, query: Query, *, max_vm_steps: int = 2_000_000) -> list[Event]:
        self._check()
        if not isinstance(query, Query):
            raise TemporalError("invalid_query", "Expected a validated Query")
        if type(max_vm_steps) is not int or not 100 <= max_vm_steps <= 10_000_000:
            raise TemporalError("invalid_budget", "Query VM budget must be in [100,10000000]")
        ticks = 0

        def progress():
            nonlocal ticks
            ticks += 1
            return int(ticks * 100 >= max_vm_steps)

        self.connection.set_progress_handler(progress, 100)
        started = time.perf_counter()
        try:
            rows = self.connection.execute(_SELECT, self._parameters(query)).fetchall()
        except sqlite3.OperationalError as error:
            raise TemporalError("query_budget", "SQLite query was interrupted or exceeded its VM budget") from error
        finally:
            self.connection.set_progress_handler(None, 0)
        sql_seconds = time.perf_counter() - started
        evidence = [_event(row) for row in rows]
        self.last_metrics = {
            "query_seconds": time.perf_counter() - started,
            "sql_execute_fetch_seconds": sql_seconds,
            "sqlite_vm_steps_estimate": ticks * 100,
            "vm_measurement_granularity": 100,
            "rows_returned": len(rows),
            "snapshot_version": self.version,
        }
        return evidence

    def query_plan(self, query: Query) -> list[str]:
        self._check()
        return [row[3] for row in self.connection.execute("EXPLAIN QUERY PLAN " + _SELECT, self._parameters(query))]

    def correction_chain(self, event: Event, query: Query, *, max_depth: int = 1000) -> list[Event]:
        self._check()
        if type(max_depth) is not int or not 1 <= max_depth <= 10_000:
            raise TemporalError("invalid_budget", "Provenance depth must be in [1,10000]")
        validate_context(query, [event])
        stored = self.connection.execute(
            f"SELECT {_COLUMNS} FROM events WHERE id=? AND recorded_us<=? AND commit_version<=?",
            (event.id, utc_microseconds(query.known_at), self.version),
        ).fetchone()
        if stored is None or _event(stored) != event:
            raise TemporalError("invalid_context", "Provenance must start from an event in this snapshot")
        chain, current = [], event
        while current is not None:
            if len(chain) == max_depth:
                raise TemporalError("provenance_limit", "Correction provenance exceeds its depth budget")
            chain.append(current)
            if current.corrects is None:
                break
            row = self.connection.execute(
                f"SELECT {_COLUMNS} FROM events WHERE id=? AND recorded_us<=? AND commit_version<=?",
                (current.corrects, utc_microseconds(query.known_at), self.version),
            ).fetchone()
            if row is None:
                raise TemporalError("invalid_correction", "Visible correction provenance is incomplete")
            current = _event(row)
        return list(reversed(chain))

    def answer(self, query: Query) -> dict:
        evidence = self.retrieve(query)
        result = ReferenceAnswerer().answer(query, evidence)
        return {
            **result,
            "snapshot_version": self.version,
            "storage_metrics": dict(self.last_metrics),
            "correction_chains": {
                event.id: [item.to_dict() for item in self.correction_chain(event, query)] for event in evidence
            },
        }
