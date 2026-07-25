# Indexed ASOF benchmark

## Storage operations exercised

- `sqlite_store.py`: a real SQLite WAL event log, immutable-event triggers, explicit foreign keys, and indexed bitemporal SQL selection.
- Transactional batch appends with expected-version compare-and-swap, correction validation, atomic rollback, and exact commit revisions.
- Consistent multi-query read snapshots that remain pinned while another connection commits; explicit old revisions remain replayable.
- Existing reference answerer integration with selected source IDs, original-to-corrected provenance, strict temporal support, and snapshot identity.
- Explicit event/batch, provenance-depth, SQLite VM, and busy-wait budgets; errors fail visibly without truncation or reference substitution.
- `store-import`, `store-query`, and `store-benchmark` CLI paths plus a per-repository CI storage smoke run.

These are executable storage/query implementations. They do not require a hosted model, generated explanatory prose, or a running website.

## Executed workload

```bash
PYTHONPATH=src python -m temporal_rag store-benchmark --entities 128 --versions 64 --queries 128 --output reports/storage-benchmark.json
PYTHONPATH=src python -m unittest discover -s test -v
```

The seeded run contains **11,497 immutable events across 128 entities**, including delayed ingestion, future announcements, correction chains, and divergent corrections. The database is populated through 12 versioned transactions. All **128 indexed query results exactly match an independent Python scan oracle by source IDs**. [The report](../reports/storage-benchmark.json) retains dataset fingerprint, every query's cutoffs/results/work, real append/query timings, the actual query plan, SQLite/Python versions, and explicit limitations.

The observed plan searches a fact-key/known-time index and a covering correction index; materialized subsets still scan eligible events for the requested key. The measured logical database pages occupy 3,981,312 bytes in this run. VM estimates are sampled every 100 instructions. They are not disk reads or directly comparable to the scan oracle's exact Python-events-examined count.

Timings are local warm-cache observations, including the SQL progress observer. Total retrieval time includes Event reconstruction/validation; SQL execute/fetch time is recorded separately. The comparison is against a straightforward independent correctness scan, not an optimized temporal tree. Exact current timing numbers are in the machine-readable report; no general serving SLA or hardware-independent speedup is claimed.

## Behavioral and concurrency verification

The complete suite has **58 passing tests**. Thirteen new storage tests cover:

- All 19 independently labeled historical/reference cases through the persistent path.
- Four concurrent CAS writers with one successful commit and three stale-version failures.
- A WAL reader retaining its old answer while a second connection commits a correction.
- Old commit-version replay after newer writes.
- Out-of-order batch correction chains with complete provenance.
- Whole-batch rollback on an invalid correction and trigger rejection of direct update/delete attempts.
- Integer microsecond and timezone boundaries, including future announcements known before they take effect.
- Explicit VM interruption followed by a successful read through the same pinned snapshot.
- Forged/future provenance rejection, indexed query plans, and independent seeded-history agreement.

The existing reference tests, model-boundary tests, and prior review regressions remain active. CI additionally executes a small storage benchmark on Python 3.11 and 3.12. Ruff formatting/static checks and compilation are run locally before handoff.

## Limits and tradeoffs

This is a local embedded storage engine with one connection per thread/instance, not a distributed database. SQLite serializes writers; CAS adds application-level stale-write detection. Long-held read snapshots can retain WAL pages. The implementation has no retention/backup service, online schema migrations, authenticated source ingestion, tenant authorization, or external-source truth verification. Same-boundary corrections are explicit; arbitrary retroactive boundary shifts require a different event contract. The optional HF answer generator was not trained or evaluated for quality during this upgrade.
