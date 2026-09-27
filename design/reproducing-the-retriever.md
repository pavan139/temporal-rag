# Reproducing source selection and storage checks

The recorded run executed 58 tests on Python 3.12.14. It covers the 19-case reference fixture, correction/clock boundaries, answer validation and persistent-store behavior. Thirteen storage tests include concurrent compare-and-swap writers, pinned WAL readers, rollback and VM interruption. The machine record is [reports/verification.json](../reports/verification.json).

```bash
RUN_ML_SMOKE=1 PYTHONPATH=src python -m unittest discover -s test -v
python -m temporal_rag store-benchmark --entities 16 --versions 16 --queries 24 --output artifacts/storage-check.json
python tools/check_retrieval.py --require-lint --wheel
```

The benchmark's oracle is an independent history scan. Compare IDs, not merely equal values, and retain the query plans and work units. [The 11,497-event run](asof-index-results.md) explains the existing measured workload and local timing limits.

No local checkpoint has been evaluated for generated-answer quality. The optional adapter's unit coverage must not be counted as real inference. Software packaging checks build and import a wheel outside this tree; the report's source fingerprint determines which code was checked.
