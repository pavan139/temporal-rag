# Preserve the two-cutoff query

A temporal regression should specify the fact key, `as_of`, `known_at`, commit revision and expected source IDs. Add an event chain that makes the wrong selection observable. Keep expected labels outside the retriever.

For SQL changes, compare source sets with the independent scan, not only the returned value. Different source sets can happen to carry the same value. Include a pinned reader while a second connection appends, and retain the failed-write rollback cases.

```bash
python -m pip install -e . -r requirements-dev.txt
PYTHONPATH=src python -m unittest discover -s test -v
python -m temporal_rag store-benchmark --entities 16 --versions 16 --queries 24 --output artifacts/storage-check.json
```

The full package check is `python tools/check_retrieval.py --require-lint --wheel`. Do not silently truncate context or replace a failed model call with a reference answer. Local generation tests establish parser/loader behavior with doubles; a claim about generated answers needs a separately executed checkpoint run.

When adding fixtures, use synthetic or authorized non-sensitive text. Exported provenance retains source content. Document changes to correction semantics, because they can change historical answers even when the latest answer is unchanged.
