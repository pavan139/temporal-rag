# Sharing an event log and its answers

A source release should carry the structured fixture, its source-set labels and the cutoffs used for each answer. `walkthrough/events.json` is synthetic. An imported event log may contain sensitive source text in provenance, so inspect it before making it public.

For a portable source checkout:

```bash
python -m pip install -e . -r requirements-dev.txt
python tools/check_retrieval.py --require-lint --wheel
python -m temporal_rag evaluate --output artifacts/fixture-evaluation.json
```

Generated SQLite files are run artifacts. Keep them outside the Git history unless you intentionally release a separate, inspected dataset. The event log plus commit/version rules explains the experiment more clearly than an unexplained database file.

This directory can be connected to an empty remote after you have made and inspected your own commit:

```bash
git remote add origin https://github.com/YOUR_USERNAME/temporal-rag.git
git push -u origin main
```

An extracted archive may first need `git init -b main`; an existing repository does not. Preserve the record that the optional local-model path is unrun unless you replace it with an actual checkpoint evaluation. The fixture retrieval result and the indexed storage comparison answer different questions and should remain separate in release notes.
