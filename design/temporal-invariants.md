# Independent implementation review

## Reproduced defects and fixes

1. **Future evidence could produce a supported-looking historical answer.** Supplying a same-key event with a 2026 effective/recorded time to a 2025 query returned `answered`, its value, and an existing citation. The retriever filtered correctly, but neither answer boundary independently checked time cutoffs. `validate_context` now rejects future knowledge, not-yet-effective facts, duplicate IDs, wrong keys, and unvalidated records before reference answering or local-model prompting. The baseline evaluator records rejected contexts as failures instead of aborting or fabricating an answer.
2. **UTC conversion could leak an unstructured overflow exception.** A valid ISO timestamp at year 0001 with a positive offset converted outside Python's datetime range. UTC conversion now returns `TemporalError(code='invalid_timestamp')` for this boundary.
3. **Duplicate JSON fields could silently override imported metadata.** Event import now rejects duplicate object keys. The local-model output parser already rejected duplicates; both ingestion and generation now have explicit behavior.
4. **Answer validation assumed a dict and valid event instances.** Direct malformed boundary calls could raise incidental attribute errors. Structured answer/evidence types and duplicate evidence IDs are now checked explicitly.

## Added capabilities

- An explicit `query` command accepts imported entity/attribute keys and both bitemporal cutoffs; it is not limited to the fixed demo fact.
- `TemporalRetriever.explain` reports selection/exclusion reasons for visible candidates. Future-knowledge IDs are excluded, so auditing does not undo snapshot isolation.
- Evaluation reports context rejections separately from abstention, includes per-tag slice counts, and uses produced-answer denominators for citation validation.

The same-effective-time correction-chain semantics were independently inspected: strict recording-time increase prevents cycles; all visible ancestors are retired; divergent corrections remain a conflict rather than being silently resolved by recency. Those semantics were preserved.

## Executed verification

```bash
PYTHONPATH=src python -m unittest discover -s test -v
PYTHONPATH=src python -m temporal_rag evaluate --output walkthrough/evaluation.json
PYTHONPATH=src python -m temporal_rag query --entity atlas --attribute owner --as-of 2025-03-15T00:00:00Z --known-at 2025-04-15T00:00:00Z --explain
ruff check --select E9,F src tests
ruff format --check src tests
```

The reviewed suite has **45 passing tests**. Ten new behavioral tests cover temporal boundary bypasses, model-boundary context rejection, duplicate records, UTC overflow, audit leakage, correction explanations, duplicate JSON fields, arbitrary imported keys, and evaluation denominators. The 19-case synthetic reference evaluation still yields 19/19 correct timestamp-aware answers; six temporally invalid baseline contexts are now rejected explicitly. These are deterministic contract checks, not learned-model accuracy.

Three deliberately faulty variants were executed in isolated temporary source copies: removing the knowledge guard, removing the effective-time guard, and exposing unpublished audit candidates. The focused behavioral suite detected all three. [Spot-check evidence](mutation-spotcheck.json) records the mutations and failing tests; the original source was never mutated.

TestProof itself was unavailable: `npx --no-install testproof --format json --fail-under 100` reported a missing executable package. It was not downloaded. The three manual spot checks are **not a full mutation score or exhaustive proof**.

## Remaining limitations

Metadata and structured values are application-owned; no source authentication, natural-language fact extraction, database concurrency, tenant ACL, or general prompt-injection guarantee is implemented. Cutoff checks do not prove source truth or discover a newer event absent from supplied context. The optional local-HF generation workflow remains a separate resource-dependent experiment; reference fixture accuracy does not measure an LLM. No pretrained weights or paid requests were used in this review.
