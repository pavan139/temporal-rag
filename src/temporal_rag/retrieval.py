"""Bounded bitemporal fact retrieval and an explicitly deterministic answerer.

`effective_at` is valid time; `recorded_at` is when the system learned a fact.
Corrections replace facts at the same valid-time boundary, rather than changing
history for callers whose knowledge cutoff precedes publication.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import re
from typing import Iterable

MAX_EVENTS = 5000
MAX_QUERY_LENGTH = 500


class TemporalError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def timestamp(value: str | datetime) -> datetime:
    if not isinstance(value, (str, datetime)):
        raise TemporalError("invalid_timestamp", "Timestamp must be an aware ISO-8601 string or datetime")
    if isinstance(value, str) and len(value) > 64:
        raise TemporalError("invalid_timestamp", "Timestamp exceeds 64 characters")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    except ValueError as exc:
        raise TemporalError("invalid_timestamp", "Timestamp must be ISO-8601") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise TemporalError("invalid_timestamp", "Timestamp must include an explicit timezone")
    try:
        return result.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise TemporalError("invalid_timestamp", "Timestamp cannot be represented in UTC") from exc


def bounded_text(value: str, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise TemporalError("invalid_input", f"{field} must contain 1 to {limit} characters")
    return value.strip()


@dataclass(frozen=True)
class Event:
    id: str
    entity: str
    attribute: str
    value: str
    effective_at: str
    recorded_at: str
    text: str
    source_uri: str
    corrects: str | None = None

    def __post_init__(self) -> None:
        for field, limit in (
            ("id", 80),
            ("entity", 80),
            ("attribute", 64),
            ("value", 200),
            ("text", 2000),
            ("source_uri", 500),
        ):
            bounded_text(getattr(self, field), field, limit)
        timestamp(self.effective_at)
        timestamp(self.recorded_at)
        if self.corrects is not None:
            bounded_text(self.corrects, "corrects", 80)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Query:
    entity: str
    attribute: str
    as_of: str | datetime
    known_at: str | datetime
    text: str = ""

    def __post_init__(self) -> None:
        bounded_text(self.entity, "entity", 80)
        bounded_text(self.attribute, "attribute", 64)
        timestamp(self.as_of)
        timestamp(self.known_at)
        if not isinstance(self.text, str) or len(self.text) > MAX_QUERY_LENGTH:
            raise TemporalError("invalid_input", f"Query text is limited to {MAX_QUERY_LENGTH} characters")


class EventStore:
    def __init__(self, events: Iterable[Event]):
        self.events: list[Event] = []
        for event in events:
            if len(self.events) == MAX_EVENTS:
                raise TemporalError("event_limit", f"At most {MAX_EVENTS} events are supported")
            if not isinstance(event, Event):
                raise TemporalError("invalid_event", "Expected Event instances")
            self.events.append(event)
        self.by_id = {event.id: event for event in self.events}
        if len(self.by_id) != len(self.events):
            raise TemporalError("duplicate_id", "Event IDs must be unique")
        for event in self.events:
            if event.corrects:
                previous = self.by_id.get(event.corrects)
                if previous is None:
                    raise TemporalError("invalid_correction", "Correction target does not exist")
                same_fact = (previous.entity, previous.attribute, timestamp(previous.effective_at)) == (
                    event.entity,
                    event.attribute,
                    timestamp(event.effective_at),
                )
                later = timestamp(event.recorded_at) > timestamp(previous.recorded_at)
                if not same_fact or not later:
                    raise TemporalError(
                        "invalid_correction",
                        "Corrections must amend the same valid-time boundary and be recorded later",
                    )

    def visible(self, query: Query) -> list[Event]:
        return [
            event
            for event in self.events
            if event.entity == query.entity
            and event.attribute == query.attribute
            and timestamp(event.recorded_at) <= timestamp(query.known_at)
        ]


class TemporalRetriever:
    name = "timestamp_aware"

    def __init__(self, events: Iterable[Event]):
        self.store = EventStore(events)

    def retrieve(self, query: Query) -> list[Event]:
        evidence, _ = self._retrieve_with_audit(query)
        return evidence

    def _retrieve_with_audit(self, query: Query) -> tuple[list[Event], list[dict]]:
        visible = self.store.visible(query)
        eligible = [event for event in visible if timestamp(event.effective_at) <= timestamp(query.as_of)]
        superseded = {event.corrects for event in eligible if event.corrects}
        active = [event for event in eligible if event.id not in superseded]
        latest = max((timestamp(event.effective_at) for event in active), default=None)
        evidence = sorted(
            (event for event in active if timestamp(event.effective_at) == latest), key=lambda event: event.id
        )
        selected = {event.id for event in evidence}
        audit = []
        for event in sorted(visible, key=lambda item: (timestamp(item.effective_at), item.id)):
            if timestamp(event.effective_at) > timestamp(query.as_of):
                decision = "not_yet_effective"
            elif event.id in superseded:
                decision = "superseded_by_visible_correction"
            elif event.id not in selected:
                decision = "older_effective_boundary"
            else:
                decision = "selected"
            audit.append(
                {
                    "event_id": event.id,
                    "decision": decision,
                    "effective_at": timestamp(event.effective_at).isoformat(),
                    "recorded_at": timestamp(event.recorded_at).isoformat(),
                    "corrects": event.corrects,
                }
            )
        return evidence, audit

    def explain(self, query: Query) -> dict:
        """Explain visible selection decisions without exposing unpublished facts.

        Only events matching the requested key and known at the requested
        snapshot enter the audit. Future knowledge is not emitted, even as IDs.
        """
        evidence, audit = self._retrieve_with_audit(query)
        return {
            "selected_ids": [event.id for event in evidence],
            "visible_candidate_count": len(audit),
            "decisions": audit,
            "as_of": timestamp(query.as_of).isoformat(),
            "known_at": timestamp(query.known_at).isoformat(),
        }

    def query(
        self, entity: str, attribute: str, as_of: str | datetime | None = None, known_at: str | datetime | None = None
    ) -> dict:
        """Current queries use UTC now; explicitly set both cutoffs for replay."""
        now = datetime.now(timezone.utc)
        request = Query(
            entity, attribute, as_of if as_of is not None else now, known_at if known_at is not None else now
        )
        evidence = self.retrieve(request)
        return ReferenceAnswerer().answer(request, evidence)


class LexicalLatestBaseline:
    """Lexical overlap then recorded-time ordering; ignores valid-time semantics.

    Both engines respect the same entity/attribute and knowledge filters. The
    baseline intentionally lacks correction and effective-time reasoning.
    """

    name = "lexical_latest"

    def __init__(self, events: Iterable[Event]):
        self.store = EventStore(events)

    def retrieve(self, query: Query) -> list[Event]:
        terms = set(re.findall(r"\w+", f"{query.entity} {query.attribute} {query.text}".lower()))
        candidates = self.store.visible(query)

        def score(event: Event) -> tuple:
            overlap = len(terms & set(re.findall(r"\w+", event.text.lower())))
            return overlap, timestamp(event.recorded_at), event.id

        return sorted(candidates, key=score, reverse=True)[:1]


class ReferenceAnswerer:
    """Structured reference behavior, deliberately not presented as generation."""

    name = "deterministic_reference_v1"

    def answer(self, query: Query, evidence: list[Event]) -> dict:
        validate_context(query, evidence)
        values = {event.value for event in evidence}
        status = "answered" if len(values) == 1 else "abstained"
        reason = "supported" if len(values) == 1 else ("contradictory_context" if values else "missing_context")
        result = {
            "answerer": self.name,
            "status": status,
            "reason": reason,
            "value": next(iter(values)) if len(values) == 1 else None,
            "citations": [event.id for event in evidence],
            "as_of": timestamp(query.as_of).isoformat(),
            "known_at": timestamp(query.known_at).isoformat(),
            "provenance": [event.to_dict() for event in evidence],
        }
        validate_answer(result, evidence)
        return result


def validate_context(query: Query, evidence: list[Event]) -> None:
    """Enforce temporal support again at the answer boundary, not just retrieval."""
    if not isinstance(evidence, list) or len(evidence) > MAX_EVENTS:
        raise TemporalError("context_limit", "Evidence must be a bounded event list")
    seen = set()
    for event in evidence:
        if not isinstance(event, Event):
            raise TemporalError("invalid_context", "Evidence must contain validated Event records")
        if event.id in seen:
            raise TemporalError("invalid_context", "Duplicate evidence IDs are forbidden")
        seen.add(event.id)
        if event.entity != query.entity or event.attribute != query.attribute:
            raise TemporalError("invalid_context", "Evidence does not match the requested fact")
        if timestamp(event.recorded_at) > timestamp(query.known_at):
            raise TemporalError("future_knowledge", "Evidence was not recorded at the requested knowledge snapshot")
        if timestamp(event.effective_at) > timestamp(query.as_of):
            raise TemporalError("future_effective_context", "Evidence is not yet effective at the requested valid time")


def validate_answer(result: dict, evidence: list[Event]) -> None:
    """Check citation membership and structured value support, not prose meaning."""
    if not isinstance(result, dict):
        raise TemporalError("invalid_answer", "Answer must be a structured object")
    if (
        not isinstance(evidence, list)
        or len(evidence) > MAX_EVENTS
        or any(not isinstance(event, Event) for event in evidence)
    ):
        raise TemporalError("invalid_context", "Evidence must be a bounded list of validated events")
    by_id = {event.id: event for event in evidence}
    if len(by_id) != len(evidence):
        raise TemporalError("invalid_context", "Evidence IDs must be unique")
    citations = result.get("citations")
    if not isinstance(citations, list) or any(not isinstance(cid, str) or cid not in by_id for cid in citations):
        raise TemporalError("invalid_citation", "Every citation must identify retrieved evidence")
    if len(citations) != len(set(citations)):
        raise TemporalError("invalid_citation", "Duplicate citations are not permitted")
    status, value = result.get("status"), result.get("value")
    if status == "answered":
        if (
            not citations
            or len({event.value for event in evidence}) != 1
            or any(by_id[cid].value != value for cid in citations)
        ):
            raise TemporalError("unsupported_answer", "Answer value must be supported by every cited fact")
    elif status == "abstained":
        if value is not None:
            raise TemporalError("unsupported_answer", "Abstained answers must not contain a value")
    else:
        raise TemporalError("invalid_answer", "Unknown answer status")
