"""Planning-only handler for ``calendar.move_after_audit``."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from typing import Any
from zoneinfo import ZoneInfo

from ..models import (
    AgentResult,
    AgentTask,
    AuditReference,
    CalendarEventPage,
    CalendarEventRef,
    EventMove,
    ResultStatus,
)
from ..tools import CalendarEventListError, CalendarTool


GOAL = "calendar.move_after_audit"
_EVENT_LIST_LIMIT = 1000
_MAX_EVENT_LIST_PAGES = 100


def _event_ref(event: dict[str, object]) -> CalendarEventRef:
    return CalendarEventRef(
        id=str(event.get("id", "")),
        title=str(event.get("title", "")),
        start_at=str(event.get("start_at", "")),
        end_at=str(event.get("end_at", "")),
        calendar_id=event.get("calendar_id") if isinstance(event.get("calendar_id"), str) else None,
        status=event.get("status") if isinstance(event.get("status"), str) else None,
        timezone=event.get("timezone") if isinstance(event.get("timezone"), str) else None,
    )


def _event_id(event: dict[str, Any]) -> str | None:
    value = event.get("id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _is_audit_candidate(event: dict[str, Any]) -> bool:
    text = " ".join(str(event.get(key) or "") for key in ("title", "description"))
    return "audit" in text.lower()


def _parse_event_interval(event: dict[str, Any]) -> tuple[datetime, datetime] | None:
    start_value = event.get("start_at")
    end_value = event.get("end_at")
    if not isinstance(start_value, str) or not isinstance(end_value, str):
        return None
    try:
        start = datetime.fromisoformat(start_value.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        return None
    return start, end


def _parse_event_times(event: dict[str, Any]) -> tuple[datetime, datetime] | None:
    interval = _parse_event_interval(event)
    if interval is None:
        return None
    event_timezone = _event_timezone(event)
    if event_timezone is None:
        return None
    return tuple(value.astimezone(timezone.utc) for value in interval)


def _event_timezone(event: dict[str, Any]) -> ZoneInfo | None:
    timezone_name = event.get("timezone")
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        return None
    try:
        return ZoneInfo(timezone_name.strip())
    except (KeyError, OSError, ValueError):
        return None


def _parse_event_start(event: dict[str, Any]) -> datetime | None:
    value = event.get("start_at")
    if not isinstance(value, str):
        return None
    try:
        start = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return start if start.tzinfo is not None else None


def _intervals_overlap(
    first_start: datetime,
    first_end: datetime,
    second_start: datetime,
    second_end: datetime,
) -> bool:
    return first_start < second_end and first_end > second_start


def _clarification(
    task: AgentTask,
    audit_reference: AuditReference,
    summary: str,
    *ambiguities: str,
) -> AgentResult:
    return AgentResult(
        goal=task.goal,
        status=ResultStatus.NEEDS_CLARIFICATION,
        summary=summary,
        ambiguities=list(ambiguities),
        audit_reference=audit_reference,
    )


def _list_complete_events(
    tools: CalendarTool,
    *,
    calendar_id: str,
    search: str | None = None,
) -> list[dict[str, Any]]:
    page_reader = getattr(tools, "list_events_page", None)
    if not callable(page_reader):
        raise CalendarEventListError(
            "The calendar adapter does not expose pagination metadata."
        )

    records: list[dict[str, Any]] = []
    expected_total: int | None = None
    seen_page_signatures: set[str] = set()
    seen_event_ids: set[str] = set()
    offset = 0

    for _ in range(_MAX_EVENT_LIST_PAGES):
        arguments: dict[str, Any] = {
            "calendar_id": calendar_id,
            "limit": _EVENT_LIST_LIMIT,
            "offset": offset,
        }
        if search is not None:
            arguments["search"] = search

        page = page_reader(**arguments)
        if not isinstance(page, CalendarEventPage):
            raise CalendarEventListError(
                "The calendar adapter returned no typed event-page metadata."
            )
        try:
            page.validate(
                requested_limit=_EVENT_LIST_LIMIT,
                requested_offset=offset,
            )
        except (TypeError, ValueError) as exc:
            raise CalendarEventListError(
                f"CalendarEvent page metadata is invalid: {exc}"
            ) from exc

        if expected_total is None:
            expected_total = page.total
        elif page.total != expected_total:
            raise CalendarEventListError(
                "CalendarEvent total changed during pagination."
            )
        if page.total >= _EVENT_LIST_LIMIT:
            raise CalendarEventListError(
                f"CalendarEvent listing reached the {_EVENT_LIST_LIMIT}-record safety limit."
            )

        page_event_ids: set[str] = set()
        for event in page.records:
            if event.get("calendar_id") != calendar_id:
                raise CalendarEventListError(
                    "CalendarEvent page returned a record outside the requested calendar."
                )
            event_id = _event_id(event)
            if (
                event_id is not None
                and event_id in seen_event_ids
                and event_id not in page_event_ids
            ):
                raise CalendarEventListError(
                    "CalendarEvent pagination returned a repeated event ID."
                )
            if event_id is not None:
                page_event_ids.add(event_id)
        seen_event_ids.update(page_event_ids)

        try:
            page_signature = json.dumps(
                page.records,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        except (TypeError, ValueError) as exc:
            raise CalendarEventListError(
                "CalendarEvent page records are not valid JSON objects."
            ) from exc
        if page_signature in seen_page_signatures:
            raise CalendarEventListError(
                "CalendarEvent pagination returned a repeated page."
            )
        seen_page_signatures.add(page_signature)

        records.extend(page.records)
        if not page.has_more:
            if len(records) != page.total:
                raise CalendarEventListError(
                    "CalendarEvent pagination ended before the reported total."
                )
            return records

        next_offset = page.next_offset
        if next_offset <= offset:
            raise CalendarEventListError(
                "CalendarEvent pagination did not advance."
            )
        offset = next_offset

    raise CalendarEventListError(
        f"CalendarEvent pagination reached the {_MAX_EVENT_LIST_PAGES}-page safety limit."
    )


def plan(task: AgentTask, tools: CalendarTool) -> AgentResult:
    """Build a deterministic, read-only plan for events after one audit."""

    reference = AuditReference(candidates=(), search_terms=("audit",))
    raw_calendar_id = task.context.get("calendar_id")
    if not isinstance(raw_calendar_id, str) or not raw_calendar_id.strip():
        return _clarification(
            task,
            reference,
            "A calendar scope is required.",
            "Provide the calendar_id whose events should be searched and planned.",
        )
    calendar_id = raw_calendar_id.strip()

    try:
        audit_records = _list_complete_events(
            tools,
            calendar_id=calendar_id,
            search="audit",
        )
    except CalendarEventListError as exc:
        return _clarification(
            task,
            reference,
            "Audit resolution is blocked because the search may be incomplete.",
            str(exc),
        )

    audit_like = [event for event in audit_records if _is_audit_candidate(event)]
    if any(event.get("calendar_id") != calendar_id for event in audit_like):
        return _clarification(
            task,
            reference,
            "Audit search returned an event outside the requested calendar.",
            "The audit calendar scope could not be verified.",
        )
    candidates = tuple(_event_ref(event) for event in audit_like)
    reference = AuditReference(candidates=candidates, search_terms=("audit",))
    if not audit_like:
        return _clarification(
            task,
            reference,
            "No audit candidate was found.",
            "An event whose title or description contains 'audit' is required.",
        )

    parsed_audits = [(event, _parse_event_times(event)) for event in audit_like]
    malformed_audits = [
        event
        for event, parsed in parsed_audits
        if parsed is None or _event_id(event) is None
    ]
    valid_audits = [
        (event, parsed)
        for event, parsed in parsed_audits
        if parsed is not None and _event_id(event) is not None
    ]
    if malformed_audits:
        return _clarification(
            task,
            reference,
            "Audit resolution is ambiguous because an audit candidate has invalid or timezone-ambiguous datetimes.",
            f"{len(malformed_audits)} audit candidate(s) could not be validated.",
        )
    if len(valid_audits) != 1:
        return _clarification(
            task,
            reference,
            "The audit cannot be resolved deterministically.",
            f"Found {len(valid_audits)} valid audit candidates; exactly one is required.",
        )

    audit_event, audit_times = valid_audits[0]
    assert audit_times is not None
    _, audit_end = audit_times
    shift = task.context.get("shift")
    if not isinstance(shift, timedelta) or shift <= timedelta(0):
        return _clarification(
            task,
            reference,
            "The audit is resolved, but the amount to move events is not specified.",
            "Provide a positive timedelta shift; the planner will not invent one.",
        )

    try:
        events = _list_complete_events(tools, calendar_id=calendar_id)
    except CalendarEventListError as exc:
        return _clarification(
            task,
            reference,
            "The affected-event set may be incomplete.",
            str(exc),
        )
    audit_id = _event_id(audit_event)
    assert audit_id is not None
    matching_audits = [event for event in events if _event_id(event) == audit_id]
    if len(matching_audits) != 1:
        occurrence = "missing from" if not matching_audits else "duplicated in"
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because the resolved audit is not present exactly once.",
            f"The selected audit ID {audit_id[:100]!r} was {occurrence} the full event listing "
            f"({len(matching_audits)} matches; expected exactly one).",
        )
    full_listing_audit_times = _parse_event_times(matching_audits[0])
    if full_listing_audit_times is None or full_listing_audit_times != audit_times:
        return _clarification(
            task,
            reference,
            "Audit resolution is ambiguous because its interval changed between reads.",
            "The selected audit interval must match in the audit search and full event listing.",
        )
    event_id_counts: dict[str, int] = {}
    for event in events:
        event_id = _event_id(event)
        if event_id is not None:
            event_id_counts[event_id] = event_id_counts.get(event_id, 0) + 1
    duplicate_event_ids = sorted(
        event_id for event_id, count in event_id_counts.items() if count > 1
    )
    if duplicate_event_ids:
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because event IDs are duplicated.",
            f"Duplicate event IDs: {', '.join(duplicate_event_ids)}.",
        )

    moves: list[EventMove] = []
    move_ids: set[str] = set()
    timezone_ambiguous: list[dict[str, Any]] = []
    malformed_events: list[dict[str, Any]] = []
    malformed_after_audit: list[dict[str, Any]] = []
    identity_ambiguous: list[dict[str, Any]] = []
    for event in events:
        event_id = _event_id(event)
        if event_id == audit_id:
            continue
        parsed = _parse_event_times(event)
        if parsed is None:
            interval = _parse_event_interval(event)
            if interval is not None:
                if interval[0] > audit_end:
                    timezone_ambiguous.append(event)
            else:
                start = _parse_event_start(event)
                if start is not None and start > audit_end:
                    malformed_after_audit.append(event)
                else:
                    malformed_events.append(event)
            continue

        event_start, event_end = parsed
        if event_start <= audit_end:
            continue
        event_timezone = _event_timezone(event)
        if event_timezone is None:
            timezone_ambiguous.append(event)
            continue
        if event_id is None:
            identity_ambiguous.append(event)
            continue
        proposed_start = (event_start + shift).astimezone(event_timezone)
        proposed_end = (event_end + shift).astimezone(event_timezone)
        move_ids.add(event_id)
        moves.append(
            EventMove(
                event_id=event_id,
                title=str(event.get("title", "")),
                original_start_at=str(event["start_at"]),
                original_end_at=str(event["end_at"]),
                proposed_start_at=proposed_start.isoformat(),
                proposed_end_at=proposed_end.isoformat(),
                duration_seconds=(event_end - event_start).total_seconds(),
                calendar_id=event.get("calendar_id") if isinstance(event.get("calendar_id"), str) else None,
                timezone=event.get("timezone") if isinstance(event.get("timezone"), str) else None,
            )
        )

    moves.sort(key=lambda move: datetime.fromisoformat(move.original_start_at.replace("Z", "+00:00")))

    if timezone_ambiguous:
        return _clarification(
            task,
            reference,
            "The affected-event boundary is ambiguous because event timezone data is missing.",
            f"{len(timezone_ambiguous)} event(s) have datetime values without an explicit timezone.",
        )
    if malformed_after_audit:
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because an event after the audit is malformed.",
            f"{len(malformed_after_audit)} event(s) that would be moved could not be validated.",
        )
    if malformed_events:
        sample_ids = [_event_id(event) or "<missing id>" for event in malformed_events[:5]]
        remaining = len(malformed_events) - len(sample_ids)
        sample = ", ".join(sample_ids)
        if remaining:
            sample += f", and {remaining} more"
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because event timestamps are unusable.",
            f"{len(malformed_events)} event(s) could not be classified or checked for destination conflicts; sample IDs: {sample}.",
        )
    if identity_ambiguous:
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because a movable event has no valid identity.",
            f"{len(identity_ambiguous)} event(s) that would be moved have a missing or blank ID.",
        )

    conflicts: set[str] = set()
    for index, move in enumerate(moves):
        proposed_start = datetime.fromisoformat(move.proposed_start_at.replace("Z", "+00:00"))
        proposed_end = datetime.fromisoformat(move.proposed_end_at.replace("Z", "+00:00"))
        for other_move in moves[index + 1:]:
            other_start = datetime.fromisoformat(other_move.proposed_start_at.replace("Z", "+00:00"))
            other_end = datetime.fromisoformat(other_move.proposed_end_at.replace("Z", "+00:00"))
            if _intervals_overlap(proposed_start, proposed_end, other_start, other_end):
                conflicts.add(f"{move.event_id} with {other_move.event_id}")
        for event in events:
            event_id = _event_id(event)
            if event_id in move_ids or event_id == audit_id:
                continue
            parsed = _parse_event_interval(event)
            if parsed is not None and _intervals_overlap(proposed_start, proposed_end, *parsed):
                conflicts.add(f"{move.event_id} with {event_id or 'an unidentified event'}")
    if conflicts:
        return _clarification(
            task,
            reference,
            "The rescheduling proposal conflicts with an existing event.",
            "Conflicting proposed intervals: " + ", ".join(sorted(conflicts)) + ".",
        )

    if not moves:
        return _clarification(
            task,
            reference,
            "No valid events start strictly after the audit end.",
            "Events overlapping the audit or ending at its boundary are not affected.",
        )

    return AgentResult(
        goal=task.goal,
        status=ResultStatus.PLANNED,
        summary="A read-only rescheduling plan was prepared; no events were moved.",
        audit_reference=reference,
        rescheduling_plan=moves,
        claimed_outcome={"moved": False, "planned_event_count": len(moves), "shift": shift},
    )