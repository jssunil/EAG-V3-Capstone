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
_DEFAULT_PROPOSED_SHIFT = timedelta(minutes=30)


def _event_ref(
    event: dict[str, object],
    fallback_timezone: ZoneInfo | None = None,
) -> CalendarEventRef:
    event_timezone = _event_timezone(event, fallback_timezone)
    start_value = event.get("start_at")
    end_value = event.get("end_at")
    start_at = str(start_value or "")
    end_at = str(end_value or "")
    if event_timezone is not None:
        if isinstance(start_value, str):
            start_at = _normalized_timestamp(start_value, event_timezone)
        if isinstance(end_value, str):
            end_at = _normalized_timestamp(end_value, event_timezone)
    return CalendarEventRef(
        id=str(event.get("number") or event.get("id", "")),
        title=str(event.get("title", "")),
        start_at=start_at,
        end_at=end_at,
        calendar_id=event.get("calendar_id") if isinstance(event.get("calendar_id"), str) else None,
        status=event.get("status") if isinstance(event.get("status"), str) else None,
        timezone=event_timezone.key if event_timezone is not None else None,
    )


def _event_id(event: dict[str, Any]) -> str | None:
    value = event.get("id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _is_audit_candidate(event: dict[str, Any]) -> bool:
    text = " ".join(str(event.get(key) or "") for key in ("title", "description"))
    return "audit" in text.lower()


def _parse_timezone(timezone_name: object) -> ZoneInfo | None:
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        return None
    try:
        return ZoneInfo(timezone_name.strip())
    except (KeyError, OSError, ValueError):
        return None


def _event_timezone(
    event: dict[str, Any],
    fallback_timezone: ZoneInfo | None = None,
) -> ZoneInfo | None:
    timezone_name = event.get("timezone")
    if timezone_name is None:
        return fallback_timezone
    return _parse_timezone(timezone_name)


def _localize_wall_time(value: datetime, event_timezone: ZoneInfo) -> datetime | None:
    instants: dict[datetime, datetime] = {}
    for fold in (0, 1):
        candidate = value.replace(tzinfo=event_timezone, fold=fold)
        instant = candidate.astimezone(timezone.utc)
        round_trip = instant.astimezone(event_timezone)
        if round_trip.replace(tzinfo=None) == value:
            instants[instant] = candidate
    if len(instants) != 1:
        return None
    return next(iter(instants.values()))


def _parse_event_datetime(value: object, event_timezone: ZoneInfo | None) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed if parsed.utcoffset() is not None else None
    if event_timezone is None:
        return None
    return _localize_wall_time(parsed, event_timezone)


def _normalized_timestamp(value: str, event_timezone: ZoneInfo) -> str:
    parsed = _parse_event_datetime(value, event_timezone)
    if parsed is None:
        return value
    try:
        original = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if original.tzinfo is not None and original.utcoffset() is not None:
        return value
    return parsed.isoformat()


def _parse_event_interval(
    event: dict[str, Any],
    fallback_timezone: ZoneInfo | None = None,
) -> tuple[datetime, datetime] | None:
    start_value = event.get("start_at")
    end_value = event.get("end_at")
    event_timezone = _event_timezone(event, fallback_timezone)
    if event_timezone is None:
        return None
    start = _parse_event_datetime(start_value, event_timezone)
    end = _parse_event_datetime(end_value, event_timezone)
    if start is None or end is None:
        return None
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    if end_utc <= start_utc:
        return None
    return start_utc, end_utc


def _parse_event_times(
    event: dict[str, Any],
    fallback_timezone: ZoneInfo | None = None,
) -> tuple[datetime, datetime] | None:
    return _parse_event_interval(event, fallback_timezone)


def _resolve_calendar_timezone(tools: CalendarTool, calendar_id: str) -> ZoneInfo:
    calendar = tools.get_calendar(calendar_id)
    if not isinstance(calendar, dict) or calendar.get("id") != calendar_id:
        raise CalendarEventListError(
            "The owning calendar could not be resolved uniquely for timezone fallback."
        )
    event_timezone = _parse_timezone(calendar.get("timezone"))
    if event_timezone is None:
        raise CalendarEventListError(
            "The owning calendar has no valid IANA timezone for timestamp fallback."
        )
    return event_timezone


def _explicit_timezone_fallback(task: AgentTask) -> ZoneInfo | None:
    timezone_name = task.context.get("naive_timestamp_timezone")
    use_calendar_timezone = task.context.get("use_calendar_timezone_fallback", False)
    if type(use_calendar_timezone) is not bool:
        raise ValueError("use_calendar_timezone_fallback must be a boolean.")
    if timezone_name is not None and use_calendar_timezone:
        raise ValueError(
            "Configure either a fixed timestamp fallback or a calendar timezone fallback, not both."
        )
    if timezone_name is None:
        return None
    event_timezone = _parse_timezone(timezone_name)
    if event_timezone is None:
        raise ValueError("naive_timestamp_timezone must be a valid IANA timezone.")
    return event_timezone


def _parse_event_start(
    event: dict[str, Any],
    fallback_timezone: ZoneInfo | None = None,
) -> datetime | None:
    event_timezone = _event_timezone(event, fallback_timezone)
    start = _parse_event_datetime(event.get("start_at"), event_timezone)
    return start.astimezone(timezone.utc) if start is not None else None


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
    calendar_id: str | None = None,
    search: str | None = None,
    event_number: str | None = None,
) -> list[dict[str, Any]]:
    if calendar_id is None and search is None and event_number is None:
        raise CalendarEventListError("CalendarEvent listing requires an explicit filter.")
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
            "limit": _EVENT_LIST_LIMIT,
            "offset": offset,
        }
        if calendar_id is not None:
            arguments["calendar_id"] = calendar_id
        if search is not None:
            arguments["search"] = search
        if event_number is not None:
            arguments["number"] = event_number

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
            if calendar_id is not None and event.get("calendar_id") != calendar_id:
                raise CalendarEventListError(
                    "CalendarEvent page returned a record outside the requested calendar."
                )
            if event_number is not None and event.get("number") != event_number:
                raise CalendarEventListError(
                    "CalendarEvent number filter returned a different event number."
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


def _resolve_exact_audit(
    task: AgentTask,
    tools: CalendarTool,
    *,
    audit_number: str,
    audit_title: str,
) -> tuple[dict[str, Any], str, AuditReference] | AgentResult:
    reference = AuditReference(candidates=(), search_terms=(audit_number,))
    try:
        records = _list_complete_events(tools, event_number=audit_number)
    except CalendarEventListError as exc:
        return _clarification(
            task,
            reference,
            "The specified audit could not be read completely.",
            str(exc),
        )

    candidates = [event for event in records if event.get("number") == audit_number]
    reference = AuditReference(
        candidates=tuple(_event_ref(event) for event in candidates),
        search_terms=(audit_number,),
    )
    if len(candidates) != 1:
        return _clarification(
            task,
            reference,
            "The specified audit could not be resolved uniquely.",
            f"Expected one event numbered {audit_number!r}; found {len(candidates)}.",
        )

    audit_event = candidates[0]
    if audit_event.get("title") != audit_title:
        return _clarification(
            task,
            reference,
            "The event number resolved to a different audit title.",
            "The audit title must match the explicit task reference exactly.",
        )
    status = audit_event.get("status")
    if not isinstance(status, str) or status.strip().casefold() != "confirmed":
        return _clarification(
            task,
            reference,
            "The specified audit is not verified as confirmed.",
            "A confirmed audit status is required before preparing a move plan.",
        )
    calendar_id = audit_event.get("calendar_id")
    if not isinstance(calendar_id, str) or not calendar_id.strip():
        return _clarification(
            task,
            reference,
            "The specified audit has no verified calendar association.",
            "The event's calendar_id is required; no calendar will be guessed.",
        )

    requested_calendar_id = task.context.get("calendar_id")
    if (
        requested_calendar_id is not None
        and requested_calendar_id != calendar_id
    ):
        return _clarification(
            task,
            reference,
            "The supplied calendar does not match the audit's associated calendar.",
            "The calendar must be taken from the exact audit record.",
        )

    return audit_event, calendar_id.strip(), reference


def plan(task: AgentTask, tools: CalendarTool) -> AgentResult:
    """Build a deterministic, read-only plan for events after one audit."""

    fallback_reference = AuditReference(candidates=(), search_terms=("timezone",))
    try:
        fallback_timezone = _explicit_timezone_fallback(task)
    except ValueError as exc:
        return _clarification(
            task,
            fallback_reference,
            "The configured timestamp timezone fallback is invalid.",
            str(exc),
        )
    use_calendar_timezone = task.context.get("use_calendar_timezone_fallback", False)

    audit_number = task.context.get("audit_number")
    if audit_number is not None:
        raw_title = task.context.get("audit_title")
        reference = AuditReference(
            candidates=(),
            search_terms=(audit_number,) if isinstance(audit_number, str) else (),
        )
        if (
            not isinstance(audit_number, str)
            or not audit_number.strip()
            or not isinstance(raw_title, str)
            or not raw_title.strip()
        ):
            return _clarification(
                task,
                reference,
                "The exact audit reference is incomplete.",
                "A nonblank audit_number and its expected audit_title are required.",
            )
        exact_resolution = _resolve_exact_audit(
            task,
            tools,
            audit_number=audit_number.strip(),
            audit_title=raw_title,
        )
        if isinstance(exact_resolution, AgentResult):
            return exact_resolution
        audit_event, calendar_id, reference = exact_resolution
        if (
            fallback_timezone is None
            and use_calendar_timezone
            and audit_event.get("timezone") is None
        ):
            try:
                fallback_timezone = _resolve_calendar_timezone(tools, calendar_id)
            except CalendarEventListError as exc:
                return _clarification(
                    task,
                    reference,
                    "The owning calendar timezone could not be verified for timestamp fallback.",
                    str(exc),
                )
        audit_times = _parse_event_times(audit_event, fallback_timezone)
        if audit_times is None:
            reference = AuditReference(
                candidates=(_event_ref(audit_event, fallback_timezone),),
                search_terms=(audit_number.strip(),),
            )
            return _clarification(
                task,
                reference,
                "The specified audit interval or timezone cannot be verified.",
                "Timestamps must be valid, unambiguous, and resolvable with the event timezone or an explicitly configured fallback.",
            )
        reference = AuditReference(
            candidates=(_event_ref(audit_event, fallback_timezone),),
            search_terms=(audit_number.strip(),),
        )
    else:
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
        if (
            fallback_timezone is None
            and use_calendar_timezone
            and any(event.get("timezone") is None for event in audit_like)
        ):
            try:
                fallback_timezone = _resolve_calendar_timezone(tools, calendar_id)
            except CalendarEventListError as exc:
                return _clarification(
                    task,
                    reference,
                    "The owning calendar timezone could not be verified for timestamp fallback.",
                    str(exc),
                )
        reference = AuditReference(
            candidates=tuple(
                _event_ref(event, fallback_timezone) for event in audit_like
            ),
            search_terms=("audit",),
        )
        if not audit_like:
            return _clarification(
                task,
                reference,
                "No audit candidate was found.",
                "An event whose title or description contains 'audit' is required.",
            )

        parsed_audits = [
            (event, _parse_event_times(event, fallback_timezone))
            for event in audit_like
        ]
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
                "Which audit should be used?",
                f"Found {len(valid_audits)} valid audit candidates; exactly one must be selected.",
            )

        audit_event, audit_times = valid_audits[0]
        assert audit_times is not None

    _, audit_end = audit_times
    shift = task.context.get("shift")
    shift_was_proposed = shift is None
    if shift_was_proposed:
        shift = _DEFAULT_PROPOSED_SHIFT
    elif not isinstance(shift, timedelta) or shift <= timedelta(0):
        return _clarification(
            task,
            reference,
            "What positive shift duration amount should be used?",
            "No shift duration was provided; the planner will not invent one.",
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
    if (
        fallback_timezone is None
        and use_calendar_timezone
        and any(event.get("timezone") is None for event in events)
    ):
        try:
            fallback_timezone = _resolve_calendar_timezone(tools, calendar_id)
        except CalendarEventListError as exc:
            return _clarification(
                task,
                reference,
                "The owning calendar timezone could not be verified for timestamp fallback.",
                str(exc),
            )
    audit_record_id = _event_id(audit_event)
    assert audit_record_id is not None
    matching_audits = [event for event in events if _event_id(event) == audit_record_id]
    if len(matching_audits) != 1:
        occurrence = "missing from" if not matching_audits else "duplicated in"
        selected_audit_id = (
            audit_number.strip()
            if isinstance(audit_number, str)
            else audit_record_id
        )
        return _clarification(
            task,
            reference,
            "The affected-event state is ambiguous because the resolved audit is not present exactly once.",
            f"The selected audit ID {selected_audit_id[:100]!r} was {occurrence} the full event listing "
            f"({len(matching_audits)} matches; expected exactly one).",
        )
    full_listing_audit_times = _parse_event_times(
        matching_audits[0], fallback_timezone
    )
    if (
        full_listing_audit_times is None
        or full_listing_audit_times != audit_times
        or (
            isinstance(audit_number, str)
            and matching_audits[0].get("number") != audit_number.strip()
        )
    ):
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

    moves_with_starts: list[tuple[datetime, EventMove]] = []
    move_ids: set[str] = set()
    timezone_ambiguous: list[dict[str, Any]] = []
    malformed_events: list[dict[str, Any]] = []
    malformed_after_audit: list[dict[str, Any]] = []
    identity_ambiguous: list[dict[str, Any]] = []
    for event in events:
        event_id = _event_id(event)
        if event_id == audit_record_id:
            continue
        parsed = _parse_event_times(event, fallback_timezone)
        if parsed is None:
            interval = _parse_event_interval(event, fallback_timezone)
            if interval is not None:
                if interval[0] > audit_end:
                    timezone_ambiguous.append(event)
            else:
                start = _parse_event_start(event, fallback_timezone)
                if start is not None and start > audit_end:
                    malformed_after_audit.append(event)
                else:
                    malformed_events.append(event)
            continue

        event_start, event_end = parsed
        if event_start <= audit_end:
            continue
        event_timezone = _event_timezone(event, fallback_timezone)
        if event_timezone is None:
            timezone_ambiguous.append(event)
            continue
        if event_id is None:
            identity_ambiguous.append(event)
            continue
        proposed_start = (event_start + shift).astimezone(event_timezone)
        proposed_end = (event_end + shift).astimezone(event_timezone)
        move_ids.add(event_id)
        moves_with_starts.append(
            (
                event_start,
                EventMove(
                event_id=event_id,
                title=str(event.get("title", "")),
                original_start_at=_normalized_timestamp(
                    str(event["start_at"]), event_timezone
                ),
                original_end_at=_normalized_timestamp(
                    str(event["end_at"]), event_timezone
                ),
                proposed_start_at=proposed_start.isoformat(),
                proposed_end_at=proposed_end.isoformat(),
                duration_seconds=(event_end - event_start).total_seconds(),
                calendar_id=event.get("calendar_id") if isinstance(event.get("calendar_id"), str) else None,
                timezone=event_timezone.key,
                ),
            )
        )

    moves_with_starts.sort(key=lambda item: item[0])
    moves = [move for _, move in moves_with_starts]

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
            "The affected-event state is ambiguous because an event after the audit has malformed timestamp or timezone data.",
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
            if event_id in move_ids or event_id == audit_record_id:
                continue
            parsed = _parse_event_interval(event, fallback_timezone)
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

    fallback_note = (
        f" The configured {fallback_timezone.key} timezone fallback is an explicit planning assumption, not a verified platform contract."
        if fallback_timezone is not None
        else ""
    )
    confirmation_prompt = (
        "Confirm the proposed 30-minute shift or provide a different positive duration before any update."
        if shift_was_proposed
        else "Confirm this plan or provide a different positive shift duration before any update."
    )
    return AgentResult(
        goal=task.goal,
        status=ResultStatus.PLANNED,
        summary=(
            "A read-only rescheduling plan is ready."
            + fallback_note
            + " Confirm this plan or request a different shift; no events have been moved."
        ),
        audit_reference=reference,
        rescheduling_plan=moves,
        claimed_outcome={
            "moved": False,
            "planned_event_count": len(moves),
            "shift": shift,
            "shift_was_proposed": shift_was_proposed,
            "confirmation_required": True,
            "confirmation_prompt": confirmation_prompt,
            "timestamp_timezone_fallback": (
                fallback_timezone.key if fallback_timezone is not None else None
            ),
            "timestamp_timezone_fallback_is_platform_contract": False,
        },
    )