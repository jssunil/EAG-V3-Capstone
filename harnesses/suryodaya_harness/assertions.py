"""Independent checks for planning-only scenarios."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import math
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from calendar_agent.models import (
    AgentResult,
    AuditReference,
    CalendarEventPage,
    CalendarEventRef,
    CandidateSlot,
    EventMove,
    ResultStatus,
    ToolCallRecord,
    VerificationResult,
)
from calendar_agent.tools import CalendarEventListError, CalendarTool


_BACKEND_EVENT_PAGE_LIMIT = 1000
_BACKEND_EVENT_MAX_PAGES = 100


@dataclass(frozen=True)
class BackendCalendarSnapshot:
    """Complete, read-only event state captured from the backend API."""

    calendar_id: str
    events: tuple[dict[str, Any], ...]
    tool_calls: tuple[ToolCallRecord, ...]


def read_backend_calendar_snapshot(
    tools: CalendarTool,
    *,
    calendar_id: str,
) -> BackendCalendarSnapshot:
    """Read a complete calendar snapshot through the backend CalendarEvent API."""

    if not isinstance(calendar_id, str) or not calendar_id.strip():
        raise ValueError("calendar_id must be non-empty text")
    page_reader = getattr(tools, "list_events_page", None)
    if not callable(page_reader):
        raise CalendarEventListError(
            "Backend verification requires paginated CalendarEvent reads."
        )

    trace_start = len(tools.trace)
    records: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    expected_total: int | None = None
    offset = 0

    for _ in range(_BACKEND_EVENT_MAX_PAGES):
        page = page_reader(
            calendar_id=calendar_id,
            limit=_BACKEND_EVENT_PAGE_LIMIT,
            offset=offset,
        )
        if not isinstance(page, CalendarEventPage):
            raise CalendarEventListError(
                "Backend CalendarEvent read returned no typed page metadata."
            )
        try:
            page.validate(
                requested_limit=_BACKEND_EVENT_PAGE_LIMIT,
                requested_offset=offset,
            )
        except (TypeError, ValueError) as exc:
            raise CalendarEventListError(
                "Backend CalendarEvent page metadata is invalid."
            ) from exc

        if expected_total is None:
            expected_total = page.total
        elif page.total != expected_total:
            raise CalendarEventListError(
                "Backend CalendarEvent total changed during pagination."
            )
        if page.total >= _BACKEND_EVENT_PAGE_LIMIT:
            raise CalendarEventListError(
                "Backend CalendarEvent result reached the local safety limit."
            )

        for event in page.records:
            event_id = event.get("id")
            if (
                event.get("calendar_id") != calendar_id
                or not isinstance(event_id, str)
                or not event_id.strip()
                or event_id in seen_event_ids
            ):
                raise CalendarEventListError(
                    "Backend CalendarEvent results contain an invalid, duplicate, or out-of-scope event."
                )
            seen_event_ids.add(event_id)
        records.extend(deepcopy(page.records))

        if not page.has_more:
            if len(records) != page.total:
                raise CalendarEventListError(
                    "Backend CalendarEvent pagination ended before the reported total."
                )
            return BackendCalendarSnapshot(
                calendar_id=calendar_id,
                events=tuple(records),
                tool_calls=tuple(tools.trace[trace_start:]),
            )

        next_offset = page.next_offset
        if next_offset <= offset:
            raise CalendarEventListError(
                "Backend CalendarEvent pagination did not advance."
            )
        offset = next_offset

    raise CalendarEventListError(
        "Backend CalendarEvent pagination reached the local page limit."
    )


def verify_backend_rescheduling_state(
    before: BackendCalendarSnapshot,
    after: BackendCalendarSnapshot,
    *,
    expected_moves: Sequence[EventMove],
) -> VerificationResult:
    """Verify backend state against expected event moves, not agent prose."""

    details: list[str] = []
    integrity_ok = (
        before.calendar_id == after.calendar_id
        and all(
            not call.mutating and call.succeeded
            for call in (*before.tool_calls, *after.tool_calls)
        )
    )
    if before.calendar_id != after.calendar_id:
        details.append("Backend snapshots refer to different calendars.")

    before_by_id = _unique_snapshot_events(before.events)
    after_by_id = _unique_snapshot_events(after.events)
    state_ok = before_by_id is not None and after_by_id is not None
    if not state_ok:
        details.append("A backend snapshot contains invalid or duplicate event identities.")
        before_by_id = before_by_id or {}
        after_by_id = after_by_id or {}
    if set(before_by_id) != set(after_by_id):
        state_ok = False
        details.append("The backend event set changed between snapshots.")

    moves_by_id: dict[str, EventMove] = {}
    for move in expected_moves:
        if not isinstance(move, EventMove) or not isinstance(move.event_id, str):
            state_ok = False
            details.append("The expected move list contains an invalid event.")
            continue
        if move.event_id in moves_by_id:
            state_ok = False
            details.append("The expected move list contains a duplicate event.")
            continue
        moves_by_id[move.event_id] = move

    for event_id, move in moves_by_id.items():
        before_event = before_by_id.get(event_id)
        after_event = after_by_id.get(event_id)
        if before_event is None or after_event is None:
            state_ok = False
            details.append("An expected moved event is missing from backend state.")
            continue
        original_start = _parse_aware_datetime(before_event.get("start_at"))
        original_end = _parse_aware_datetime(before_event.get("end_at"))
        planned_start = _parse_aware_datetime(move.original_start_at)
        planned_end = _parse_aware_datetime(move.original_end_at)
        actual_start = _parse_aware_datetime(after_event.get("start_at"))
        actual_end = _parse_aware_datetime(after_event.get("end_at"))
        expected_start = _parse_aware_datetime(move.proposed_start_at)
        expected_end = _parse_aware_datetime(move.proposed_end_at)
        if (
            None in (
                original_start,
                original_end,
                planned_start,
                planned_end,
                actual_start,
                actual_end,
                expected_start,
                expected_end,
            )
            or original_start != planned_start
            or original_end != planned_end
            or actual_start != expected_start
            or actual_end != expected_end
        ):
            state_ok = False
            details.append("A moved event's backend interval does not match the expected interval.")

    for event_id in set(before_by_id) - set(moves_by_id):
        if before_by_id[event_id] != after_by_id.get(event_id):
            state_ok = False
            details.append("A non-target backend event changed.")

    if not integrity_ok:
        details.append("Backend verification used a failed or mutating tool call.")
    if not details:
        details.append(
            f"Backend state verified for {len(moves_by_id)} moved event(s); "
            f"{len(before_by_id) - len(moves_by_id)} non-target event(s) were unchanged."
        )
    return VerificationResult(
        passed=state_ok and integrity_ok,
        outcome_ok=state_ok,
        integrity_ok=integrity_ok,
        details=tuple(details),
    )


def _unique_snapshot_events(
    events: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]] | None:
    by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        event_id = event.get("id")
        if not isinstance(event_id, str) or not event_id.strip() or event_id in by_id:
            return None
        by_id[event_id] = event
    return by_id


def _parse_aware_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _valid_candidate_slot(slot: object) -> bool:
    if not isinstance(slot, CandidateSlot):
        return False
    start = _parse_aware_datetime(slot.start_at)
    end = _parse_aware_datetime(slot.end_at)
    return (
        start is not None
        and end is not None
        and end - start == timedelta(minutes=30)
        and isinstance(slot.timezone, str)
        and bool(slot.timezone.strip())
        and slot.availability_confirmed is True
        and isinstance(slot.reason, str)
        and bool(slot.reason.strip())
        and isinstance(slot.party_id, str)
        and bool(slot.party_id.strip())
        and isinstance(slot.calendar_id, str)
        and bool(slot.calendar_id.strip())
    )


def _valid_event_ref(reference: object) -> bool:
    if not isinstance(reference, CalendarEventRef):
        return False
    start = _parse_aware_datetime(reference.start_at)
    end = _parse_aware_datetime(reference.end_at)
    return (
        isinstance(reference.id, str)
        and bool(reference.id.strip())
        and start is not None
        and end is not None
        and end > start
        and isinstance(reference.timezone, str)
        and bool(reference.timezone.strip())
    )


def _valid_event_move(move: object, audit_end: datetime, audit_id: str) -> bool:
    if (
        not isinstance(move, EventMove)
        or not isinstance(move.event_id, str)
        or not move.event_id.strip()
        or move.event_id == audit_id
    ):
        return False
    original_start = _parse_aware_datetime(move.original_start_at)
    original_end = _parse_aware_datetime(move.original_end_at)
    proposed_start = _parse_aware_datetime(move.proposed_start_at)
    proposed_end = _parse_aware_datetime(move.proposed_end_at)
    if None in (original_start, original_end, proposed_start, proposed_end):
        return False
    assert original_start is not None
    assert original_end is not None
    assert proposed_start is not None
    assert proposed_end is not None
    original_duration = original_end - original_start
    proposed_duration = proposed_end - proposed_start
    shift_start = proposed_start - original_start
    shift_end = proposed_end - original_end
    try:
        duration_seconds = float(move.duration_seconds)
        duration_matches = math.isfinite(duration_seconds) and timedelta(seconds=duration_seconds) == original_duration
    except (OverflowError, TypeError, ValueError):
        duration_matches = False
    return (
        original_start > audit_end
        and original_duration > timedelta(0)
        and original_duration == proposed_duration
        and shift_start == shift_end
        and shift_start > timedelta(0)
        and isinstance(move.timezone, str)
        and bool(move.timezone.strip())
        and duration_matches
    )


def _planned_state_errors(result: AgentResult) -> list[str]:
    if result.goal == "calendar.find_30_minutes":
        if (
            not isinstance(result.candidate_slots, list)
            or len(result.candidate_slots) != 1
            or not _valid_candidate_slot(result.candidate_slots[0])
        ):
            return ["A planned find-30-minutes result must contain one valid 30-minute candidate slot."]
        if (
            result.audit_reference is not None
            or not isinstance(result.rescheduling_plan, list)
            or result.rescheduling_plan
        ):
            return ["A find-30-minutes result contains contradictory rescheduling state."]
        if not isinstance(result.claimed_outcome, dict):
            return ["A planned find-30-minutes result has malformed claimed state."]
        claimed_slot = result.claimed_outcome.get("slot")
        if claimed_slot is not None and claimed_slot != result.candidate_slots[0]:
            return ["The claimed slot contradicts the returned candidate slot."]
        return []

    if result.goal == "calendar.move_after_audit":
        reference = result.audit_reference
        if (
            not isinstance(reference, AuditReference)
            or not isinstance(reference.candidates, tuple)
            or len(reference.candidates) != 1
            or not _valid_event_ref(reference.candidates[0])
        ):
            return ["A planned move result must resolve one valid audit reference."]
        audit_end = _parse_aware_datetime(reference.candidates[0].end_at)
        assert audit_end is not None
        audit_id = reference.candidates[0].id
        if not isinstance(result.rescheduling_plan, list) or not result.rescheduling_plan:
            return ["A planned move result must contain a non-empty event plan."]
        if any(not isinstance(move, EventMove) for move in result.rescheduling_plan):
            return ["A planned move result contains an invalid event move."]
        if any(not isinstance(move.event_id, str) for move in result.rescheduling_plan):
            return ["A planned move result contains an event with an invalid identity."]
        if len({move.event_id for move in result.rescheduling_plan}) != len(result.rescheduling_plan):
            return ["A planned move result contains duplicate event IDs."]
        if any(not _valid_event_move(move, audit_end, audit_id) for move in result.rescheduling_plan):
            return ["A planned move result contains an invalid or contradictory event move."]
        if not isinstance(result.candidate_slots, list) or result.candidate_slots:
            return ["A move result contains contradictory candidate-slot state."]
        claimed = result.claimed_outcome
        if not isinstance(claimed, dict):
            return ["A planned move result has malformed claimed state."]
        if "moved" in claimed and claimed["moved"] is not False:
            return ["A planned move result cannot claim that events were moved."]
        planned_count = claimed.get("planned_event_count")
        if planned_count is not None and planned_count != len(result.rescheduling_plan):
            return ["The claimed planned event count contradicts the event plan."]
        claimed_shift = claimed.get("shift")
        if claimed_shift is not None and (
            not isinstance(claimed_shift, timedelta) or claimed_shift <= timedelta(0)
        ):
            return ["The claimed shift is invalid."]
        if isinstance(claimed_shift, timedelta):
            for move in result.rescheduling_plan:
                proposed_start = _parse_aware_datetime(move.proposed_start_at)
                original_start = _parse_aware_datetime(move.original_start_at)
                assert proposed_start is not None
                assert original_start is not None
                if proposed_start - original_start != claimed_shift:
                    return ["The claimed shift contradicts the event plan."]
        return []

    return [f"No structural validator exists for planned goal {result.goal!r}."]


def verify_result(result: AgentResult, *, expected_status: ResultStatus) -> VerificationResult:
    """Verify observable result state and tool integrity, not prose claims."""

    outcome_ok = result.status == expected_status
    mutation_calls = [call.tool for call in result.tool_calls if call.mutating]
    integrity_ok = not mutation_calls and all(call.succeeded for call in result.tool_calls)
    details: list[str] = []
    if result.status is ResultStatus.PLANNED:
        structural_errors = _planned_state_errors(result)
        if structural_errors:
            outcome_ok = False
            details.extend(structural_errors)
    if not outcome_ok:
        details.append(f"Expected status {expected_status.value}, got {result.status.value}.")
    if mutation_calls:
        details.append(f"Mutation calls observed: {', '.join(mutation_calls)}.")
    if not integrity_ok and not mutation_calls:
        details.append("At least one read tool call failed.")
    return VerificationResult(
        passed=outcome_ok and integrity_ok,
        outcome_ok=outcome_ok,
        integrity_ok=integrity_ok,
        details=tuple(details),
    )


def verify_no_mutations(result: AgentResult) -> VerificationResult:
    mutation_calls = [call.tool for call in result.tool_calls if call.mutating]
    return VerificationResult(
        passed=not mutation_calls,
        outcome_ok=True,
        integrity_ok=not mutation_calls,
        details=tuple(f"Mutation observed: {name}" for name in mutation_calls),
    )


def verify_supported_slot(
    result: AgentResult,
    *,
    expected_start: str,
    expected_end: str,
    minimum_notice_boundary: datetime | None = None,
    expected_calendar_id: str | None = None,
) -> VerificationResult:
    """Verify the returned slot's observable shape independently of its prose."""

    if result.status is not ResultStatus.PLANNED or len(result.candidate_slots) != 1:
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=not any(call.mutating for call in result.tool_calls),
            details=("A planned result must contain exactly one candidate slot.",),
        )
    slot = result.candidate_slots[0]
    slot_ok = (
        slot.start_at == expected_start
        and slot.end_at == expected_end
        and slot.availability_confirmed
    )
    if minimum_notice_boundary is not None:
        try:
            slot_ok = slot_ok and datetime.fromisoformat(slot.start_at) >= minimum_notice_boundary
        except ValueError:
            slot_ok = False
    if expected_calendar_id is not None:
        slot_ok = slot_ok and slot.calendar_id == expected_calendar_id
    return VerificationResult(
        passed=slot_ok and not any(call.mutating for call in result.tool_calls),
        outcome_ok=slot_ok,
        integrity_ok=not any(call.mutating for call in result.tool_calls),
        details=() if slot_ok else ("Candidate slot does not match independently expected state.",),
    )


def _rule_allows_slot(rule: dict[str, Any], start: datetime, end: datetime) -> bool:
    timezone_name = rule.get("timezone")
    weekly_hours = rule.get("weekly_hours")
    if not isinstance(timezone_name, str) or not isinstance(weekly_hours, list):
        return False
    try:
        local_start = start.astimezone(ZoneInfo(timezone_name))
        local_end = end.astimezone(ZoneInfo(timezone_name))
    except (ValueError, ZoneInfoNotFoundError):
        return False
    if local_start.date() != local_end.date():
        return False

    for override in rule.get("date_overrides") or []:
        if not isinstance(override, dict) or override.get("date") != local_start.date().isoformat():
            continue
        if override.get("type") in {"unavailable", "blocked"}:
            return False
        if override.get("type") == "available":
            return True

    day = local_start.strftime("%A").lower()
    for period in weekly_hours:
        if not isinstance(period, dict) or str(period.get("day", "")).lower() != day:
            continue
        if period.get("enabled") is not True:
            continue
        try:
            period_start = time.fromisoformat(str(period["start"]))
            period_end = time.fromisoformat(str(period["end"]))
        except (KeyError, ValueError):
            continue
        if (
            period_start <= local_start.timetz().replace(tzinfo=None)
            and period_end >= local_end.timetz().replace(tzinfo=None)
        ):
            return True
    return False


def verify_goal_one_candidate_against_reads(
    result: AgentResult,
    tools: Any,
    *,
    now: datetime,
    search_start: datetime,
    search_end: datetime,
    prior_tool_calls: list[ToolCallRecord] | None = None,
) -> VerificationResult:
    """Re-read the relevant scheduling state and verify one proposed slot."""

    if (
        now.tzinfo is None
        or now.utcoffset() is None
        or search_start.tzinfo is None
        or search_start.utcoffset() is None
        or search_end.tzinfo is None
        or search_end.utcoffset() is None
        or search_end <= search_start
    ):
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("The verifier requires a valid timezone-aware search window.",),
        )

    structural = verify_result(result, expected_status=ResultStatus.PLANNED)
    if not structural.passed:
        return structural

    slot = result.candidate_slots[0]
    start = _parse_aware_datetime(slot.start_at)
    end = _parse_aware_datetime(slot.end_at)
    if (
        start is None
        or end is None
        or start < search_start
        or end > search_end
        or end - start != timedelta(minutes=30)
    ):
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("Candidate is not a 30-minute slot inside the requested window.",),
        )

    parties_by_id: dict[str, dict[str, Any]] = {}
    for title in ("Plant Manager", "Plant Head"):
        parties = tools.find_parties(job_title=title, limit=1000)
        if not isinstance(parties, list) or len(parties) >= 1000:
            return VerificationResult(
                passed=False,
                outcome_ok=False,
                integrity_ok=True,
                details=("Plant-head Party data is incomplete.",),
            )
        for party in parties:
            if not isinstance(party, dict):
                return VerificationResult(
                    passed=False,
                    outcome_ok=False,
                    integrity_ok=True,
                    details=("Plant-head Party data contains an invalid record.",),
                )
            party_id = party.get("id")
            email = party.get("email")
            job_title = party.get("job_title")
            if (
                not isinstance(job_title, str)
                or job_title.strip().casefold() != title.casefold()
                or not isinstance(party_id, str)
                or not party_id.strip()
                or not isinstance(email, str)
                or not email.strip()
            ):
                continue
            previous = parties_by_id.get(party_id)
            if previous is not None and (
                previous.get("email") != email
                or previous.get("job_title") != job_title
            ):
                return VerificationResult(
                    passed=False,
                    outcome_ok=False,
                    integrity_ok=True,
                    details=("Plant-head identity records conflict.",),
                )
            parties_by_id[party_id] = party

    if len(parties_by_id) != 1 or slot.party_id not in parties_by_id:
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("Candidate does not resolve to one verified plant-head Party.",),
        )
    participant = parties_by_id[slot.party_id]

    calendars = tools.list_calendars(owner_party_id=slot.party_id, limit=1000)
    if not isinstance(calendars, list) or len(calendars) >= 1000:
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("Owned Calendar data is incomplete.",),
        )
    owned_calendars = [
        calendar
        for calendar in calendars
        if isinstance(calendar, dict)
        and calendar.get("owner_party_id") == slot.party_id
        and isinstance(calendar.get("id"), str)
        and calendar.get("id")
    ]
    if len(owned_calendars) != 1 or owned_calendars[0]["id"] != slot.calendar_id:
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("Candidate does not use the unique owned Calendar.",),
        )

    preferences = tools.list_scheduling_preferences(limit=1000)
    if not isinstance(preferences, list) or len(preferences) != 1:
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("Scheduling preferences are not uniquely available.",),
        )
    preference = preferences[0]
    if (
        isinstance(preference.get("default_meeting_duration"), bool)
        or isinstance(preference.get("min_notice_hours"), bool)
    ):
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("The meeting duration preference is invalid.",),
        )
    try:
        duration = float(preference.get("default_meeting_duration"))
        notice_hours = float(preference.get("min_notice_hours"))
        notice = timedelta(hours=notice_hours)
    except (OverflowError, TypeError, ValueError):
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("Scheduling preferences are invalid.",),
        )
    if (
        not math.isfinite(duration)
        or duration != 30
        or not math.isfinite(notice_hours)
        or notice_hours < 0
        or start < max(search_start, now + notice)
    ):
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("Candidate does not satisfy duration or minimum-notice preferences.",),
        )

    rules = tools.list_availability_rules(limit=1000)
    if not isinstance(rules, list) or len(rules) >= 1000:
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("Availability-rule data is incomplete.",),
        )
    linked_rules = [
        rule
        for rule in rules
        if isinstance(rule, dict)
        and (
            (rule.get("party_id") or rule.get("owner_party_id")) == slot.party_id
            or rule.get("calendar_id") == slot.calendar_id
        )
    ]
    if not any(
        rule.get("timezone") == slot.timezone and _rule_allows_slot(rule, start, end)
        for rule in linked_rules
    ):
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("Candidate is outside a linked availability rule.",),
        )

    response = tools.free_busy(
        start=search_start,
        end=search_end,
        emails=[participant["email"]],
        calendar_ids=[slot.calendar_id],
    )
    free_busy = response.get("result") if isinstance(response, dict) else None
    window = free_busy.get("window") if isinstance(free_busy, dict) else None
    response_start = _parse_aware_datetime(window.get("start")) if isinstance(window, dict) else None
    response_end = _parse_aware_datetime(window.get("end")) if isinstance(window, dict) else None
    if (
        not isinstance(free_busy, dict)
        or free_busy.get("ok") is not True
        or response_start != search_start
        or response_end != search_end
    ):
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("Fresh free/busy evidence is missing or covers a different window.",),
        )

    subjects = free_busy.get("subjects")
    matching_subjects = [
        subject
        for subject in subjects
        if isinstance(subject, dict)
        and subject.get("subject_kind") == "email"
        and subject.get("subject") == participant["email"]
    ] if isinstance(subjects, list) else []
    if len(matching_subjects) != 1:
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("Fresh free/busy evidence does not identify one participant.",),
        )

    subject = matching_subjects[0]
    subject_calendars = subject.get("calendars")
    matching_calendars = [
        calendar
        for calendar in subject_calendars
        if isinstance(calendar, dict) and calendar.get("calendar_id") == slot.calendar_id
    ] if isinstance(subject_calendars, list) else []
    if len(matching_calendars) != 1:
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("Fresh free/busy evidence does not identify the owned Calendar.",),
        )

    calendar_access = matching_calendars[0]
    valid_access = {"free_busy", "read", "write", "owner"}
    if (
        subject.get("visibility") not in {"visible", "free_busy", "read", "write"}
        or subject.get("visibility_code") in {"freebusy.notShared", "freebusy.noCalendar"}
        or calendar_access.get("access_level") not in valid_access
        or (
            "actual_access_level" in calendar_access
            and calendar_access.get("actual_access_level") not in valid_access
        )
    ):
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("Fresh free/busy evidence does not confirm visible Calendar access.",),
        )

    busy_intervals = subject.get("busy")
    if not isinstance(busy_intervals, list):
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=True,
            details=("Fresh free/busy intervals are malformed.",),
        )
    for interval in busy_intervals:
        if not isinstance(interval, dict):
            return VerificationResult(
                passed=False,
                outcome_ok=False,
                integrity_ok=True,
                details=("Fresh free/busy interval is malformed.",),
            )
        busy_calendar = interval.get("calendar_id")
        busy_start = _parse_aware_datetime(interval.get("start") or interval.get("start_at"))
        busy_end = _parse_aware_datetime(interval.get("end") or interval.get("end_at"))
        if (
            not isinstance(busy_calendar, str)
            or not busy_calendar.strip()
            or busy_start is None
            or busy_end is None
            or busy_end <= busy_start
        ):
            return VerificationResult(
                passed=False,
                outcome_ok=False,
                integrity_ok=True,
                details=("Fresh free/busy interval is malformed.",),
            )
        if (
            busy_calendar == slot.calendar_id
            and start < busy_end
            and end > busy_start
        ):
            return VerificationResult(
                passed=False,
                outcome_ok=False,
                integrity_ok=True,
                details=("Candidate overlaps a fresh busy interval.",),
            )

    calls = [*(prior_tool_calls or ()), *getattr(tools, "trace", ())]
    integrity_ok = all(not call.mutating and call.succeeded for call in calls)
    return VerificationResult(
        passed=integrity_ok,
        outcome_ok=True,
        integrity_ok=integrity_ok,
        details=() if integrity_ok else ("A verifier tool call failed or attempted a mutation.",),
    )


def verify_rescheduling_plan(
    result: AgentResult,
    *,
    expected_audit_id: str,
    expected_event_ids: list[str],
    audit_end: datetime,
    shift: timedelta,
) -> VerificationResult:
    """Verify a move plan from its event state, not its summary or claimed outcome."""

    mutation_free = not any(call.mutating for call in result.tool_calls)
    reference = result.audit_reference
    if result.status is not ResultStatus.PLANNED or reference is None or len(reference.candidates) != 1:
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=mutation_free,
            details=("A planned rescheduling result must resolve exactly one audit.",),
        )
    if reference.candidates[0].id != expected_audit_id:
        return VerificationResult(
            passed=False,
            outcome_ok=False,
            integrity_ok=mutation_free,
            details=("The resolved audit does not match independent state.",),
        )

    plan = result.rescheduling_plan
    plan_ok = [move.event_id for move in plan] == expected_event_ids
    details: list[str] = []
    for move in plan:
        try:
            original_start = datetime.fromisoformat(move.original_start_at)
            original_end = datetime.fromisoformat(move.original_end_at)
            proposed_start = datetime.fromisoformat(move.proposed_start_at)
            proposed_end = datetime.fromisoformat(move.proposed_end_at)
        except ValueError:
            plan_ok = False
            details.append(f"Malformed datetime in plan for {move.event_id}.")
            continue
        if original_start <= audit_end:
            plan_ok = False
            details.append(f"Event {move.event_id} is not strictly after the audit end.")
        if original_end - original_start != proposed_end - proposed_start:
            plan_ok = False
            details.append(f"Event {move.event_id} duration changed.")
        if proposed_start - original_start != shift or proposed_end - original_end != shift:
            plan_ok = False
            details.append(f"Event {move.event_id} shift changed.")
    if not plan_ok:
        details.append("Rescheduling plan does not match independently expected ordering or state.")
    return VerificationResult(
        passed=plan_ok and mutation_free,
        outcome_ok=plan_ok,
        integrity_ok=mutation_free,
        details=tuple(details),
    )