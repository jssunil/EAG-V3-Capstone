from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

import calendar_agent.goals.move_after_audit as move_after_audit
from calendar_agent.agent import CalendarAgent
from calendar_agent.models import (
    AgentResult,
    AgentTask,
    CalendarEventPage,
    CalendarMutationError,
    ResultStatus,
)
from calendar_agent.tools import CalendarEventListError
from harnesses.suryodaya_harness.assertions import verify_rescheduling_plan, verify_result

from .fakes import FakeCalendarTools


AUDIT = {
    "id": "audit-1",
    "title": "Internal quality audit",
    "description": "ISO review",
    "start_at": "2026-09-22T09:00:00+00:00",
    "end_at": "2026-09-22T10:00:00+00:00",
    "timezone": "UTC",
    "calendar_id": "calendar-1",
}
SHIFT = timedelta(hours=1)
AUDIT_END = datetime(2026, 9, 22, 10, tzinfo=timezone.utc)


def event(
    event_id: str,
    start: str,
    end: str,
    *,
    title: str = "Work",
    timezone_name: str = "UTC",
) -> dict[str, str]:
    return {
        "id": event_id,
        "title": title,
        "start_at": start,
        "end_at": end,
        "timezone": timezone_name,
        "calendar_id": "calendar-1",
    }


class SplitAuditListingTools(FakeCalendarTools):
    def __init__(self, full_listing: list[dict[str, object]]) -> None:
        super().__init__()
        self.full_listing = full_listing

    def list_events(self, **filters: object) -> list[dict[str, object]]:
        self._record("CalendarEvent.list", filters)
        if filters.get("search") == "audit":
            return [AUDIT]
        return self.full_listing


def run(
    events: list[dict[str, object]],
    *,
    shift: timedelta | None = SHIFT,
    fallback_timezone: str | None = None,
):
    context = {"calendar_id": "calendar-1"}
    if shift is not None:
        context["shift"] = shift
    if fallback_timezone is not None:
        context["naive_timestamp_timezone"] = fallback_timezone
    return CalendarAgent(FakeCalendarTools(events=events)).run(
        AgentTask("calendar.move_after_audit", "Move everything after the audit", context)
    )


def run_with_split_listings(full_listing: list[dict[str, object]]):
    return CalendarAgent(SplitAuditListingTools(full_listing)).run(
        AgentTask(
            "calendar.move_after_audit",
            "Move everything after the audit",
            {"shift": SHIFT, "calendar_id": "calendar-1"},
        )
    )


class ScriptedEventPages(FakeCalendarTools):
    def __init__(
        self,
        search_pages: list[CalendarEventPage | Exception],
        full_pages: list[CalendarEventPage | Exception],
    ) -> None:
        super().__init__()
        self.pages = {"audit": list(search_pages), "full": list(full_pages)}

    def list_events_page(self, **filters: object) -> CalendarEventPage:
        self._record("CalendarEvent.list", filters)
        key = (
            "audit"
            if filters.get("search") == "audit" or filters.get("number") is not None
            else "full"
        )
        response = self.pages[key].pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class WritableFakeCalendarTools(FakeCalendarTools):
    disposable_mutations_enabled = True

    def __init__(
        self,
        *,
        events: list[dict[str, object]],
        fail_update_at: int | None = None,
        mutate_unplanned_event: bool = False,
    ) -> None:
        super().__init__(events=events)
        self.fail_update_at = fail_update_at
        self.mutate_unplanned_event = mutate_unplanned_event
        self.update_attempts = 0

    def list_events_page(self, **filters: object) -> CalendarEventPage:
        page_result = super().list_events_page(**filters)
        return CalendarEventPage(
            records=tuple(deepcopy(record) for record in page_result.records),
            total=page_result.total,
            limit=page_result.limit,
            offset=page_result.offset,
        )

    def update_event(self, **arguments: object) -> dict[str, object]:
        self.update_attempts += 1
        if self.update_attempts == self.fail_update_at:
            raise CalendarMutationError("synthetic update failure")
        event_id = arguments["id"]
        target = next(record for record in self.events if record.get("id") == event_id)
        target["start_at"] = arguments["start_at"]
        target["end_at"] = arguments["end_at"]
        if self.mutate_unplanned_event:
            self.events[1]["title"] = "Changed unexpectedly"
        return {"id": event_id}


def page(
    *records: dict[str, object],
    total: int,
    limit: int,
    offset: int,
) -> CalendarEventPage:
    return CalendarEventPage(tuple(records), total, limit, offset)


def run_with_pages(
    search_pages: list[CalendarEventPage | Exception],
    full_pages: list[CalendarEventPage | Exception],
    *,
    calendar_id: str = "calendar-1",
) -> tuple[AgentResult, ScriptedEventPages]:
    tools = ScriptedEventPages(search_pages, full_pages)
    result = CalendarAgent(tools).run(
        AgentTask(
            "calendar.move_after_audit",
            "Move everything after the audit",
            {"shift": SHIFT, "calendar_id": calendar_id},
        )
    )
    return result, tools


def run_with_exact_audit_pages(
    audit_pages: list[CalendarEventPage | Exception],
    full_pages: list[CalendarEventPage | Exception],
    *,
    shift: timedelta | None = SHIFT,
    calendar_id: str | None = None,
    fallback_timezone: str | None = None,
) -> tuple[AgentResult, ScriptedEventPages]:
    tools = ScriptedEventPages(audit_pages, full_pages)
    context: dict[str, object] = {
        "audit_number": "EVT-2026-00007",
        "audit_title": "Internal quality audit — press shop and welding",
    }
    if shift is not None:
        context["shift"] = shift
    if calendar_id is not None:
        context["calendar_id"] = calendar_id
    if fallback_timezone is not None:
        context["naive_timestamp_timezone"] = fallback_timezone
    result = CalendarAgent(tools).run(
        AgentTask(
            "calendar.move_after_audit",
            "Move everything after the specified internal quality audit",
            context,
        )
    )
    return result, tools


def exact_audit(**overrides: object) -> dict[str, object]:
    return {
        "id": "audit-uuid",
        "number": "EVT-2026-00007",
        "title": "Internal quality audit — press shop and welding",
        "start_at": "2026-09-19T09:30:00+05:30",
        "end_at": "2026-09-19T12:30:00+05:30",
        "timezone": "Asia/Kolkata",
        "calendar_id": "calendar-1",
        "status": "confirmed",
        **overrides,
    }


def test_exactly_one_audit_plans_events_strictly_after_audit_end() -> None:
    result = run([
        AUDIT,
        event("before", "2026-09-22T08:00:00+00:00", "2026-09-22T08:30:00+00:00"),
        event("after-1", "2026-09-22T11:00:00+00:00", "2026-09-22T11:45:00+00:00"),
        event("after-2", "2026-09-22T12:00:00+00:00", "2026-09-22T13:30:00+00:00"),
    ])

    verification = verify_rescheduling_plan(
        result,
        expected_audit_id="audit-1",
        expected_event_ids=["after-1", "after-2"],
        audit_end=AUDIT_END,
        shift=SHIFT,
    )
    assert verification.passed
    assert verify_result(result, expected_status=ResultStatus.PLANNED).passed
    assert result.claimed_outcome["moved"] is False
    assert not any(call.mutating for call in result.tool_calls)


def test_equivalent_offsets_use_the_same_instant_for_audit_boundary() -> None:
    result = run([
        AUDIT,
        event(
            "at-audit-end",
            "2026-09-22T11:00:00+01:00",
            "2026-09-22T11:30:00+01:00",
            timezone_name="Europe/London",
        ),
        event(
            "after",
            "2026-09-22T12:00:00+01:00",
            "2026-09-22T13:00:00+01:00",
            timezone_name="Europe/London",
        ),
    ])

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["after"]
    assert result.rescheduling_plan[0].proposed_start_at == "2026-09-22T13:00:00+01:00"


@pytest.mark.parametrize(
    ("start", "end", "expected_start", "expected_end"),
    [
        (
            "2026-03-08T01:30:00-05:00",
            "2026-03-08T03:30:00-04:00",
            "2026-03-08T03:30:00-04:00",
            "2026-03-08T04:30:00-04:00",
        ),
        (
            "2026-11-01T01:30:00-04:00",
            "2026-11-01T01:30:00-05:00",
            "2026-11-01T01:30:00-05:00",
            "2026-11-01T02:30:00-05:00",
        ),
    ],
)
def test_dst_shift_preserves_elapsed_time_and_emits_event_zone_offsets(
    start: str,
    end: str,
    expected_start: str,
    expected_end: str,
) -> None:
    audit = {
        **AUDIT,
        "start_at": "2026-03-07T23:00:00-05:00",
        "end_at": "2026-03-07T23:30:00-05:00",
    }
    if "2026-11" in start:
        audit = {
            **AUDIT,
            "start_at": "2026-10-31T23:00:00-04:00",
            "end_at": "2026-10-31T23:30:00-04:00",
        }
    after = {
        **event("after", start, end),
        "timezone": None,
    }
    result = run(
        [audit, after],
        fallback_timezone="America/New_York",
    )

    assert result.status is ResultStatus.PLANNED
    move = result.rescheduling_plan[0]
    assert move.proposed_start_at == expected_start
    assert move.proposed_end_at == expected_end
    assert move.duration_seconds == 3600
    assert datetime.fromisoformat(move.proposed_start_at) - datetime.fromisoformat(move.original_start_at) == SHIFT
    assert datetime.fromisoformat(move.proposed_end_at) - datetime.fromisoformat(move.original_end_at) == SHIFT


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2026-11-01T01:30:00", "2026-11-01T02:30:00"),
        ("2026-03-08T02:30:00", "2026-03-08T03:30:00"),
    ],
)
def test_ambiguous_or_nonexistent_local_timestamps_remain_fail_closed(
    start: str,
    end: str,
) -> None:
    audit = {
        **AUDIT,
        "start_at": "2026-10-31T12:00:00-04:00",
        "end_at": "2026-10-31T12:30:00-04:00",
    }
    if "2026-03" in start:
        audit = {
            **AUDIT,
            "start_at": "2026-03-07T12:00:00-05:00",
            "end_at": "2026-03-07T12:30:00-05:00",
        }
    result = run([
        audit,
        event("after", start, end, timezone_name="America/New_York"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_invalid_iana_zone_for_a_movable_event_requires_clarification() -> None:
    result = run([
        AUDIT,
        event(
            "after",
            "2026-09-22T11:00:00+00:00",
            "2026-09-22T12:00:00+00:00",
            timezone_name="Not/A_Zone",
        ),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_destination_conflict_is_detected_across_spring_forward_transition() -> None:
    audit = {
        **AUDIT,
        "start_at": "2026-03-08T00:30:00-05:00",
        "end_at": "2026-03-08T01:30:00-05:00",
        "timezone": "America/New_York",
    }
    result = run([
        audit,
        event(
            "overlap-audit",
            "2026-03-08T01:00:00-05:00",
            "2026-03-08T03:45:00-04:00",
            timezone_name="America/New_York",
        ),
        event(
            "after",
            "2026-03-08T03:00:00-04:00",
            "2026-03-08T03:30:00-04:00",
            timezone_name="America/New_York",
        ),
    ], shift=timedelta(minutes=30))

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "conflict" in result.summary.lower()


def test_audit_missing_from_full_listing_requires_clarification() -> None:
    result = run_with_split_listings([
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "audit-1" in " ".join(result.ambiguities)
    assert "0 matches" in " ".join(result.ambiguities)


def test_audit_duplicated_in_full_listing_requires_clarification() -> None:
    result = run_with_split_listings([
        AUDIT,
        {**AUDIT, "title": "Duplicate audit copy"},
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "audit-1" in " ".join(result.ambiguities)
    assert "2 matches" in " ".join(result.ambiguities)


def test_exactly_one_audit_in_full_listing_continues_valid_planning() -> None:
    result = run_with_split_listings([
        AUDIT,
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["after"]


def test_audit_search_at_record_limit_requires_clarification() -> None:
    tools = FakeCalendarTools(events=[AUDIT.copy() for _ in range(1000)])

    result = CalendarAgent(tools).run(
        AgentTask(
            "calendar.move_after_audit",
            "Move everything after the audit",
            {"shift": SHIFT, "calendar_id": "calendar-1"},
        )
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "may be incomplete" in result.summary
    assert len(tools.trace) == 1


def test_full_event_listing_at_record_limit_produces_no_plan() -> None:
    events = [AUDIT, *(
        event(f"after-{index}", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")
        for index in range(999)
    )]
    result = run_with_split_listings(events)

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "may be incomplete" in result.summary


def test_multiple_audits_do_not_produce_a_plan() -> None:
    result = run([AUDIT, {**AUDIT, "id": "audit-2", "title": "Supplier audit"}])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert len(result.audit_reference.candidates) == 2


def test_no_audit_does_not_produce_a_plan() -> None:
    result = run([event("work", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_event_overlapping_audit_is_not_selected() -> None:
    result = run([
        AUDIT,
        event("overlap", "2026-09-22T09:30:00+00:00", "2026-09-22T10:30:00+00:00"),
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["after"]


def test_malformed_event_with_unknown_position_requires_clarification() -> None:
    result = run([
        AUDIT,
        {
            "id": "malformed",
            "title": "Broken",
            "start_at": "not-a-date",
            "end_at": "also-broken",
            "timezone": "UTC",
            "calendar_id": "calendar-1",
        },
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "malformed" in " ".join(result.ambiguities)


def test_malformed_end_before_audit_cannot_hide_a_destination_conflict() -> None:
    result = run([
        AUDIT,
        event("uncertain", "2026-09-22T08:00:00+00:00", "not-a-date"),
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert any("uncertain" in ambiguity for ambiguity in result.ambiguities)


def test_unrelated_pre_audit_interval_uses_explicit_fallback() -> None:
    without_timezone = event("before", "2026-09-22T07:00:00+00:00", "2026-09-22T08:00:00+00:00")
    del without_timezone["timezone"]

    result = run([
        AUDIT,
        without_timezone,
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ], fallback_timezone="UTC")

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["after"]


def test_malformed_audit_candidate_requires_clarification() -> None:
    malformed_audit = {**AUDIT, "start_at": "not-a-date"}

    result = run([malformed_audit, event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_malformed_affected_event_that_starts_after_audit_requires_clarification() -> None:
    malformed = event(
        "broken",
        "2026-09-22T11:00:00+00:00",
        "2026-09-22T10:30:00+00:00",
    )

    result = run([AUDIT, malformed, event("after", "2026-09-22T12:00:00+00:00", "2026-09-22T13:00:00+00:00")])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_no_events_after_audit_is_clarification() -> None:
    result = run([
        AUDIT,
        event("before", "2026-09-22T08:00:00+00:00", "2026-09-22T09:00:00+00:00"),
        event("at-end", "2026-09-22T10:00:00+00:00", "2026-09-22T10:30:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_timezone_null_event_is_not_silently_reinterpreted() -> None:
    ambiguous = event("ambiguous", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")
    ambiguous["timezone"] = None
    result = run([AUDIT, ambiguous])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "timezone" in result.summary.lower()


def test_missing_audit_id_does_not_produce_a_plan() -> None:
    result = run([
        {**AUDIT, "id": ""},
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_missing_affected_event_id_does_not_produce_a_plan() -> None:
    missing_id = event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")
    del missing_id["id"]

    result = run([AUDIT, missing_id])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_duplicate_affected_event_ids_do_not_produce_a_plan() -> None:
    result = run([
        AUDIT,
        event("duplicate", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
        event("duplicate", "2026-09-22T13:00:00+00:00", "2026-09-22T14:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_duplicate_event_ids_across_audit_boundary_do_not_produce_a_plan() -> None:
    result = run([
        AUDIT,
        event("x", "2026-09-22T09:00:00+00:00", "2026-09-22T09:30:00+00:00"),
        event("x", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_affected_event_sharing_audit_id_does_not_produce_a_plan() -> None:
    result = run([
        AUDIT,
        event("audit-1", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
        event("after", "2026-09-22T13:00:00+00:00", "2026-09-22T14:00:00+00:00"),
    ])

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_successful_planning_does_not_mutate_source_events() -> None:
    from copy import deepcopy

    events = [
        AUDIT,
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ]
    tools = FakeCalendarTools(events=events)
    before = deepcopy(tools.events)

    result = CalendarAgent(tools).run(
        AgentTask(
            "calendar.move_after_audit",
            "Move everything after the audit",
            {"shift": SHIFT, "calendar_id": "calendar-1"},
        )
    )

    assert result.status is ResultStatus.PLANNED
    assert result.status is not ResultStatus.COMPLETED
    assert tools.events == before
    assert not any(call.mutating for call in result.tool_calls)


def test_missing_shift_proposes_thirty_minutes_and_requires_confirmation() -> None:
    result = run(
        [AUDIT, event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")],
        shift=None,
    )

    assert result.status is ResultStatus.PLANNED
    assert result.claimed_outcome["shift"] == timedelta(minutes=30)
    assert result.claimed_outcome["shift_was_proposed"] is True
    assert result.claimed_outcome["confirmation_required"] is True
    assert "proposed 30-minute shift" in result.claimed_outcome["confirmation_prompt"]
    assert result.claimed_outcome["moved"] is False
    assert not any(call.mutating for call in result.tool_calls)


def test_exact_audit_number_derives_calendar_and_returns_confirmation_request() -> None:
    target = exact_audit()
    before = event("before", "2026-09-19T08:00:00+05:30", "2026-09-19T09:00:00+05:30")
    boundary = event("boundary", "2026-09-19T12:30:00+05:30", "2026-09-19T13:00:00+05:30")
    after = {
        **event("work-uuid", "2026-09-19T13:30:00+05:30", "2026-09-19T14:30:00+05:30"),
        "number": "EVT-2026-00008",
        "status": "confirmed",
    }
    result, tools = run_with_exact_audit_pages(
        [page(target, total=1, limit=10, offset=0)],
        [page(target, before, boundary, after, total=4, limit=10, offset=0)],
    )

    assert result.status is ResultStatus.PLANNED
    assert result.audit_reference is not None
    assert [candidate.id for candidate in result.audit_reference.candidates] == [
        "EVT-2026-00007"
    ]
    assert [move.event_id for move in result.rescheduling_plan] == ["work-uuid"]
    assert result.rescheduling_plan[0].calendar_id == "calendar-1"
    assert result.rescheduling_plan[0].duration_seconds == 3600
    assert result.claimed_outcome["moved"] is False
    assert result.claimed_outcome["confirmation_required"] is True
    assert "confirm" in result.claimed_outcome["confirmation_prompt"].lower()
    assert not any(call.mutating for call in result.tool_calls)
    assert tools.trace[0].arguments["number"] == "EVT-2026-00007"
    assert "calendar_id" not in tools.trace[0].arguments
    assert all(
        call.arguments["calendar_id"] == "calendar-1"
        for call in tools.trace[1:]
    )


@pytest.mark.parametrize(
    ("audit_overrides", "summary_fragment"),
    [
        ({"title": "Other audit"}, "different audit title"),
        ({"status": "tentative"}, "not verified as confirmed"),
        ({"calendar_id": None}, "no verified calendar association"),
    ],
)
def test_exact_audit_requires_expected_title_confirmed_status_and_calendar(
    audit_overrides: dict[str, object],
    summary_fragment: str,
) -> None:
    result, tools = run_with_exact_audit_pages(
        [page(exact_audit(**audit_overrides), total=1, limit=10, offset=0)],
        [],
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert summary_fragment in result.summary
    assert len(tools.trace) == 1
    assert not any(call.mutating for call in result.tool_calls)


def test_exact_audit_rejects_a_caller_calendar_that_does_not_match() -> None:
    result, tools = run_with_exact_audit_pages(
        [page(exact_audit(), total=1, limit=10, offset=0)],
        [],
        calendar_id="another-calendar",
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "does not match the audit's associated calendar" in result.summary
    assert len(tools.trace) == 1
    assert not any(call.mutating for call in result.tool_calls)


def test_duplicate_exact_audit_numbers_require_clarification() -> None:
    duplicate = exact_audit(id="audit-uuid-duplicate")
    result, tools = run_with_exact_audit_pages(
        [
            page(
                exact_audit(),
                duplicate,
                total=2,
                limit=10,
                offset=0,
            )
        ],
        [],
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert len(result.audit_reference.candidates) == 2
    assert "found 2" in " ".join(result.ambiguities)
    assert len(tools.trace) == 1
    assert not any(call.mutating for call in result.tool_calls)


def test_exact_audit_missing_shift_proposes_thirty_minutes_and_waits() -> None:
    target = exact_audit()
    after = {
        **event(
            "work-uuid",
            "2026-09-19T13:30:00+05:30",
            "2026-09-19T14:30:00+05:30",
        ),
        "number": "EVT-2026-00008",
    }
    result, tools = run_with_exact_audit_pages(
        [page(target, total=1, limit=10, offset=0)],
        [page(target, after, total=2, limit=10, offset=0)],
        shift=None,
    )

    assert result.status is ResultStatus.PLANNED
    assert result.claimed_outcome["shift"] == timedelta(minutes=30)
    assert result.claimed_outcome["shift_was_proposed"] is True
    assert result.claimed_outcome["confirmation_required"] is True
    assert result.claimed_outcome["moved"] is False
    assert [move.event_id for move in result.rescheduling_plan] == ["work-uuid"]
    assert not any(call.mutating for call in result.tool_calls)
    assert len(tools.trace) == 2


def test_exact_audit_naive_interval_uses_event_level_timezone() -> None:
    target = exact_audit(
        start_at="2026-09-19T09:30:00",
        end_at="2026-09-19T12:30:00",
    )
    after = {
        **event(
            "work-uuid",
            "2026-09-19T13:30:00",
            "2026-09-19T14:30:00",
            timezone_name="Asia/Kolkata",
        ),
        "number": "EVT-2026-00008",
    }
    result, tools = run_with_exact_audit_pages(
        [page(target, total=1, limit=10, offset=0)],
        [page(target, after, total=2, limit=10, offset=0)],
    )

    assert result.status is ResultStatus.PLANNED
    assert result.rescheduling_plan[0].original_start_at == "2026-09-19T13:30:00+05:30"
    assert result.rescheduling_plan[0].proposed_start_at == "2026-09-19T14:30:00+05:30"
    assert len(tools.trace) == 2
    assert not any(call.mutating for call in result.tool_calls)


def test_naive_suryodaya_timestamps_use_explicit_kolkata_fallback() -> None:
    target = exact_audit(
        start_at="2026-09-19T09:30:00",
        end_at="2026-09-19T12:30:00",
        timezone=None,
    )
    after = {
        **event(
            "work-uuid",
            "2026-09-19T13:30:00",
            "2026-09-19T14:30:00",
            timezone_name="UTC",
        ),
        "timezone": None,
        "number": "EVT-2026-00008",
    }
    result, _ = run_with_exact_audit_pages(
        [page(target, total=1, limit=10, offset=0)],
        [page(target, after, total=2, limit=10, offset=0)],
        fallback_timezone="Asia/Kolkata",
    )

    assert result.status is ResultStatus.PLANNED
    move = result.rescheduling_plan[0]
    assert move.original_start_at == "2026-09-19T13:30:00+05:30"
    assert move.proposed_start_at == "2026-09-19T14:30:00+05:30"
    assert move.original_end_at == "2026-09-19T14:30:00+05:30"
    assert move.duration_seconds == 3600
    assert result.audit_reference is not None
    assert result.audit_reference.candidates[0].start_at == "2026-09-19T09:30:00+05:30"
    assert result.claimed_outcome["moved"] is False
    assert result.claimed_outcome["confirmation_required"] is True


def test_confirm_move_stops_on_partial_update_and_reports_attempt_count() -> None:
    events = [
        AUDIT.copy(),
        event("after-1", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
        event("after-2", "2026-09-22T13:00:00+00:00", "2026-09-22T14:00:00+00:00"),
    ]
    tools = WritableFakeCalendarTools(events=events, fail_update_at=2)
    agent = CalendarAgent(tools, allow_disposable_writes=True)
    task = AgentTask(
        "calendar.move_after_audit",
        "Move everything after the audit",
        {"calendar_id": "calendar-1", "shift": SHIFT},
    )
    proposal = agent.run(task)

    result = agent.confirm_move(task, proposal, "Confirm 1 hour")

    assert proposal.status is ResultStatus.PLANNED
    assert result.status is ResultStatus.FAILED
    assert tools.update_attempts == 2
    assert result.claimed_outcome["updates_attempted"] == 2
    assert result.claimed_outcome["updates_completed"] == 1
    assert "partial test changes" in result.summary
    assert "before any retry" in result.summary


def test_confirm_move_detects_non_target_metadata_changes() -> None:
    events = [
        AUDIT.copy(),
        event("before", "2026-09-22T08:00:00+00:00", "2026-09-22T09:00:00+00:00"),
        event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00"),
    ]
    tools = WritableFakeCalendarTools(
        events=events,
        mutate_unplanned_event=True,
    )
    agent = CalendarAgent(tools, allow_disposable_writes=True)
    task = AgentTask(
        "calendar.move_after_audit",
        "Move everything after the audit",
        {"calendar_id": "calendar-1", "shift": SHIFT},
    )
    proposal = agent.run(task)

    result = agent.confirm_move(task, proposal, "Confirm 1 hour")

    assert proposal.status is ResultStatus.PLANNED
    assert result.status is ResultStatus.FAILED
    assert "outside the confirmed move set changed" in result.summary


def test_missing_or_invalid_timezone_fails_closed() -> None:
    target = exact_audit(
        start_at="2026-09-19T09:30:00",
        end_at="2026-09-19T12:30:00",
        timezone=None,
    )
    after = {
        **event("work-uuid", "2026-09-19T13:30:00", "2026-09-19T14:30:00"),
        "timezone": None,
        "number": "EVT-2026-00008",
    }
    unconfigured, _ = run_with_exact_audit_pages(
        [page(target, total=1, limit=10, offset=0)],
        [page(target, after, total=2, limit=10, offset=0)],
    )
    invalid_fallback, _ = run_with_exact_audit_pages(
        [page(target, total=1, limit=10, offset=0)],
        [page(target, after, total=2, limit=10, offset=0)],
        fallback_timezone="Mars/Phobos",
    )
    invalid_event_zone, _ = run_with_exact_audit_pages(
        [page(exact_audit(), total=1, limit=10, offset=0)],
        [page(
            exact_audit(),
            {**after, "timezone": "Mars/Phobos"},
            total=2,
            limit=10,
            offset=0,
        )],
        fallback_timezone="Asia/Kolkata",
    )

    assert unconfigured.status is ResultStatus.NEEDS_CLARIFICATION
    assert invalid_fallback.status is ResultStatus.NEEDS_CLARIFICATION
    assert invalid_event_zone.status is ResultStatus.NEEDS_CLARIFICATION
    assert not unconfigured.rescheduling_plan
    assert not invalid_fallback.rescheduling_plan
    assert not invalid_event_zone.rescheduling_plan


def test_mixed_aware_and_naive_interval_preserves_offset_instant_and_duration() -> None:
    result = run([
        {
            **AUDIT,
            "start_at": "2026-09-22T09:00:00",
            "end_at": "2026-09-22T10:00:00+05:30",
            "timezone": None,
        },
        {
            **event(
                "mixed",
                "2026-09-22T11:00:00",
                "2026-09-22T06:30:00Z",
                timezone_name="UTC",
            ),
            "timezone": None,
        },
    ], fallback_timezone="Asia/Kolkata")

    assert result.status is ResultStatus.PLANNED
    move = result.rescheduling_plan[0]
    assert move.original_start_at == "2026-09-22T11:00:00+05:30"
    assert move.original_end_at == "2026-09-22T06:30:00Z"
    assert move.proposed_start_at == "2026-09-22T12:00:00+05:30"
    assert move.duration_seconds == 3600


def test_fallback_events_are_sorted_chronologically_and_conflicts_are_checked() -> None:
    result = run([
        {
            **AUDIT,
            "start_at": "2026-09-22T09:00:00",
            "end_at": "2026-09-22T10:00:00",
            "timezone": None,
        },
        {
            **event("later", "2026-09-22T13:15:00Z", "2026-09-22T13:45:00Z"),
            "timezone": None,
        },
        {
            **event("earlier", "2026-09-22T11:00:00", "2026-09-22T11:30:00"),
            "timezone": None,
        },
    ], fallback_timezone="Asia/Kolkata")

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["earlier", "later"]
    assert [move.duration_seconds for move in result.rescheduling_plan] == [1800, 1800]

    conflict = run([
        {
            **AUDIT,
            "start_at": "2026-09-22T09:00:00",
            "end_at": "2026-09-22T10:00:00",
            "timezone": None,
        },
        {
            **event("overlap-audit", "2026-09-22T09:30:00", "2026-09-22T11:45:00"),
            "timezone": None,
        },
        {
            **event("after", "2026-09-22T11:00:00", "2026-09-22T12:00:00"),
            "timezone": None,
        },
    ], shift=timedelta(minutes=30), fallback_timezone="Asia/Kolkata")

    assert conflict.status is ResultStatus.NEEDS_CLARIFICATION
    assert not conflict.rescheduling_plan
    assert "conflict" in conflict.summary.lower()


def test_complete_pagination_plans_after_short_first_page() -> None:
    before = event("before", "2026-09-22T08:00:00+00:00", "2026-09-22T08:30:00+00:00")
    after = event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")
    result, tools = run_with_pages(
        [page(AUDIT, total=1, limit=2, offset=0)],
        [
            page(AUDIT, before, total=3, limit=2, offset=0),
            page(after, total=3, limit=2, offset=2),
        ],
    )

    assert result.status is ResultStatus.PLANNED
    assert [move.event_id for move in result.rescheduling_plan] == ["after"]
    assert [call.arguments["offset"] for call in tools.trace] == [0, 0, 2]
    assert all(call.arguments["calendar_id"] == "calendar-1" for call in tools.trace)


def test_changed_total_between_pages_fails_closed() -> None:
    after = event("after", "2026-09-22T11:00:00+00:00", "2026-09-22T12:00:00+00:00")
    result, _ = run_with_pages(
        [page(AUDIT, total=1, limit=2, offset=0)],
        [
            page(AUDIT, total=2, limit=1, offset=0),
            page(after, total=3, limit=1, offset=1),
        ],
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "total changed" in " ".join(result.ambiguities)


@pytest.mark.parametrize(
    "invalid_page",
    [
        page(AUDIT, total=2, limit=1, offset=1),
        page(AUDIT, total=2, limit=1001, offset=0),
        page(AUDIT, total=0, limit=1, offset=0),
    ],
)
def test_invalid_page_offsets_and_limits_fail_closed(
    invalid_page: CalendarEventPage,
) -> None:
    result, _ = run_with_pages(
        [page(AUDIT, total=1, limit=2, offset=0)],
        [invalid_page],
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_repeated_event_page_fails_closed() -> None:
    repeated = {
        "title": "Repeated page record",
        "start_at": "2026-09-22T11:00:00+00:00",
        "end_at": "2026-09-22T12:00:00+00:00",
        "timezone": "UTC",
        "calendar_id": "calendar-1",
    }
    result, _ = run_with_pages(
        [page(AUDIT, total=1, limit=1, offset=0)],
        [
            page(repeated, total=2, limit=1, offset=0),
            page(repeated, total=2, limit=1, offset=1),
        ],
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "repeated page" in " ".join(result.ambiguities)


def test_empty_page_before_reported_total_fails_closed() -> None:
    result, _ = run_with_pages(
        [page(AUDIT, total=1, limit=1, offset=0)],
        [page(total=2, limit=1, offset=0)],
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan


def test_fetch_failure_mid_pagination_fails_closed() -> None:
    result, tools = run_with_pages(
        [page(AUDIT, total=1, limit=1, offset=0)],
        [
            page(AUDIT, total=2, limit=1, offset=0),
            CalendarEventListError("synthetic second-page failure"),
        ],
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "incomplete" in result.summary.lower()
    assert len(tools.trace) == 3


def test_missing_calendar_scope_does_not_query_global_events() -> None:
    tools = FakeCalendarTools(events=[AUDIT])
    result = CalendarAgent(tools).run(
        AgentTask(
            "calendar.move_after_audit",
            "Move everything after the audit",
            {"shift": SHIFT},
        )
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert tools.trace == []


def test_legacy_list_without_page_metadata_cannot_produce_a_plan() -> None:
    tools = FakeCalendarTools(events=[AUDIT, event(
        "after",
        "2026-09-22T11:00:00+00:00",
        "2026-09-22T12:00:00+00:00",
    )])
    tools.list_events_page = None

    result = CalendarAgent(tools).run(
        AgentTask(
            "calendar.move_after_audit",
            "Move everything after the audit",
            {"shift": SHIFT, "calendar_id": "calendar-1"},
        )
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "pagination metadata" in " ".join(result.ambiguities)
    assert tools.trace == []


def test_page_count_safety_limit_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(move_after_audit, "_MAX_EVENT_LIST_PAGES", 1)
    result, _ = run_with_pages(
        [page(AUDIT, total=1, limit=1, offset=0)],
        [page(AUDIT, total=2, limit=1, offset=0)],
    )

    assert result.status is ResultStatus.NEEDS_CLARIFICATION
    assert not result.rescheduling_plan
    assert "page safety limit" in " ".join(result.ambiguities)
