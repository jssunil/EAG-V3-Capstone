from copy import deepcopy

from calendar_agent.models import EventMove
from harnesses.suryodaya_harness.assertions import (
    read_backend_calendar_snapshot,
    verify_backend_rescheduling_state,
)

from tests.unit.fakes import FakeCalendarTools


CALENDAR_ID = "test-calendar"
BEFORE_EVENTS = [
    {
        "id": "audit",
        "title": "Synthetic confirmed audit",
        "start_at": "2026-10-30T09:00:00+00:00",
        "end_at": "2026-10-30T10:00:00+00:00",
        "calendar_id": CALENDAR_ID,
        "status": "confirmed",
    },
    {
        "id": "moving",
        "title": "Synthetic event to move",
        "start_at": "2026-10-30T12:00:00+00:00",
        "end_at": "2026-10-30T13:00:00+00:00",
        "calendar_id": CALENDAR_ID,
        "status": "confirmed",
    },
    {
        "id": "untouched",
        "title": "Synthetic event that must not change",
        "start_at": "2026-10-30T08:00:00+00:00",
        "end_at": "2026-10-30T08:30:00+00:00",
        "calendar_id": CALENDAR_ID,
        "status": "confirmed",
    },
]
MOVE = EventMove(
    event_id="moving",
    title="Synthetic event to move",
    original_start_at="2026-10-30T12:00:00+00:00",
    original_end_at="2026-10-30T13:00:00+00:00",
    proposed_start_at="2026-10-30T12:30:00+00:00",
    proposed_end_at="2026-10-30T13:30:00+00:00",
    duration_seconds=3600,
    calendar_id=CALENDAR_ID,
    timezone="UTC",
)


def _snapshots(after_events: list[dict[str, object]]):
    tools = FakeCalendarTools(events=deepcopy(BEFORE_EVENTS))
    before = read_backend_calendar_snapshot(tools, calendar_id=CALENDAR_ID)
    tools.events = deepcopy(after_events)
    after = read_backend_calendar_snapshot(tools, calendar_id=CALENDAR_ID)
    return before, after


def test_backend_verifier_accepts_exact_move_and_unchanged_non_targets() -> None:
    after_events = deepcopy(BEFORE_EVENTS)
    after_events[1]["start_at"] = MOVE.proposed_start_at
    after_events[1]["end_at"] = MOVE.proposed_end_at
    before, after = _snapshots(after_events)

    verification = verify_backend_rescheduling_state(
        before,
        after,
        expected_moves=(MOVE,),
    )

    assert verification.passed
    assert verification.integrity_ok


def test_backend_verifier_rejects_non_target_database_change() -> None:
    after_events = deepcopy(BEFORE_EVENTS)
    after_events[1]["start_at"] = MOVE.proposed_start_at
    after_events[1]["end_at"] = MOVE.proposed_end_at
    after_events[2]["title"] = "Unexpected backend edit"
    before, after = _snapshots(after_events)

    verification = verify_backend_rescheduling_state(
        before,
        after,
        expected_moves=(MOVE,),
    )

    assert not verification.passed
    assert "A non-target backend event changed." in verification.details


def test_backend_verifier_rejects_incorrect_target_interval() -> None:
    after_events = deepcopy(BEFORE_EVENTS)
    after_events[1]["start_at"] = "2026-10-30T12:45:00+00:00"
    after_events[1]["end_at"] = "2026-10-30T13:45:00+00:00"
    before, after = _snapshots(after_events)

    verification = verify_backend_rescheduling_state(
        before,
        after,
        expected_moves=(MOVE,),
    )

    assert not verification.passed
    assert "A moved event's backend interval does not match the expected interval." in verification.details
