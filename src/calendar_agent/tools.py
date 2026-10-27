"""Explicit Calendar tool boundary.

The adapter exposes read operations and records every call. Mutation methods are
present as a visible boundary, but fail closed and are never used by the first
planning-only goal handlers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, Sequence
from urllib.parse import urlsplit

from .agentswitch_client import (
    AgentSwitchError,
    AgentSwitchAuthenticationError,
    AgentSwitchBudgetExhaustedError,
    AgentSwitchClient,
)
from .models import (
    CalendarEventPage,
    CalendarMutationError,
    MutationDisabledError,
    ToolCallRecord,
)


_MAX_EVENT_PAGE_SIZE = 1000


class CalendarEventListError(RuntimeError):
    """Raised when a CalendarEvent listing cannot be proven complete."""


class CalendarTool(Protocol):
    """Operations the agent may use without knowing about HTTP or MCP."""

    trace: list[ToolCallRecord]

    def find_parties(self, **filters: Any) -> list[dict[str, Any]]: ...

    def list_calendars(self, **filters: Any) -> list[dict[str, Any]]: ...

    def get_calendar(self, calendar_id: str) -> dict[str, Any]: ...

    def list_events(self, **filters: Any) -> list[dict[str, Any]]: ...

    def get_event(self, event_id: str) -> dict[str, Any]: ...

    def free_busy(
        self,
        *,
        start: datetime,
        end: datetime,
        emails: Sequence[str] | None = None,
        calendar_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]: ...

    def list_availability_rules(self, **filters: Any) -> list[dict[str, Any]]: ...

    def list_scheduling_preferences(self, **filters: Any) -> list[dict[str, Any]]: ...

    def create_event(self, **arguments: Any) -> dict[str, Any]: ...

    def update_event(self, **arguments: Any) -> dict[str, Any]: ...

    def delete_event(self, event_id: str) -> dict[str, Any]: ...

    def confirm_event(self, event_id: str) -> dict[str, Any]: ...

    def mark_event_tentative(self, event_id: str) -> dict[str, Any]: ...

    def cancel_event(self, event_id: str) -> dict[str, Any]: ...


class ReadOnlyCalendarTool:
    """Delegate reads while preventing mutations from reaching the wrapped tool."""

    def __init__(self, delegate: CalendarTool) -> None:
        self._delegate = delegate

    @property
    def disposable_mutations_enabled(self) -> bool:
        return False

    @property
    def trace(self) -> list[ToolCallRecord]:
        return self._delegate.trace

    def __getattr__(self, name: str) -> Any:
        if name not in {"find_parties_page", "list_events_page"}:
            raise AttributeError(name)
        return getattr(self._delegate, name)

    def find_parties(self, **filters: Any) -> list[dict[str, Any]]:
        return self._delegate.find_parties(**filters)

    def list_calendars(self, **filters: Any) -> list[dict[str, Any]]:
        return self._delegate.list_calendars(**filters)

    def get_calendar(self, calendar_id: str) -> dict[str, Any]:
        return self._delegate.get_calendar(calendar_id)

    def list_events(self, **filters: Any) -> list[dict[str, Any]]:
        return self._delegate.list_events(**filters)

    def list_events_page(self, **filters: Any) -> CalendarEventPage:
        return self._delegate.list_events_page(**filters)

    def get_event(self, event_id: str) -> dict[str, Any]:
        return self._delegate.get_event(event_id)

    def free_busy(
        self,
        *,
        start: datetime,
        end: datetime,
        emails: Sequence[str] | None = None,
        calendar_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        return self._delegate.free_busy(
            start=start,
            end=end,
            emails=emails,
            calendar_ids=calendar_ids,
        )

    def list_availability_rules(self, **filters: Any) -> list[dict[str, Any]]:
        return self._delegate.list_availability_rules(**filters)

    def list_scheduling_preferences(self, **filters: Any) -> list[dict[str, Any]]:
        return self._delegate.list_scheduling_preferences(**filters)

    def _mutation_disabled(self, tool: str, arguments: dict[str, Any]) -> None:
        self.trace.append(ToolCallRecord(tool, arguments, mutating=True, succeeded=False, error="disabled"))
        raise MutationDisabledError(f"Mutation tool disabled: {tool}")

    def create_event(self, **arguments: Any) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.create", arguments)
        return {}

    def update_event(self, **arguments: Any) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.update", arguments)
        return {}

    def delete_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.delete", {"id": event_id})
        return {}

    def confirm_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.confirm", {"id": event_id})
        return {}

    def mark_event_tentative(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.mark_tentative", {"id": event_id})
        return {}

    def cancel_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.cancel", {"id": event_id})
        return {}


class AgentSwitchCalendarTools:
    """Translate typed calendar operations into MCP tool calls."""

    def __init__(
        self,
        client: AgentSwitchClient,
        *,
        enable_disposable_mutations: bool = False,
    ) -> None:
        if type(enable_disposable_mutations) is not bool:
            raise TypeError("enable_disposable_mutations must be a boolean")
        if enable_disposable_mutations:
            hostname = urlsplit(client.config.base_url).hostname
            if hostname in {
                "agentswitch.theschoolofai.in",
                "class.agentswitch.theschoolofai.in",
            }:
                raise MutationDisabledError(
                    "Calendar mutations are disabled for shared live AgentSwitch origins."
                )
        self._client = client
        self._disposable_mutations_enabled = enable_disposable_mutations
        self.trace: list[ToolCallRecord] = []

    @property
    def disposable_mutations_enabled(self) -> bool:
        return self._disposable_mutations_enabled

    def _read(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self._client.call_tool(tool, arguments)
        except Exception as exc:
            self.trace.append(ToolCallRecord(tool, arguments, error=type(exc).__name__, succeeded=False))
            raise
        self.trace.append(ToolCallRecord(tool, arguments))
        return result

    @staticmethod
    def _records(result: dict[str, Any]) -> list[dict[str, Any]]:
        structured = result.get("structuredContent", {})
        records = structured.get("data", []) if isinstance(structured, dict) else []
        return records if isinstance(records, list) else []

    def find_parties(self, **filters: Any) -> list[dict[str, Any]]:
        return self._records(self._read("Party.list", filters))

    def list_calendars(self, **filters: Any) -> list[dict[str, Any]]:
        return self._records(self._read("Calendar.list", filters))

    def get_calendar(self, calendar_id: str) -> dict[str, Any]:
        return self._read("Calendar.get", {"id": calendar_id}).get("structuredContent", {})

    def list_events(self, **filters: Any) -> list[dict[str, Any]]:
        return self._records(self._read("CalendarEvent.list", filters))

    def list_events_page(self, **filters: Any) -> CalendarEventPage:
        requested_limit = filters.get("limit")
        requested_offset = filters.get("offset")
        if (
            type(requested_limit) is not int
            or requested_limit < 1
            or requested_limit > _MAX_EVENT_PAGE_SIZE
            or type(requested_offset) is not int
            or requested_offset < 0
        ):
            raise CalendarEventListError(
                "CalendarEvent pagination requires a positive limit and non-negative offset."
            )

        try:
            result = self._read("CalendarEvent.list", filters)
        except (AgentSwitchAuthenticationError, AgentSwitchBudgetExhaustedError):
            raise
        except AgentSwitchError as exc:
            raise CalendarEventListError(
                "CalendarEvent page retrieval failed."
            ) from exc

        try:
            structured = result.get("structuredContent")
            if not isinstance(structured, dict):
                raise ValueError("structuredContent is missing")
            records = structured.get("data")
            total = structured.get("total")
            limit = structured.get("limit")
            offset = structured.get("offset")
            if not isinstance(records, list):
                raise ValueError("data is missing or not a list")
            page = CalendarEventPage(tuple(records), total, limit, offset)
            page.validate(
                requested_limit=requested_limit,
                requested_offset=requested_offset,
            )
            return page
        except (TypeError, ValueError) as exc:
            error = CalendarEventListError(
                f"CalendarEvent page metadata is invalid: {exc}"
            )
            self._mark_last_read_failed(error)
            raise error from exc

    def get_event(self, event_id: str) -> dict[str, Any]:
        return self._read("CalendarEvent.get", {"id": event_id}).get("structuredContent", {})

    def free_busy(
        self,
        *,
        start: datetime,
        end: datetime,
        emails: Sequence[str] | None = None,
        calendar_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        if emails is not None:
            arguments["emails"] = list(emails)
        if calendar_ids is not None:
            arguments["calendar_ids"] = list(calendar_ids)
        return self._read("endpoint.calendar.free_busy", arguments).get("structuredContent", {})

    def list_availability_rules(self, **filters: Any) -> list[dict[str, Any]]:
        return self._records(self._read("AvailabilityRule.list", filters))

    def list_scheduling_preferences(self, **filters: Any) -> list[dict[str, Any]]:
        return self._records(self._read("SchedulingPreferences.list", filters))

    def _mark_last_read_failed(self, error: Exception) -> None:
        if not self.trace:
            return
        last_call = self.trace[-1]
        if last_call.tool == "CalendarEvent.list" and last_call.succeeded:
            self.trace[-1] = ToolCallRecord(
                tool=last_call.tool,
                arguments=last_call.arguments,
                succeeded=False,
                error=type(error).__name__,
            )

    def _mutation_disabled(self, tool: str, arguments: dict[str, Any]) -> None:
        self.trace.append(ToolCallRecord(tool, arguments, mutating=True, succeeded=False, error="disabled"))
        raise MutationDisabledError(f"Mutation tool disabled: {tool}")

    def _write(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._disposable_mutations_enabled:
            self._mutation_disabled(tool, arguments)
        try:
            result = self._client.call_tool(tool, arguments)
        except AgentSwitchError as exc:
            self.trace.append(
                ToolCallRecord(
                    tool,
                    arguments,
                    mutating=True,
                    succeeded=False,
                    error=type(exc).__name__,
                )
            )
            raise CalendarMutationError(
                f"{tool} request failed; verify state before retrying."
            ) from exc
        structured = result.get("structuredContent")
        if not isinstance(structured, dict) or not isinstance(structured.get("id"), str):
            self.trace.append(
                ToolCallRecord(
                    tool,
                    arguments,
                    mutating=True,
                    succeeded=False,
                    error="unverified_response",
                )
            )
            raise CalendarMutationError(
                f"{tool} returned no verifiable event ID; re-read state before retrying."
            )
        self.trace.append(ToolCallRecord(tool, arguments, mutating=True))
        return structured

    def create_event(self, **arguments: Any) -> dict[str, Any]:
        required = {"calendar_id", "title", "start_at", "end_at"}
        if not required.issubset(arguments):
            raise ValueError("CalendarEvent.create requires calendar_id, title, start_at, and end_at.")
        return self._write("CalendarEvent.create", arguments)

    def update_event(self, **arguments: Any) -> dict[str, Any]:
        if not self._disposable_mutations_enabled:
            self._mutation_disabled("CalendarEvent.update", arguments)
        if set(arguments) - {"id", "start_at", "end_at"}:
            raise ValueError("CalendarEvent.update accepts only ID and scheduled start/end fields.")
        if not {"id", "start_at", "end_at"}.issubset(arguments):
            raise ValueError("CalendarEvent.update requires ID, start_at, and end_at.")
        return self._write("CalendarEvent.update", arguments)

    def delete_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.delete", {"id": event_id})
        return {}

    def confirm_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.confirm", {"id": event_id})
        return {}

    def mark_event_tentative(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.mark_tentative", {"id": event_id})
        return {}

    def cancel_event(self, event_id: str) -> dict[str, Any]:
        self._mutation_disabled("CalendarEvent.cancel", {"id": event_id})
        return {}