from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from calendar_agent.models import CalendarEventRef, EventAttendee, PartyRecord
from harnesses.keystone_harness import (
    CollectionEvidence,
    CompletenessResult,
    CompletenessStatus,
    DiscoveredTool,
    KeystoneCalendarAdapter,
    KeystoneClient,
    KeystoneConfig,
    KeystoneOperationNotAllowedError,
    KeystoneProtocolError,
    KeystoneTransportError,
    ToolDiscoveryResult,
    map_event_attendee_record,
)


_LIST_FILTERS = {
    "Party.list": {
        "job_title": {"type": "string"},
        "email": {"type": "string", "format": "email"},
    },
    "Calendar.list": {
        "id": {"type": "string", "minLength": 1},
        "owner_party_id": {"type": "string"},
        "timezone": {"type": "string"},
    },
    "CalendarEvent.list": {
        "search": {"type": "string"},
        "calendar_id": {"type": "string"},
        "start_at": {"type": "string"},
        "end_at": {"type": "string"},
    },
    "EventAttendee.list": {
        "event_id": {"type": "string"},
        "email": {"type": "string", "format": "email"},
        "party_id": {"type": "string"},
        "role": {"type": "string", "enum": ["chair", "req", "opt"]},
        "status": {
            "type": "string",
            "enum": ["needs_action", "accepted", "declined", "tentative"],
        },
    },
}


def _tool(name: str, *, output_schema: dict[str, Any] | None = None) -> DiscoveredTool:
    properties = {
        "limit": {
            "type": "integer",
            "minimum": 0,
            "maximum": 1000,
            "default": 20,
        },
        "offset": {"type": "integer", "minimum": 0, "default": 0},
        **_LIST_FILTERS.get(name, {}),
    }
    return DiscoveredTool(
        name=name,
        read_only_hint=True,
        description=f"List {name.split('.')[0]} records with optional filters",
        input_schema={
            "type": "object",
            "properties": properties,
            "required": [],
            "additionalProperties": False,
        },
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
        },
        output_schema=output_schema or {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                        },
                        "required": ["id"],
                        "additionalProperties": True,
                    },
                },
                "total": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 0},
                "offset": {"type": "integer", "minimum": 0},
            },
            "required": ["data", "total", "limit", "offset"],
            "additionalProperties": True,
        },
        meta={
            "agentswitch": {
                "permission": "read",
                "discovery": "caller",
                "risk": "READ",
            }
        },
    )


def _catalog(names: tuple[str, ...] = tuple(_LIST_FILTERS)) -> ToolDiscoveryResult:
    return ToolDiscoveryResult(
        tools=tuple(_tool(name) for name in names),
        collection_evidence=CollectionEvidence(()),
        completeness=CompletenessResult(
            CompletenessStatus.UNKNOWN,
            observed_record_count=None,
            expected_record_count=None,
            reasons=("synthetic catalog fixture",),
        ),
    )


def _page(records: list[dict[str, Any]], *, total: int, limit: int, offset: int) -> dict[str, Any]:
    return {"data": records, "total": total, "limit": limit, "offset": offset}


class SyntheticReadClient:
    def __init__(
        self,
        pages: Mapping[tuple[str, int], Mapping[str, Any] | Exception] | None,
    ) -> None:
        self.pages = pages


class SyntheticDiscoveryResponse:
    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class SyntheticDiscoverySession:
    def __init__(self, descriptors: tuple[DiscoveredTool, ...]) -> None:
        self.descriptors = descriptors
        self.methods: list[str] = []

    def post(self, url: str, **kwargs: Any) -> SyntheticDiscoveryResponse:
        request = kwargs["json"]
        self.methods.append(request["method"])
        if request["method"] == "initialize":
            return SyntheticDiscoveryResponse({
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "serverInfo": {"name": "test-server", "version": "1"},
                },
            })
        if request["method"] == "notifications/initialized":
            return SyntheticDiscoveryResponse({})
        if request["method"] != "tools/list":
            raise AssertionError("synthetic discovery session cannot call tools")
        return SyntheticDiscoveryResponse({
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {
                "tools": [
                    {
                        "name": descriptor.name,
                        "description": descriptor.description,
                        "inputSchema": descriptor.input_schema,
                        "outputSchema": descriptor.output_schema,
                        "annotations": descriptor.annotations,
                        "_meta": descriptor.meta,
                    }
                    for descriptor in self.descriptors
                ],
                "nextCursor": None,
            },
        })


class HostedReadSession(SyntheticDiscoverySession):
    def __init__(
        self,
        descriptors: tuple[DiscoveredTool, ...],
        pages: Mapping[tuple[str, int], Mapping[str, Any]],
    ) -> None:
        super().__init__(descriptors)
        self.pages = pages

    def post(self, url: str, **kwargs: Any) -> SyntheticDiscoveryResponse:
        request = kwargs["json"]
        if request["method"] in {
            "initialize",
            "notifications/initialized",
            "tools/list",
        }:
            return super().post(url, **kwargs)
        if request["method"] != "tools/call":
            raise AssertionError("hosted read fixture received an unexpected method")
        self.methods.append(request["method"])
        name = request["params"]["name"]
        offset = request["params"]["arguments"]["offset"]
        page = self.pages[(name, offset)]
        return SyntheticDiscoveryResponse({
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {
                "content": [],
                "structuredContent": page,
            },
        })


def _build_adapter(
    fixture: SyntheticReadClient,
    *,
    catalog: ToolDiscoveryResult | None = None,
) -> tuple[KeystoneCalendarAdapter, SyntheticDiscoverySession]:
    catalog = _catalog() if catalog is None else catalog
    session = SyntheticDiscoverySession(catalog.tools)
    client = KeystoneClient(
        KeystoneConfig(token="offline-test-token"),
        session=session,  # type: ignore[arg-type]
    )
    client.list_tools()
    adapter = KeystoneCalendarAdapter(
        client,
        synthetic_response_pages=fixture.pages,
    )
    return adapter, session


def test_opted_in_hosted_adapter_uses_only_validated_read_pages() -> None:
    catalog = _catalog()
    session = HostedReadSession(
        catalog.tools,
        {
            ("Calendar.list", 0): _page(
                [
                    {
                        "id": "calendar-1",
                        "owner_party_id": "party-1",
                        "timezone": "UTC",
                    }
                ],
                total=1,
                limit=2,
                offset=0,
            )
        },
    )
    client = KeystoneClient(
        KeystoneConfig.for_hosted_runner(
            base_url="http://runner.internal:8000/keystone",
            token="offline-hosted-test-token",
        ),
        session=session,  # type: ignore[arg-type]
        enable_hosted_read_transport=True,
    )
    client.list_tools()
    adapter = KeystoneCalendarAdapter(client)

    result = adapter.list_calendars(filters={"owner_party_id": "party-1"}, limit=2)

    assert result.collection_complete
    assert result.records is not None
    assert result.records[0].value is not None
    assert result.records[0].value.id == "calendar-1"
    assert session.methods == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    ]


@pytest.mark.parametrize(
    ("method", "tool_name", "filters", "record", "model_type"),
    [
        (
            "list_parties",
            "Party.list",
            {"job_title": "Plant Manager"},
            {
                "id": "party-1",
                "name": "Plant Manager",
                "email": "manager@example.test",
                "job_title": "Plant Manager",
            },
            PartyRecord,
        ),
        (
            "list_calendars",
            "Calendar.list",
            {"owner_party_id": "party-1"},
            {
                "id": "calendar-1",
                "owner_party_id": "party-1",
                "timezone": "Asia/Kolkata",
            },
            None,
        ),
        (
            "list_events",
            "CalendarEvent.list",
            {"calendar_id": "calendar-1"},
            {
                "id": "event-1",
                "title": "Review",
                "start_at": "2026-10-17T09:00:00+05:30",
                "end_at": "2026-10-17T09:30:00+05:30",
                "calendar_id": "calendar-1",
                "status": "confirmed",
                "timezone": "Asia/Kolkata",
            },
            CalendarEventRef,
        ),
        (
            "list_event_attendees",
            "EventAttendee.list",
            {"event_id": "event-1"},
            {
                "id": "attendee-1",
                "event_id": "event-1",
                "email": "manager@example.test",
                "party_id": "party-1",
                "role": "req",
                "status": "accepted",
                "is_organizer": False,
            },
            EventAttendee,
        ),
    ],
)
def test_each_allowlisted_list_operation_maps_declared_fields_and_forwards_filters(
    method: str,
    tool_name: str,
    filters: dict[str, Any],
    record: dict[str, Any],
    model_type: type[Any] | None,
) -> None:
    fixture = SyntheticReadClient(
        {(tool_name, 0): _page([record], total=1, limit=2, offset=0)}
    )
    adapter, session = _build_adapter(fixture)

    result = getattr(adapter, method)(filters=filters, limit=2)

    assert [request.arguments for request in result.requests] == [
        {**filters, "limit": 2, "offset": 0}
    ]
    assert session.methods == ["tools/list"]
    assert result.collection_complete
    assert result.is_complete
    assert result.records is not None
    assert result.records[0].source_id == record["id"]
    if model_type is not None:
        assert isinstance(result.records[0].value, model_type)
    else:
        assert result.records[0].value.owner_party_id == "party-1"


def test_multpage_list_forwards_offset_and_preserves_pagination_evidence() -> None:
    fixture = SyntheticReadClient(
        {
            ("Party.list", 0): _page(
                [
                    {
                        "id": "party-1",
                        "name": "One",
                        "email": "one@example.test",
                        "job_title": "Plant Manager",
                    },
                    {
                        "id": "party-2",
                        "name": "Two",
                        "email": "two@example.test",
                        "job_title": "Plant Manager",
                    },
                ],
                total=3,
                limit=2,
                offset=0,
            ),
            ("Party.list", 2): _page(
                [{
                    "id": "party-3",
                    "name": "Three",
                    "email": "three@example.test",
                    "job_title": "Plant Manager",
                }],
                total=3,
                limit=2,
                offset=2,
            ),
        }
    )
    adapter, session = _build_adapter(fixture)

    result = adapter.list_parties(filters={"job_title": "Plant Manager"}, limit=2)

    assert [request.arguments["offset"] for request in result.requests] == [0, 2]
    assert all(request.arguments["limit"] == 2 for request in result.requests)
    assert all(
        request.arguments["job_title"] == "Plant Manager"
        for request in result.requests
    )
    assert session.methods == ["tools/list"]
    assert [request.arguments["offset"] for request in result.requests] == [0, 2]
    assert result.collection_complete
    assert result.is_complete
    assert result.pagination.completeness.expected_record_count == 3
    assert len(result.records or ()) == 3


def test_unlisted_or_undiscovered_operations_are_rejected_before_read() -> None:
    adapter, session = _build_adapter(
        SyntheticReadClient({}),
        catalog=_catalog(("Party.create",)),
    )

    with pytest.raises(KeystoneOperationNotAllowedError):
        adapter.list_parties(filters={"job_title": "Plant Manager"})

    assert session.methods == ["tools/list"]
    assert not hasattr(adapter, "call_tool")
    assert not hasattr(adapter, "free_busy")
    assert not hasattr(adapter, "create_event")
    assert not hasattr(adapter, "update_event")


def test_filters_not_declared_by_the_catalog_are_rejected() -> None:
    adapter, session = _build_adapter(SyntheticReadClient({}))

    with pytest.raises(ValueError, match="not allowed"):
        adapter.list_parties(filters={"invented_field": "value"})
    with pytest.raises(ValueError, match="limit and offset"):
        adapter.list_parties(filters={"offset": 5})

    assert session.methods == ["tools/list"]


def test_list_output_schema_must_declare_pagination_and_record_ids() -> None:
    malformed = _tool(
        "Party.list",
        output_schema={
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                },
            },
            "required": ["data"],
        },
    )
    catalog = _catalog(("Party.list",))
    catalog = ToolDiscoveryResult(
        tools=(malformed,),
        collection_evidence=catalog.collection_evidence,
        completeness=catalog.completeness,
    )
    adapter, session = _build_adapter(SyntheticReadClient({}), catalog=catalog)

    with pytest.raises(KeystoneProtocolError, match="pagination contract"):
        adapter.list_parties()

    assert session.methods == ["tools/list"]


def test_output_schema_missing_page_metadata_fails_before_pagination() -> None:
    fixture = SyntheticReadClient(
        {
            ("Party.list", 0): {
                "data": [{
                    "id": "party-1",
                    "name": "Manager",
                    "email": "manager@example.test",
                    "job_title": "Plant Manager",
                }],
                "limit": 1,
                "offset": 0,
            }
        }
    )
    adapter, session = _build_adapter(fixture)

    with pytest.raises(KeystoneProtocolError, match="does not satisfy"):
        adapter.list_parties(limit=1)
    assert session.methods == ["tools/list"]


def test_followup_read_failure_returns_incomplete_evidence_not_partial_records() -> None:
    fixture = SyntheticReadClient(
        {
            ("Party.list", 0): _page(
                [{
                    "id": "party-1",
                    "name": "Manager",
                    "email": "manager@example.test",
                    "job_title": "Plant Manager",
                }],
                total=2,
                limit=1,
                offset=0,
            ),
            ("Party.list", 1): KeystoneTransportError("synthetic transport failure"),
        }
    )
    adapter, session = _build_adapter(fixture)

    result = adapter.list_parties(limit=1)

    assert [request.arguments["offset"] for request in result.requests] == [0, 1]
    assert result.read_error == "KeystoneTransportError"
    assert result.pagination.completeness.status is CompletenessStatus.INCOMPLETE
    assert result.records is None
    assert session.methods == ["tools/list"]


@pytest.mark.parametrize(
    "responses",
    [
        {
            0: _page(
                [{"id": "event-1"}, {"id": "event-1"}],
                total=2,
                limit=2,
                offset=0,
            )
        },
        {
            0: _page([{"id": "event-1"}, {"id": "event-2"}], total=3, limit=2, offset=0),
            2: _page([{"id": "event-2"}], total=3, limit=2, offset=1),
        },
        {
            0: _page([{"id": "event-1"}], total=2, limit=1, offset=0),
            1: _page([{"id": "event-2"}], total=3, limit=1, offset=1),
        },
    ],
)
def test_duplicate_overlap_and_changing_total_pages_fail_closed(
    responses: dict[int, dict[str, Any]],
) -> None:
    fixture = SyntheticReadClient(
        {("CalendarEvent.list", offset): response for offset, response in responses.items()}
    )
    adapter, session = _build_adapter(fixture)

    result = adapter.list_events(limit=2)

    assert result.pagination.completeness.status is CompletenessStatus.INCOMPLETE
    assert result.records is None
    assert session.methods == ["tools/list"]


def test_party_mapping_does_not_invent_missing_name_email_or_job_title() -> None:
    fixture = SyntheticReadClient({
        ("Party.list", 0): _page([{"id": "party-1"}], total=1, limit=1, offset=0)
    })
    adapter, _ = _build_adapter(fixture)

    result = adapter.list_parties(limit=1)

    assert result.collection_complete
    assert not result.is_complete
    assert result.records is not None
    mapping = result.records[0]
    assert mapping.source_id == "party-1"
    assert mapping.value is None
    assert set(mapping.errors) == {
        "Required field name is absent.",
        "Required field email is absent.",
        "Required field job_title is absent.",
    }


def test_calendar_event_timezone_is_optional_for_explicit_fallback_resolution() -> None:
    fixture = SyntheticReadClient({
        ("CalendarEvent.list", 0): _page(
            [{
                "id": "event-1",
                "title": "Review",
                "start_at": "2026-10-17T09:00:00+05:30",
                "end_at": "2026-10-17T09:30:00+05:30",
                "calendar_id": "calendar-1",
                "status": "confirmed",
            }],
            total=1,
            limit=1,
            offset=0,
        )
    })
    adapter, _ = _build_adapter(fixture)

    result = adapter.list_events(limit=1)

    assert result.records is not None
    mapping = result.records[0]
    assert mapping.value is not None
    assert mapping.value.timezone is None
    assert "timezone" not in mapping.unresolved_fields
    assert result.is_complete


def test_malformed_event_fields_are_reported_without_a_fabricated_model() -> None:
    fixture = SyntheticReadClient({
        ("CalendarEvent.list", 0): _page(
            [{
                "id": "event-1",
                "title": "Review",
                "start_at": 123,
                "end_at": "2026-10-17T09:30:00+05:30",
            }],
            total=1,
            limit=1,
            offset=0,
        )
    })
    adapter, _ = _build_adapter(fixture)

    result = adapter.list_events(limit=1)

    assert result.collection_complete
    assert result.records is not None
    assert result.records[0].value is None
    assert any("start_at" in error for error in result.records[0].errors)
    assert not result.is_complete


@pytest.mark.parametrize(
    ("source_role", "organizer", "expected_role"),
    [
        ("req", False, "required"),
        ("opt", False, "optional"),
        ("chair", False, "chair"),
        (None, True, "organizer"),
    ],
)
def test_attendee_roles_map_only_from_supported_explicit_values(
    source_role: str | None,
    organizer: bool,
    expected_role: str,
) -> None:
    record: dict[str, Any] = {
        "id": "attendee-1",
        "event_id": "event-1",
        "email": "person@example.test",
        "status": "accepted",
        "is_organizer": organizer,
    }
    if source_role is not None:
        record["role"] = source_role

    result = map_event_attendee_record(record)

    assert result.source_id == "attendee-1"
    assert result.value is not None
    assert result.value.role == expected_role


@pytest.mark.parametrize(
    "record",
    [
        {
            "id": "attendee-1",
            "event_id": "event-1",
            "email": "person@example.test",
            "role": "organizer",
            "status": "accepted",
        },
        {
            "id": "attendee-1",
            "event_id": "event-1",
            "email": "person@example.test",
            "role": "req",
            "is_organizer": True,
            "status": "accepted",
        },
        {
            "id": "attendee-1",
            "event_id": "event-1",
            "email": "person@example.test",
            "role": "opt",
            "status": "unknown",
        },
    ],
)
def test_unsupported_or_conflicting_attendee_fields_remain_unresolved(
    record: dict[str, Any],
) -> None:
    result = map_event_attendee_record(record)

    assert result.value is None
    assert result.errors
    assert not result.is_resolved


def test_real_keystone_client_keeps_business_reads_disabled() -> None:
    adapter, session = _build_adapter(SyntheticReadClient(None))

    result = adapter.list_parties(limit=1)

    assert result.read_error == "KeystoneOperationNotAllowedError"
    assert result.records is None
    assert result.pagination.completeness.status is CompletenessStatus.UNKNOWN
    assert session.methods == ["tools/list"]


def test_adapter_rejects_custom_reader_objects() -> None:
    with pytest.raises(TypeError, match="concrete KeystoneClient"):
        KeystoneCalendarAdapter(SyntheticReadClient({}))  # type: ignore[arg-type]
