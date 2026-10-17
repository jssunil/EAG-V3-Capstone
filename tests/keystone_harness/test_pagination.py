from __future__ import annotations

import pytest

from harnesses.keystone_harness import (
    CompletenessStatus,
    collect_resource_pages,
)


def page(
    ids: tuple[str, ...],
    *,
    total: int,
    limit: int = 2,
    offset: int = 0,
) -> dict[str, object]:
    return {
        "data": [{"id": record_id} for record_id in ids],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def test_empty_resource_collection_is_complete() -> None:
    result = collect_resource_pages([page((), total=0)])

    assert result.is_complete
    assert result.records == ()
    assert result.completeness.observed_record_count == 0
    assert result.completeness.expected_record_count == 0
    assert result.collection_evidence.pages[0].reported_total_count == 0


def test_zero_limit_is_valid_only_for_an_empty_collection() -> None:
    result = collect_resource_pages([page((), total=0, limit=0)])

    assert result.is_complete
    assert result.records == ()

    invalid = collect_resource_pages([page(("r1",), total=1, limit=0)])
    assert invalid.completeness.status is CompletenessStatus.INCOMPLETE
    assert invalid.records is None


def test_single_page_resource_collection_is_complete() -> None:
    result = collect_resource_pages(
        [page(("r1", "r2", "r3"), total=3, limit=5)],
        collection_reference="synthetic:single",
    )

    assert result.is_complete
    assert [record["id"] for record in result.records or ()] == ["r1", "r2", "r3"]
    assert result.collection_evidence.collection_reference == "synthetic:single"


def test_multi_page_resource_collection_is_complete() -> None:
    result = collect_resource_pages(
        [
            page(("r1", "r2"), total=5, limit=2, offset=0),
            page(("r3", "r4"), total=5, limit=2, offset=2),
            page(("r5",), total=5, limit=2, offset=4),
        ]
    )

    assert result.is_complete
    assert [record["id"] for record in result.records or ()] == [
        "r1", "r2", "r3", "r4", "r5"
    ]
    assert result.completeness.observed_record_count == 5
    assert [item.page_id for item in result.collection_evidence.pages] == [
        "offset:0", "offset:2", "offset:4"
    ]


def test_resource_collection_can_exceed_one_thousand_records() -> None:
    first_ids = tuple(f"r{index}" for index in range(1000))
    result = collect_resource_pages(
        [
            page(first_ids, total=1001, limit=1000, offset=0),
            page(("r1000",), total=1001, limit=1000, offset=1000),
        ]
    )

    assert result.is_complete
    assert len(result.records or ()) == 1001


def test_missing_pagination_metadata_is_unknown_and_returns_no_partial_records() -> None:
    result = collect_resource_pages([
        {"data": [{"id": "r1"}], "limit": 1, "offset": 0},
    ])

    assert result.completeness.status is CompletenessStatus.UNKNOWN
    assert result.records is None
    assert any("missing pagination metadata: total" in reason for reason in result.completeness.reasons)


@pytest.mark.parametrize(
    "bad_page",
    [
        {"data": {}, "total": 0, "limit": 1, "offset": 0},
        {"data": [], "total": True, "limit": 1, "offset": 0},
        {"data": [], "total": -1, "limit": 1, "offset": 0},
        {"data": [{"id": "r1"}, {"id": "r2"}], "total": 2, "limit": 1, "offset": 0},
        {"data": [{"id": 1}], "total": 1, "limit": 1, "offset": 0},
        {"data": [{"other": "r1"}], "total": 1, "limit": 1, "offset": 0},
    ],
)
def test_malformed_resource_page_fails_closed(bad_page: dict[str, object]) -> None:
    result = collect_resource_pages([bad_page])

    assert result.completeness.status is CompletenessStatus.INCOMPLETE
    assert result.records is None
    assert result.collection_evidence is not None


def test_empty_page_list_is_not_treated_as_an_empty_collection() -> None:
    result = collect_resource_pages([])

    assert result.completeness.status is CompletenessStatus.UNKNOWN
    assert result.records is None
    assert not result.collection_evidence.pages


def test_premature_termination_is_incomplete() -> None:
    result = collect_resource_pages([
        page(("r1",), total=2, limit=1),
    ])

    assert result.completeness.status is CompletenessStatus.INCOMPLETE
    assert result.records is None
    assert any("final observed page says collection should continue" in reason for reason in result.completeness.reasons)


def test_overlapping_or_skipped_offsets_are_incomplete() -> None:
    overlap = collect_resource_pages([
        page(("r1", "r2"), total=3, limit=2, offset=0),
        page(("r2", "r3"), total=3, limit=2, offset=1),
    ])
    skipped = collect_resource_pages([
        page(("r1",), total=3, limit=1, offset=0),
        page(("r3",), total=3, limit=1, offset=2),
    ])

    assert overlap.completeness.status is CompletenessStatus.INCOMPLETE
    assert skipped.completeness.status is CompletenessStatus.INCOMPLETE
    assert overlap.records is None
    assert skipped.records is None


def test_duplicate_ids_and_repeated_pages_are_incomplete() -> None:
    duplicate_ids = collect_resource_pages([
        page(("r1", "r1"), total=2, limit=2),
    ])
    repeated_page = page(("r1",), total=2, limit=1)
    repeated = collect_resource_pages([repeated_page, repeated_page])

    assert duplicate_ids.completeness.status is CompletenessStatus.INCOMPLETE
    assert duplicate_ids.completeness.duplicate_record_ids == ("r1",)
    assert repeated.completeness.status is CompletenessStatus.INCOMPLETE
    assert repeated.records is None


def test_changing_totals_are_incomplete() -> None:
    result = collect_resource_pages([
        page(("r1",), total=2, limit=1, offset=0),
        page(("r2", "r3"), total=3, limit=2, offset=1),
    ])

    assert result.completeness.status is CompletenessStatus.INCOMPLETE
    assert result.records is None
    assert any("total inconsistent" in reason for reason in result.completeness.reasons)


def test_page_and_record_safety_limits_fail_closed() -> None:
    too_many_pages = collect_resource_pages(
        [
            page(("r1",), total=2, limit=1, offset=0),
            page(("r2",), total=2, limit=1, offset=1),
        ],
        max_pages=1,
    )
    too_many_records = collect_resource_pages(
        [page(("r1", "r2"), total=3, limit=2)],
        max_records=2,
    )

    assert too_many_pages.completeness.status is CompletenessStatus.INCOMPLETE
    assert too_many_records.completeness.status is CompletenessStatus.INCOMPLETE
    assert too_many_pages.records is None
    assert too_many_records.records is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_pages": 0}, "max_pages"),
        ({"max_pages": 101}, "max_pages"),
        ({"max_records": 0}, "max_records"),
        ({"max_records": 10_001}, "max_records"),
        ({"max_records": True}, "max_records"),
    ],
)
def test_resource_safety_limits_are_strict(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        collect_resource_pages([], **kwargs)  # type: ignore[arg-type]
