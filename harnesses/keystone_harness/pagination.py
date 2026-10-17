"""Offline completeness checks for offset-paginated resource responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .discovery import (
    CollectionEvidence,
    CompletenessContract,
    CompletenessResult,
    CompletenessStatus,
    PageEvidence,
    validate_completeness,
)


MAX_RESOURCE_PAGES = 100
MAX_RESOURCE_RECORDS = 10_000


@dataclass(frozen=True)
class ResourceCollectionResult:
    """A complete record set, or evidence explaining why it is unavailable."""

    records: tuple[dict[str, Any], ...] | None
    collection_evidence: CollectionEvidence
    completeness: CompletenessResult

    @property
    def is_complete(self) -> bool:
        return (
            self.records is not None
            and self.completeness.status is CompletenessStatus.COMPLETE
        )


def collect_resource_pages(
    pages: Sequence[Mapping[str, Any]],
    *,
    max_pages: int = MAX_RESOURCE_PAGES,
    max_records: int = MAX_RESOURCE_RECORDS,
    collection_reference: str | None = None,
) -> ResourceCollectionResult:
    """Validate already-fetched ``data/total/limit/offset`` pages.

    This function performs no I/O. Synthetic offset markers are used only to
    express page progression in the protocol-neutral completeness evidence.
    Partial records are never returned.
    """

    if isinstance(pages, (str, bytes, Mapping)) or not isinstance(pages, Sequence):
        raise TypeError("pages must be a sequence of resource response mappings")
    if type(max_pages) is not int or not 1 <= max_pages <= MAX_RESOURCE_PAGES:
        raise ValueError(f"max_pages must be between 1 and {MAX_RESOURCE_PAGES}")
    if type(max_records) is not int or not 1 <= max_records <= MAX_RESOURCE_RECORDS:
        raise ValueError(f"max_records must be between 1 and {MAX_RESOURCE_RECORDS}")

    page_evidence: list[PageEvidence] = []
    records: list[dict[str, Any]] = []
    detected_duplicate_ids: set[str] = set()
    first_total: int | None = None
    expected_offset = 0
    failure_status: CompletenessStatus | None = None
    failure_reason: str | None = None

    for page_index, response in enumerate(pages):
        page_number = page_index + 1
        if page_index >= max_pages:
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = f"Resource collection exceeded the {max_pages}-page safety limit."
            break
        if not isinstance(response, Mapping):
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = f"Resource page {page_number} is not an object."
            break

        required_fields = ("data", "total", "limit", "offset")
        missing_fields = [field for field in required_fields if field not in response]
        if missing_fields:
            failure_status = CompletenessStatus.UNKNOWN
            failure_reason = (
                f"Resource page {page_number} is missing pagination metadata: "
                + ", ".join(missing_fields)
                + "."
            )
            break

        data = response["data"]
        total = response["total"]
        limit = response["limit"]
        offset = response["offset"]
        if not isinstance(data, list):
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = f"Resource page {page_number} data is not an array."
            break
        if any(type(value) is not int for value in (total, limit, offset)):
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = (
                f"Resource page {page_number} total, limit, and offset must be integers."
            )
            break
        if total < 0 or limit < 0 or offset < 0:
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = (
                f"Resource page {page_number} total, limit, and offset must be non-negative."
            )
            break
        if limit == 0 and (total != 0 or data):
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = (
                f"Resource page {page_number} has a zero limit for a non-empty collection."
            )
            break
        if len(data) > limit:
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = (
                f"Resource page {page_number} contains more records than its declared limit."
            )
            break
        if len(data) > max_records:
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = (
                f"Resource page {page_number} exceeds the {max_records}-record safety limit."
            )
            break
        if len(records) + len(data) > max_records:
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = (
                f"Resource collection exceeded the {max_records}-record safety limit."
            )
            break

        record_ids: list[str] = []
        page_ids: set[str] = set()
        duplicate_page_ids: set[str] = set()
        invalid_record = False
        for record in data:
            if not isinstance(record, Mapping):
                invalid_record = True
                break
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id.strip():
                invalid_record = True
                break
            if record_id in page_ids:
                duplicate_page_ids.add(record_id)
            else:
                page_ids.add(record_id)
                record_ids.append(record_id)
        if invalid_record:
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = (
                f"Resource page {page_number} contains a record without a non-empty string ID."
            )
            break

        next_offset = offset + len(data)
        has_more = next_offset < total
        page_evidence.append(
            PageEvidence(
                continuation_metadata_present=True,
                has_more=has_more,
                request_cursor=(
                    None if page_index == 0 else f"offset:{offset}"
                ),
                next_cursor=f"offset:{next_offset}" if has_more else None,
                page_id=f"offset:{offset}",
                reported_total_count=total,
                observed_record_count=len(data),
                record_ids=tuple(record_ids),
                truncated=None,
            )
        )

        if first_total is None:
            first_total = total
        elif total != first_total:
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = (
                f"Resource page {page_number} reported a total inconsistent with earlier pages."
            )
            break
        if offset != expected_offset:
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = (
                f"Resource page {page_number} overlaps or skips the expected offset "
                f"{expected_offset}."
            )
            break
        if duplicate_page_ids:
            detected_duplicate_ids.update(duplicate_page_ids)
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = (
                f"Resource page {page_number} contains duplicate record IDs: "
                + ", ".join(sorted(duplicate_page_ids))
                + "."
            )
            break
        if next_offset > total:
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = (
                f"Resource page {page_number} extends beyond its reported total."
            )
            break
        if total > max_records:
            failure_status = CompletenessStatus.INCOMPLETE
            failure_reason = (
                f"Resource total exceeds the {max_records}-record safety limit."
            )
            break

        expected_offset = next_offset
        records.extend(dict(record) for record in data)

    collection_evidence = CollectionEvidence(
        tuple(page_evidence),
        collection_reference=collection_reference,
    )
    contract = CompletenessContract(
        expected_record_count=first_total,
        require_total_count=True,
        require_truncation_metadata=False,
        initial_position_known=True,
        max_iterations=max_pages,
    )
    completeness = validate_completeness(collection_evidence, contract)

    if failure_reason is not None:
        status = (
            CompletenessStatus.INCOMPLETE
            if failure_status is CompletenessStatus.INCOMPLETE
            or completeness.status is CompletenessStatus.INCOMPLETE
            else CompletenessStatus.UNKNOWN
        )
        completeness = CompletenessResult(
            status=status,
            observed_record_count=completeness.observed_record_count,
            expected_record_count=completeness.expected_record_count,
            duplicate_record_ids=tuple(
                sorted(set(completeness.duplicate_record_ids) | detected_duplicate_ids)
            ),
            reasons=tuple(dict.fromkeys((*completeness.reasons, failure_reason))),
        )

    complete_records = (
        tuple(records)
        if completeness.status is CompletenessStatus.COMPLETE
        else None
    )
    return ResourceCollectionResult(
        records=complete_records,
        collection_evidence=collection_evidence,
        completeness=completeness,
    )
