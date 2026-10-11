"""Protocol-neutral discovery evidence and offline validation primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime
from enum import Enum
import json
import math
import re
from types import MappingProxyType
from typing import Any


_SECRET_KEY_PARTS = (
    "authorization",
    "apikey",
    "token",
    "password",
    "passwd",
    "cookie",
    "secret",
    "credential",
)
_BEARER_PATTERN = re.compile(r"(\bbearer\s+)[^\s,;]+", re.IGNORECASE)
_CREDENTIAL_PATTERN = re.compile(
    r"(\b(?:authorization|api[_ -]?key|token|password|passwd|cookie|secret|credential)\s*[:=]\s*)"
    r"[^,;\r\n]+",
    re.IGNORECASE,
)
_GENERIC_DISPATCH_IDENTIFIERS = frozenset({"toolscall", "calltool", "mcpcall"})
_CURSOR_FIELDS = frozenset({"request_cursor", "next_cursor", "initial_cursor"})


class EvidenceOrigin(str, Enum):
    """Caller-asserted evidence source; values do not authenticate provenance."""

    SYNTHETIC_FIXTURE = "synthetic_fixture"
    LOCAL_DOCUMENTATION = "local_documentation"
    VERIFIED_OBSERVATION = "verified_observation"


class VerificationStatus(str, Enum):
    """Caller-asserted verification state, not an independent verifier result."""

    UNVERIFIED = "unverified"
    VERIFIED = "verified"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class CompletenessStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


class FindingEligibilityStatus(str, Enum):
    ELIGIBLE_FOR_REVIEW = "eligible_for_review"
    INCONCLUSIVE = "inconclusive"
    REJECTED = "rejected"


def _sanitize_text(value: str) -> str:
    sanitized = value
    sanitized = _BEARER_PATTERN.sub(r"\1[REDACTED]", sanitized)
    return _CREDENTIAL_PATTERN.sub(r"\1[REDACTED]", sanitized)


def _is_secret_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return any(part in normalized for part in _SECRET_KEY_PARTS)


def _sanitize_value(value: Any, *, key: object | None = None) -> Any:
    if isinstance(key, str) and key in _CURSOR_FIELDS:
        return "[REDACTED_CURSOR]" if value is not None else None
    if _is_secret_key(key):
        return "[REDACTED]"
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        if not _has_utc_offset(value):
            raise ValueError("evidence timestamps must be timezone-aware")
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return _sanitize_text(value) if isinstance(value, str) else value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("evidence numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(item_key, str) for item_key in value):
            raise ValueError("evidence object keys must be strings")
        return {
            item_key: _sanitize_value(item_value, key=item_key)
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    if is_dataclass(value):
        return {
            item.name: _sanitize_value(getattr(value, item.name), key=item.name)
            for item in fields(value)
        }
    raise ValueError(f"unsupported evidence value type: {type(value).__name__}")


def _has_utc_offset(value: datetime) -> bool:
    if value.tzinfo is None:
        return False
    try:
        return value.utcoffset() is not None
    except (OverflowError, TypeError, ValueError):
        return False


def serialize_evidence(value: Any) -> str:
    """Serialize supported evidence deterministically with credential redaction."""

    return json.dumps(
        _sanitize_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return _sanitize_text(value.strip())


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name)


def _opaque_cursor(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty opaque cursor")
    return value


def _text_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(_required_text(value, name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class DiscoveryObservation:
    """Sanitized provenance for one discovery observation."""

    observation_id: str
    observed_at: datetime
    origin: EvidenceOrigin
    verification_status: VerificationStatus
    operation_id: str | None = None
    evidence_reference: str | None = None
    evidence_payload_json: str | None = None
    unknowns: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_id", _required_text(self.observation_id, "observation_id"))
        object.__setattr__(self, "operation_id", _optional_text(self.operation_id, "operation_id"))
        object.__setattr__(self, "evidence_reference", _optional_text(self.evidence_reference, "evidence_reference"))
        if not isinstance(self.observed_at, datetime):
            raise TypeError("observed_at must be a datetime")
        if not _has_utc_offset(self.observed_at):
            raise ValueError("observed_at must be timezone-aware")
        if not isinstance(self.origin, EvidenceOrigin):
            raise TypeError("origin must be an EvidenceOrigin")
        if not isinstance(self.verification_status, VerificationStatus):
            raise TypeError("verification_status must be a VerificationStatus")
        if (
            self.origin is EvidenceOrigin.SYNTHETIC_FIXTURE
            and self.verification_status is VerificationStatus.VERIFIED
        ):
            raise ValueError("synthetic fixtures cannot be labeled as verified observations")
        if self.evidence_payload_json is not None:
            try:
                payload = json.loads(self.evidence_payload_json)
            except (TypeError, json.JSONDecodeError):
                raise ValueError("evidence_payload_json must contain valid JSON") from None
            object.__setattr__(self, "evidence_payload_json", serialize_evidence(payload))
        if self.evidence_reference is None and self.evidence_payload_json is None:
            if not self.unknowns and not self.limitations:
                raise ValueError("an observation without evidence must state unknowns or limitations")
        object.__setattr__(self, "unknowns", _text_tuple(self.unknowns, "unknowns"))
        object.__setattr__(self, "limitations", _text_tuple(self.limitations, "limitations"))

    @classmethod
    def from_payload(
        cls,
        *,
        observation_id: str,
        observed_at: datetime,
        origin: EvidenceOrigin,
        verification_status: VerificationStatus,
        payload: Any,
        operation_id: str | None = None,
        evidence_reference: str | None = None,
        unknowns: tuple[str, ...] = (),
        limitations: tuple[str, ...] = (),
    ) -> DiscoveryObservation:
        return cls(
            observation_id=observation_id,
            observed_at=observed_at,
            origin=origin,
            verification_status=verification_status,
            operation_id=operation_id,
            evidence_reference=evidence_reference,
            evidence_payload_json=serialize_evidence(payload),
            unknowns=unknowns,
            limitations=limitations,
        )

    def to_json(self) -> str:
        return serialize_evidence(self)


@dataclass(frozen=True)
class OperationContract:
    """Caller-supplied operation claims, distinct from trusted approval."""

    operation_id: str
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    request_schema_reference: str | None = None
    response_schema_reference: str | None = None
    documented_behavior: str | None = None
    read_only: bool | None = None
    read_only_documented: bool = False
    supporting_observation_ids: tuple[str, ...] = ()
    unresolved_assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _required_text(self.operation_id, "operation_id"))
        object.__setattr__(self, "request_schema_reference", _optional_text(self.request_schema_reference, "request_schema_reference"))
        object.__setattr__(self, "response_schema_reference", _optional_text(self.response_schema_reference, "response_schema_reference"))
        object.__setattr__(self, "documented_behavior", _optional_text(self.documented_behavior, "documented_behavior"))
        object.__setattr__(self, "supporting_observation_ids", _text_tuple(self.supporting_observation_ids, "supporting_observation_ids"))
        object.__setattr__(self, "unresolved_assumptions", _text_tuple(self.unresolved_assumptions, "unresolved_assumptions"))
        if self.read_only is not None and not isinstance(self.read_only, bool):
            raise TypeError("read_only must be boolean or None")
        if not isinstance(self.read_only_documented, bool):
            raise TypeError("read_only_documented must be boolean")
        if not isinstance(self.verification_status, VerificationStatus):
            raise TypeError("verification_status must be a VerificationStatus")
        if not isinstance(self.approval_status, ApprovalStatus):
            raise TypeError("approval_status must be an ApprovalStatus")
        if self.approval_status is ApprovalStatus.APPROVED:
            raise ValueError("operation approval requires a trusted reviewer boundary")

    def to_json(self) -> str:
        return serialize_evidence(self)


def _is_generic_dispatch(operation_id: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", operation_id.casefold())
    return normalized in _GENERIC_DISPATCH_IDENTIFIERS


class OperationRegistryError(ValueError):
    """Raised when a contract cannot be safely registered or approved."""


class OperationNotExecutableError(PermissionError):
    """Raised when an operation is not approved for the current registry scope."""


class OperationRegistry:
    """Immutable evidence registry; no trusted approval authority is configured."""

    __slots__ = ("_observations", "_contracts")

    def __init__(self) -> None:
        object.__setattr__(self, "_observations", MappingProxyType({}))
        object.__setattr__(self, "_contracts", MappingProxyType({}))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("OperationRegistry instances are immutable")

    @classmethod
    def _from_state(
        cls,
        observations: Mapping[str, DiscoveryObservation],
        contracts: Mapping[str, OperationContract],
    ) -> OperationRegistry:
        registry = cls()
        object.__setattr__(registry, "_observations", MappingProxyType(dict(observations)))
        object.__setattr__(registry, "_contracts", MappingProxyType(dict(contracts)))
        return registry

    @property
    def observations(self) -> Mapping[str, DiscoveryObservation]:
        return self._observations

    @property
    def contracts(self) -> Mapping[str, OperationContract]:
        return self._contracts

    def with_observation(self, observation: DiscoveryObservation) -> OperationRegistry:
        if not isinstance(observation, DiscoveryObservation):
            raise TypeError("observation must be a DiscoveryObservation")
        if observation.observation_id in self._observations:
            raise OperationRegistryError("observation identifier is already registered")
        observations = dict(self._observations)
        observations[observation.observation_id] = observation
        return self._from_state(observations, self._contracts)

    def with_contract(self, contract: OperationContract) -> OperationRegistry:
        if not isinstance(contract, OperationContract):
            raise TypeError("contract must be an OperationContract")
        if contract.operation_id in self._contracts:
            raise OperationRegistryError("operation contract is already registered")
        if contract.approval_status not in (ApprovalStatus.PENDING, ApprovalStatus.REJECTED):
            raise OperationRegistryError("contracts must be registered pending or rejected")
        if _is_generic_dispatch(contract.operation_id):
            raise OperationRegistryError("generic dispatch operations cannot be registered")
        for observation_id in contract.supporting_observation_ids:
            observation = self._observations.get(observation_id)
            if observation is None:
                raise OperationRegistryError("a supporting observation is missing")
            if observation.operation_id not in (None, contract.operation_id):
                raise OperationRegistryError("supporting observation refers to a different operation")
        contracts = dict(self._contracts)
        contracts[contract.operation_id] = contract
        return self._from_state(self._observations, contracts)

    def approve(self, operation_id: str) -> OperationRegistry:
        raise OperationRegistryError(
            "No trusted approval authority is configured; operation contracts remain pending."
        )

    def reject(self, operation_id: str) -> OperationRegistry:
        contract = self._contracts.get(operation_id)
        if contract is None:
            raise OperationRegistryError("operation must have an explicit registered contract")
        if contract.approval_status not in (ApprovalStatus.PENDING, ApprovalStatus.REJECTED):
            raise OperationRegistryError("only pending operation contracts can be rejected")
        contracts = dict(self._contracts)
        contracts[operation_id] = replace(contract, approval_status=ApprovalStatus.REJECTED)
        return self._from_state(self._observations, contracts)

    def is_executable(self, operation_id: str) -> bool:
        return False

    def require_executable(self, operation_id: str) -> OperationContract:
        if not self.is_executable(operation_id):
            raise OperationNotExecutableError("operation is not approved for this registry scope")
        return self._contracts[operation_id]


PRODUCTION_OPERATION_REGISTRY = OperationRegistry()


@dataclass(frozen=True)
class CompletenessContract:
    """Protocol-neutral completeness requirements asserted by a caller."""

    expected_record_count: int | None = None
    require_total_count: bool = False
    require_snapshot_consistency: bool = False
    require_truncation_metadata: bool = True
    initial_position_known: bool = False
    initial_cursor: str | None = field(default=None, repr=False)
    max_iterations: int = 100

    def __post_init__(self) -> None:
        if self.expected_record_count is not None and (
            type(self.expected_record_count) is not int or self.expected_record_count < 0
        ):
            raise ValueError("expected_record_count must be a non-negative integer or None")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not isinstance(self.initial_position_known, bool):
            raise TypeError("initial_position_known must be boolean")
        if self.initial_cursor is not None:
            object.__setattr__(self, "initial_cursor", _opaque_cursor(self.initial_cursor, "initial_cursor"))
        for name in (
            "require_total_count",
            "require_snapshot_consistency",
            "require_truncation_metadata",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")


@dataclass(frozen=True)
class PageEvidence:
    """One protocol-neutral collection iteration."""

    continuation_metadata_present: bool
    has_more: bool | None
    request_cursor: str | None = field(default=None, repr=False)
    next_cursor: str | None = field(default=None, repr=False)
    page_id: str | None = None
    reported_total_count: int | None = None
    observed_record_count: int | None = None
    record_ids: tuple[str, ...] = ()
    snapshot_id: str | None = None
    truncated: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.continuation_metadata_present, bool):
            raise TypeError("continuation_metadata_present must be boolean")
        if self.has_more is not None and not isinstance(self.has_more, bool):
            raise TypeError("has_more must be boolean or None")
        for name in ("page_id", "snapshot_id"):
            object.__setattr__(self, name, _optional_text(getattr(self, name), name))
        for name in ("request_cursor", "next_cursor"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _opaque_cursor(value, name))
        for name in ("reported_total_count", "observed_record_count"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.truncated is not None and not isinstance(self.truncated, bool):
            raise TypeError("truncated must be boolean or None")
        object.__setattr__(self, "record_ids", _text_tuple(self.record_ids, "record_ids"))


@dataclass(frozen=True)
class CollectionEvidence:
    """Ordered page observations provided to the completeness validator."""

    pages: tuple[PageEvidence, ...]
    collection_reference: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pages", tuple(self.pages))
        if any(not isinstance(page, PageEvidence) for page in self.pages):
            raise TypeError("pages must contain PageEvidence values")
        object.__setattr__(self, "collection_reference", _optional_text(self.collection_reference, "collection_reference"))


@dataclass(frozen=True)
class CompletenessResult:
    status: CompletenessStatus
    observed_record_count: int | None
    expected_record_count: int | None
    duplicate_record_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, CompletenessStatus):
            raise TypeError("status must be a CompletenessStatus")
        for name in ("observed_record_count", "expected_record_count"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        object.__setattr__(self, "duplicate_record_ids", _text_tuple(self.duplicate_record_ids, "duplicate_record_ids"))
        object.__setattr__(self, "reasons", _text_tuple(self.reasons, "reasons"))


def validate_completeness(
    collection: CollectionEvidence,
    contract: CompletenessContract,
) -> CompletenessResult:
    """Assess completeness without assuming a particular pagination protocol."""

    if not isinstance(collection, CollectionEvidence) or not isinstance(contract, CompletenessContract):
        raise TypeError("collection and contract must be Keystone completeness evidence types")
    if not collection.pages:
        return CompletenessResult(
            CompletenessStatus.UNKNOWN,
            None,
            contract.expected_record_count,
            reasons=("No page evidence was supplied.",),
        )

    incomplete: list[str] = []
    unknown: list[str] = []
    duplicate_ids: set[str] = set()
    seen_ids: set[str] = set()
    page_ids: set[str] = set()
    request_cursors: set[str] = set()
    next_cursors: set[str] = set()
    totals: set[int] = set()
    snapshots: set[str] = set()
    observed_counts: list[int] = []
    page_count = len(collection.pages)

    if not contract.initial_position_known:
        unknown.append("The contract does not establish the collection's initial position.")
    elif collection.pages[0].request_cursor != contract.initial_cursor:
        incomplete.append("The first page does not start at the contract's initial position.")

    if page_count > contract.max_iterations:
        incomplete.append("The collection exceeded its configured iteration limit.")

    for index, page in enumerate(collection.pages):
        if page.page_id is not None:
            if page.page_id in page_ids:
                incomplete.append("A page identifier was repeated.")
            page_ids.add(page.page_id)

        if page.request_cursor is not None:
            if page.request_cursor in request_cursors:
                incomplete.append("A request continuation token was repeated.")
            request_cursors.add(page.request_cursor)

        if page.next_cursor is not None:
            if page.next_cursor in next_cursors or page.next_cursor in request_cursors:
                incomplete.append("A continuation token formed a repeated progression.")
            next_cursors.add(page.next_cursor)

        if page.reported_total_count is not None:
            totals.add(page.reported_total_count)
        elif contract.require_total_count:
            unknown.append("A required total-count value is missing.")

        if page.snapshot_id is not None:
            snapshots.add(page.snapshot_id)
        elif contract.require_snapshot_consistency:
            unknown.append("A required snapshot identifier is missing.")

        if page.truncated is True:
            incomplete.append("The response explicitly indicates truncation.")
        elif page.truncated is None and contract.require_truncation_metadata:
            unknown.append("Required truncation metadata is missing.")

        if page.observed_record_count is None:
            unknown.append("An observed record count is missing.")
        else:
            observed_counts.append(page.observed_record_count)
            if page.observed_record_count != len(page.record_ids):
                incomplete.append("Observed record count does not match the supplied record identities.")
        for record_id in page.record_ids:
            if record_id in seen_ids:
                duplicate_ids.add(record_id)
            seen_ids.add(record_id)

        if not page.continuation_metadata_present and (
            page.has_more is not None or page.next_cursor is not None
        ):
            incomplete.append("Continuation fields contradict the missing-metadata indicator.")
        if not page.continuation_metadata_present or page.has_more is None:
            unknown.append("Continuation metadata is missing.")
            continue
        if page.has_more and page.next_cursor is None:
            incomplete.append("A continuing page has no next continuation token.")
        if not page.has_more and page.next_cursor is not None:
            incomplete.append("A terminal page contradictorily supplies a next continuation token.")
        if page.has_more and page.observed_record_count == 0:
            incomplete.append("An empty page claims that more records remain.")
        if not page.has_more and index < page_count - 1:
            incomplete.append("Additional pages follow a terminal page.")
        if page.has_more and index == page_count - 1:
            incomplete.append("The final observed page says collection should continue.")

        if index > 0:
            previous = collection.pages[index - 1]
            if previous.next_cursor is not None:
                if page.request_cursor is None:
                    unknown.append("A continued page is missing its request cursor.")
                elif page.request_cursor != previous.next_cursor:
                    incomplete.append("A page request cursor does not match the prior continuation token.")

    if len(totals) > 1:
        incomplete.append("Reported total counts differ across pages.")
    if totals and any(page.reported_total_count is None for page in collection.pages):
        unknown.append("Reported total-count metadata is missing from one or more pages.")
    reported_total = next(iter(totals)) if len(totals) == 1 else None
    if contract.expected_record_count is not None and reported_total is not None:
        if reported_total != contract.expected_record_count:
            incomplete.append("The reported total differs from the contract's expected record count.")
    if duplicate_ids:
        incomplete.append("Duplicate record identifiers were observed across pages.")
    if len(snapshots) > 1:
        incomplete.append("Snapshot identifiers differ across pages.")

    observed_total = sum(observed_counts) if len(observed_counts) == page_count else None
    expected_total = contract.expected_record_count if contract.expected_record_count is not None else reported_total
    if expected_total is not None and observed_total is not None and observed_total != expected_total:
        incomplete.append("Observed record count does not match the expected total.")

    if incomplete:
        status = CompletenessStatus.INCOMPLETE
    elif unknown:
        status = CompletenessStatus.UNKNOWN
    else:
        status = CompletenessStatus.COMPLETE
    reasons = tuple(dict.fromkeys(incomplete + unknown))
    return CompletenessResult(
        status=status,
        observed_record_count=observed_total,
        expected_record_count=expected_total,
        duplicate_record_ids=tuple(sorted(duplicate_ids)),
        reasons=reasons,
    )


@dataclass(frozen=True)
class FindingEligibility:
    status: FindingEligibilityStatus
    reasons: tuple[str, ...]
    eligible_for_review: bool
    confirmed_defect: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, FindingEligibilityStatus):
            raise TypeError("status must be a FindingEligibilityStatus")
        if type(self.eligible_for_review) is not bool or type(self.confirmed_defect) is not bool:
            raise TypeError("finding eligibility flags must be booleans")
        object.__setattr__(self, "reasons", _text_tuple(self.reasons, "reasons"))
        if self.eligible_for_review != (self.status is FindingEligibilityStatus.ELIGIBLE_FOR_REVIEW):
            raise ValueError("eligible_for_review must agree with status")
        if self.confirmed_defect:
            raise ValueError("eligibility assessment cannot confirm a defect")


def assess_finding_eligibility(
    *,
    collection: CollectionEvidence,
    completeness_contract: CompletenessContract,
    contract: OperationContract | None,
    observations: Sequence[DiscoveryObservation],
    record_provenance: Mapping[str, str],
    invariant_observation_ids: Sequence[str],
    invariant_documented: bool,
    independently_established: bool = False,
    unsupported_assumptions: Sequence[str] = (),
) -> FindingEligibility:
    """Assess review eligibility from supplied evidence objects, never confirm a bug.

    Verification labels and origins remain caller claims; this function does not
    authenticate source material or provide a trusted verification authority.
    """

    if not isinstance(collection, CollectionEvidence):
        raise TypeError("collection must be CollectionEvidence")
    if not isinstance(completeness_contract, CompletenessContract):
        raise TypeError("completeness_contract must be CompletenessContract")
    if contract is not None and not isinstance(contract, OperationContract):
        raise TypeError("contract must be OperationContract or None")
    if not isinstance(observations, (tuple, list)) or any(
        not isinstance(item, DiscoveryObservation) for item in observations
    ):
        raise TypeError("observations must be a tuple or list of DiscoveryObservation values")
    if not isinstance(invariant_documented, bool) or not isinstance(independently_established, bool):
        raise TypeError("invariant evidence flags must be boolean")

    completeness = validate_completeness(collection, completeness_contract)
    reasons: list[str] = []
    rejected = contract is not None and contract.read_only is False
    if rejected:
        reasons.append("The operation contract explicitly identifies a non-read-only operation.")
    if completeness.status is not CompletenessStatus.COMPLETE:
        reasons.append("Collection completeness is not established.")
    if (
        contract is None
        or contract.verification_status is not VerificationStatus.VERIFIED
        or contract.read_only is not True
        or not contract.read_only_documented
    ):
        reasons.append("A verified contract with documented read-only semantics is unavailable.")
    observations_by_id = {item.observation_id: item for item in observations}
    if len(observations_by_id) != len(observations):
        reasons.append("Observation identifiers are duplicated.")

    def verified_reference(
        reference_id: str,
        *,
        allowed_origins: frozenset[EvidenceOrigin],
    ) -> bool:
        evidence = observations_by_id.get(reference_id)
        return bool(
            evidence is not None
            and evidence.origin in allowed_origins
            and evidence.verification_status is VerificationStatus.VERIFIED
            and not evidence.unknowns
            and not evidence.limitations
            and (evidence.evidence_reference is not None or evidence.evidence_payload_json is not None)
            and (contract is None or evidence.operation_id in (None, contract.operation_id))
        )

    if not isinstance(record_provenance, Mapping) or any(
        not isinstance(record_id, str) or not record_id.strip()
        or not isinstance(observation_id, str) or not observation_id.strip()
        for record_id, observation_id in record_provenance.items()
    ):
        raise TypeError("record_provenance must map non-empty record IDs to observation IDs")
    collected_record_ids = {
        record_id
        for page in collection.pages
        for record_id in page.record_ids
    }
    if not record_provenance:
        reasons.append("Record-level provenance is missing.")
    elif set(record_provenance) != collected_record_ids:
        reasons.append("Record provenance does not exactly cover the collected record identifiers.")
    elif any(
        not verified_reference(
            record_provenance[record_id],
            allowed_origins=frozenset({EvidenceOrigin.VERIFIED_OBSERVATION}),
        )
        for record_id in collected_record_ids
    ):
        reasons.append("Record provenance does not resolve to verified observation evidence.")

    if contract is not None:
        if not contract.supporting_observation_ids:
            reasons.append("The operation contract has no supporting observation references.")
        elif any(
            not verified_reference(
                reference_id,
                allowed_origins=frozenset({
                    EvidenceOrigin.LOCAL_DOCUMENTATION,
                    EvidenceOrigin.VERIFIED_OBSERVATION,
                }),
            )
            for reference_id in contract.supporting_observation_ids
        ):
            reasons.append("Operation contract references unresolved or unverified observations.")
        if contract.unresolved_assumptions:
            reasons.append("The operation contract contains unresolved assumptions.")

    invariant_ids = _text_tuple(invariant_observation_ids, "invariant_observation_ids")
    if not invariant_documented and not independently_established:
        reasons.append("The claimed invariant is neither documented nor independently established.")
    if invariant_documented or independently_established:
        if not invariant_ids:
            reasons.append("The invariant has no supporting observation references.")
        elif any(
            not verified_reference(
                reference_id,
                allowed_origins=frozenset({
                    EvidenceOrigin.LOCAL_DOCUMENTATION,
                    EvidenceOrigin.VERIFIED_OBSERVATION,
                }),
            )
            for reference_id in invariant_ids
        ):
            reasons.append("Invariant references do not resolve to verified evidence.")
    assumptions = _text_tuple(unsupported_assumptions, "unsupported_assumptions")
    if assumptions:
        reasons.append("The finding depends on unsupported assumptions.")

    if reasons:
        return FindingEligibility(
            FindingEligibilityStatus.REJECTED if rejected else FindingEligibilityStatus.INCONCLUSIVE,
            tuple(reasons),
            eligible_for_review=False,
        )
    return FindingEligibility(
        FindingEligibilityStatus.ELIGIBLE_FOR_REVIEW,
        ("Evidence may be reviewed; this assessment does not confirm a defect.",),
        eligible_for_review=True,
    )