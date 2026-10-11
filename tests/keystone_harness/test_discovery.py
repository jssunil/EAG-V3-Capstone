from __future__ import annotations

from datetime import datetime, timezone, tzinfo
import json

import pytest

from harnesses.keystone_harness import (
    PRODUCTION_OPERATION_REGISTRY,
    ApprovalStatus,
    CollectionEvidence,
    CompletenessContract,
    CompletenessStatus,
    DiscoveryObservation,
    EvidenceOrigin,
    FindingEligibilityStatus,
    KeystoneClient,
    KeystoneConfig,
    KeystoneOperationNotAllowedError,
    OperationContract,
    OperationNotExecutableError,
    OperationRegistry,
    OperationRegistryError,
    PageEvidence,
    VerificationStatus,
    assess_finding_eligibility,
    serialize_evidence,
    validate_completeness,
)


NOW = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
SECRET = "synthetic-discovery-secret"


def observation(
    observation_id: str = "doc-1",
    *,
    origin: EvidenceOrigin = EvidenceOrigin.LOCAL_DOCUMENTATION,
    verification: VerificationStatus | None = None,
    operation_id: str | None = "sample.read",
) -> DiscoveryObservation:
    if verification is None:
        verification = (
            VerificationStatus.UNVERIFIED
            if origin is EvidenceOrigin.SYNTHETIC_FIXTURE
            else VerificationStatus.VERIFIED
        )
    return DiscoveryObservation.from_payload(
        observation_id=observation_id,
        observed_at=NOW,
        origin=origin,
        verification_status=verification,
        operation_id=operation_id,
        evidence_reference="local-doc:read-contract",
        payload={"description": "synthetic read behavior", "token": SECRET},
    )


def contract(
    *,
    operation_id: str = "sample.read",
    verification: VerificationStatus = VerificationStatus.VERIFIED,
    read_only: bool | None = True,
    documented: bool = True,
    observation_ids: tuple[str, ...] = ("doc-1",),
    assumptions: tuple[str, ...] = (),
) -> OperationContract:
    return OperationContract(
        operation_id=operation_id,
        verification_status=verification,
        read_only=read_only,
        read_only_documented=documented,
        documented_behavior="Returns fixture data without mutation.",
        request_schema_reference="fixture:request",
        response_schema_reference="fixture:response",
        supporting_observation_ids=observation_ids,
        unresolved_assumptions=assumptions,
    )


def page(
    *,
    page_id: str,
    ids: tuple[str, ...],
    more: bool = False,
    request: str | None = None,
    next_token: str | None = None,
    total: int | None = None,
    count: int | None = None,
    snapshot: str | None = None,
    truncated: bool | None = False,
    metadata: bool = True,
) -> PageEvidence:
    return PageEvidence(
        page_id=page_id,
        request_cursor=request,
        next_cursor=next_token,
        continuation_metadata_present=metadata,
        has_more=more if metadata else None,
        reported_total_count=total,
        observed_record_count=len(ids) if count is None else count,
        record_ids=ids,
        snapshot_id=snapshot,
        truncated=truncated,
    )


def complete_contract(**overrides: object) -> CompletenessContract:
    values: dict[str, object] = {"initial_position_known": True}
    values.update(overrides)
    return CompletenessContract(**values)  # type: ignore[arg-type]


def complete_page_collection() -> CollectionEvidence:
    return CollectionEvidence(
        pages=(page(page_id="page-1", ids=("r1",), total=1, snapshot="v1"),),
        collection_reference="fixture:collection-1",
    )


def test_observation_and_contract_retain_unknowns_and_serialize_stably() -> None:
    item = DiscoveryObservation.from_payload(
        observation_id="unknown-1",
        observed_at=NOW,
        origin=EvidenceOrigin.SYNTHETIC_FIXTURE,
        verification_status=VerificationStatus.UNVERIFIED,
        payload={"Authorization": f"Bearer {SECRET}", "details": "shape unknown"},
        unknowns=("response schema is unknown",),
        limitations=("synthetic fixture only",),
    )
    contract_value = OperationContract(operation_id="sample.read")

    first = item.to_json()
    second = serialize_evidence(item)

    assert first == second
    assert SECRET not in first
    assert json.loads(first)["evidence_payload_json"] == '{"Authorization":"[REDACTED]","details":"shape unknown"}'
    assert item.unknowns == ("response schema is unknown",)
    assert contract_value.request_schema_reference is None
    assert contract_value.read_only is None
    assert contract_value.approval_status is ApprovalStatus.PENDING


def test_observation_without_payload_must_explain_missing_evidence() -> None:
    with pytest.raises(ValueError, match="state unknowns or limitations"):
        DiscoveryObservation(
            observation_id="empty",
            observed_at=NOW,
            origin=EvidenceOrigin.LOCAL_DOCUMENTATION,
            verification_status=VerificationStatus.UNVERIFIED,
        )


def test_observation_rejects_naive_timestamp_and_malformed_payload() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        DiscoveryObservation(
            observation_id="naive",
            observed_at=datetime(2026, 10, 11),
            origin=EvidenceOrigin.LOCAL_DOCUMENTATION,
            verification_status=VerificationStatus.UNVERIFIED,
            unknowns=("timestamp invalid",),
        )
    with pytest.raises(ValueError, match="valid JSON"):
        DiscoveryObservation(
            observation_id="bad-json",
            observed_at=NOW,
            origin=EvidenceOrigin.LOCAL_DOCUMENTATION,
            verification_status=VerificationStatus.UNVERIFIED,
            evidence_payload_json="not-json",
        )


def test_observation_rejects_tzinfo_without_utc_offset() -> None:
    class EmptyOffset(tzinfo):
        def utcoffset(self, dt: datetime | None):
            return None

        def dst(self, dt: datetime | None):
            return None

    with pytest.raises(ValueError, match="timezone-aware"):
        DiscoveryObservation(
            observation_id="empty-offset",
            observed_at=datetime(2026, 10, 11, tzinfo=EmptyOffset()),
            origin=EvidenceOrigin.LOCAL_DOCUMENTATION,
            verification_status=VerificationStatus.UNVERIFIED,
            unknowns=("timestamp offset is invalid",),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"operation_id": ""},
        {"operation_id": "sample.read", "approval_status": ApprovalStatus.APPROVED},
        {
            "operation_id": "sample.read",
            "approval_status": ApprovalStatus.APPROVED,
            "verification_status": VerificationStatus.VERIFIED,
            "read_only": True,
            "read_only_documented": True,
            "supporting_observation_ids": (),
        },
    ],
)
def test_contract_rejects_missing_identity_or_invalid_approval_combination(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        OperationContract(**kwargs)  # type: ignore[arg-type]


def test_contract_serialization_redacts_secret_shaped_fields() -> None:
    value = OperationContract(
        operation_id="sample.read",
        documented_behavior=f"Authorization: Bearer {SECRET}",
        request_schema_reference="local://schema?token=" + SECRET,
    )

    serialized = value.to_json()

    assert SECRET not in serialized
    assert "[REDACTED]" in serialized


def test_production_registry_is_empty_and_observations_do_not_register_operations() -> None:
    registry = OperationRegistry().with_observation(observation())
    observed_only = PRODUCTION_OPERATION_REGISTRY.with_observation(observation("prod-observation"))

    assert dict(PRODUCTION_OPERATION_REGISTRY.contracts) == {}
    assert dict(PRODUCTION_OPERATION_REGISTRY.observations) == {}
    assert dict(observed_only.contracts) == {}
    assert not registry.is_executable("sample.read")


def test_pending_and_unknown_operations_cannot_execute() -> None:
    registry = OperationRegistry().with_observation(observation()).with_contract(contract())

    with pytest.raises(OperationNotExecutableError):
        registry.require_executable("sample.read")
    with pytest.raises(OperationNotExecutableError):
        registry.require_executable("not.registered")


def test_registry_cannot_be_mutated_through_public_state() -> None:
    registry = OperationRegistry()

    with pytest.raises(AttributeError, match="immutable"):
        registry._contracts = {}  # type: ignore[misc]


@pytest.mark.parametrize("operation_id", ["tools/call", "TOOLS/CALL", "tools.call", "call_tool", "callTool"])
def test_generic_dispatch_aliases_are_rejected(operation_id: str) -> None:
    with pytest.raises(OperationRegistryError, match="generic dispatch"):
        OperationRegistry().with_contract(contract(operation_id=operation_id, observation_ids=()))


def test_mutation_names_are_not_used_to_infer_or_grant_safety() -> None:
    pending = OperationRegistry().with_contract(
        contract(operation_id="CalendarEvent.confirm", observation_ids=())
    )

    assert not pending.is_executable("CalendarEvent.confirm")
    with pytest.raises(OperationRegistryError, match="trusted approval authority"):
        pending.approve("CalendarEvent.confirm")


def test_unverified_or_undocumented_contract_cannot_be_approved() -> None:
    registry = OperationRegistry().with_observation(observation()).with_contract(
        contract(verification=VerificationStatus.UNVERIFIED)
    )

    with pytest.raises(OperationRegistryError, match="trusted approval authority"):
        registry.approve("sample.read")

    undocumented = OperationRegistry().with_observation(observation()).with_contract(
        contract(documented=False)
    )
    with pytest.raises(OperationRegistryError, match="trusted approval authority"):
        undocumented.approve("sample.read")


def test_contract_references_must_resolve_when_registered() -> None:
    with pytest.raises(OperationRegistryError, match="supporting observation is missing"):
        OperationRegistry().with_contract(contract())

    registry = OperationRegistry().with_observation(observation()).with_contract(contract())
    with pytest.raises(OperationRegistryError, match="trusted approval authority"):
        registry.approve("sample.read")


def test_synthetic_observations_cannot_claim_verified_status() -> None:
    with pytest.raises(ValueError, match="synthetic fixtures cannot be labeled"):
        observation(origin=EvidenceOrigin.SYNTHETIC_FIXTURE, verification=VerificationStatus.VERIFIED)


def test_unknowns_and_limitations_are_retained_and_approval_remains_pending() -> None:
    evidence = observation().from_payload(
        observation_id="limited",
        observed_at=NOW,
        origin=EvidenceOrigin.LOCAL_DOCUMENTATION,
        verification_status=VerificationStatus.VERIFIED,
        operation_id="sample.read",
        evidence_reference="local-doc:read-contract",
        payload={"read_only": True},
        unknowns=("mutation semantics not independently checked",),
    )
    registry = OperationRegistry().with_observation(evidence).with_contract(
        contract(observation_ids=("limited",))
    )

    with pytest.raises(OperationRegistryError, match="trusted approval authority"):
        registry.approve("sample.read")
    assert registry.contracts["sample.read"].approval_status is ApprovalStatus.PENDING


def test_test_mode_bypass_does_not_exist_and_contract_cannot_self_approve() -> None:
    with pytest.raises(TypeError):
        OperationRegistry(test_mode=True)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="trusted reviewer boundary"):
        OperationContract("sample.read", approval_status=ApprovalStatus.APPROVED)


def test_valid_documented_contract_stays_pending_without_trusted_reviewer() -> None:
    registry = OperationRegistry().with_observation(observation()).with_contract(contract())

    with pytest.raises(OperationRegistryError, match="trusted approval authority"):
        registry.approve("sample.read")
    assert registry.contracts["sample.read"].approval_status is ApprovalStatus.PENDING
    assert not registry.is_executable("sample.read")


def test_unknown_and_case_variant_ids_are_never_executable() -> None:
    registry = OperationRegistry().with_contract(contract(observation_ids=()))

    assert not registry.is_executable("sample.read")
    assert not registry.is_executable("SAMPLE.READ")


def test_rejected_contract_cannot_execute() -> None:
    registry = OperationRegistry().with_contract(contract(observation_ids=())).reject("sample.read")

    assert registry.contracts["sample.read"].approval_status is ApprovalStatus.REJECTED
    assert not registry.is_executable("sample.read")


def test_contract_enum_values_are_runtime_validated() -> None:
    with pytest.raises(TypeError, match="VerificationStatus"):
        OperationContract("sample.read", verification_status="verified")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ApprovalStatus"):
        OperationContract("sample.read", approval_status="pending")  # type: ignore[arg-type]


def test_finding_assessment_does_not_accept_forged_completeness_result() -> None:
    forged = validate_completeness(complete_page_collection(), complete_contract())
    with pytest.raises(TypeError, match="collection must be CollectionEvidence"):
        assess_finding_eligibility(
            collection=forged,  # type: ignore[arg-type]
            completeness_contract=complete_contract(),
            contract=contract(),
            observations=(),
            record_provenance={},
            invariant_observation_ids=(),
            invariant_documented=False,
        )


def test_unverified_or_undocumented_contract_cannot_be_approved() -> None:
    registry = OperationRegistry().with_observation(observation()).with_contract(
        contract(verification=VerificationStatus.UNVERIFIED)
    )

    with pytest.raises(OperationRegistryError, match="trusted approval authority"):
        registry.approve("sample.read")

    undocumented = OperationRegistry().with_observation(observation()).with_contract(
        contract(documented=False)
    )
    with pytest.raises(OperationRegistryError, match="trusted approval authority"):
        undocumented.approve("sample.read")


def test_observation_must_exist_be_verified_and_match_operation_before_approval() -> None:
    with pytest.raises(OperationRegistryError, match="supporting observation is missing"):
        OperationRegistry().with_contract(contract())

    unverified = OperationRegistry().with_observation(
        observation(verification=VerificationStatus.UNVERIFIED)
    ).with_contract(contract())
    with pytest.raises(OperationRegistryError, match="trusted approval authority"):
        unverified.approve("sample.read")

    with pytest.raises(OperationRegistryError, match="different operation"):
        OperationRegistry().with_observation(
            observation(operation_id="different.read")
        ).with_contract(contract())


def test_synthetic_contract_approval_is_scoped_to_test_registry() -> None:
    fixture_observation = observation(origin=EvidenceOrigin.SYNTHETIC_FIXTURE)
    registry = OperationRegistry().with_observation(fixture_observation).with_contract(contract())

    assert not registry.is_executable("sample.read")
    assert not PRODUCTION_OPERATION_REGISTRY.is_executable("sample.read")
    with pytest.raises(OperationRegistryError, match="trusted approval authority"):
        registry.approve("sample.read")


def test_synthetic_test_registry_cannot_change_client_production_guard() -> None:
    fixture_observation = observation(origin=EvidenceOrigin.SYNTHETIC_FIXTURE)
    registry = OperationRegistry().with_observation(fixture_observation).with_contract(contract())
    client = KeystoneClient(KeystoneConfig(token=SECRET))

    assert not registry.is_executable("sample.read")
    with pytest.raises(KeystoneOperationNotAllowedError):
        client.call_read("sample.read")


def test_contract_with_unresolved_assumptions_cannot_be_approved() -> None:
    registry = OperationRegistry().with_observation(observation()).with_contract(
        contract(assumptions=("response side effect behavior",))
    )

    with pytest.raises(OperationRegistryError, match="trusted approval authority"):
        registry.approve("sample.read")


def test_valid_documented_contract_requires_explicit_approval() -> None:
    registry = OperationRegistry().with_observation(observation()).with_contract(contract())

    assert not registry.is_executable("sample.read")
    assert registry.contracts["sample.read"].approval_status is ApprovalStatus.PENDING
    with pytest.raises(OperationRegistryError, match="trusted approval authority"):
        registry.approve("sample.read")


def test_completeness_accepts_consistent_terminal_page() -> None:
    result = validate_completeness(
        complete_page_collection(),
        complete_contract(expected_record_count=1, require_total_count=True, require_snapshot_consistency=True, require_truncation_metadata=True),
    )

    assert result.status is CompletenessStatus.COMPLETE
    assert result.observed_record_count == result.expected_record_count == 1
    assert not result.reasons


def test_completeness_accepts_verified_initial_page_through_terminal_page() -> None:
    collection = CollectionEvidence((
        page(page_id="p1", ids=("r1",), more=True, next_token="cursor-2", total=2),
        page(page_id="p2", ids=("r2",), request="cursor-2", total=2),
    ))
    result = validate_completeness(
        collection,
        complete_contract(expected_record_count=2, require_total_count=True),
    )

    assert result.status is CompletenessStatus.COMPLETE
    assert result.observed_record_count == 2


def test_unknown_or_noninitial_start_never_becomes_complete() -> None:
    collection = CollectionEvidence((
        page(page_id="p1", ids=("r2",), request="cursor-2", total=1),
    ))
    unknown_start = validate_completeness(collection, CompletenessContract())
    wrong_start = validate_completeness(
        collection,
        complete_contract(initial_cursor=None, expected_record_count=1),
    )

    assert unknown_start.status is CompletenessStatus.UNKNOWN
    assert wrong_start.status is CompletenessStatus.INCOMPLETE


def test_terminal_page_with_a_continuation_token_is_incomplete() -> None:
    result = validate_completeness(
        CollectionEvidence((page(page_id="p1", ids=("r1",), next_token="unexpected", total=1),)),
        complete_contract(expected_record_count=1),
    )

    assert result.status is CompletenessStatus.INCOMPLETE
    assert any("terminal page" in reason for reason in result.reasons)


def test_completeness_detects_missing_expected_records_and_inconsistent_totals() -> None:
    missing = validate_completeness(
        complete_page_collection(),
        complete_contract(expected_record_count=2),
    )
    inconsistent = validate_completeness(
        CollectionEvidence((
            page(page_id="p1", ids=("r1",), more=True, next_token="next", total=2),
            page(page_id="p2", ids=("r2",), request="next", total=3),
        )),
        complete_contract(),
    )

    assert missing.status is CompletenessStatus.INCOMPLETE
    assert any("does not match" in reason for reason in missing.reasons)
    assert inconsistent.status is CompletenessStatus.INCOMPLETE
    assert any("differ" in reason for reason in inconsistent.reasons)


def test_completeness_detects_duplicate_record_ids_and_page_ids() -> None:
    result = validate_completeness(
        CollectionEvidence((
            page(page_id="same-page", ids=("r1",), more=True, next_token="c2"),
            page(page_id="same-page", ids=("r1",), request="c2"),
        )),
        complete_contract(),
    )

    assert result.status is CompletenessStatus.INCOMPLETE
    assert result.duplicate_record_ids == ("r1",)
    assert any("page identifier" in reason for reason in result.reasons)


def test_completeness_detects_repeated_cursor_and_mismatched_progression() -> None:
    repeated = validate_completeness(
        CollectionEvidence((
            page(page_id="p1", ids=("r1",), more=True, next_token="c2"),
            page(page_id="p2", ids=("r2",), request="c2", more=True, next_token="c2"),
        )),
        complete_contract(),
    )
    mismatched = validate_completeness(
        CollectionEvidence((
            page(page_id="p1", ids=("r1",), more=True, next_token="c2"),
            page(page_id="p2", ids=("r2",), request="other"),
        )),
        complete_contract(),
    )

    assert repeated.status is CompletenessStatus.INCOMPLETE
    assert any("repeated" in reason for reason in repeated.reasons)
    assert mismatched.status is CompletenessStatus.INCOMPLETE
    assert any("does not match" in reason for reason in mismatched.reasons)


def test_completeness_detects_empty_intermediate_page() -> None:
    result = validate_completeness(
        CollectionEvidence((
            page(page_id="p1", ids=("r1",), more=True, next_token="c2"),
            page(page_id="p2", ids=(), request="c2", more=True, next_token="c3"),
            page(page_id="p3", ids=("r2",), request="c3"),
        )),
        complete_contract(),
    )

    assert result.status is CompletenessStatus.INCOMPLETE
    assert any("empty page" in reason for reason in result.reasons)


def test_missing_continuation_or_required_metadata_is_unknown() -> None:
    missing_continuation = validate_completeness(
        CollectionEvidence((page(page_id="p1", ids=("r1",), metadata=False),)),
        complete_contract(),
    )
    missing_total = validate_completeness(
        CollectionEvidence((page(page_id="p1", ids=("r1",), total=None),)),
        complete_contract(require_total_count=True),
    )
    missing_truncation = validate_completeness(
        CollectionEvidence((page(page_id="p1", ids=("r1",), truncated=None),)),
        complete_contract(require_truncation_metadata=True),
    )

    assert missing_continuation.status is CompletenessStatus.UNKNOWN
    assert missing_total.status is CompletenessStatus.UNKNOWN
    assert missing_truncation.status is CompletenessStatus.UNKNOWN


def test_contradictory_continuation_and_truncation_are_incomplete() -> None:
    contradiction = validate_completeness(
        CollectionEvidence((page(page_id="p1", ids=("r1",), next_token="unexpected"),)),
        complete_contract(),
    )
    truncated = validate_completeness(
        CollectionEvidence((page(page_id="p1", ids=("r1",), truncated=True),)),
        complete_contract(),
    )
    metadata_contradiction = validate_completeness(
        CollectionEvidence((page(page_id="p1", ids=("r1",), more=True, next_token="c2", metadata=False),)),
        complete_contract(),
    )

    assert contradiction.status is CompletenessStatus.INCOMPLETE
    assert truncated.status is CompletenessStatus.INCOMPLETE
    assert metadata_contradiction.status is CompletenessStatus.INCOMPLETE


def test_snapshot_inconsistency_and_missing_required_snapshot_are_separate() -> None:
    changed = validate_completeness(
        CollectionEvidence((
            page(page_id="p1", ids=("r1",), more=True, next_token="c2", snapshot="v1"),
            page(page_id="p2", ids=("r2",), request="c2", snapshot="v2"),
        )),
        complete_contract(require_snapshot_consistency=True),
    )
    missing = validate_completeness(
        CollectionEvidence((page(page_id="p1", ids=("r1",), snapshot=None),)),
        complete_contract(require_snapshot_consistency=True),
    )
    optional_but_inconsistent = validate_completeness(
        CollectionEvidence((
            page(page_id="p1", ids=("r1",), more=True, next_token="c2", snapshot="v1"),
            page(page_id="p2", ids=("r2",), request="c2", snapshot="v2"),
        )),
        complete_contract(),
    )

    assert changed.status is CompletenessStatus.INCOMPLETE
    assert any("differ" in reason for reason in changed.reasons)
    assert optional_but_inconsistent.status is CompletenessStatus.INCOMPLETE
    assert missing.status is CompletenessStatus.UNKNOWN
    assert any("missing" in reason for reason in missing.reasons)


def test_page_limit_and_continuing_final_page_are_incomplete() -> None:
    pages = CollectionEvidence((
        page(page_id="p1", ids=("r1",), more=True, next_token="c2"),
        page(page_id="p2", ids=("r2",), request="c2", more=True, next_token="c3"),
    ))
    limited = validate_completeness(pages, complete_contract(max_iterations=1))
    continuing = validate_completeness(pages, complete_contract(max_iterations=2))

    assert limited.status is CompletenessStatus.INCOMPLETE
    assert any("iteration limit" in reason for reason in limited.reasons)
    assert continuing.status is CompletenessStatus.INCOMPLETE
    assert any("should continue" in reason for reason in continuing.reasons)


def test_missing_collection_evidence_is_unknown_not_complete() -> None:
    result = validate_completeness(CollectionEvidence(()), complete_contract())

    assert result.status is CompletenessStatus.UNKNOWN


def test_invalid_completeness_contract_and_page_metadata_are_rejected() -> None:
    with pytest.raises(ValueError, match="expected_record_count"):
        CompletenessContract(expected_record_count=-1)
    with pytest.raises(ValueError, match="max_iterations"):
        CompletenessContract(max_iterations=0)
    with pytest.raises(ValueError, match="reported_total_count"):
        PageEvidence(True, False, reported_total_count=True)
    with pytest.raises(ValueError, match="observed_record_count"):
        PageEvidence(True, False, observed_record_count=-1)
    with pytest.raises(ValueError, match="reported_total_count"):
        page(page_id="bad", ids=(), total=-1)


def test_identity_count_mismatch_is_incomplete() -> None:
    result = validate_completeness(
        CollectionEvidence((page(page_id="p1", ids=("r1",), count=2),)),
        complete_contract(expected_record_count=2),
    )

    assert result.status is CompletenessStatus.INCOMPLETE
    assert any("record identities" in reason for reason in result.reasons)


def record_observation(
    observation_id: str = "record-1",
    *,
    verification: VerificationStatus = VerificationStatus.VERIFIED,
    origin: EvidenceOrigin = EvidenceOrigin.VERIFIED_OBSERVATION,
) -> DiscoveryObservation:
    return DiscoveryObservation.from_payload(
        observation_id=observation_id,
        observed_at=NOW,
        origin=origin,
        verification_status=verification,
        operation_id="sample.read",
        evidence_reference="fixture-record:record-1",
        payload={"record_id": "record-1"},
    )


def eligibility(**overrides: object):
    arguments: dict[str, object] = {
        "collection": complete_page_collection(),
        "completeness_contract": complete_contract(),
        "contract": contract(),
        "observations": (observation(), record_observation()),
        "record_provenance": {"r1": "record-1"},
        "invariant_observation_ids": ("doc-1",),
        "invariant_documented": True,
    }
    arguments.update(overrides)
    return assess_finding_eligibility(**arguments)  # type: ignore[arg-type]


def test_sufficient_fixture_evidence_is_eligible_for_review_not_confirmed() -> None:
    result = eligibility()

    assert result.status is FindingEligibilityStatus.ELIGIBLE_FOR_REVIEW
    assert result.eligible_for_review
    assert not result.confirmed_defect


def test_finding_provenance_must_match_collected_record_ids() -> None:
    missing = eligibility(record_provenance={})
    extra = eligibility(record_provenance={"different-record": "record-1"})
    untrusted_origin = eligibility(
        observations=(observation(), record_observation(origin=EvidenceOrigin.LOCAL_DOCUMENTATION))
    )

    assert missing.status is FindingEligibilityStatus.INCONCLUSIVE
    assert extra.status is FindingEligibilityStatus.INCONCLUSIVE
    assert untrusted_origin.status is FindingEligibilityStatus.INCONCLUSIVE


@pytest.mark.parametrize(
    "overrides",
    [
        {"observations": (observation(origin=EvidenceOrigin.SYNTHETIC_FIXTURE),)},
        {"contract": contract(verification=VerificationStatus.UNVERIFIED)},
        {"invariant_documented": False, "invariant_observation_ids": ()},
        {"record_provenance": {}},
        {"unsupported_assumptions": ("status has inferred severity",)},
    ],
)
def test_weak_provenance_or_contract_makes_finding_inconclusive(overrides: dict[str, object]) -> None:
    result = eligibility(**overrides)

    assert result.status is FindingEligibilityStatus.INCONCLUSIVE
    assert not result.eligible_for_review
    assert not result.confirmed_defect


def test_incomplete_collection_cannot_become_confirmed_finding() -> None:
    result = eligibility(
        collection=CollectionEvidence((page(page_id="p1", ids=("r1",), more=True, next_token="c2"),))
    )

    assert result.status is FindingEligibilityStatus.INCONCLUSIVE
    assert any("completeness" in reason for reason in result.reasons)
    assert not result.confirmed_defect


def test_invariant_requires_resolvable_observation_references() -> None:
    missing_reference = eligibility(
        invariant_documented=False,
        independently_established=True,
        invariant_observation_ids=(),
    )
    unresolved_reference = eligibility(
        invariant_documented=False,
        independently_established=True,
        invariant_observation_ids=("missing-observation",),
    )

    assert missing_reference.status is FindingEligibilityStatus.INCONCLUSIVE
    assert unresolved_reference.status is FindingEligibilityStatus.INCONCLUSIVE


def test_unverified_record_observation_cannot_support_review_eligibility() -> None:
    result = eligibility(
        observations=(observation(), record_observation(verification=VerificationStatus.UNVERIFIED))
    )

    assert result.status is FindingEligibilityStatus.INCONCLUSIVE


def test_unknown_collection_cannot_support_review_eligibility() -> None:
    result = eligibility(
        collection=complete_page_collection(),
        completeness_contract=CompletenessContract(),
    )

    assert result.status is FindingEligibilityStatus.INCONCLUSIVE
    assert not result.confirmed_defect


def test_unresolved_contract_assumptions_block_review_eligibility() -> None:
    result = eligibility(contract=contract(assumptions=("read semantics not reviewed",)))

    assert result.status is FindingEligibilityStatus.INCONCLUSIVE


def test_invalid_observation_origin_or_status_is_rejected() -> None:
    with pytest.raises(TypeError, match="EvidenceOrigin"):
        DiscoveryObservation(
            observation_id="bad-origin",
            observed_at=NOW,
            origin="live",  # type: ignore[arg-type]
            verification_status=VerificationStatus.UNVERIFIED,
            unknowns=("origin is not validated",),
        )
    with pytest.raises(TypeError, match="VerificationStatus"):
        DiscoveryObservation(
            observation_id="bad-status",
            observed_at=NOW,
            origin=EvidenceOrigin.LOCAL_DOCUMENTATION,
            verification_status="verified",  # type: ignore[arg-type]
            unknowns=("status is not validated",),
        )


def test_manually_constructed_complete_result_is_not_accepted_as_evidence() -> None:
    from harnesses.keystone_harness import CompletenessResult

    manually_marked = CompletenessResult(CompletenessStatus.COMPLETE, 1, 1)
    with pytest.raises(TypeError, match="collection must be CollectionEvidence"):
        assess_finding_eligibility(
            collection=manually_marked,  # type: ignore[arg-type]
            completeness_contract=complete_contract(),
            contract=contract(),
            observations=(observation(), record_observation()),
            record_provenance={"r1": "record-1"},
            invariant_observation_ids=("doc-1",),
            invariant_documented=True,
        )


def test_explicitly_non_read_only_contract_rejects_finding_eligibility() -> None:
    result = eligibility(contract=contract(read_only=False))

    assert result.status is FindingEligibilityStatus.REJECTED
    assert not result.eligible_for_review
    assert not result.confirmed_defect


def test_serialized_collection_evidence_is_stable_and_redacted() -> None:
    cursor = "cursor-token=" + SECRET
    collection = CollectionEvidence((
        page(page_id="p1", ids=("r1",), more=True, next_token=cursor, total=2),
        page(page_id="p2", ids=("r2",), request=cursor, total=2),
    ))

    serialized = serialize_evidence(collection)
    result = validate_completeness(collection, complete_contract(expected_record_count=2))

    assert result.status is CompletenessStatus.COMPLETE
    assert SECRET not in serialized
    assert "[REDACTED_CURSOR]" in serialized
    assert SECRET not in repr(collection)
    assert serialized == serialize_evidence(collection)