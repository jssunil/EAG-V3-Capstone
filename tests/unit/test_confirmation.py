from datetime import timedelta

import pytest

from calendar_agent.confirmation import confirms_shift


@pytest.mark.parametrize(
    "message",
    [
        "Confirm 30 minutes",
        "30 minutes is okay",
        "Yes, 30 minutes is fine.",
        "I approve the shift of 0.5 hours",
        "Go ahead with 30 mins",
    ],
)
def test_confirms_shift_accepts_only_clear_matching_affirmations(message: str) -> None:
    assert confirms_shift(message, timedelta(minutes=30))


@pytest.mark.parametrize(
    "message",
    [
        "No, 30 minutes",
        "Don't move it; 30 minutes is okay",
        "Confirm 30 minutes instead of 60 minutes",
        "Maybe 30 minutes?",
        "Yes, 30 minutes, and update the title",
        "30 minutes",
        "Confirm 45 minutes",
    ],
)
def test_confirms_shift_rejects_negation_ambiguity_and_unrelated_text(
    message: str,
) -> None:
    assert not confirms_shift(message, timedelta(minutes=30))
