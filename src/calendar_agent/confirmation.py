"""Deterministic confirmation checks for proposed Calendar changes."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal, InvalidOperation
import re


_DURATION = re.compile(
    r"(?<![\w.])(?P<amount>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>minutes?|mins?|hours?|hrs?)\b",
    re.IGNORECASE,
)
_AFFIRMATIVE = re.compile(
    r"\b(confirm(?:ed)?|approve(?:d)?|yes|yeah|yep|ok|okay|fine|"
    r"agree(?:d)?|accept(?:ed)?|go ahead|proceed|acceptable|works?)\b",
    re.IGNORECASE,
)
_CONFIRMATION = re.compile(
    r"^\s*(?:(?:yes|yeah|yep)\s*[,!.]?\s*)?"
    r"(?:(?:i\s+)?(?:confirm(?:ed)?|approve(?:d)?|agree(?:d)?|"
    r"accept(?:ed)?|proceed(?:\s+with)?|go\s+ahead(?:\s+with)?)"
    r"\s+(?:the\s+)?(?:shift\s+of\s+)?)?"
    r"(?P<duration>\d+(?:\.\d+)?\s*(?:minutes?|mins?|hours?|hrs?))"
    r"(?:\s+(?:shift|move))?"
    r"(?:\s+(?:is\s+)?(?:okay|ok|fine|confirmed|approved|agreed|"
    r"acceptable|works?))?"
    r"(?:\s+please)?[.!]*\s*$",
    re.IGNORECASE,
)


def confirms_shift(message: object, proposed_shift: timedelta) -> bool:
    """Accept only an unambiguous affirmative that repeats the proposed shift."""

    if (
        not isinstance(message, str)
        or not message.strip()
        or not isinstance(proposed_shift, timedelta)
        or proposed_shift <= timedelta(0)
    ):
        return False
    text = message.strip()
    match = _CONFIRMATION.fullmatch(text)
    if match is None or not _AFFIRMATIVE.search(text):
        return False

    expected_minutes = Decimal(str(proposed_shift.total_seconds())) / Decimal(60)
    duration = _DURATION.fullmatch(match.group("duration"))
    if duration is None:
        return False
    try:
        amount = Decimal(duration.group("amount"))
    except InvalidOperation:
        return False
    unit = duration.group("unit").casefold()
    minutes = amount * (Decimal(60) if unit.startswith("h") else Decimal(1))
    return minutes == expected_minutes
