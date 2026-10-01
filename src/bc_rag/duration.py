"""Durations written by people, such as 5m, 90s, 2h, or a bare number of seconds."""

from __future__ import annotations

_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0}


def parse_duration(text: str) -> float:
    """Seconds for `text`. Raises ValueError on an empty, malformed, or non-positive value."""
    raw = text.strip().lower()
    if not raw:
        raise ValueError("a duration is required, like 5m, 90s, or 2h")
    suffix = raw[-1]
    try:
        if suffix in _UNITS and raw[:-1]:
            seconds = _UNITS[suffix] * float(raw[:-1])
        else:
            seconds = float(raw)
    except ValueError:
        raise ValueError(f"{text!r} is not a duration like 5m, 90s, or 2h") from None
    if seconds <= 0:
        raise ValueError(f"duration {text!r} must be greater than 0")
    return seconds
