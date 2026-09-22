import pytest

from bc_rag.cli import parse_every


def test_parse_every_minutes_and_seconds() -> None:
    assert parse_every("5m") == 300
    assert parse_every("90s") == 90
    assert parse_every("2h") == 7200
    assert parse_every("45") == 45


def test_parse_every_rejects_zero() -> None:
    with pytest.raises(ValueError):
        parse_every("0")
    with pytest.raises(ValueError):
        parse_every("")
