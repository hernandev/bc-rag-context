import pytest

from bc_rag.duration import parse_duration


def test_parse_duration_units_and_bare_seconds() -> None:
    assert parse_duration("5m") == 300
    assert parse_duration("90s") == 90
    assert parse_duration("2h") == 7200
    assert parse_duration("45") == 45
    assert parse_duration(" 1.5M ") == 90


@pytest.mark.parametrize("text", ["0", "", "  ", "-5m", "m", "five", "5d"])
def test_parse_duration_rejects_bad_values(text: str) -> None:
    with pytest.raises(ValueError):
        parse_duration(text)
