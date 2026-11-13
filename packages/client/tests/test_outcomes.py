"""The `busy` wait an app reads from a refusal."""

from http import HTTPStatus

import pytest

from speech_client import outcomes, wire
from speech_client.outcomes import BusyWait, RetryAfterError, busy_wait
from speech_client.wire import REFUSAL_STATUS, RETRY_AFTER_HEADER, RefusalCode

BUSY = REFUSAL_STATUS[RefusalCode.BUSY]


def test_busy_answer_gives_the_wait_in_seconds() -> None:
    """WIRE-104: a busy answer gives the wait in whole seconds."""
    assert busy_wait(BUSY, {RETRY_AFTER_HEADER: "5"}) == BusyWait(retry_after_seconds=5)


@pytest.mark.parametrize("name", ["BusyWait", "RetryAfterError", "busy_wait"])
def test_wire_module_serves_the_busy_wait(name: str) -> None:
    """WIRE-104: `busy_wait` is imported from `speech_client.wire`."""
    assert getattr(wire, name) is getattr(outcomes, name)


def test_wire_module_has_no_unknown_attribute() -> None:
    with pytest.raises(AttributeError):
        wire.not_a_wire_name  # noqa: B018 - the attribute read is the call under test


def test_header_name_is_matched_case_insensitively() -> None:
    assert busy_wait(int(BUSY), {RETRY_AFTER_HEADER.lower(): "1"}) == BusyWait(1)


@pytest.mark.parametrize(
    "status",
    [s for s in (*REFUSAL_STATUS.values(), HTTPStatus.OK) if s != BUSY],
)
def test_other_statuses_give_no_wait(status: HTTPStatus) -> None:
    """WIRE-104: any status but busy gives no wait."""
    assert busy_wait(status, {RETRY_AFTER_HEADER: "5"}) is None


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {RETRY_AFTER_HEADER: "-1"},
        {RETRY_AFTER_HEADER: "1.5"},
        {RETRY_AFTER_HEADER: "Wed, 21 Oct 2026 07:28:00 GMT"},
        {RETRY_AFTER_HEADER: ""},
        {RETRY_AFTER_HEADER: "0"},
    ],
    ids=["missing", "negative", "fraction", "http-date", "empty", "zero"],
)
def test_busy_answer_without_whole_seconds_fails_loud(headers: dict[str, str]) -> None:
    """WIRE-104: a busy answer without whole seconds fails loud."""
    with pytest.raises(RetryAfterError, match=RETRY_AFTER_HEADER):
        busy_wait(BUSY, headers)
