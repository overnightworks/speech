"""The hearing proof's client reads the answer that refuses a live upgrade."""

from __future__ import annotations

import asyncio
from http import HTTPStatus
from types import SimpleNamespace

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from speech_runtime_client.wire import RefusalCode, refusal
from tests.support.identity_support import MINTING
from tests.support.script_support import script_module

CLIENT = script_module("hearing_client")


class Upgrade:
    """A stand-in for the WebSocket connection of a service."""

    def __init__(self, outcome: Exception | None) -> None:
        """Remember the outcome."""
        self.outcome = outcome

    async def __aenter__(self) -> None:
        """Refuse the upgrade with the outcome, if there is one."""
        if self.outcome is not None:
            raise self.outcome

    async def __aexit__(self, *_: object) -> None:
        """Close nothing."""


SERVICE = SimpleNamespace(base_url="http://127.0.0.1:1", minting=MINTING)


def refused_with(status: HTTPStatus, body: bytes = b"") -> InvalidStatus:
    return InvalidStatus(Response(status, status.phrase, Headers(), body))


def unavailable_body() -> bytes:
    return refusal(RefusalCode.UNAVAILABLE).model_dump_json().encode()


def test_a_refused_upgrade_gives_the_status_and_the_code_that_refused_it() -> None:
    refused = Upgrade(refused_with(HTTPStatus.SERVICE_UNAVAILABLE, unavailable_body()))

    answer = asyncio.run(CLIENT.upgrade_refusal(SERVICE, lambda *_, **__: refused))

    assert answer == CLIENT.UpgradeRefusal(
        HTTPStatus.SERVICE_UNAVAILABLE, RefusalCode.UNAVAILABLE
    )


def test_a_refused_upgrade_without_the_contract_body_gives_no_code() -> None:
    refused = Upgrade(refused_with(HTTPStatus.SERVICE_UNAVAILABLE))

    answer = asyncio.run(CLIENT.upgrade_refusal(SERVICE, lambda *_, **__: refused))

    assert answer == CLIENT.UpgradeRefusal(HTTPStatus.SERVICE_UNAVAILABLE, None)


@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        (HTTPStatus.SERVICE_UNAVAILABLE, unavailable_body(), RefusalCode.UNAVAILABLE),
        (
            HTTPStatus.SERVICE_UNAVAILABLE,
            refusal(RefusalCode.BUSY).model_dump_json().encode(),
            RefusalCode.BUSY,
        ),
        (HTTPStatus.SERVICE_UNAVAILABLE, b"", None),
        (HTTPStatus.SERVICE_UNAVAILABLE, b"not json", None),
        (HTTPStatus.OK, unavailable_body(), None),
    ],
)
def test_only_a_503_with_the_contract_body_gives_a_refusal_code(
    status: HTTPStatus, body: bytes, code: RefusalCode | None
) -> None:
    assert CLIENT.refusal_code(status, body) == code


def test_an_accepted_upgrade_gives_no_refusal() -> None:
    accepted = Upgrade(None)

    assert (
        asyncio.run(CLIENT.upgrade_refusal(SERVICE, lambda *_, **__: accepted)) is None
    )
