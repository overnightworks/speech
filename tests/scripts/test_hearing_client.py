"""The hearing proof's client reads the status that refuses a live upgrade."""

from __future__ import annotations

import asyncio
from http import HTTPStatus
from types import SimpleNamespace

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

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


SERVICE = SimpleNamespace(base_url="http://127.0.0.1:1", minting=None)


@pytest.fixture(autouse=True)
def tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CLIENT, "token_headers", lambda _: {})


def refused_with(status: HTTPStatus) -> InvalidStatus:
    return InvalidStatus(Response(status, status.phrase, Headers()))


def test_a_refused_upgrade_gives_the_status_that_refused_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        CLIENT,
        "connect",
        lambda *_, **__: Upgrade(refused_with(HTTPStatus.SERVICE_UNAVAILABLE)),
    )

    assert asyncio.run(CLIENT.upgrade_status(SERVICE)) == HTTPStatus.SERVICE_UNAVAILABLE


def test_an_accepted_upgrade_gives_no_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CLIENT, "connect", lambda *_, **__: Upgrade(None))

    assert asyncio.run(CLIENT.upgrade_status(SERVICE)) is None
