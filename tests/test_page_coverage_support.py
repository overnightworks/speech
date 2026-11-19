"""The page coverage harness takes every worklet that still lives."""

from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from tests.support.page_coverage_support import PageCoverage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from playwright.async_api import Page

TAKE_COVERAGE = "Profiler.takePreciseCoverage"


class FakeSession:
    """A page session whose worklets answer, except one that detaches mid-request."""

    def __init__(self, detaching: str) -> None:
        """Detach `detaching` when a request reaches it."""
        self.handlers: dict[
            str, Callable[[dict[str, object]], Awaitable[None] | None]
        ] = {}
        self._detaching = detaching

    def on(
        self,
        event: str,
        handler: Callable[[dict[str, object]], Awaitable[None] | None],
    ) -> None:
        """Register the handler like a CDP session."""
        self.handlers[event] = handler

    async def send(
        self, method: str, params: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Answer a send; a request to the detaching target detaches it instead."""
        if method != "Target.sendMessageToTarget":
            return {}
        assert params is not None
        target = str(params["sessionId"])
        message = json.loads(str(params["message"]))
        if target == self._detaching and message["method"] == TAKE_COVERAGE:
            self.handlers["Target.detachedFromTarget"]({"sessionId": target})
        else:
            reply = {"id": message["id"], "result": {"result": [target]}}
            self.handlers["Target.receivedMessageFromTarget"](
                {"message": json.dumps(reply)}
            )
        return {}


def test_worklet_still_living_is_written_when_the_one_before_it_detaches(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        session = FakeSession(detaching="first")
        page = SimpleNamespace(
            context=SimpleNamespace(new_cdp_session=_returning(session))
        )
        coverage = PageCoverage(tmp_path)
        await coverage.watch(cast("Page", page))
        for target in ("first", "second"):
            attached = session.handlers["Target.attachedToTarget"](
                {"sessionId": target, "targetInfo": {"type": "worklet"}}
            )
            assert inspect.isawaitable(attached)
            await attached
        await coverage.take_worklets()

    asyncio.run(scenario())

    written = [json.loads(file.read_text()) for file in tmp_path.glob("*.json")]
    assert written == [{"result": ["second"]}]


def _returning(session: FakeSession) -> Callable[[object], Awaitable[FakeSession]]:
    async def new_cdp_session(_page: object) -> FakeSession:
        return session

    return new_cdp_session
