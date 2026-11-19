"""The page coverage harness takes every worklet that still lives."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

from tests.support.page_coverage_support import PageCoverage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


class FakeSession:
    """A page session whose worklets answer, except one that detaches mid-request."""

    def __init__(self, detaching: str) -> None:
        """Detach `detaching` when a request reaches it."""
        self.handlers: dict[str, object] = {}
        self._detaching = detaching

    def on(self, event: str, handler: object) -> None:
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
        if target == self._detaching:
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
        await coverage.watch(page)  # type: ignore[arg-type]
        coverage._worklets.extend(["first", "second"])  # noqa: SLF001
        await coverage.take_worklets()

    asyncio.run(scenario())

    written = [json.loads(file.read_text()) for file in tmp_path.glob("*.json")]
    assert written == [{"result": ["second"]}]


def _returning(session: FakeSession) -> Callable[[object], Awaitable[FakeSession]]:
    async def new_cdp_session(_page: object) -> FakeSession:
        return session

    return new_cdp_session
