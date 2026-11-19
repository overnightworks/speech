"""Chromium's precise V8 coverage of a test's page and its worklet, for the CI report.

Collecting is off unless `SPEECH_BROWSER_COVERAGE_DIR` names a directory, so local runs
pay nothing. The worklet runs in its own global scope: the page's session attaches to it
before it runs, starts the same coverage there and resumes it. Its scope ends with its
audio context, so a test takes the worklet's coverage while the capture still runs. A
target that detached before or while it is addressed is gone: it gets no coverage and
nothing waits for its reply.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from playwright.async_api import Error as PlaywrightError

if TYPE_CHECKING:
    from playwright.async_api import CDPSession, Page

COVERAGE_DIRECTORY_VARIABLE = "SPEECH_BROWSER_COVERAGE_DIR"
WORKLET_TARGET = "worklet"
PRECISE_COVERAGE = {"callCount": True, "detailed": True}
NO_SUCH_SESSION = "No session with given id"


class PageCoverage:
    """The coverage of one page and of each worklet it starts."""

    def __init__(self, directory: Path | None) -> None:
        """Write into `directory`; collect nothing when it is None."""
        self._directory = directory
        self._session: CDPSession | None = None
        self._worklets: list[str] = []
        self._replies: dict[
            int, tuple[str, asyncio.Future[dict[str, object] | None]]
        ] = {}
        self._message_ids = itertools.count(1)

    @classmethod
    def from_environment(cls) -> PageCoverage:
        """Collect into the directory the environment names, if any."""
        directory = os.environ.get(COVERAGE_DIRECTORY_VARIABLE)
        return cls(Path(directory) if directory else None)

    async def watch(self, page: Page) -> None:
        """Start the profiler on the page and on every worklet it starts."""
        if self._directory is None:
            return
        session = await page.context.new_cdp_session(page)
        self._session = session
        await session.send("Profiler.enable")
        await session.send("Profiler.startPreciseCoverage", PRECISE_COVERAGE)
        session.on("Target.attachedToTarget", self._attached)
        session.on("Target.receivedMessageFromTarget", self._received)
        session.on("Target.detachedFromTarget", self._detached)
        # Non-flat: Playwright for Python's CDPSession has no child sessions.
        await session.send(
            "Target.setAutoAttach",
            {"autoAttach": True, "waitForDebuggerOnStart": True, "flatten": False},
        )

    async def take_worklets(self) -> None:
        """Write the coverage of every worklet still running."""
        for target in self._worklets:
            taken = await self._to_target(target, "Profiler.takePreciseCoverage")
            if taken is not None:
                self._write(taken["result"])
        self._worklets.clear()

    async def write(self) -> None:
        """Write the page's coverage; call it before its context closes."""
        if self._session is not None:
            self._write(await self._session.send("Profiler.takePreciseCoverage"))

    async def _attached(self, event: dict[str, object]) -> None:
        target = str(event["sessionId"])
        info = event["targetInfo"]
        if isinstance(info, dict) and info.get("type") == WORKLET_TARGET:
            await self._to_target(target, "Profiler.enable")
            await self._to_target(
                target, "Profiler.startPreciseCoverage", PRECISE_COVERAGE
            )
            if target not in self._worklets:
                self._worklets.append(target)
        await self._to_target(target, "Runtime.runIfWaitingForDebugger")

    def _received(self, event: dict[str, object]) -> None:
        message = json.loads(str(event["message"]))
        awaited = self._replies.pop(message.get("id"), None)
        if awaited is not None and not awaited[1].done():
            awaited[1].set_result(message)

    def _detached(self, event: dict[str, object]) -> None:
        target = str(event["sessionId"])
        if target in self._worklets:
            self._worklets.remove(target)
        for message_id, (awaited_target, reply) in list(self._replies.items()):
            if awaited_target == target:
                del self._replies[message_id]
                if not reply.done():
                    reply.set_result(None)

    async def _to_target(
        self, target: str, method: str, params: dict[str, object] | None = None
    ) -> dict[str, object] | None:
        """Send to a child target; None when it is gone."""
        assert self._session is not None
        message_id = next(self._message_ids)
        reply = asyncio.get_running_loop().create_future()
        self._replies[message_id] = (target, reply)
        message = {"id": message_id, "method": method, "params": params or {}}
        try:
            await self._session.send(
                "Target.sendMessageToTarget",
                {"sessionId": target, "message": json.dumps(message)},
            )
        except PlaywrightError as error:
            self._replies.pop(message_id, None)
            if NO_SUCH_SESSION not in str(error):
                raise
            return None
        return await reply

    def _write(self, coverage: object) -> None:
        assert self._directory is not None
        self._directory.mkdir(parents=True, exist_ok=True)
        (self._directory / f"{uuid4().hex}.json").write_text(json.dumps(coverage))
