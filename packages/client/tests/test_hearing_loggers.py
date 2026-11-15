"""The hearing helper and its import change no logger of the process."""

from __future__ import annotations

import subprocess
import sys

# Run in a fresh interpreter: the helper's import is part of what it must not do to a
# logger. hearing and relay set their own loggers up first, which is allowed.
_LOGGER_PROBE = """
import asyncio, logging, sys
from websockets.asyncio.server import serve
import speech_client.hearing, speech_client.relay
from speech_client.hearing import SessionTokens
from speech_client.wire import (
    DurationUsage, Language, SessionCreated, TranscriptCompleted,
    TranscriptionSession, TranscriptLanguage,
)

CREATED_EVENT = SessionCreated(
    session=TranscriptionSession(type="transcription")
).model_dump_json()

QUIET = (logging.NOTSET, (), (), False)


def snapshot():
    loggers = {"": logging.root} | {
        name: logger
        for name, logger in logging.root.manager.loggerDict.items()
        if isinstance(logger, logging.Logger)
    }
    return {
        name: (logger.level, tuple(logger.handlers), tuple(logger.filters),
               logger.disabled)
        for name, logger in loggers.items()
    }


class Browser:
    async def receive(self):
        await asyncio.Event().wait()

    async def close(self, code):
        pass


async def hear():
    from speech_client.utterances import hear_utterances

    async def tokens():
        return SessionTokens("w", "p")

    async def service(connection):
        await connection.recv()
        await connection.send(CREATED_EVENT)
        await connection.send(TranscriptCompleted(
            item_id="a", transcript="x", usage=DurationUsage(seconds=1),
            languages=(TranscriptLanguage(code=Language.DE),),
        ).model_dump_json())
        await connection.wait_closed()

    ended = asyncio.Event()

    async def on_utterance(utterance):
        ended.set()

    async with serve(service, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        await asyncio.wait_for(hear_utterances(
            Browser(), f"http://127.0.0.1:{port}",
            token_source=tokens, languages=(Language.DE,),
            deadline_seconds=5, held_limit_bytes=1 << 20, opening_seconds=5,
            ended=ended, on_utterance=on_utterance, on_partial=None,
        ), 10)


before = snapshot()
asyncio.run(hear())
after = snapshot()
changed = {
    name: state for name, state in after.items()
    if state != before.get(name, QUIET)
}
sys.exit(repr(sorted(changed)) if changed else 0)
"""


def test_a_hearing_changes_no_logger_and_its_import_changes_none() -> None:
    done = subprocess.run(  # noqa: S603 - a fixed probe of this interpreter
        [sys.executable, "-c", _LOGGER_PROBE],
        check=False,
        capture_output=True,
        text=True,
    )

    assert done.returncode == 0, done.stderr
