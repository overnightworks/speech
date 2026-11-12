"""The finished-utterance helper against a scripted browser and a scripted service."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from packages.client.tests.support.hearing_support import (
    HEARD_TEXT,
    LOOPBACK,
    PERSON_TOKEN,
    WORKLOAD_TOKEN,
    silent_peer,
)
from speech_client.hearing import Closed, Refused, session_update
from speech_client.outcomes import (
    Busy,
    Cut,
    Malformed,
    OverCap,
    PastDeadline,
    Unreachable,
)
from speech_client.relay import (
    HOP_FRAME_BYTES,
    AppEnded,
    BrowserLeft,
    HopClose,
    RuleBroken,
)
from speech_client.utterances import (
    Partial,
    ServiceFailed,
    Utterance,
    hear_utterances,
)
from speech_client.wire import (
    BusyWait,
    DurationUsage,
    Language,
    RefusalCode,
    SpeechStarted,
    SpeechStopped,
    TranscriptCompleted,
    TranscriptDelta,
    TranscriptFailed,
    TranscriptLanguage,
    error_event,
    item_failure,
    token_headers,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

    from websockets.asyncio.server import ServerConnection
    from websockets.http11 import Request

    from speech_client.utterances import UtteranceOutcome
    from speech_client.wire import ServerEvent

type Peer = Callable[[], contextlib.AbstractAsyncContextManager[int]]

DISCONNECT = {"type": "websocket.disconnect", "code": 1001}
# Far below the session deadline, so a stalled opening is told from a late one.
STALLED_OPENING_SECONDS = 0.3
WAIT_SECONDS = 10


def _ended_already() -> asyncio.Event:
    ended = asyncio.Event()
    ended.set()
    return ended


def _binary(size: int) -> dict[str, object]:
    return {"type": "websocket.receive", "bytes": b"\0" * size, "text": None}


def _text() -> dict[str, object]:
    return {"type": "websocket.receive", "bytes": None, "text": "hello"}


def _delta(item_id: str, text: str) -> TranscriptDelta:
    return TranscriptDelta(item_id=item_id, delta=text)


def _started(item_id: str, at_ms: int) -> SpeechStarted:
    return SpeechStarted(item_id=item_id, audio_start_ms=at_ms)


def _stopped(item_id: str, at_ms: int) -> SpeechStopped:
    return SpeechStopped(item_id=item_id, audio_end_ms=at_ms)


def _completed(
    item_id: str, text: str, language: Language = Language.DE
) -> TranscriptCompleted:
    return TranscriptCompleted(
        item_id=item_id,
        transcript=text,
        usage=DurationUsage(seconds=1),
        languages=(TranscriptLanguage(code=language),),
    )


def _utterance(
    text: str,
    language: Language = Language.DE,
    start: int | None = None,
    end: int | None = None,
) -> Utterance:
    return Utterance(text, language, audio_start_ms=start, audio_end_ms=end)


class _Browser:
    """A browser socket that plays its messages, then waits for its peer.

    It offers only what the relay calls: nothing can be written to it but a close.
    """

    def __init__(self, messages: tuple[Mapping[str, object], ...] = ()) -> None:
        self._messages = list(messages)
        self.closed_with: list[int] = []

    async def receive(self) -> Mapping[str, object]:
        if self._messages:
            return self._messages.pop(0)
        await asyncio.Event().wait()
        raise AssertionError

    async def close(self, code: int) -> None:
        self.closed_with.append(code)


@dataclass
class _Recorder:
    """The app's side: what its callbacks got, in order, and how they behave."""

    ended: asyncio.Event = field(default_factory=asyncio.Event)
    ends_after_first: bool = False
    raises: Exception | None = None
    got: list[Utterance | Partial] = field(default_factory=list)

    async def on_utterance(self, utterance: Utterance) -> None:
        await self._take(utterance)

    async def on_partial(self, partial: Partial) -> None:
        await self._take(partial)

    async def _take(self, heard: Utterance | Partial) -> None:
        self.got.append(heard)
        if self.ends_after_first:
            self.ended.set()
        if self.raises is not None:
            raise self.raises


@dataclass(frozen=True)
class _Bounds:
    deadline_seconds: float = 5
    held_limit_bytes: int = 2 * HOP_FRAME_BYTES
    opening_seconds: float = 5


def _says(
    *events: ServerEvent, stays_open: bool = False
) -> Callable[[ServerConnection], Awaitable[None]]:
    async def handler(connection: ServerConnection) -> None:
        await connection.recv()
        for event in events:
            await connection.send(event.model_dump_json())
        if stays_open:
            await connection.wait_closed()

    return handler


async def _speaks_nonsense(connection: ServerConnection) -> None:
    await connection.send("not json")
    await connection.wait_closed()


async def _drops_the_connection(connection: ServerConnection) -> None:
    connection.transport.abort()


@contextlib.asynccontextmanager
async def _service_saying(
    handler: Callable[[ServerConnection], Awaitable[None]],
) -> AsyncIterator[int]:
    async with serve(handler, LOOPBACK, 0) as server:
        yield server.sockets[0].getsockname()[1]


@contextlib.asynccontextmanager
async def _service_refusing(
    status: HTTPStatus, *headers: tuple[str, str]
) -> AsyncIterator[int]:
    def refuse(_: ServerConnection, __: Request) -> Response:
        return Response(status.value, status.phrase, Headers(headers), b"")

    async with serve(_says(), LOOPBACK, 0, process_request=refuse) as server:
        yield server.sockets[0].getsockname()[1]


@contextlib.asynccontextmanager
async def _service_not_answering() -> AsyncIterator[int]:
    async with await asyncio.start_server(silent_peer, LOOPBACK, 0) as server:
        yield server.sockets[0].getsockname()[1]


async def _hear(  # noqa: PLR0913 - one keyword per thing a scenario varies
    peer: Peer,
    browser: _Browser | None = None,
    recorder: _Recorder | None = None,
    *,
    partials: bool = True,
    languages: tuple[Language, ...] = (Language.DE,),
    bounds: _Bounds = _Bounds(),  # noqa: B008 - immutable
) -> UtteranceOutcome:
    recorder = recorder or _Recorder()
    async with peer() as port:
        return await asyncio.wait_for(
            hear_utterances(
                browser or _Browser(),
                f"http://{LOOPBACK}:{port}",
                workload_token=WORKLOAD_TOKEN,
                person_token=PERSON_TOKEN,
                languages=languages,
                deadline_seconds=bounds.deadline_seconds,
                held_limit_bytes=bounds.held_limit_bytes,
                opening_seconds=bounds.opening_seconds,
                ended=recorder.ended,
                on_utterance=recorder.on_utterance,
                on_partial=recorder.on_partial if partials else None,
            ),
            WAIT_SECONDS,
        )


def _hearing(*events: ServerEvent, stays_open: bool = False) -> Peer:
    return functools.partial(_service_saying, _says(*events, stays_open=stays_open))


def test_one_session_goes_to_the_origin_with_both_tokens_and_the_languages() -> None:
    languages = (Language.EN, Language.DE)
    arrived: list[tuple[Mapping[str, str], str | bytes]] = []

    async def records(connection: ServerConnection) -> None:
        arrived.append((dict(connection.request.headers), await connection.recv()))

    outcome = asyncio.run(
        _hear(functools.partial(_service_saying, records), languages=languages)
    )

    (headers, first), *_ = arrived
    assert outcome == Closed()
    assert (
        token_headers(workload_token=WORKLOAD_TOKEN, person_token=PERSON_TOKEN).items()
        <= {name.title(): value for name, value in headers.items()}.items()
    )
    assert first == session_update(languages)


@pytest.mark.parametrize("language", list(Language))
def test_a_completed_transcript_reaches_the_app_once_with_its_text_and_language(
    language: Language,
) -> None:
    recorder = _Recorder()

    asyncio.run(_hear(_hearing(_completed("a", "guten Tag", language)), None, recorder))

    assert recorder.got == [_utterance("guten Tag", language)]


def test_an_empty_transcript_reaches_nobody_and_the_next_one_still_does() -> None:
    recorder = _Recorder()

    asyncio.run(
        _hear(_hearing(_completed("a", ""), _completed("b", "weiter")), None, recorder)
    )

    assert recorder.got == [_utterance("weiter")]


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        pytest.param(
            [_started("a", 100), _stopped("a", 900), _completed("a", "x")],
            [_utterance("x", start=100, end=900)],
            id="both-positions",
        ),
        pytest.param(
            [_stopped("a", 900), _completed("a", "x")],
            [_utterance("x", end=900)],
            id="no-speech-started",
        ),
        pytest.param(
            [_started("a", 100), _completed("a", "x")],
            [_utterance("x", start=100)],
            id="no-speech-stopped",
        ),
        pytest.param([_completed("a", "x")], [_utterance("x")], id="no-position"),
        pytest.param(
            [
                _started("a", 100),
                _started("b", 500),
                _stopped("a", 400),
                _stopped("b", 900),
                _completed("b", "second"),
                _completed("a", "first"),
            ],
            [
                _utterance("second", start=500, end=900),
                _utterance("first", start=100, end=400),
            ],
            id="two-items-interleaved",
        ),
        pytest.param(
            [
                _started("a", 100),
                _stopped("a", 400),
                _completed("a", "first"),
                _completed("a", "again"),
            ],
            [_utterance("first", start=100, end=400), _utterance("again")],
            id="nothing-kept-once-an-item-completed",
        ),
        pytest.param(
            [
                _started("a", 100),
                _stopped("a", 400),
                _completed("a", ""),
                _completed("a", "again"),
            ],
            [_utterance("again")],
            id="nothing-kept-of-an-item-that-completed-empty",
        ),
    ],
)
def test_an_utterance_carries_the_positions_of_its_own_item(
    events: list[ServerEvent], expected: list[Utterance]
) -> None:
    recorder = _Recorder()

    asyncio.run(_hear(_hearing(*events), None, recorder))

    assert recorder.got == expected


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        pytest.param(
            [_delta("a", "gu"), _delta("a", "ten"), _completed("a", "guten")],
            [Partial("gu"), Partial("guten"), _utterance("guten")],
            id="the-text-so-far",
        ),
        pytest.param(
            [_delta("a", "gu"), _delta("b", "he"), _delta("a", "t"), _delta("b", "y")],
            [Partial("gu"), Partial("he"), Partial("gut"), Partial("hey")],
            id="two-items-interleaved",
        ),
        pytest.param(
            [_delta("a", "one"), _completed("a", "one"), _delta("b", "two")],
            [Partial("one"), _utterance("one"), Partial("two")],
            id="the-next-item-starts-anew",
        ),
        pytest.param(
            [_delta("a", "one"), _completed("a", ""), _delta("a", "two")],
            [Partial("one"), Partial("two")],
            id="the-same-id-starts-anew-after-an-empty-one",
        ),
    ],
)
def test_with_on_partial_each_delta_arrives_as_its_items_text_so_far(
    events: list[ServerEvent], expected: list[Utterance | Partial]
) -> None:
    recorder = _Recorder()

    asyncio.run(_hear(_hearing(*events), None, recorder))

    assert recorder.got == expected


def test_without_on_partial_nothing_of_a_delta_reaches_the_app() -> None:
    recorder = _Recorder()

    asyncio.run(
        _hear(
            _hearing(_delta("a", "gu"), _delta("a", "ten"), _completed("a", "guten")),
            None,
            recorder,
            partials=False,
        )
    )

    assert recorder.got == [_utterance("guten")]


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(
            TranscriptFailed(
                item_id="a", error=item_failure(RefusalCode.UNAVAILABLE, "model")
            ),
            id="a-failed-item",
        ),
        pytest.param(
            error_event(RefusalCode.UNAVAILABLE, "model"), id="an-error-event"
        ),
    ],
)
def test_a_failure_event_ends_the_hearing_as_a_service_failure_naming_only_its_code(
    failure: ServerEvent,
) -> None:
    recorder = _Recorder()
    browser = _Browser()
    after = _completed("b", "never")

    outcome = asyncio.run(
        _hear(
            _hearing(_completed("a", "before"), failure, after, stays_open=True),
            browser,
            recorder,
        )
    )

    assert outcome == ServiceFailed(RefusalCode.UNAVAILABLE)
    assert recorder.got == [_utterance("before")]
    assert browser.closed_with == [HopClose.SERVICE_FAILED]
    assert "model" not in repr(outcome)
    assert failure.error.message not in repr(outcome)


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(_completed("a", "x"), id="the-utterance-callback"),
        pytest.param(_delta("a", "x"), id="the-partial-callback"),
    ],
)
def test_a_failure_of_the_apps_callback_is_raised_unchanged(
    event: ServerEvent,
) -> None:
    failure = RuntimeError("the app failed")
    browser = _Browser()

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(
            _hear(_hearing(event, stays_open=True), browser, _Recorder(raises=failure))
        )

    assert raised.value is failure
    assert browser.closed_with == []


@pytest.mark.parametrize(
    ("peer", "messages", "bounds", "outcome", "close"),
    [
        pytest.param(_hearing(), (), _Bounds(), Closed(), HopClose.NORMAL, id="closed"),
        pytest.param(
            functools.partial(_service_refusing, HTTPStatus.FORBIDDEN),
            (),
            _Bounds(),
            Refused(403, None),
            HopClose.SERVICE_FAILED,
            id="refused",
        ),
        pytest.param(
            functools.partial(
                _service_refusing, HTTPStatus.TOO_MANY_REQUESTS, ("Retry-After", "7")
            ),
            (),
            _Bounds(),
            Busy(BusyWait(retry_after_seconds=7)),
            HopClose.BUSY,
            id="busy",
        ),
        pytest.param(
            functools.partial(_service_saying, _speaks_nonsense),
            (),
            _Bounds(),
            Malformed(),
            HopClose.SERVICE_FAILED,
            id="malformed",
        ),
        pytest.param(
            functools.partial(_service_saying, _drops_the_connection),
            (),
            _Bounds(),
            Cut(),
            HopClose.SERVICE_FAILED,
            id="cut",
        ),
        pytest.param(
            _service_not_answering,
            (),
            _Bounds(opening_seconds=STALLED_OPENING_SECONDS),
            Unreachable(),
            HopClose.SERVICE_FAILED,
            id="the-opening-bound-passes",
        ),
        pytest.param(
            _service_not_answering,
            (),
            _Bounds(deadline_seconds=STALLED_OPENING_SECONDS),
            PastDeadline(),
            HopClose.PAST_DEADLINE,
            id="the-deadline-passes",
        ),
        pytest.param(
            _hearing(stays_open=True),
            (_binary(2),),
            _Bounds(held_limit_bytes=1),
            OverCap(),
            HopClose.OVER_CAP,
            id="the-held-limit-is-passed",
        ),
        pytest.param(
            _hearing(stays_open=True),
            (_binary(HOP_FRAME_BYTES + 1),),
            _Bounds(),
            RuleBroken(HopClose.FRAME_TOO_LARGE),
            HopClose.FRAME_TOO_LARGE,
            id="a-frame-over-the-hop-frame-limit",
        ),
        pytest.param(
            _hearing(stays_open=True),
            (_text(),),
            _Bounds(),
            RuleBroken(HopClose.TEXT_FRAME),
            HopClose.TEXT_FRAME,
            id="a-text-frame",
        ),
        pytest.param(
            _hearing(stays_open=True),
            (DISCONNECT,),
            _Bounds(),
            BrowserLeft(),
            None,
            id="the-browser-leaves",
        ),
        pytest.param(
            _hearing(stays_open=True),
            (_binary(HOP_FRAME_BYTES), DISCONNECT),
            _Bounds(),
            BrowserLeft(),
            None,
            id="a-frame-exactly-the-hop-frame-limit-does-not-end-the-hop",
        ),
    ],
)
def test_every_other_end_is_the_relays_outcome_and_closes_the_browser_with_its_code(
    peer: Peer,
    messages: tuple[Mapping[str, object], ...],
    bounds: _Bounds,
    outcome: UtteranceOutcome,
    close: HopClose | None,
) -> None:
    browser = _Browser(messages)

    heard = asyncio.run(_hear(peer, browser, bounds=bounds))

    assert heard == outcome
    assert browser.closed_with == ([] if close is None else [close])


@pytest.mark.parametrize(
    ("events", "recorder", "expected"),
    [
        pytest.param(
            [_completed("a", "late")],
            _Recorder(ended=_ended_already()),
            [],
            id="an-utterance-after-the-end-set-before-the-call",
        ),
        pytest.param(
            [_delta("a", "late")],
            _Recorder(ended=_ended_already()),
            [],
            id="a-partial-after-the-end-set-before-the-call",
        ),
        pytest.param(
            [_completed("a", "first"), _completed("b", "late")],
            _Recorder(ends_after_first=True),
            [_utterance("first")],
            id="an-utterance-after-the-end-set-by-the-first",
        ),
        pytest.param(
            [_delta("a", "first"), _delta("b", "late")],
            _Recorder(ends_after_first=True),
            [Partial("first")],
            id="a-partial-after-the-end-set-by-the-first",
        ),
    ],
)
def test_once_ended_is_set_no_utterance_and_no_partial_reaches_the_app(
    events: list[ServerEvent],
    recorder: _Recorder,
    expected: list[Utterance | Partial],
) -> None:
    browser = _Browser()

    outcome = asyncio.run(_hear(_hearing(*events, stays_open=True), browser, recorder))

    assert outcome == AppEnded()
    assert recorder.got == expected
    assert browser.closed_with == [HopClose.NORMAL]


def test_no_log_and_no_outcome_carries_a_token_or_heard_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    browser = _Browser()

    outcome = asyncio.run(
        _hear(
            _hearing(
                _delta("a", HEARD_TEXT),
                _completed("a", HEARD_TEXT),
                error_event(RefusalCode.UNAVAILABLE),
                stays_open=True,
            ),
            browser,
        )
    )

    # The scripted service logs its own side of the socket.
    client_log = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name != "websockets.server"
    )
    for secret in (WORKLOAD_TOKEN, PERSON_TOKEN, HEARD_TEXT):
        assert secret not in client_log
        assert secret not in repr(outcome)
        assert secret not in repr(browser.closed_with)


# Run in a fresh interpreter: the helper's import is part of what it must not do to a
# logger. hearing and relay set their own loggers up first, which is allowed.
_LOGGER_PROBE = """
import asyncio, logging, sys
from websockets.asyncio.server import serve
import speech_client.hearing, speech_client.relay
from speech_client.wire import (
    DurationUsage, Language, TranscriptCompleted, TranscriptLanguage,
)

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

    async def service(connection):
        await connection.recv()
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
            workload_token="w", person_token="p", languages=(Language.DE,),
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
