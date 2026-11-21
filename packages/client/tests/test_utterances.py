"""The finished-utterance helper against a scripted browser and a scripted service."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest

import speech_client.hearing
from packages.client.tests.support.hearing_support import (
    HEARD_TEXT,
    PERSON_TOKEN,
    WORKLOAD_TOKEN,
)
from packages.client.tests.support.scripted_hearing_support import (
    Refusal,
    drains,
    says,
    scripted,
)
from packages.client.tests.support.utterances_support import (
    Bounds,
    Browser,
    Recorder,
    binary_message,
    completed,
    delta,
    drops_the_connection,
    ended_already,
    hear,
    hearing,
    service_not_answering,
    service_refusing,
    service_saying,
    sound_detected,
    speaks_nonsense,
    started,
    stopped,
    text_message,
    utterance_of,
)
from speech_client.hearing import (
    Closed,
    NoToken,
    ReconnectBound,
    ReconnectCause,
    Reconnected,
    ReconnectFailed,
    Refused,
    SessionTokens,
    session_update,
)
from speech_client.outcomes import (
    Busy,
    BusyWait,
    Cut,
    Malformed,
    OverCap,
    PastDeadline,
    Unreachable,
)
from speech_client.realtime_wire import (
    ServerEventError,
    SoundName,
    TranscriptFailed,
    error_event,
    item_failure,
    parse_client_event,
)
from speech_client.relay import (
    HOP_FRAME_BYTES,
    AppEnded,
    BrowserLeft,
    HopClose,
    RuleBroken,
)
from speech_client.utterances import (
    HeardSound,
    Partial,
    ServiceFailed,
    Utterance,
)
from speech_client.wire import DRAIN_CLOSE_CODE, Language, RefusalCode, token_headers

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping

    from websockets.asyncio.server import ServerConnection

    from packages.client.tests.support.scripted_hearing_support import Script
    from speech_client.realtime_wire import ServerEvent
    from speech_client.utterances import UtteranceOutcome

type Peer = Callable[[], contextlib.AbstractAsyncContextManager[int]]

DISCONNECT = {"type": "websocket.disconnect", "code": 1001}
# Far below the session deadline, so a stalled opening is told from a late one.
STALLED_OPENING_SECONDS = 0.3


def test_one_session_goes_to_the_origin_with_both_tokens_and_the_languages() -> None:
    languages = (Language.EN, Language.DE)
    arrived: list[tuple[Mapping[str, str], str | bytes]] = []

    async def records(connection: ServerConnection) -> None:
        arrived.append((dict(connection.request.headers), await connection.recv()))

    outcome = asyncio.run(
        hear(functools.partial(service_saying, records), languages=languages)
    )

    (headers, first), *_ = arrived
    assert outcome == Closed()
    assert (
        token_headers(workload_token=WORKLOAD_TOKEN, person_token=PERSON_TOKEN).items()
        <= {name.title(): value for name, value in headers.items()}.items()
    )
    assert first == session_update(languages)


@pytest.mark.parametrize("asked", [True, False], ids=["with-on-sound", "without"])
def test_the_session_asks_for_sounds_only_when_the_app_passes_on_sound(
    *, asked: bool
) -> None:
    arrived: list[str | bytes] = []

    async def records(connection: ServerConnection) -> None:
        arrived.append(await connection.recv())

    asyncio.run(hear(functools.partial(service_saying, records), sounds=asked))

    (first,) = arrived
    asked_for = parse_client_event(first).session.audio.input.sound_detection
    assert (asked_for is not None) is asked
    if not asked:
        assert first == session_update((Language.DE,))


def test_a_sound_reaches_on_sound_in_event_order_with_its_positions() -> None:
    recorder = Recorder()

    asyncio.run(
        hear(
            hearing(
                sound_detected(SoundName.LAUGHTER, 500, 2500),
                completed("a", "ha"),
                sound_detected(SoundName.COUGH, 3000, 5000),
            ),
            None,
            recorder,
            sounds=True,
        )
    )

    assert recorder.got == [
        HeardSound(SoundName.LAUGHTER, 500, 2500),
        utterance_of("ha"),
        HeardSound(SoundName.COUGH, 3000, 5000),
    ]


def test_a_sound_the_app_did_not_ask_for_raises_a_server_event_error() -> None:
    recorder = Recorder()

    with pytest.raises(ServerEventError):
        asyncio.run(
            hear(
                hearing(sound_detected(SoundName.LAUGHTER, 0, 2000), stays_open=True),
                None,
                recorder,
            )
        )

    assert recorder.got == []


@pytest.mark.parametrize("language", list(Language))
def test_a_completed_transcript_reaches_the_app_once_with_its_text_and_language(
    language: Language,
) -> None:
    recorder = Recorder()

    asyncio.run(hear(hearing(completed("a", "guten Tag", language)), None, recorder))

    assert recorder.got == [utterance_of("guten Tag", language)]


def test_an_empty_transcript_reaches_nobody_and_the_next_one_still_does() -> None:
    recorder = Recorder()

    asyncio.run(
        hear(hearing(completed("a", ""), completed("b", "weiter")), None, recorder)
    )

    assert recorder.got == [utterance_of("weiter", item_id="b")]


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        pytest.param(
            [started("a", 100), stopped("a", 900), completed("a", "x")],
            [utterance_of("x", start=100, end=900)],
            id="both-positions",
        ),
        pytest.param(
            [stopped("a", 900), completed("a", "x")],
            [utterance_of("x", end=900)],
            id="no-speech-started",
        ),
        pytest.param(
            [started("a", 100), completed("a", "x")],
            [utterance_of("x", start=100)],
            id="no-speech-stopped",
        ),
        pytest.param([completed("a", "x")], [utterance_of("x")], id="no-position"),
        pytest.param(
            [
                started("a", 100),
                started("b", 500),
                stopped("a", 400),
                stopped("b", 900),
                completed("b", "second"),
                completed("a", "first"),
            ],
            [
                utterance_of("second", start=500, end=900, item_id="b"),
                utterance_of("first", start=100, end=400),
            ],
            id="two-items-interleaved",
        ),
        pytest.param(
            [
                started("a", 100),
                stopped("a", 400),
                completed("a", "first"),
                completed("a", "again"),
            ],
            [utterance_of("first", start=100, end=400), utterance_of("again")],
            id="nothing-kept-once-an-item-completed",
        ),
        pytest.param(
            [
                started("a", 100),
                stopped("a", 400),
                completed("a", ""),
                completed("a", "again"),
            ],
            [utterance_of("again")],
            id="nothing-kept-of-an-item-that-completed-empty",
        ),
    ],
)
def test_an_utterance_carries_the_positions_of_its_own_item(
    events: list[ServerEvent], expected: list[Utterance]
) -> None:
    recorder = Recorder()

    asyncio.run(hear(hearing(*events), None, recorder))

    assert recorder.got == expected


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        pytest.param(
            [delta("a", "gu"), delta("a", "ten"), completed("a", "guten")],
            [Partial("gu"), Partial("guten"), utterance_of("guten")],
            id="the-text-so-far",
        ),
        pytest.param(
            [delta("a", "gu"), delta("b", "he"), delta("a", "t"), delta("b", "y")],
            [Partial("gu"), Partial("he"), Partial("gut"), Partial("hey")],
            id="two-items-interleaved",
        ),
        pytest.param(
            [delta("a", "one"), completed("a", "one"), delta("b", "two")],
            [Partial("one"), utterance_of("one"), Partial("two")],
            id="the-next-item-starts-anew",
        ),
        pytest.param(
            [delta("a", "one"), completed("a", ""), delta("a", "two")],
            [Partial("one"), Partial("two")],
            id="the-same-id-starts-anew-after-an-empty-one",
        ),
    ],
)
def test_with_on_partial_each_delta_arrives_as_its_items_text_so_far(
    events: list[ServerEvent], expected: list[Utterance | Partial]
) -> None:
    recorder = Recorder()

    asyncio.run(hear(hearing(*events), None, recorder))

    assert recorder.got == expected


def test_without_on_partial_nothing_of_a_delta_reaches_the_app() -> None:
    recorder = Recorder()

    asyncio.run(
        hear(
            hearing(delta("a", "gu"), delta("a", "ten"), completed("a", "guten")),
            None,
            recorder,
            partials=False,
        )
    )

    assert recorder.got == [utterance_of("guten")]


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
    recorder = Recorder()
    browser = Browser()
    after = completed("b", "never")

    outcome = asyncio.run(
        hear(
            hearing(completed("a", "before"), failure, after, stays_open=True),
            browser,
            recorder,
        )
    )

    assert outcome == ServiceFailed(RefusalCode.UNAVAILABLE)
    assert recorder.got == [utterance_of("before")]
    assert browser.closed_with == [HopClose.SERVICE_FAILED]
    assert "model" not in repr(outcome)
    assert failure.error.message not in repr(outcome)


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(completed("a", "x"), id="the-utterance-callback"),
        pytest.param(delta("a", "x"), id="the-partial-callback"),
    ],
)
def test_a_failure_of_the_apps_callback_is_raised_unchanged(
    event: ServerEvent,
) -> None:
    failure = RuntimeError("the app failed")
    browser = Browser()

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(
            hear(hearing(event, stays_open=True), browser, Recorder(raises=failure))
        )

    assert raised.value is failure
    assert browser.closed_with == []


@pytest.mark.parametrize(
    ("peer", "messages", "bounds", "outcome", "close"),
    [
        pytest.param(hearing(), (), Bounds(), Closed(), HopClose.NORMAL, id="closed"),
        pytest.param(
            functools.partial(service_refusing, HTTPStatus.FORBIDDEN),
            (),
            Bounds(),
            Refused(403, None),
            HopClose.SERVICE_FAILED,
            id="refused",
        ),
        pytest.param(
            functools.partial(
                service_refusing, HTTPStatus.TOO_MANY_REQUESTS, ("Retry-After", "7")
            ),
            (),
            Bounds(),
            Busy(BusyWait(retry_after_seconds=7)),
            HopClose.BUSY,
            id="busy",
        ),
        pytest.param(
            functools.partial(service_saying, speaks_nonsense),
            (),
            Bounds(),
            Malformed(),
            HopClose.SERVICE_FAILED,
            id="malformed",
        ),
        pytest.param(
            functools.partial(service_saying, drops_the_connection),
            (),
            Bounds(),
            Cut(),
            HopClose.SERVICE_FAILED,
            id="cut",
        ),
        pytest.param(
            service_not_answering,
            (),
            Bounds(opening_seconds=STALLED_OPENING_SECONDS),
            Unreachable(),
            HopClose.SERVICE_FAILED,
            id="the-opening-bound-passes",
        ),
        pytest.param(
            service_not_answering,
            (),
            Bounds(deadline_seconds=STALLED_OPENING_SECONDS),
            PastDeadline(),
            HopClose.PAST_DEADLINE,
            id="the-deadline-passes",
        ),
        pytest.param(
            hearing(stays_open=True),
            (binary_message(2),),
            Bounds(held_limit_bytes=1),
            OverCap(),
            HopClose.OVER_CAP,
            id="the-held-limit-is-passed",
        ),
        pytest.param(
            hearing(stays_open=True),
            (binary_message(HOP_FRAME_BYTES + 1),),
            Bounds(),
            RuleBroken(HopClose.FRAME_TOO_LARGE),
            HopClose.FRAME_TOO_LARGE,
            id="a-frame-over-the-hop-frame-limit",
        ),
        pytest.param(
            hearing(stays_open=True),
            (text_message(),),
            Bounds(),
            RuleBroken(HopClose.TEXT_FRAME),
            HopClose.TEXT_FRAME,
            id="a-text-frame",
        ),
        pytest.param(
            hearing(stays_open=True),
            (DISCONNECT,),
            Bounds(),
            BrowserLeft(),
            None,
            id="the-browser-leaves",
        ),
        pytest.param(
            hearing(stays_open=True),
            (binary_message(HOP_FRAME_BYTES), DISCONNECT),
            Bounds(),
            BrowserLeft(),
            None,
            id="a-frame-exactly-the-hop-frame-limit-does-not-end-the-hop",
        ),
    ],
)
def test_every_other_end_is_the_relays_outcome_and_closes_the_browser_with_its_code(
    peer: Peer,
    messages: tuple[Mapping[str, object], ...],
    bounds: Bounds,
    outcome: UtteranceOutcome,
    close: HopClose | None,
) -> None:
    browser = Browser(messages)

    heard = asyncio.run(hear(peer, browser, bounds=bounds))

    assert heard == outcome
    assert browser.closed_with == ([] if close is None else [close])


@pytest.mark.parametrize(
    ("events", "recorder", "expected"),
    [
        pytest.param(
            [completed("a", "late")],
            Recorder(ended=ended_already()),
            [],
            id="an-utterance-after-the-end-set-before-the-call",
        ),
        pytest.param(
            [delta("a", "late")],
            Recorder(ended=ended_already()),
            [],
            id="a-partial-after-the-end-set-before-the-call",
        ),
        pytest.param(
            [completed("a", "first"), completed("b", "late")],
            Recorder(ends_after_first=True),
            [utterance_of("first")],
            id="an-utterance-after-the-end-set-by-the-first",
        ),
        pytest.param(
            [delta("a", "first"), delta("b", "late")],
            Recorder(ends_after_first=True),
            [Partial("first")],
            id="a-partial-after-the-end-set-by-the-first",
        ),
    ],
)
def test_once_ended_is_set_no_utterance_and_no_partial_reaches_the_app(
    events: list[ServerEvent],
    recorder: Recorder,
    expected: list[Utterance | Partial],
) -> None:
    browser = Browser()

    outcome = asyncio.run(hear(hearing(*events, stays_open=True), browser, recorder))

    assert outcome == AppEnded()
    assert recorder.got == expected
    assert browser.closed_with == [HopClose.NORMAL]


def test_no_turn_end_reaches_the_app_after_its_utterance_set_ended() -> None:
    recorder = Recorder(ends_after_first=True)

    outcome = asyncio.run(
        hear(hearing(completed("a", "first"), stays_open=True), recorder=recorder)
    )

    assert outcome == AppEnded()
    assert recorder.got == [utterance_of("first")]
    assert recorder.turn_ends == []


def test_no_log_and_no_outcome_carries_a_token_or_heard_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    browser = Browser()

    outcome = asyncio.run(
        hear(
            hearing(
                delta("a", HEARD_TEXT),
                completed("a", HEARD_TEXT),
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


@contextlib.asynccontextmanager
async def _playing(*scripts: Script) -> AsyncIterator[int]:
    async with scripted(*scripts) as (port, _):
        yield port


@pytest.fixture
def quick_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        speech_client.hearing,
        "RECONNECT_BOUND",
        ReconnectBound(attempts=2, pause_seconds=0.01, within_seconds=3),
    )


@pytest.mark.usefixtures("quick_reconnect")
def test_a_reconnect_reaches_on_reconnected_and_forgets_the_old_items() -> None:
    recorder = Recorder()
    first = says(
        started("a", 100).model_dump_json(),
        delta("a", "hel").model_dump_json(),
        then_close=DRAIN_CLOSE_CODE,
    )
    second = says(
        delta("a", "lo").model_dump_json(),
        completed("a", "lo").model_dump_json(),
        then_close=1000,
    )

    outcome = asyncio.run(
        hear(functools.partial(_playing, first, second), None, recorder)
    )

    assert outcome == Closed()
    assert recorder.reconnected == [Reconnected(ReconnectCause.DRAIN, 0, 0)]
    assert recorder.got == [Partial("hel"), Partial("lo"), utterance_of("lo")]


@pytest.mark.usefixtures("quick_reconnect")
def test_a_hearing_that_cannot_reopen_closes_the_browser_as_a_service_failure() -> None:
    browser = Browser()
    refusing = functools.partial(_playing, drains, Refusal(RefusalCode.UNAVAILABLE))

    outcome = asyncio.run(hear(refusing, browser))

    assert outcome == ReconnectFailed()
    assert browser.closed_with == [HopClose.SERVICE_FAILED]


def test_a_source_without_tokens_closes_the_browser_in_order() -> None:
    browser = Browser()

    async def lapsed() -> None:
        return None

    outcome = asyncio.run(hear(hearing(), browser, token_source=lapsed))

    assert outcome == NoToken()
    assert browser.closed_with == [HopClose.NORMAL]


def test_a_token_source_failure_is_raised_and_the_browser_is_left_to_the_app() -> None:
    browser = Browser()
    failure = OSError("the token exchange is down")

    async def raises() -> SessionTokens:
        raise failure

    with pytest.raises(OSError, match="token exchange") as raised:
        asyncio.run(hear(hearing(), browser, token_source=raises))

    assert raised.value is failure
    assert browser.closed_with == []
