"""An app's browser streams its microphone through the package and relay to the service.

Chromium runs the built package on the stand-in app's page, under webauth's real
headers, against the real service over TCP with a fake hearing engine.
"""

from __future__ import annotations

import asyncio
import json
import math
import wave
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from speech_client.hearing import HearingSession
from speech_client.realtime_wire import (
    PcmFormat,
    SessionInput,
    TranscriptCompleted,
    TranscriptDelta,
)
from speech_client.relay import HOP_FRAME_BYTES, HopClose
from speech_client.wire import Language
from speech_service.config import DEFAULT_LIVE_SILENCE_SECONDS
from speech_service.voice_activity import PRE_ROLL_SECONDS
from tests.support.browser_capture_support import (
    FRAMES_BEFORE_STOP,
    MESSAGING,
    Stage,
    capture_stage,
)
from tests.support.browser_support import (
    HELD_WORKLET_PATH,
    MESSAGING_HOP_PATH,
    RECORDING,
    REFUSING_HOP_PATH,
    AppSide,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


pytestmark = pytest.mark.browser

WAIT_SECONDS = 20
WIRE_RATE = SessionInput().format.rate
FRAME_SECONDS = 0.1
FULL_SCALE = 32767
RECORDING_RATE = 48_000
TONE_HZ = 300
QUIET_RMS = 0.01
SPEECH_RMS = 0.14
SILENT_TAIL_SECONDS = 1.5
LAYOUTS = [(2.0, 1.0), (1.0, 2.5)]
# The turn's seconds beyond the layout's speech: the pre-roll before it and the
# silence that ends it, give or take a frame and a window.
TURN_EXTRA_SECONDS = PRE_ROLL_SECONDS + DEFAULT_LIVE_SILENCE_SECONDS
TURN_TOLERANCE_SECONDS = 0.15
STOP_BOUND_MS = 1000
CLOSED = 3
BURST = 100


def _tone(seconds: float, rms: float) -> np.ndarray:
    times = np.arange(round(seconds * RECORDING_RATE)) / RECORDING_RATE
    return rms * math.sqrt(2) * np.sin(2 * math.pi * TONE_HZ * times)


def _write_layout(path: Path, quiet_seconds: float, speech_seconds: float) -> None:
    samples = np.concatenate(
        [
            _tone(quiet_seconds, QUIET_RMS),
            _tone(speech_seconds, SPEECH_RMS),
            np.zeros(round(SILENT_TAIL_SECONDS * RECORDING_RATE)),
        ]
    )
    with wave.open(str(path), "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(PcmFormat.sample_bytes)
        recording.setframerate(RECORDING_RATE)
        recording.writeframes((samples * FULL_SCALE).astype("<i2").tobytes())


def _pcm(frames: list[bytes | str]) -> list[np.ndarray]:
    return [np.frombuffer(frame, dtype="<i2").astype(float) for frame in frames]


def _decibels(ratio: float) -> float:
    return 20 * math.log10(ratio)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples**2)))


def _amplitude_at(samples: np.ndarray, frequency: float) -> float:
    times = np.arange(len(samples)) / WIRE_RATE
    return (
        2
        * abs(np.sum(samples * np.exp(-2j * math.pi * frequency * times)))
        / len(samples)
    )


def _longest_quiet_seconds(frames: list[bytes | str]) -> float:
    reference = QUIET_RMS * FULL_SCALE
    longest = run = 0
    for window in _pcm(frames):
        level = _rms(window)
        quiet = level > 0 and abs(_decibels(level / reference)) <= 1
        run = run + 1 if quiet else 0
        longest = max(longest, run)
    return longest * FRAME_SECONDS


async def _play(tmp_path: Path, quiet: float, speech: float) -> tuple[AppSide, float]:
    """Play one layout through the page; the app's side and the turn's seconds."""
    _write_layout(tmp_path / RECORDING, quiet, speech)
    async with capture_stage(tmp_path) as stage:
        await stage.open(source="wav")
        await stage.switch()
        await asyncio.wait_for(stage.side.completed.wait(), WAIT_SECONDS)
        await stage.switch_off()
        [completed] = [
            event
            for event in stage.side.events
            if isinstance(event, TranscriptCompleted)
        ]
        return stage.side, completed.usage.seconds


@asynccontextmanager
async def _service_taken(stage: Stage) -> AsyncIterator[None]:
    """Another session holds the service's one hearing."""
    other = AppSide.new()
    taking = HearingSession(
        stage.service_url,
        token_source=other.token_source,
        languages=(Language.EN,),
        deadline_seconds=WAIT_SECONDS,
        held_limit_bytes=HOP_FRAME_BYTES,
    )
    async with taking:
        await asyncio.wait_for(anext(taking.events()), WAIT_SECONDS)
        yield


@dataclass(frozen=True)
class Streamed:
    """What the page and the app's side saw of a capture switched on and off."""

    seen: dict[str, object]
    frames: list[bytes | str]
    side: AppSide
    held: str
    policy: str


async def _stream_and_stop(tmp_path: Path, source: str) -> Streamed:
    async with capture_stage(tmp_path) as stage:
        page = await stage.page.request.get(f"{stage.base_url}/capture.html")
        await stage.switch_on_until_streaming(source=source)
        seen = await stage.switch_off()
        return Streamed(
            seen=seen,
            frames=stage.side.frames,
            side=stage.side,
            held=await stage.held(),
            policy=page.headers["content-security-policy"],
        )


@pytest.mark.parametrize(("quiet", "speech"), LAYOUTS)
def test_every_quiet_frame_of_a_recording_reaches_the_app(
    tmp_path: Path, quiet: float, speech: float
) -> None:
    side, _ = asyncio.run(_play(tmp_path, quiet, speech))

    assert _longest_quiet_seconds(side.frames) >= quiet - FRAME_SECONDS - 1e-9


@pytest.mark.parametrize(("quiet", "speech"), LAYOUTS)
def test_the_turn_holds_the_speech_with_its_pre_roll_and_closing_silence(
    tmp_path: Path, quiet: float, speech: float
) -> None:
    """WIRE-66: the quiet lead is dropped; the turn is the speech, pre-roll and end."""
    _, seconds = asyncio.run(_play(tmp_path, quiet, speech))

    assert seconds - speech == pytest.approx(
        TURN_EXTRA_SECONDS, abs=TURN_TOLERANCE_SECONDS
    )


def test_stop_leaves_the_apps_device_track_live_and_opens_no_device(
    tmp_path: Path,
) -> None:
    streamed = asyncio.run(_stream_and_stop(tmp_path, "device"))

    assert streamed.seen["trackState"] == "live"
    assert streamed.seen["packageGetUserMediaCalls"] == 0


def test_the_tracks_ended_event_stops_capture_after_the_last_partial_frame(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], list[bytes | str]]:
        async with capture_stage(tmp_path) as stage:
            await stage.switch_on_until_streaming(source="device")
            await stage.coverage.take_worklets()
            await stage.page.evaluate("window.endTrack()")
            await stage.until("seen.ended !== null")
            return await stage.seen(), stage.side.frames

    seen, frames = asyncio.run(scenario())

    assert seen["ended"] == {"kind": "hop", "close": HopClose.NORMAL}
    assert seen["endedCount"] == 1
    assert seen["errors"] == []
    assert seen["trackState"] == "live"
    assert [len(frame) for frame in frames] == seen["sentBytes"]
    assert len(frames) > seen["sentBeforeEnd"]
    assert 0 < len(frames[-1]) <= HOP_FRAME_BYTES


@pytest.mark.parametrize("how", ["suspend", "close"])
def test_a_context_that_stops_running_after_it_streamed_ends_the_capture_once(
    tmp_path: Path, how: str
) -> None:
    async def scenario() -> tuple[dict[str, object], list[bytes | str]]:
        async with capture_stage(tmp_path) as stage:
            await stage.switch_on_until_streaming(source="tone")
            await stage.page.evaluate(f"window.stopContext({how!r})")
            await stage.until("seen.ended !== null")
            return await stage.seen(), stage.side.frames

    seen, frames = asyncio.run(scenario())

    assert seen["ended"] == {"kind": "local", "failure": "context-stopped"}
    assert seen["endedCount"] == 1
    assert seen["errors"] == []
    assert set(seen["socketStatesAtEnd"]) <= {CLOSED}
    assert set(seen["contextStatesAtEnd"]) <= {"closed"}
    assert [len(frame) for frame in frames] == seen["sentBytes"]
    assert 0 < len(frames[-1]) <= HOP_FRAME_BYTES


def test_the_worklet_loads_from_its_file_under_webauths_policy(tmp_path: Path) -> None:
    streamed = asyncio.run(_stream_and_stop(tmp_path, "tone"))

    assert "script-src 'self'" in streamed.policy
    assert streamed.seen["started"] == "streaming"


def test_every_frame_is_binary_whole_samples_of_the_frame_size_and_the_last_arrives(
    tmp_path: Path,
) -> None:
    streamed = asyncio.run(_stream_and_stop(tmp_path, "tone"))
    *whole, last = streamed.frames

    assert all(isinstance(frame, bytes) for frame in streamed.frames)
    assert all(len(frame) % PcmFormat.sample_bytes == 0 for frame in streamed.frames)
    assert {len(frame) for frame in whole} == {HOP_FRAME_BYTES}
    assert 0 < len(last) <= HOP_FRAME_BYTES
    assert [len(frame) for frame in streamed.frames] == streamed.seen["sentBytes"]


def test_on_a_hop_that_sends_none_the_page_receives_no_message_and_holds_no_token(
    tmp_path: Path,
) -> None:
    streamed = asyncio.run(_stream_and_stop(tmp_path, "tone"))

    assert streamed.side.events
    assert streamed.seen["messagesReceived"] == 0
    assert not any(token in streamed.held for token in streamed.side.tokens)


def test_the_apps_end_signal_reaches_ended_as_the_hops_normal_close(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with capture_stage(tmp_path) as stage:
            await stage.switch_on_until_streaming(source="tone")
            await stage.coverage.take_worklets()
            stage.side.app_ended.set()
            await stage.until("seen.ended !== null")
            return await stage.seen()

    seen = asyncio.run(scenario())

    assert seen["ended"] == {"kind": "hop", "close": HopClose.NORMAL}


def test_a_busy_service_reaches_ended_as_the_hops_busy_close(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with capture_stage(tmp_path) as stage, _service_taken(stage):
            await stage.open(source="tone")
            await stage.switch()
            await stage.until("seen.ended !== null")
            return await stage.seen()

    seen = asyncio.run(scenario())

    assert seen["ended"] == {"kind": "hop", "close": HopClose.BUSY}


def test_stop_closes_the_socket_and_the_packages_context_within_a_second(
    tmp_path: Path,
) -> None:
    streamed = asyncio.run(_stream_and_stop(tmp_path, "tone"))

    assert streamed.seen["stopToEndedMs"] < STOP_BOUND_MS
    assert streamed.seen["socketStatesAtEnd"] == [CLOSED]
    assert streamed.seen["contextStatesAtEnd"] == ["closed"]
    assert streamed.seen["ended"] == {"kind": "hop", "close": HopClose.NORMAL}


def test_started_resolves_after_the_hop_opened_and_before_the_first_frame(
    tmp_path: Path,
) -> None:
    seen = asyncio.run(_stream_and_stop(tmp_path, "tone")).seen

    assert seen["started"] == "streaming"
    assert seen["openedAt"] <= seen["startedAt"] < seen["firstSendAt"]


@pytest.mark.parametrize(
    ("query", "failure"),
    [
        ({"source": "none"}, "no-audio-track"),
        # The policy refuses a blob: worklet; Chromium reports that refusal to no
        # listener of the page, so the failure value is what the page sees.
        ({"source": "tone", "worklet": "blob"}, "worklet-refused"),
        ({"source": "tone", "start": "load"}, "context-failed"),
        ({"source": "tone", "context": "throws"}, "context-failed"),
        ({"source": "tone", "context": "rejects"}, "context-failed"),
        ({"source": "tone", "worklet": "ear"}, "worklet-refused"),
    ],
    ids=[
        "no-audio-track",
        "worklet-blob-refused-by-policy",
        "suspended-without-activation",
        "context-constructor-throws",
        "resume-rejects",
        "worklet-registers-no-such-processor",
    ],
)
def test_a_local_failure_ends_as_its_value_and_leaves_nothing_open(
    tmp_path: Path, query: dict[str, str], failure: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with capture_stage(tmp_path) as stage:
            await stage.open(**query)
            if query.get("start") != "load":
                await stage.switch()
            await stage.until("seen.ended !== null")
            return await stage.seen()

    seen = asyncio.run(scenario())

    assert seen["ended"] == {"kind": "local", "failure": failure}
    assert seen["started"] == "ended"
    assert seen["errors"] == []
    assert set(seen["socketStatesAtEnd"]) <= {CLOSED}
    assert set(seen["contextStatesAtEnd"]) <= {"closed"}


def test_a_hop_url_the_websocket_refuses_throws_and_closes_the_packages_context(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with capture_stage(tmp_path) as stage:
            await stage.open(source="tone", hop="ftp://hop.invalid/hear")
            await stage.switch()
            await stage.until("seen.contextStatesAfterThrow !== null")
            return await stage.seen()

    seen = asyncio.run(scenario())

    assert seen["started"] is None
    assert seen["ended"] is None
    assert seen["socketsOpened"] == 0
    assert seen["startThrew"] is True
    assert seen["contextStatesAfterThrow"] == ["closed"]


def test_a_suspended_start_resumes_the_context_and_streams(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with capture_stage(tmp_path) as stage:
            await stage.switch_on_until_streaming(source="tone", context="suspended")
            return await stage.switch_off()

    seen = asyncio.run(scenario())

    assert seen["started"] == "streaming"
    assert seen["contextStatesAtStart"] == ["running"]


def test_a_statechange_while_the_context_runs_leaves_the_capture_streaming(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], int]:
        async with capture_stage(tmp_path) as stage:
            await stage.switch_on_until_streaming(source="tone")
            await stage.page.evaluate("window.dispatchStateChange()")
            before = len(await stage.page.evaluate("window.capturePage.sentBytes"))
            await stage.until(f"seen.sentBytes.length >= {before + FRAMES_BEFORE_STOP}")
            return await stage.seen(), before

    seen, before = asyncio.run(scenario())

    assert seen["ended"] is None
    assert len(seen["sentBytes"]) > before


def test_an_app_route_that_refuses_the_upgrade_ends_as_socket_1006(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with capture_stage(tmp_path) as stage:
            await stage.open(source="tone", hop=REFUSING_HOP_PATH)
            await stage.switch()
            await stage.until("seen.ended !== null")
            return await stage.seen()

    seen = asyncio.run(scenario())

    assert seen["ended"] == {"kind": "socket", "code": 1006}
    assert seen["started"] == "ended"


def test_a_click_under_the_gesture_policy_runs_the_context_and_frames_flow(
    tmp_path: Path,
) -> None:
    streamed = asyncio.run(_stream_and_stop(tmp_path, "tone"))

    assert streamed.seen["contextStatesAtStart"] == ["running"]
    assert streamed.seen["started"] == "streaming"
    assert len(streamed.frames) >= FRAMES_BEFORE_STOP


def test_the_first_audio_track_of_two_is_the_one_streamed(tmp_path: Path) -> None:
    streamed = asyncio.run(_stream_and_stop(tmp_path, "two-tones"))
    samples = np.concatenate(_pcm(streamed.frames[1:-1]))
    first, second = streamed.seen["trackTones"]

    assert _amplitude_at(samples, first) > 10 * _amplitude_at(samples, second)


def _texts(seen: dict[str, object]) -> list[object]:
    return [message["text"] for message in seen["messages"]]


def _numbered(count: int) -> list[str]:
    return [str(number) for number in range(count)]


async def _transcribed_turn(tmp_path: Path) -> tuple[Streamed, list[str]]:
    """A recorded turn on the messaging route, ended after its transcripts arrived."""
    _write_layout(tmp_path / RECORDING, 1.0, 1.0)
    async with capture_stage(tmp_path) as stage:
        page = await stage.page.request.get(f"{stage.base_url}/capture.html")
        await stage.open(source="wav", **MESSAGING)
        await stage.switch()
        await asyncio.wait_for(stage.side.completed.wait(), WAIT_SECONDS)
        transcripts = [
            event.type
            for event in stage.side.events
            if isinstance(event, TranscriptDelta | TranscriptCompleted)
        ]
        await stage.until(f"seen.messages.length >= {len(transcripts)}")
        before = len((await stage.seen())["sentBytes"])
        await stage.until(f"seen.sentBytes.length > {before}")
        seen = await stage.switch_off()
        turn = Streamed(
            seen=seen,
            frames=stage.side.frames,
            side=stage.side,
            held=await stage.held(),
            policy=page.headers["content-security-policy"],
        )
        return turn, transcripts


def test_the_routes_transcripts_reach_on_message_in_order_while_frames_stream(
    tmp_path: Path,
) -> None:
    turn, transcripts = asyncio.run(_transcribed_turn(tmp_path))
    seen = turn.seen

    assert transcripts
    assert [message["kind"] for message in seen["messages"]] == ["text"] * len(
        transcripts
    )
    assert [json.loads(text)["type"] for text in _texts(seen)] == transcripts
    assert seen["errors"] == []


def test_text_and_binary_messages_interleave_as_strings_and_exact_array_buffers(
    tmp_path: Path,
) -> None:
    sent: list[str | bytes] = ["one", bytes(range(256)), "two", b"\x00\xff", "three"]

    async def scenario() -> dict[str, object]:
        async with capture_stage(tmp_path) as stage:
            stage.send(*sent)
            await stage.switch_on_until_streaming(source="tone", **MESSAGING)
            await stage.until(f"seen.messages.length >= {len(sent)}")
            return await stage.switch_off()

    seen = asyncio.run(scenario())

    assert seen["messages"] == [
        {"kind": "text", "text": message}
        if isinstance(message, str)
        else {"kind": "binary", "bytes": list(message)}
        for message in sent
    ]


def test_a_message_sent_at_accept_arrives_while_the_worklet_is_held_and_before_started(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with capture_stage(tmp_path) as stage:
            stage.send("at accept")
            await stage.open(source="tone", worklet="held", **MESSAGING)
            await stage.switch()
            await stage.side.arrival(HELD_WORKLET_PATH)
            await stage.until("seen.messages.length >= 1")
            while_held = await stage.seen()
            stage.side.worklet_release.set()
            await stage.until("seen.started !== null")
            return while_held, await stage.switch_off()

    while_held, seen = asyncio.run(scenario())

    assert while_held["started"] is None
    assert seen["startedAtFirstMessage"] == "pending"
    assert seen["started"] == "streaming"
    assert _texts(seen) == ["at accept"]


def test_a_last_message_before_the_apps_end_signal_arrives_before_ended(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with capture_stage(tmp_path) as stage:
            await stage.switch_on_until_streaming(source="tone", **MESSAGING)
            await stage.coverage.take_worklets()
            stage.send("last", None)
            await stage.until("seen.ended !== null")
            return await stage.seen()

    seen = asyncio.run(scenario())

    assert _texts(seen) == ["last"]
    assert seen["messagesAfterEnded"] == 0
    assert seen["ended"] == {"kind": "hop", "close": HopClose.NORMAL}


@pytest.mark.parametrize(
    ("failure", "hearing_available", "taken"),
    [(HopClose.SERVICE_FAILED, False, False), (HopClose.BUSY, True, True)],
    ids=["service-failed", "busy"],
)
def test_the_message_sent_at_accept_arrives_before_a_failing_hop_ends(
    tmp_path: Path, failure: HopClose, *, hearing_available: bool, taken: bool
) -> None:
    async def scenario() -> dict[str, object]:
        async with capture_stage(
            tmp_path, hearing_available=hearing_available
        ) as stage:
            stage.send("at accept")
            async with AsyncExitStack() as arrangement:
                if taken:
                    await arrangement.enter_async_context(_service_taken(stage))
                await stage.open(source="tone", **MESSAGING)
                await stage.switch()
                await stage.until("seen.ended !== null")
                return await stage.seen()

    seen = asyncio.run(scenario())

    assert _texts(seen)[0] == "at accept"
    assert seen["ended"] == {"kind": "hop", "close": failure}
    assert seen["messagesAfterEnded"] == 0


def test_a_stop_from_inside_the_callback_ends_delivery_at_once(tmp_path: Path) -> None:
    stop_at = 5

    async def scenario() -> dict[str, object]:
        async with capture_stage(tmp_path) as stage:
            stage.send(*_numbered(BURST))
            await stage.open(source="tone", stopAt=str(stop_at), **MESSAGING)
            await stage.switch()
            await stage.until("seen.ended !== null")
            return await stage.seen()

    seen = asyncio.run(scenario())

    assert _texts(seen) == _numbered(stop_at)
    assert seen["ended"] == {"kind": "hop", "close": HopClose.NORMAL}
    assert seen["errors"] == []


def test_a_throwing_callback_surfaces_one_page_error_and_the_capture_goes_on(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with capture_stage(tmp_path) as stage:
            stage.send("first", "second", "third")
            await stage.switch_on_until_streaming(
                source="tone", throwFirst="1", **MESSAGING
            )
            await stage.until("seen.messages.length >= 3")
            return await stage.switch_off()

    seen = asyncio.run(scenario())

    assert len(seen["errors"]) == 1
    assert "the page's callback threw" in seen["errors"][0]
    assert _texts(seen) == ["first", "second", "third"]
    assert seen["started"] == "streaming"
    assert seen["ended"] == {"kind": "hop", "close": HopClose.NORMAL}


def test_without_on_message_a_route_that_sends_changes_nothing_for_the_capture(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with capture_stage(tmp_path) as stage:
            stage.send("dropped", b"also dropped")
            await stage.switch_on_until_streaming(source="tone", hop=MESSAGING_HOP_PATH)
            await stage.until("seen.messagesReceived >= 2")
            return await stage.switch_off()

    seen = asyncio.run(scenario())

    assert seen["messages"] == []
    assert seen["started"] == "streaming"
    assert seen["errors"] == []
    assert seen["ended"] == {"kind": "hop", "close": HopClose.NORMAL}


def test_no_callback_follows_a_local_failure_on_a_route_that_sends(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with capture_stage(tmp_path) as stage:
            stage.send(*_numbered(BURST))
            await stage.open(source="tone", worklet="blob", **MESSAGING)
            await stage.switch()
            await stage.until("seen.ended !== null")
            return await stage.seen()

    seen = asyncio.run(scenario())

    assert seen["ended"] == {"kind": "local", "failure": "worklet-refused"}
    assert seen["messagesAfterEnded"] == 0
    assert seen["errors"] == []


def test_a_burst_of_small_messages_arrives_in_order(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with capture_stage(tmp_path) as stage:
            stage.send(*_numbered(BURST))
            await stage.switch_on_until_streaming(source="tone", **MESSAGING)
            await stage.until(f"seen.messages.length >= {BURST}")
            return await stage.switch_off()

    assert _texts(asyncio.run(scenario())) == _numbered(BURST)


def test_no_token_appears_in_any_value_the_page_holds_while_messages_flow(
    tmp_path: Path,
) -> None:
    turn, _ = asyncio.run(_transcribed_turn(tmp_path))

    assert turn.seen["messages"]
    assert not any(token in turn.held for token in turn.side.tokens)
