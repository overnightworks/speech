"""An app's browser streams its microphone through the package and relay to the service.

Chromium runs the built package on the stand-in app's page, under webauth's real
headers, against the real service over TCP with a fake hearing engine.
"""

from __future__ import annotations

import asyncio
import math
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import numpy as np
import pytest
from playwright.async_api import async_playwright

from speech_client.hearing import HearingSession
from speech_client.relay import HOP_FRAME_BYTES, HopClose
from speech_client.wire import (
    HearingLanguage,
    PcmFormat,
    SessionInput,
    TranscriptCompleted,
)
from tests.support.browser_support import (
    RECORDING,
    REFUSING_HOP_PATH,
    AppSide,
    stand_in_app,
)
from tests.support.hear_support import FakeHearing, hear_runtime
from tests.support.page_coverage_support import PageCoverage
from tests.support.tcp_server_support import serving_asgi, serving_over_tcp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from playwright.async_api import Page

pytestmark = pytest.mark.browser

CHROMIUM_FLAGS = (
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    "--autoplay-policy=document-user-activation-required",
)
WAIT_MS = 20_000
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
# The turn's seconds beyond the layout's quiet and speech: the silent tail the
# service waits for, up to a poll interval, a frame, the lead before the recording
# and jitter.
TURN_EXTRA_SECONDS = (0.3, 1.6)
FRAMES_BEFORE_STOP = 3
STOP_BOUND_MS = 1000
CLOSED = 3


@dataclass(frozen=True)
class Stage:
    """One test's service, app side and page."""

    side: AppSide
    page: Page
    base_url: str
    service_url: str
    coverage: PageCoverage

    async def open(self, **query: str) -> None:
        """Load the app's page for the scenario `query` names."""
        await self.page.goto(f"{self.base_url}/capture.html?{urlencode(query)}")

    async def switch(self) -> None:
        """Click the app's hearing switch."""
        await self.page.click("#switch")

    async def until(self, condition: str) -> None:
        """Wait until `condition` holds on `seen`, what the page saw."""
        await self.page.wait_for_function(
            f"() => (seen => {condition})(window.capturePage)", timeout=WAIT_MS
        )

    async def seen(self) -> dict[str, object]:
        """What the page saw."""
        return await self.page.evaluate("window.capturePage")

    async def switch_on_until_streaming(self, **query: str) -> None:
        """Open the page, switch hearing on and wait until frames leave."""
        await self.open(**query)
        await self.switch()
        await self.until(f"seen.sentBytes.length >= {FRAMES_BEFORE_STOP}")

    async def switch_off(self) -> dict[str, object]:
        """Switch hearing off and return what the page saw once the capture ended."""
        await self.coverage.take_worklets()
        await self.switch()
        await self.until("seen.ended !== null")
        return await self.seen()


@asynccontextmanager
async def _stage(tmp_path: Path) -> AsyncIterator[Stage]:
    async with serving_over_tcp(hear_runtime(FakeHearing())) as service:
        side = AppSide.new()
        app = stand_in_app(service.base_url, side, tmp_path)
        async with serving_asgi(app) as base_url, async_playwright() as playwright:
            browser = await playwright.chromium.launch(args=CHROMIUM_FLAGS)
            try:
                page = await browser.new_page()
                coverage = PageCoverage.from_environment()
                await coverage.watch(page)
                yield Stage(side, page, base_url, service.base_url, coverage)
                await coverage.write()
            finally:
                await browser.close()


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
    async with _stage(tmp_path) as stage:
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


@dataclass(frozen=True)
class Streamed:
    """What the page and the app's side saw of a capture switched on and off."""

    seen: dict[str, object]
    frames: list[bytes | str]
    side: AppSide
    held: str
    policy: str


async def _stream_and_stop(tmp_path: Path, source: str) -> Streamed:
    async with _stage(tmp_path) as stage:
        page = await stage.page.request.get(f"{stage.base_url}/capture.html")
        await stage.switch_on_until_streaming(source=source)
        seen = await stage.switch_off()
        held = await stage.page.evaluate(
            "() => [document.documentElement.outerHTML, location.href,"
            " JSON.stringify(localStorage), JSON.stringify(sessionStorage),"
            " document.cookie].join('\\n')"
        )
        cookies = await stage.page.context.cookies()
        return Streamed(
            seen=seen,
            frames=stage.side.frames,
            side=stage.side,
            held=held + repr(cookies),
            policy=page.headers["content-security-policy"],
        )


@pytest.mark.parametrize(("quiet", "speech"), LAYOUTS)
def test_every_quiet_frame_of_a_recording_reaches_the_app(
    tmp_path: Path, quiet: float, speech: float
) -> None:
    side, _ = asyncio.run(_play(tmp_path, quiet, speech))

    assert _longest_quiet_seconds(side.frames) >= quiet - FRAME_SECONDS - 1e-9


@pytest.mark.parametrize(("quiet", "speech"), LAYOUTS)
def test_the_turn_holds_the_whole_recording_at_the_wire_rate(
    tmp_path: Path, quiet: float, speech: float
) -> None:
    _, seconds = asyncio.run(_play(tmp_path, quiet, speech))

    least, most = TURN_EXTRA_SECONDS
    assert least <= seconds - (quiet + speech) <= most


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
        async with _stage(tmp_path) as stage:
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


def test_the_page_receives_no_message_and_holds_no_token(tmp_path: Path) -> None:
    streamed = asyncio.run(_stream_and_stop(tmp_path, "tone"))

    assert streamed.side.events
    assert streamed.seen["messagesReceived"] == 0
    assert not any(token in streamed.held for token in streamed.side.tokens)


def test_the_apps_end_signal_reaches_ended_as_the_hops_normal_close(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with _stage(tmp_path) as stage:
            await stage.switch_on_until_streaming(source="tone")
            await stage.coverage.take_worklets()
            stage.side.app_ended.set()
            await stage.until("seen.ended !== null")
            return await stage.seen()

    seen = asyncio.run(scenario())

    assert seen["ended"] == {"kind": "hop", "close": HopClose.NORMAL}


def test_a_busy_service_reaches_ended_as_the_hops_busy_close(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with _stage(tmp_path) as stage:
            other = AppSide.new()
            taking = HearingSession(
                stage.service_url,
                workload_token=other.tokens[0],
                person_token=other.tokens[1],
                language=HearingLanguage.EN,
                deadline_seconds=WAIT_SECONDS,
                held_limit_bytes=HOP_FRAME_BYTES,
            )
            async with taking:
                await asyncio.wait_for(anext(taking.events()), WAIT_SECONDS)
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
    ],
)
def test_a_local_failure_ends_as_its_value_and_leaves_nothing_open(
    tmp_path: Path, query: dict[str, str], failure: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with _stage(tmp_path) as stage:
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


def test_an_app_route_that_refuses_the_upgrade_ends_as_socket_1006(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with _stage(tmp_path) as stage:
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
