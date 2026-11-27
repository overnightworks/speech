"""POST /v1/audio/transcriptions: the places that keep recordings in memory."""

import asyncio
import time
from typing import TYPE_CHECKING

import pytest
from starlette.status import (
    HTTP_200_OK,
    HTTP_413_CONTENT_TOO_LARGE,
    HTTP_429_TOO_MANY_REQUESTS,
)

from speech_client.wire import RETRY_AFTER_HEADER, RefusalCode
from tests.support.clock_support import FakeClock
from tests.support.hear_support import (
    WAIT_SECONDS,
    FakeHearing,
    hear_app,
    hear_call,
    hear_call_begun,
    hear_runtime,
    multipart_body,
    recording,
)
from tests.support.speak_support import SpeakCall, assert_call_refusal

if TYPE_CHECKING:
    from fastapi import FastAPI

    from speech_service.service import Runtime

SHORT_WAIT_SECONDS = 0.2


def _valid() -> bytes:
    return multipart_body(file=recording())


async def _arrived(*calls: SpeakCall) -> None:
    """Let each of `calls` run up to its first wait: for bytes, a gate or a decode."""
    await asyncio.gather(*(call.received() for call in calls))


async def _hold_every_place(app: FastAPI, waiting: int) -> list[SpeakCall]:
    """Uploads half sent, one for each of the `1 + waiting` places."""
    holders = [hear_call_begun(app) for _ in range(1 + waiting)]
    await _arrived(*holders)
    return holders


async def _finish_all(holders: list[SpeakCall]) -> None:
    for holder in holders:
        holder.finish_upload()
    await asyncio.gather(*(holder.finish() for holder in holders))


@pytest.mark.parametrize("waiting", [0, 2], ids=["no-waiting", "two-waiting"])
def test_an_upload_beyond_the_kept_recordings_is_busy_and_never_heard(
    waiting: int,
) -> None:
    """WIRE-160: a recording arriving while every place is kept is busy."""

    async def scenario() -> tuple[SpeakCall, FakeHearing, list[int]]:
        engine = FakeHearing()
        app = hear_app(hear_runtime(engine), hearing_max_waiting_requests=waiting)
        holders = await _hold_every_place(app, waiting)
        beyond = hear_call(app)
        await beyond.finish()
        heard_while_held = len(engine.heard)
        await _finish_all(holders)
        return beyond, engine, [heard_while_held, len(engine.heard)]

    beyond, engine, heard = asyncio.run(scenario())

    assert_call_refusal(beyond, HTTP_429_TOO_MANY_REQUESTS, RefusalCode.BUSY, None)
    assert int(beyond.headers[RETRY_AFTER_HEADER.lower()]) >= 1
    assert heard == [0, 1 + waiting]
    assert engine.most_concurrent == 1


def test_an_upload_beyond_the_places_is_busy_though_a_place_frees_in_its_upload() -> (
    None
):
    """WIRE-160: busy even when a place frees while the upload is still arriving."""

    async def scenario() -> tuple[SpeakCall, SpeakCall]:
        app = hear_app(hear_runtime(FakeHearing()), hearing_max_waiting_requests=0)
        (holder,) = await _hold_every_place(app, 0)
        beyond = hear_call_begun(app)
        await _arrived(beyond)
        holder.task.cancel()
        await asyncio.gather(holder.task, return_exceptions=True)
        beyond.finish_upload()
        await beyond.finish()
        return holder, beyond

    _, beyond = asyncio.run(scenario())

    assert_call_refusal(beyond, HTTP_429_TOO_MANY_REQUESTS, RefusalCode.BUSY, None)


def test_with_every_place_kept_a_body_over_the_cap_is_still_too_large() -> None:
    """WIRE-160, WIRE-30: too large is answered before busy, whatever the places."""
    cap = len(_valid())

    async def scenario() -> tuple[SpeakCall, SpeakCall]:
        app = hear_app(
            hear_runtime(FakeHearing()),
            hearing_max_waiting_requests=0,
            max_recording_bytes=cap,
        )
        holders = await _hold_every_place(app, 0)
        oversized = hear_call(app, multipart_body(file=recording(seconds=1.0)))
        fits = hear_call(app)
        await asyncio.gather(oversized.finish(), fits.finish())
        await _finish_all(holders)
        return oversized, fits

    oversized, fits = asyncio.run(scenario())

    assert_call_refusal(
        oversized, HTTP_413_CONTENT_TOO_LARGE, RefusalCode.TOO_LARGE, None
    )
    assert_call_refusal(fits, HTTP_429_TOO_MANY_REQUESTS, RefusalCode.BUSY, None)


def _first_call(app: FastAPI, ending: str) -> SpeakCall:
    """The request that `ending` brings to its end."""
    match ending:
        case "too-large":
            return hear_call(app, b"x" * (len(_valid()) + 1))
        case "malformed":
            return hear_call(app, b"x" * 9)
        case "leaves":
            return hear_call_begun(app)
        case "leaves-while-refused":
            leaver = hear_call(app)
            leaver.disconnect()
            return leaver
        case _:
            return hear_call(app)


@pytest.mark.parametrize(
    "ending",
    [
        "too-large",
        "malformed",
        "busy",
        "heard",
        "leaves",
        "cancelled-waiting",
        "leaves-mid-decode",
        "leaves-while-refused",
    ],
)
def test_a_kept_place_is_freed_when_its_request_ends(ending: str) -> None:
    """WIRE-161: a place frees after a refusal, a leave, a cancel and the decode."""

    async def scenario() -> int:
        engine = FakeHearing(hold=ending == "leaves-mid-decode")
        runtime = hear_runtime(engine)
        app = hear_app(
            runtime,
            hearing_max_waiting_requests=int(ending == "cancelled-waiting"),
            hearing_busy_wait_seconds=SHORT_WAIT_SECONDS,
            max_recording_bytes=len(_valid()),
        )
        taken = None
        if ending in {"busy", "cancelled-waiting", "leaves-while-refused"}:
            taken = runtime.hearing.gate.try_acquire()
        first = _first_call(app, ending)
        if ending == "leaves-mid-decode":
            await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        if ending in {"leaves", "cancelled-waiting", "leaves-mid-decode"}:
            await _arrived(first)
            first.task.cancel()
            await asyncio.gather(first.task, return_exceptions=True)
        else:
            await first.finish()
        if taken is not None:
            taken.close()
        if ending == "leaves-mid-decode":
            engine.release.set()
            await _decode_over(runtime)
        next_one = hear_call(app)
        await next_one.finish()
        return next_one.status_code

    assert asyncio.run(scenario()) == HTTP_200_OK


def test_a_caller_leaving_while_its_request_waits_for_the_decode_frees_its_place() -> (
    None
):
    """WIRE-161: the server reports the leave; no handler is cancelled."""

    async def scenario() -> int:
        runtime = hear_runtime(FakeHearing())
        app = hear_app(runtime, hearing_max_waiting_requests=1)
        taken = runtime.hearing.gate.try_acquire()
        leaver = hear_call(app)
        await _arrived(leaver)
        leaver.disconnect()
        await leaver.finish()
        successor = hear_call(app)
        await _arrived(successor)
        if taken is not None:
            taken.close()
        await successor.finish()
        return successor.status_code

    assert asyncio.run(scenario()) == HTTP_200_OK


def test_a_complete_upload_whose_caller_left_is_never_heard_and_frees_its_place() -> (
    None
):
    """WIRE-161: a caller gone before the decode begins costs no decode and no place."""

    async def scenario() -> tuple[int, int]:
        engine = FakeHearing()
        app = hear_app(hear_runtime(engine), hearing_max_waiting_requests=0)
        leaver = hear_call_begun(app)
        await _arrived(leaver)
        leaver.disconnect()
        leaver.finish_upload()
        await leaver.finish()
        successor = hear_call(app)
        await successor.finish()
        return len(engine.heard), successor.status_code

    assert asyncio.run(scenario()) == (1, HTTP_200_OK)


async def _decode_over(runtime: Runtime) -> None:
    deadline = time.monotonic() + WAIT_SECONDS
    lease = await runtime.hearing.gate.acquire(deadline, 1)
    assert lease is not None
    lease.close()


def test_a_place_is_kept_while_the_decode_runs_though_the_caller_left() -> None:
    """WIRE-161: a place is freed when the decode returns, not at the leave."""

    async def scenario() -> tuple[int, int]:
        engine = FakeHearing(hold=True)
        runtime = hear_runtime(engine)
        app = hear_app(runtime, hearing_max_waiting_requests=0)
        running = hear_call(app)
        await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        running.task.cancel()
        await asyncio.gather(running.task, return_exceptions=True)
        beyond = hear_call(app)
        await beyond.finish()
        engine.release.set()
        await _decode_over(runtime)
        after = hear_call(app)
        await after.finish()
        return beyond.status_code, after.status_code

    during, after = asyncio.run(scenario())

    assert (during, after) == (HTTP_429_TOO_MANY_REQUESTS, HTTP_200_OK)


def test_an_upload_whose_wait_runs_out_hands_its_place_on_and_is_busy() -> None:
    """WIRE-162: a slow upload is busy, and the request behind it is heard."""

    async def scenario() -> tuple[SpeakCall, SpeakCall, FakeHearing]:
        clock = FakeClock()
        engine = FakeHearing()
        runtime = hear_runtime(engine, clock)
        app = hear_app(
            runtime,
            hearing_max_waiting_requests=0,
            hearing_busy_wait_seconds=SHORT_WAIT_SECONDS,
        )
        slow = hear_call_begun(app)
        await clock.until_a_wait_begins()
        clock.advance(SHORT_WAIT_SECONDS)
        await _place_frees(runtime)
        behind = hear_call(app)
        await behind.finish()
        slow.finish_upload()
        await slow.finish()
        return slow, behind, engine

    slow, behind, engine = asyncio.run(scenario())

    assert behind.status_code == HTTP_200_OK
    assert_call_refusal(slow, HTTP_429_TOO_MANY_REQUESTS, RefusalCode.BUSY, None)
    assert len(engine.heard) == 1


async def _place_frees(runtime: Runtime) -> None:
    async with asyncio.timeout(WAIT_SECONDS):
        while (place := runtime.hearing.keep_recording(1)) is None:  # noqa: ASYNC110 - a place has no event to await
            await asyncio.sleep(0)
    place.close()


def test_an_upload_ending_after_its_wait_is_busy_though_its_timer_has_not_run() -> None:
    """WIRE-162: the deadline decides, not whether its timer got to run first."""

    async def scenario() -> tuple[SpeakCall, FakeHearing]:
        clock = FakeClock()
        engine = FakeHearing()
        app = hear_app(
            hear_runtime(engine, clock), hearing_busy_wait_seconds=SHORT_WAIT_SECONDS
        )
        upload = hear_call_begun(app)
        await clock.until_a_wait_begins()
        clock.jump(SHORT_WAIT_SECONDS)
        upload.finish_upload()
        await upload.finish()
        return upload, engine

    upload, engine = asyncio.run(scenario())

    assert_call_refusal(upload, HTTP_429_TOO_MANY_REQUESTS, RefusalCode.BUSY, None)
    assert not engine.heard


def test_a_whole_body_is_heard_on_an_idle_service_when_the_wait_is_zero() -> None:
    """WIRE-118: with no wait, an idle decode is still taken, as for speaking."""

    async def scenario() -> SpeakCall:
        app = hear_app(
            hear_runtime(FakeHearing(), FakeClock()), hearing_busy_wait_seconds=0
        )
        upload = hear_call(app)
        await upload.finish()
        return upload

    assert asyncio.run(scenario()).status_code == HTTP_200_OK


def test_an_upload_within_its_wait_on_an_idle_service_is_heard() -> None:
    """WIRE-162: an upload that ends within the busy wait is heard."""

    async def scenario() -> SpeakCall:
        app = hear_app(hear_runtime(FakeHearing()))
        upload = hear_call_begun(app)
        await _arrived(upload)
        upload.finish_upload()
        await upload.finish()
        return upload

    assert asyncio.run(scenario()).status_code == HTTP_200_OK


def test_a_caller_leaving_mid_upload_raises_nothing_and_frees_its_place() -> None:
    async def scenario() -> int:
        app = hear_app(hear_runtime(FakeHearing()), hearing_max_waiting_requests=0)
        leaver = hear_call_begun(app)
        await _arrived(leaver)
        leaver.disconnect()
        await asyncio.wait_for(leaver.task, timeout=WAIT_SECONDS)
        successor = hear_call(app)
        await successor.finish()
        return successor.status_code

    assert asyncio.run(scenario()) == HTTP_200_OK
