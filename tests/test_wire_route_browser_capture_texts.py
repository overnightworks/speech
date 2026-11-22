"""An app's page sends texts to the app through the capture's hop.

Chromium runs the built package on the stand-in app's page; see the shared stage.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from speech_client.relay import HOP_FRAME_BYTES, HopClose
from tests.support.browser_capture_support import (
    FRAMES_BEFORE_STOP,
    MESSAGING,
    Stage,
    capture_stage,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support.browser_support import AppSide

pytestmark = pytest.mark.browser

TEXTS_FROM_THE_PAGE = ["", "one", "héllo ✓ 😀", '{"slide": 3}', "last"]


async def _send_from_page(stage: Stage, message: str) -> dict[str, object]:
    """Have the page send `message` through its capture; what `send` answered."""
    return await stage.page.evaluate("window.sendFromPage", message)


def test_texts_from_the_page_reach_on_text_exact_and_in_order_while_frames_stream(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], AppSide, list[object]]:
        async with capture_stage(tmp_path) as stage:
            await stage.switch_on_until_streaming(source="tone", **MESSAGING)
            answers = [
                await _send_from_page(stage, text) for text in TEXTS_FROM_THE_PAGE
            ]
            return await stage.switch_off(), stage.side, answers

    seen, side, answers = asyncio.run(scenario())

    assert answers == [{"sent": True}] * len(TEXTS_FROM_THE_PAGE)
    assert side.texts == TEXTS_FROM_THE_PAGE
    assert seen["sentTexts"] == TEXTS_FROM_THE_PAGE
    assert isinstance(side.frames[0], bytes)
    assert [frame for frame in side.frames if isinstance(frame, str)] == side.texts


def test_a_text_sent_while_the_hop_connects_is_not_sent_and_not_queued(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], AppSide]:
        async with capture_stage(tmp_path) as stage:
            await stage.open(source="tone", sendWhileConnecting="1", **MESSAGING)
            await stage.switch()
            await stage.until(f"seen.sentBytes.length >= {FRAMES_BEFORE_STOP}")
            return await stage.switch_off(), stage.side

    seen, side = asyncio.run(scenario())

    assert seen["sendWhileConnecting"] is False
    assert seen["sentTexts"] == []
    assert side.texts == []


def test_a_text_sent_after_the_capture_ended_is_not_sent(tmp_path: Path) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object], AppSide]:
        async with capture_stage(tmp_path) as stage:
            await stage.switch_on_until_streaming(source="tone", **MESSAGING)
            await stage.switch_off()
            answer = await _send_from_page(stage, "too late")
            return answer, await stage.seen(), stage.side

    answer, seen, side = asyncio.run(scenario())

    assert answer == {"sent": False}
    assert seen["sentTexts"] == []
    assert side.texts == []


def test_a_message_that_is_not_a_string_throws_a_type_error_and_the_capture_goes_on(
    tmp_path: Path,
) -> None:
    odd_sized = 7  # whole samples are even, so no audio frame is this long

    async def scenario() -> tuple[object, object, dict[str, object], AppSide]:
        async with capture_stage(tmp_path) as stage:
            await stage.switch_on_until_streaming(source="tone", **MESSAGING)
            thrown = await stage.page.evaluate(
                f"() => window.sendFromPage(new ArrayBuffer({odd_sized}))"
            )
            then = await _send_from_page(stage, "still sending")
            return thrown, then, await stage.switch_off(), stage.side

    thrown, then, seen, side = asyncio.run(scenario())

    assert thrown == {"error": "TypeError"}
    assert then == {"sent": True}
    assert seen["ended"] == {"kind": "hop", "close": HopClose.NORMAL}
    assert side.texts == ["still sending"]
    assert odd_sized not in seen["sentBytes"]
    assert odd_sized not in [len(frame) for frame in side.frames]


def test_a_text_one_utf8_byte_over_the_frame_limit_ends_the_hop_as_too_large(
    tmp_path: Path,
) -> None:
    text = "é" * (HOP_FRAME_BYTES // 2) + "a"
    assert len(text) < HOP_FRAME_BYTES < len(text.encode())

    async def scenario() -> tuple[dict[str, object], AppSide]:
        async with capture_stage(tmp_path) as stage:
            await stage.switch_on_until_streaming(source="tone", **MESSAGING)
            await _send_from_page(stage, text)
            await stage.until("seen.ended !== null")
            return await stage.seen(), stage.side

    seen, side = asyncio.run(scenario())

    assert seen["ended"] == {"kind": "hop", "close": HopClose.FRAME_TOO_LARGE}
    assert side.texts == []


def test_on_a_hop_without_on_text_a_text_from_the_page_ends_it_as_a_text_frame(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with capture_stage(tmp_path) as stage:
            await stage.switch_on_until_streaming(source="tone")
            await _send_from_page(stage, "unwanted")
            await stage.until("seen.ended !== null")
            return await stage.seen()

    seen = asyncio.run(scenario())

    assert seen["ended"] == {"kind": "hop", "close": HopClose.TEXT_FRAME}
