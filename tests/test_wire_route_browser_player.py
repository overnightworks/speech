"""An app's browser plays sounds one after another, holds them and cuts them.

The flows run in Chromium on the stand-in app's page; see the shared stage.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

import pytest
from webauth.policies import default_content_security_policy

from tests.support.browser_player_support import (
    ABSENT,
    AUDIBLE_RMS,
    CLICK_STEP,
    CLOSED,
    CUT,
    CUT_LATE,
    EDGE_QUANTA,
    EDGE_SECONDS,
    FULL_PEAK,
    HELD,
    JUNK,
    LONG,
    LOST_AT_EDGES_SECONDS,
    MARKER,
    MARKER_SECONDS,
    MARKER_STREAMED,
    MIN_QUANTA_CHECKED,
    NATURAL,
    ONSET_BOUND_FRAMES,
    OTHER,
    RESUMES_FOR_START_AND_RESUME,
    RESUMES_REJECTED_THEN_PASSED,
    SHORT,
    SILENT_PEAK,
    SOUND_SECONDS,
    STOP_BOUND_MS,
    STREAMED,
    TAP_NAMES,
    TONE_PEAK,
    VOICE_BYTES,
    WAIT_SECONDS,
    action_times,
    activated,
    audible_seconds,
    bytes_of,
    edge_peaks,
    ends,
    events,
    largest_step,
    peaks_between,
    peaks_outside_edges,
    quantum_seconds,
    starts,
    tone_quanta,
    unactivated,
    url,
)
from tests.support.browser_support import (
    VOICE_PATH,
    AppSide,
)
from tests.support.speak_support import info_comment_before_data

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.browser


def test_three_sounds_play_in_the_order_handed_over(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.prepare_bytes()
            await stage.hand_over(url(SHORT), bytes_of(VOICE_BYTES), url(OTHER))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["played"] * 3
    assert events(seen) == [
        (event, sound) for sound in range(3) for event in ("started", "ended")
    ]


def test_a_sound_whose_answer_is_held_plays_before_the_later_one_that_decoded(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.prepare_bytes()
            await stage.hand_over(url(HELD), bytes_of(VOICE_BYTES))
            await stage.until("s.decodesSettled >= 1")
            while_held = await stage.snapshot()
            stage.side.held_release.set()
            return while_held, await stage.finish()

    while_held, seen = asyncio.run(scenario())

    assert while_held["sourcesStarted"] == 0
    assert starts(while_held) == [None, None]
    assert events(seen) == [
        (event, sound) for sound in range(2) for event in ("started", "ended")
    ]


def test_bytes_from_the_service_play_with_their_ai_mark_and_are_heard(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], AppSide]:
        async with activated(tmp_path) as stage:
            await stage.prepare_bytes()
            await stage.hand_over(bytes_of(VOICE_BYTES))
            return await stage.finish(), stage.side

    seen, side = asyncio.run(scenario())

    assert info_comment_before_data(side.spoken[0]) is not None
    assert ends(seen) == ["played"]
    assert seen["sounds"][0]["peakRms"] > AUDIBLE_RMS


@pytest.mark.parametrize("handed_as", ["url", "urlObject"])
def test_a_route_of_the_app_plays_once_whether_handed_as_text_or_as_a_url_object(
    tmp_path: Path, handed_as: str
) -> None:
    async def scenario() -> tuple[dict[str, object], AppSide]:
        async with activated(tmp_path) as stage:
            await stage.hand_over({handed_as: VOICE_PATH})
            return await stage.finish(), stage.side

    seen, side = asyncio.run(scenario())

    assert ends(seen) == ["played"]
    assert seen["sounds"][0]["peakRms"] > AUDIBLE_RMS
    assert side.requests == [VOICE_PATH]


def test_stop_ends_waiting_sounds_at_once_the_playing_one_after_its_edge_and_aborts(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], float]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG), url(HELD), url(SHORT))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.side.arrival(HELD)
            await stage.until(f"s.sounds[0].peakRms > {AUDIBLE_RMS}")
            await stage.act("stop")
            at_stop = (await stage.snapshot())["endedAtAction"]
            silent_ms = await stage.silent_within()
            await asyncio.wait_for(stage.side.held_aborted.wait(), WAIT_SECONDS)
            await stage.until("s.sounds[0].endedValue !== null")
            return {**await stage.snapshot(), "atStop": at_stop}, silent_ms

    seen, silent_ms = asyncio.run(scenario())

    assert seen["atStop"] == [None, "stopped", "stopped"]
    assert ends(seen) == ["stopped"] * 3
    assert starts(seen) == ["playing", "ended", "ended"]
    assert silent_ms < STOP_BOUND_MS


def test_a_stopped_sound_reports_nothing_more_at_its_natural_end_and_starts_nothing(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(NATURAL), url(SHORT))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.act("stop")
            await stage.until("s.sounds[0].endedValue !== null")
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert sorted(events(seen)) == sorted(
        (event, sound) for sound in range(2) for event in ("started", "ended")
    )
    assert ends(seen) == ["stopped", "stopped"]
    assert starts(seen) == ["playing", "ended"]


def test_a_sound_handed_over_after_stop_plays(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.act("stop")
            await stage.hand_over(url(SHORT))
            await stage.until("s.sounds[1].endedValue !== null")
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["stopped", "played"]


def test_stop_twice_on_an_idle_player_reports_nothing(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with unactivated(tmp_path) as stage:
            await stage.open()
            await stage.act("stop")
            await stage.act("stop")
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert seen["log"] == []
    assert seen["sounds"] == []


def test_muting_ends_the_playing_and_waiting_sounds_at_once_and_silences_the_output(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], float]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG), url(SHORT))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.until(f"s.sounds[0].peakRms > {AUDIBLE_RMS}")
            await stage.set_muted(muted=True)
            silent_ms = await stage.silent_within()
            return await stage.snapshot(), silent_ms

    seen, silent_ms = asyncio.run(scenario())

    assert seen["endedAtAction"] == ["muted", "muted"]
    assert starts(seen) == ["playing", "ended"]
    assert seen["gainValue"] == 0
    assert silent_ms < STOP_BOUND_MS


def test_a_sound_handed_over_while_muted_ends_muted_and_is_not_fetched(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], AppSide]:
        async with activated(tmp_path) as stage:
            await stage.set_muted(muted=True)
            await stage.hand_over(url(SHORT), url(OTHER))
            return await stage.finish(), stage.side

    seen, side = asyncio.run(scenario())

    assert ends(seen) == ["muted", "muted"]
    assert starts(seen) == ["ended", "ended"]
    assert side.requests == []
    assert seen["decodesCalled"] == 0


def test_unmuting_plays_the_next_sound_and_none_handed_over_before(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG), url(SHORT))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.set_muted(muted=True)
            await stage.set_muted(muted=False)
            await stage.hand_over(url(OTHER))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["muted", "muted", "played"]
    assert starts(seen) == ["playing", "ended", "playing"]
    assert seen["gainValue"] == 1
    assert seen["sounds"][2]["peakRms"] > AUDIBLE_RMS


@pytest.mark.parametrize(
    ("broken", "failure"),
    [(url(ABSENT), "a route answering 404"), (JUNK, "bytes that do not decode")],
)
def test_a_sound_that_cannot_load_fails_and_the_next_one_plays(
    tmp_path: Path, broken: dict[str, str], failure: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(broken, url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["failed", "played"], failure
    assert starts(seen) == ["ended", "playing"]


def test_a_streamed_wav_starts_before_its_body_has_ended_and_plays_through_to_its_end(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(STREAMED))
            await stage.until(f"s.sounds[0].peakRms > {AUDIBLE_RMS}")
            while_streaming = await stage.snapshot()
            stage.side.held_release.set()
            return while_streaming, await stage.finish()

    while_streaming, seen = asyncio.run(scenario())

    assert starts(while_streaming) == ["playing"]
    assert ends(while_streaming) == [None]
    assert ends(seen) == ["played"]
    assert events(seen) == [("started", 0), ("ended", 0)]


def test_stop_silences_a_streamed_wav_at_once_and_aborts_its_fetch(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], float]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(STREAMED))
            await stage.until(f"s.sounds[0].peakRms > {AUDIBLE_RMS}")
            await stage.act("stop")
            silent_ms = await stage.silent_within()
            await asyncio.wait_for(stage.side.held_aborted.wait(), WAIT_SECONDS)
            await stage.until("s.sounds[0].endedValue !== null")
            return await stage.snapshot(), silent_ms

    seen, silent_ms = asyncio.run(scenario())

    assert ends(seen) == ["stopped"]
    assert silent_ms < STOP_BOUND_MS


def test_a_route_that_aborts_mid_body_fails_and_the_next_one_plays(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(CUT), url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["failed", "played"]


def test_a_streamed_wav_that_breaks_off_stops_playing_where_it_broke_and_fails(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(CUT_LATE), url(SHORT))
            await stage.until(f"s.sounds[0].peakRms > {AUDIBLE_RMS}")
            stage.side.held_release.set()
            await stage.until("s.sounds[0].endedValue !== null")
            at_break = await stage.snapshot()
            return at_break, await stage.finish()

    at_break, seen = asyncio.run(scenario())

    assert ends(at_break)[0] == "failed"
    assert starts(seen) == ["playing", "playing"]
    assert ends(seen) == ["failed", "played"]


def test_a_sound_handed_over_without_a_click_is_refused_and_plays_after_one(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with unactivated(tmp_path) as stage:
            ended = asyncio.Event()
            await stage.page.expose_function("noteEnded", lambda _sound: ended.set())
            await stage.open(handOver=SHORT)
            await asyncio.wait_for(ended.wait(), WAIT_SECONDS)
            refused = await stage.snapshot()
            await stage.activate()
            await stage.hand_over(url(SHORT))
            return refused, await stage.finish()

    refused, seen = asyncio.run(scenario())

    assert ends(refused) == ["refused"]
    assert starts(refused) == ["ended"]
    assert refused["resumesCalled"] == 0
    assert ends(seen) == ["refused", "played"]
    assert seen["resumesCalled"] == 1


def test_a_rejected_resume_refuses_the_sound_and_the_next_one_plays(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.act("rejectNextResume")
            await stage.hand_over(url(SHORT))
            await stage.until("s.sounds[0].endedValue !== null")
            await stage.hand_over(url(OTHER))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["refused", "played"]
    assert seen["resumesCalled"] == RESUMES_REJECTED_THEN_PASSED


def test_every_source_connects_only_to_the_one_gain_that_is_the_outputs_only_input(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(SHORT), url(OTHER), url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())
    edges = seen["edges"]
    [into_output] = [edge for edge in edges if edge["toKind"] == "AudioDestinationNode"]
    from_sources = [
        edge for edge in edges if edge["fromKind"] == "AudioBufferSourceNode"
    ]

    assert into_output["fromKind"] == "GainNode"
    assert len(from_sources) == seen["sourcesStarted"] >= len(seen["sounds"])
    assert {edge["to"] for edge in from_sources} == {into_output["from"]}


def test_the_package_builds_no_tap_and_exports_none(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert seen["analysersCreated"] == 0
    assert "createPlayer" in seen["exportNames"]
    assert not [name for name in seen["exportNames"] if TAP_NAMES.search(name)]


def test_the_page_holds_none_of_the_tokens_the_apps_server_spoke_with(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[str, tuple[str, str]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(VOICE_PATH))
            await stage.finish()
            held = await stage.page.evaluate(
                "() => [document.documentElement.outerHTML, location.href,"
                " JSON.stringify(localStorage), JSON.stringify(sessionStorage),"
                " document.cookie].join('\\n')"
            )
            cookies = await stage.page.context.cookies()
            return held + repr(cookies), stage.side.tokens

    held, tokens = asyncio.run(scenario())

    assert not any(token in held for token in tokens)


def test_the_player_runs_under_webauths_default_policy(tmp_path: Path) -> None:
    async def scenario() -> tuple[str, dict[str, object]]:
        async with activated(tmp_path) as stage:
            page = await stage.page.request.get(f"{stage.base_url}/player.html")
            await stage.hand_over(url(SHORT))
            return page.headers["content-security-policy"], await stage.finish()

    policy, seen = asyncio.run(scenario())

    assert policy == default_content_security_policy()
    assert ends(seen) == ["played"]


def test_close_ends_every_sound_closes_the_context_and_later_sounds_end_stopped(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG), url(SHORT))
            await stage.until("s.sounds[0].startedValue === 'playing'")
            await stage.act("close")
            at_close = await stage.snapshot()
            await stage.until(f"s.contextStates.every(state => state === '{CLOSED}')")
            await stage.hand_over(url(SHORT))
            await stage.act("close")
            seen = await stage.finish()
            return {**seen, "atClose": at_close["endedAtAction"]}

    seen = asyncio.run(scenario())

    assert seen["atClose"] == ["stopped", "stopped"]
    assert ends(seen) == ["stopped"] * 3
    assert starts(seen) == ["playing", "ended", "ended"]
    assert seen["contextStates"] == [CLOSED]


def test_the_bytes_handed_over_keep_their_length(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.prepare_bytes()
            await stage.hand_over(bytes_of(VOICE_BYTES))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert seen["bytes"][VOICE_BYTES]["before"] > 0
    assert seen["bytes"][VOICE_BYTES]["after"] == seen["bytes"][VOICE_BYTES]["before"]


def test_a_stop_in_the_task_of_a_natural_end_gives_one_report_and_starts_nothing(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.act("stopOnNextSourceEnd")
            await stage.hand_over(url(SHORT), url(OTHER))
            await stage.finish()
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert seen["instrumentStops"] == 1
    assert starts(seen)[1] == "ended"
    assert ends(seen) == ["stopped", "stopped"]
    assert [sound["endedCount"] for sound in seen["sounds"]] == [1, 1]


@pytest.mark.parametrize("held", ["decode", "resume"])
@pytest.mark.parametrize("switch", ["stop", "mute"])
def test_a_switch_while_the_head_is_held_leaves_it_ended_and_the_new_sound_plays_once(
    tmp_path: Path, held: str, switch: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.prepare_bytes()
            await stage.act("hold", held)
            await stage.hand_over(bytes_of(VOICE_BYTES))
            await stage.until(f"s.{held}sCalled >= 1")
            if switch == "stop":
                await stage.act("stop")
            else:
                await stage.set_muted(muted=True)
                await stage.set_muted(muted=False)
            await stage.hand_over(bytes_of(VOICE_BYTES))
            await stage.until("s.sounds[1].endedValue !== null")
            await stage.act("release")
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["stopped" if switch == "stop" else "muted", "played"]
    assert starts(seen) == ["ended", "playing"]
    assert seen["sounds"][0]["start"] is None
    assert seen["sourcesStarted"] == 1
    assert [sound["endedCount"] for sound in seen["sounds"]] == [1, 1]


def test_a_sound_played_through_reports_its_full_length_as_played_seconds(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(
        SOUND_SECONDS["short.wav"], abs=seen["clockStep"]
    )


@pytest.mark.parametrize("cut", ["stop", "mute"])
@pytest.mark.parametrize("sound", [LONG, STREAMED], ids=["whole", "streamed"])
def test_a_cut_sound_reports_the_seconds_the_context_clock_passed_since_it_started(
    tmp_path: Path, cut: str, sound: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(sound), url(SHORT))
            await stage.until(f"s.sounds[0].peakRms > {AUDIBLE_RMS}")
            await stage.settle()
            if cut == "stop":
                await stage.act("stop")
            else:
                await stage.set_muted(muted=True)
            return await stage.finish()

    seen = asyncio.run(scenario())

    cut_sound, waiting = seen["sounds"]
    start = cut_sound["start"]["contextTime"]
    elapsed = cut_sound["contextAtEnd"] - start
    assert elapsed > 0
    if cut == "mute":
        assert cut_sound["playedSeconds"] == pytest.approx(
            elapsed, abs=seen["clockStep"]
        )
    else:
        # A stopped sound plays on through its edge, up to the end it reports.
        step = seen["clockStep"]
        assert cut_sound["playedSeconds"] >= seen["contextAtAction"] - start - step
        assert cut_sound["playedSeconds"] <= elapsed + step
    assert waiting["playedSeconds"] == 0


def test_a_sound_that_never_started_ends_with_zero_played_seconds(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.set_muted(muted=True)
            await stage.hand_over(url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert seen["sounds"][0]["playedSeconds"] == 0


@pytest.mark.parametrize("sound", [SHORT, STREAMED], ids=["whole", "streamed"])
def test_the_start_a_sound_hands_over_is_the_context_time_its_first_sample_is_heard(
    tmp_path: Path, sound: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(sound))
            await stage.until(f"s.sounds[0].peakRms > {AUDIBLE_RMS}")
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    start = seen["sounds"][0]["start"]
    assert start["onThePlayersContext"]
    # The start is never earlier than the render thread's clock, so the sound is heard
    # within one render quantum of it. The page hears the onset to one sample, and the
    # tone's first sample is silent: the onset heard is a sample later.
    assert seen["heardOnsets"][0] == pytest.approx(
        start["contextTime"], abs=ONSET_BOUND_FRAMES / seen["sampleRate"]
    )


@pytest.mark.parametrize("sound", [SHORT, STREAMED], ids=["whole", "streamed"])
def test_the_start_of_a_sound_scheduled_right_after_another_ended_is_where_it_is_heard(
    tmp_path: Path, sound: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(SHORT), url(sound))
            await stage.until("s.heardOnsets.length === 2")
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    start = seen["sounds"][1]["start"]
    # The second sound's time has passed when it is scheduled; the player still hands
    # over a start the render thread has not passed.
    assert seen["heardOnsets"][1] == pytest.approx(
        start["contextTime"], abs=ONSET_BOUND_FRAMES / seen["sampleRate"]
    )


AUDIBLE_CONDITION = f"s.sounds[0].peakRms > {AUDIBLE_RMS}"
SUSPENDED = "s.contextStates[0] === 'suspended'"
RUNNING = "s.contextStates[0] === 'running'"


def start_of(seen: dict[str, object], sound: int = 0) -> float:
    return seen["sounds"][sound]["start"]["contextTime"]


@pytest.mark.parametrize("sound", [MARKER, MARKER_STREAMED], ids=["whole", "streamed"])
def test_pause_ramps_the_output_down_without_a_click_and_holds_the_sound(
    tmp_path: Path, sound: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(sound))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.until(SUSPENDED)
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    [paused_at] = action_times(seen, "pause")
    edge = edge_peaks(seen, "pause")
    assert largest_step(edge) < CLICK_STEP
    ramping = [peak for peak in edge if SILENT_PEAK < peak < FULL_PEAK]
    assert len(ramping) >= EDGE_QUANTA
    assert edge[-1] < SILENT_PEAK
    assert audible_seconds(seen, paused_at + EDGE_SECONDS, math.inf) == 0
    assert seen["contextStates"] == ["suspended"]
    assert ends(seen) == [None]


@pytest.mark.parametrize("sound", [MARKER, MARKER_STREAMED], ids=["whole", "streamed"])
def test_resume_plays_on_from_where_it_paused_though_its_body_arrived_meanwhile(
    tmp_path: Path, sound: str
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(sound))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.until(SUSPENDED)
            stage.side.held_release.set()
            await stage.settle()
            while_paused = await stage.snapshot()
            await stage.act("resume")
            return while_paused, await stage.finish()

    while_paused, seen = asyncio.run(scenario())

    tone, gap, tail = MARKER_SECONDS
    start = start_of(seen)
    [paused_at] = action_times(seen, "pause")
    assert audible_seconds(while_paused, paused_at + EDGE_SECONDS, math.inf) == 0
    assert ends(while_paused) == [None]
    assert ends(seen) == ["played"]
    # The tone loses only the edges' slack; the gap and the tail keep their place.
    assert audible_seconds(seen, start, start + tone) == pytest.approx(
        tone, abs=LOST_AT_EDGES_SECONDS
    )
    assert audible_seconds(seen, start + tone + gap, math.inf) == pytest.approx(
        tail, abs=quantum_seconds(seen)
    )
    assert seen["heardOnsets"][-1] == pytest.approx(
        start + tone + gap, abs=ONSET_BOUND_FRAMES / seen["sampleRate"]
    )
    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(
        tone + gap + tail, abs=seen["clockStep"]
    )


def test_a_sound_handed_over_while_paused_waits_and_plays_after_resume(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.act("pause")
            await stage.hand_over(url(SHORT))
            await stage.settle()
            while_paused = await stage.snapshot()
            await stage.act("resume")
            return while_paused, await stage.finish()

    while_paused, seen = asyncio.run(scenario())

    assert starts(while_paused) == [None]
    assert ends(while_paused) == [None]
    assert while_paused["sourcesStarted"] == 0
    assert ends(seen) == ["played"]
    assert seen["sounds"][0]["peakRms"] > AUDIBLE_RMS


def test_pause_and_resume_on_an_idle_player_report_and_start_nothing(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with unactivated(tmp_path) as stage:
            await stage.open()
            await stage.act("pause")
            await stage.settle()
            await stage.act("resume")
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert seen["log"] == []
    assert seen["sourcesStarted"] == 0


@pytest.mark.parametrize(
    ("sound", "body_released"),
    [(LONG, False), (STREAMED, False), (STREAMED, True)],
    ids=["whole", "streamed", "streamed-body-scheduled"],
)
def test_stop_ramps_down_without_a_click_then_ends_and_the_next_sound_is_at_full_volume(
    tmp_path: Path, sound: str, *, body_released: bool
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(sound))
            await stage.until(AUDIBLE_CONDITION)
            if body_released:
                stage.side.held_release.set()
                await stage.settle()
            await stage.act("stop")
            at_stop = await stage.snapshot()
            await stage.hand_over(url(SHORT))
            return at_stop, await stage.finish()

    at_stop, seen = asyncio.run(scenario())

    assert at_stop["endedAtAction"] == [None]
    assert ends(seen) == ["stopped", "played"]
    # The next start is known to within a quantum; its first quanta are no edge.
    next_start = start_of(seen, 1) - 3 * quantum_seconds(seen)
    edge = edge_peaks(seen, "stop", until=next_start)
    assert seen["sounds"][0]["endedAt"] - seen["lastActionAt"] < STOP_BOUND_MS
    assert largest_step(edge) < CLICK_STEP
    assert edge[-1] < SILENT_PEAK
    short = start_of(seen, 1)
    margin = 3 * quantum_seconds(seen)
    heard = peaks_between(
        seen, short + margin, short + SOUND_SECONDS["short.wav"] - margin
    )
    assert heard
    assert min(heard) >= FULL_PEAK


def test_stop_while_paused_ends_every_sound_at_once_and_a_later_one_waits_for_resume(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG), url(SHORT))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.until(SUSPENDED)
            await stage.act("stop")
            at_stop = (await stage.snapshot())["endedAtAction"]
            await stage.hand_over(url(OTHER))
            await stage.settle()
            while_paused = await stage.snapshot()
            await stage.act("resume")
            return {**while_paused, "atStop": at_stop}, await stage.finish()

    while_paused, seen = asyncio.run(scenario())

    assert while_paused["atStop"] == ["stopped", "stopped"]
    assert starts(while_paused)[2] is None
    assert ends(seen) == ["stopped", "stopped", "played"]


def test_stop_while_the_pause_still_ramps_ends_the_sound_after_that_edge_and_holds(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("inOneTask", "pause", "stop")
            await stage.until("s.sounds[0].endedValue !== null")
            await stage.until(SUSPENDED)
            await stage.settle()
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    edge = edge_peaks(seen, "pause+stop")
    assert ends(seen) == ["stopped"]
    assert largest_step(edge) < CLICK_STEP
    assert edge[-1] < SILENT_PEAK
    [at] = action_times(seen, "pause+stop")
    assert audible_seconds(seen, at + EDGE_SECONDS, math.inf) == 0


def test_resume_while_the_pause_is_still_ramping_plays_on_without_a_click(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(MARKER))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.act("resume")
            return await stage.finish()

    seen = asyncio.run(scenario())

    start = start_of(seen)
    assert ends(seen) == ["played"]
    assert largest_step([peak for _, peak in tone_quanta(seen, start)]) < CLICK_STEP
    assert min(peaks_outside_edges(seen, start)) >= FULL_PEAK


def test_pause_while_the_resume_is_still_ramping_up_holds_the_sound_without_a_click(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(MARKER))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.until(SUSPENDED)
            await stage.act("resume")
            await stage.until(RUNNING)
            await stage.act("pause")
            await stage.until(SUSPENDED)
            await stage.settle()
            held = await stage.snapshot()
            await stage.act("resume")
            return held, await stage.finish()

    held, seen = asyncio.run(scenario())

    start = start_of(seen)
    [*_, last_pause] = action_times(held, "pause")
    assert ends(held) == [None]
    assert audible_seconds(held, last_pause + EDGE_SECONDS, math.inf) == 0
    assert ends(seen) == ["played"]
    assert largest_step([peak for _, peak in tone_quanta(seen, start)]) < CLICK_STEP
    assert min(peaks_outside_edges(seen, start)) >= FULL_PEAK


def test_resume_after_stop_changes_nothing_and_the_next_sound_plays_at_full_volume(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("stop")
            await stage.act("resume")
            await stage.hand_over(url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    short = start_of(seen, 1)
    margin = 3 * quantum_seconds(seen)
    assert ends(seen) == ["stopped", "played"]
    assert (
        min(
            peaks_between(
                seen, short + margin, short + SOUND_SECONDS["short.wav"] - margin
            )
        )
        >= FULL_PEAK
    )


def test_pause_twice_and_resume_twice_act_once(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(MARKER))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.act("pause")
            await stage.until(SUSPENDED)
            await stage.act("resume")
            await stage.act("resume")
            return await stage.finish()

    seen = asyncio.run(scenario())

    start = start_of(seen)
    assert ends(seen) == ["played"]
    assert seen["resumesCalled"] == RESUMES_FOR_START_AND_RESUME
    assert min(peaks_outside_edges(seen, start)) >= FULL_PEAK


def test_muting_while_paused_ends_the_sound_at_once_and_clears_the_pause(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.until(SUSPENDED)
            await stage.set_muted(muted=True)
            at_mute = await stage.snapshot()
            await stage.set_muted(muted=False)
            await stage.hand_over(url(SHORT))
            return at_mute, await stage.finish()

    at_mute, seen = asyncio.run(scenario())

    assert at_mute["endedAtAction"] == ["muted"]
    assert ends(seen) == ["muted", "played"]
    assert seen["gainValue"] == 1
    assert seen["sounds"][1]["peakRms"] > AUDIBLE_RMS


def test_close_while_paused_ends_the_sound_at_once_and_closes_the_context(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(LONG))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.until(SUSPENDED)
            await stage.act("close")
            at_close = (await stage.snapshot())["endedAtAction"]
            await stage.until(f"s.contextStates.every(state => state === '{CLOSED}')")
            await stage.settle()
            return {**await stage.snapshot(), "atClose": at_close}

    seen = asyncio.run(scenario())

    assert seen["atClose"] == ["stopped"]
    assert ends(seen) == ["stopped"]
    assert seen["contextStates"] == [CLOSED]


def test_the_volume_changes_in_no_way_but_at_the_edges_of_a_pause_and_a_resume(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with activated(tmp_path) as stage:
            await stage.hand_over(url(MARKER))
            await stage.until(AUDIBLE_CONDITION)
            await stage.act("pause")
            await stage.until(SUSPENDED)
            await stage.act("resume")
            return await stage.finish()

    seen = asyncio.run(scenario())

    outside = peaks_outside_edges(seen, start_of(seen))
    assert len(outside) > MIN_QUANTA_CHECKED
    assert min(outside) >= FULL_PEAK
    assert max(outside) <= 1.05 * TONE_PEAK
