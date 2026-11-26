"""An app's browser plays sounds one after another, holds them and cuts them.

The flows run in Chromium on the stand-in app's page; see the shared stage.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from webauth.policies import default_content_security_policy

from tests.support.browser_player_support import (
    ABSENT,
    ALL_ENDED,
    AUDIBLE_RMS,
    CLOSED,
    CUT,
    CUT_LATE,
    FINISH_QUANTA,
    FINISH_WITHIN_SECONDS,
    HELD,
    INTO_THE_EDGE_QUANTA,
    JUNK,
    LONG,
    NATURAL,
    OTHER,
    RESUMES_REJECTED_THEN_PASSED,
    SHORT,
    SILENT_PEAK,
    SILENT_RMS,
    SOUND_SECONDS,
    STARTED,
    STREAMED,
    STRETCH_SECONDS,
    TAP_NAMES,
    UNSTREAMED,
    VOICE_BYTES,
    WAIT_SECONDS,
    activated,
    audible_after_edge,
    bytes_of,
    ends,
    events,
    last_ramp_end,
    loudest_peak,
    markers,
    start_of,
    starts,
    stepped,
    stepped_through,
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
        async with stepped(tmp_path) as stage:
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
        async with stepped(tmp_path) as stage:
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
        async with stepped(tmp_path) as stage:
            await stage.prepare_bytes()
            await stage.hand_over(bytes_of(VOICE_BYTES))
            return await stage.finish(), stage.side

    seen, side = asyncio.run(scenario())

    assert info_comment_before_data(side.spoken[0]) is not None
    assert ends(seen) == ["played"]
    assert loudest_peak(seen, start_of(seen)) > SILENT_PEAK


@pytest.mark.parametrize("handed_as", ["url", "urlObject"])
def test_a_route_of_the_app_plays_once_whether_handed_as_text_or_as_a_url_object(
    tmp_path: Path, handed_as: str
) -> None:
    async def scenario() -> tuple[dict[str, object], AppSide]:
        async with stepped(tmp_path) as stage:
            await stage.hand_over({handed_as: VOICE_PATH})
            return await stage.finish(), stage.side

    seen, side = asyncio.run(scenario())

    assert ends(seen) == ["played"]
    assert loudest_peak(seen, start_of(seen)) > SILENT_PEAK
    assert side.requests == [VOICE_PATH]


def test_an_answer_the_player_does_not_stream_is_read_whole_decoded_and_heard(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hand_over(url(UNSTREAMED))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["played"]
    assert seen["decodesCalled"] == 1
    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(
        SOUND_SECONDS["short.wav"]
    )
    assert loudest_peak(seen, start_of(seen)) > SILENT_PEAK


def test_stop_ends_waiting_sounds_at_once_the_playing_one_after_its_edge_and_aborts(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], list[str | None]]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(LONG), url(HELD), url(SHORT))
            await stage.side.arrival(HELD)
            await stage.act("stop")
            at_stop = (await stage.snapshot())["endedAtAction"]
            await asyncio.wait_for(stage.side.held_aborted.wait(), WAIT_SECONDS)
            await stage.finish()
            await stage.step_for(STRETCH_SECONDS)
            return await stage.snapshot(), at_stop

    seen, at_stop = asyncio.run(scenario())

    assert at_stop == [None, "stopped", "stopped"]
    assert ends(seen) == ["stopped"] * 3
    assert starts(seen) == ["playing", "ended", "ended"]
    assert audible_after_edge(seen, "stop") == 0


def test_a_stopped_sound_reports_nothing_more_at_its_natural_end_and_starts_nothing(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(NATURAL), url(SHORT))
            await stage.act("stop")
            await stage.finish()
            await stage.step_for(SOUND_SECONDS["natural.wav"] + STRETCH_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert sorted(events(seen)) == sorted(
        (event, sound) for sound in range(2) for event in ("started", "ended")
    )
    assert ends(seen) == ["stopped", "stopped"]
    assert starts(seen) == ["playing", "ended"]


def test_a_sound_handed_over_after_stop_plays(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(LONG))
            await stage.act("stop")
            await stage.hand_over(url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["stopped", "played"]


def test_stop_twice_on_an_idle_player_reports_nothing(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.act("stop")
            await stage.act("stop")
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert seen["log"] == []
    assert seen["sounds"] == []


def test_a_second_stop_while_the_first_ones_edge_runs_changes_nothing(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(LONG))
            await stage.act("stop")
            await stage.step(INTO_THE_EDGE_QUANTA)
            await stage.act("stop")
            await stage.finish()
            await stage.step_for(STRETCH_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["stopped"]
    assert seen["sounds"][0]["endedCount"] == 1
    # The sound ends where the first stop's edge ends, not where a second one would.
    assert seen["sounds"][0]["playedSeconds"] == pytest.approx(
        last_ramp_end(seen, 0) - start_of(seen)
    )
    assert audible_after_edge(seen, "stop") == 0


def test_muting_ends_the_playing_and_waiting_sounds_at_once_and_silences_the_output(
    tmp_path: Path,
) -> None:
    seen = asyncio.run(
        stepped_through(
            tmp_path,
            lambda stage: stage.hear(url(LONG), url(SHORT)),
            lambda stage: stage.set_muted(muted=True),
            lambda stage: stage.step_for(STRETCH_SECONDS),
        )
    )

    assert seen["endedAtAction"] == ["muted", "muted"]
    assert starts(seen) == ["playing", "ended"]
    assert seen["gainValue"] == 0
    assert audible_after_edge(seen, "mute") == 0


def test_a_sound_handed_over_while_muted_ends_muted_and_is_not_fetched(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], AppSide]:
        async with stepped(tmp_path) as stage:
            await stage.set_muted(muted=True)
            await stage.hand_over(url(SHORT), url(OTHER))
            await stage.until(ALL_ENDED)
            return await stage.snapshot(), stage.side

    seen, side = asyncio.run(scenario())

    assert ends(seen) == ["muted", "muted"]
    assert starts(seen) == ["ended", "ended"]
    assert side.requests == []
    assert seen["decodesCalled"] == 0


def test_unmuting_plays_the_next_sound_and_none_handed_over_before(
    tmp_path: Path,
) -> None:
    seen = asyncio.run(
        stepped_through(
            tmp_path,
            lambda stage: stage.hear(url(LONG), url(SHORT)),
            lambda stage: stage.set_muted(muted=True),
            lambda stage: stage.set_muted(muted=False),
            lambda stage: stage.hand_over(url(OTHER)),
            lambda stage: stage.finish(),
        )
    )

    assert ends(seen) == ["muted", "muted", "played"]
    assert starts(seen) == ["playing", "ended", "playing"]
    assert seen["gainValue"] == 1
    assert loudest_peak(seen, start_of(seen, 2)) > SILENT_PEAK


def test_unmuting_a_player_that_is_not_muted_leaves_a_pause_held(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(LONG))
            await stage.paused()
            await stage.set_muted(muted=False)
            await stage.stand_still_for(STRETCH_SECONDS)
            held = await stage.snapshot()
            await stage.act("resume")
            return held, await stage.finish()

    held, seen = asyncio.run(scenario())

    assert held["gainValue"] == 0
    assert ends(held) == [None]
    assert held["contextStates"] == ["suspended"]
    assert ends(seen) == ["played"]


@pytest.mark.parametrize(
    ("broken", "failure"),
    [(url(ABSENT), "a route answering 404"), (JUNK, "bytes that do not decode")],
)
def test_a_sound_that_cannot_load_fails_and_the_next_one_plays(
    tmp_path: Path, broken: dict[str, str], failure: str
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hand_over(broken, url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["failed", "played"], failure
    assert starts(seen) == ["ended", "playing"]


def test_a_streamed_wav_starts_before_its_body_has_ended_and_plays_through_to_its_end(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            while_streaming = await stage.hear_then_release(url(STREAMED))
            return while_streaming, await stage.finish()

    while_streaming, seen = asyncio.run(scenario())

    assert starts(while_streaming) == ["playing"]
    assert ends(while_streaming) == [None]
    assert ends(seen) == ["played"]
    assert events(seen) == [("started", 0), ("ended", 0)]


def test_stop_silences_a_streamed_wav_at_once_and_aborts_its_fetch(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(STREAMED))
            await stage.act("stop")
            await asyncio.wait_for(stage.side.held_aborted.wait(), WAIT_SECONDS)
            await stage.finish()
            await stage.step_for(STRETCH_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["stopped"]
    assert audible_after_edge(seen, "stop") == 0


def test_a_route_that_aborts_mid_body_fails_and_the_next_one_plays(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            await stage.hand_over(url(CUT), url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["failed", "played"]


def test_a_streamed_wav_that_breaks_off_stops_playing_where_it_broke_and_fails(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.hear_then_release(url(CUT_LATE), url(SHORT))
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
        async with stepped(tmp_path) as stage:
            await stage.act("rejectNextResume")
            await stage.hand_over(url(SHORT))
            await stage.until("s.sounds[0].endedValue !== null")
            await stage.hand_over(url(OTHER))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["refused", "played"]
    assert seen["resumesCalled"] == RESUMES_REJECTED_THEN_PASSED


@pytest.mark.parametrize(
    ("garbling", "marked"),
    [("stale", ["duplicate", "gap"]), ("dropped", ["gap"])],
    ids=["a_stale_report", "a_report_that_never_arrives"],
)
def test_an_ear_report_that_is_stale_or_missing_is_marked_and_the_stepped_cell_finishes(
    tmp_path: Path, garbling: str, marked: list[str]
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
            # The last report of the first step, the one that step has to see.
            await stage.act("garbleReport", garbling, FINISH_QUANTA - 1)
            await stage.hand_over(url(SHORT))
            return await stage.finish()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["played"]
    assert [fault["kind"] for fault in markers(seen)["quantumReports"]] == marked


def test_every_source_connects_only_to_the_one_gain_that_is_the_outputs_only_input(
    tmp_path: Path,
) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
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


def test_an_analyser_on_the_tap_reads_energy_while_a_sentence_plays_and_none_after(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.act("readTap")
            await stage.hear(url(SHORT))
            seen = await stage.finish()
            await stage.step_for(STRETCH_SECONDS)
            return seen, await stage.snapshot()

    while_playing, after = asyncio.run(scenario())

    assert while_playing["tapIsOutputGain"]
    assert while_playing["sounds"][0]["peakTapRms"] > AUDIBLE_RMS
    assert after["tapRms"] < SILENT_RMS


def test_the_package_builds_no_reader_and_exports_none(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
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
        async with stepped(tmp_path) as stage:
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
        async with stepped(tmp_path) as stage:
            await stage.hand_over(url(LONG), url(SHORT))
            await stage.until(STARTED)
            await stage.act("close")
            at_close = await stage.snapshot()
            await stage.hand_over(url(SHORT))
            await stage.act("close")
            seen = await stage.finish()
            return {**seen, "atClose": at_close["endedAtAction"]}

    seen = asyncio.run(scenario())

    assert seen["atClose"] == ["stopped", "stopped"]
    assert ends(seen) == ["stopped"] * 3
    assert starts(seen) == ["playing", "ended", "ended"]
    assert seen["contextStates"] == [CLOSED]


def test_resume_after_close_changes_nothing(tmp_path: Path) -> None:
    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        async with stepped(tmp_path) as stage:
            await stage.hear(url(LONG))
            await stage.paused()
            await stage.act("close")
            at_close = await stage.snapshot()
            await stage.act("resume")
            await stage.stand_still_for(STRETCH_SECONDS)
            return at_close, await stage.snapshot()

    at_close, seen = asyncio.run(scenario())

    assert seen["resumesCalled"] == at_close["resumesCalled"]
    assert seen["contextStates"] == [CLOSED]
    assert ends(seen) == ["stopped"]


def test_the_bytes_handed_over_keep_their_length(tmp_path: Path) -> None:
    async def scenario() -> dict[str, object]:
        async with stepped(tmp_path) as stage:
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
        async with stepped(tmp_path) as stage:
            await stage.act("stopOnNextSourceEnd")
            await stage.hand_over(url(SHORT), url(OTHER))
            await stage.finish()
            await stage.step_for(STRETCH_SECONDS)
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
        async with stepped(tmp_path) as stage:
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
            await stage.until("s.decodesSettled >= 2")
            await stage.step_until(
                "s.sounds[1].endedValue !== null",
                within_seconds=FINISH_WITHIN_SECONDS,
                quanta=FINISH_QUANTA,
            )
            await stage.act("release")
            await stage.step_for(STRETCH_SECONDS)
            return await stage.snapshot()

    seen = asyncio.run(scenario())

    assert ends(seen) == ["stopped" if switch == "stop" else "muted", "played"]
    assert starts(seen) == ["ended", "playing"]
    assert seen["sounds"][0]["start"] is None
    assert seen["sourcesStarted"] == 1
    assert [sound["endedCount"] for sound in seen["sounds"]] == [1, 1]
