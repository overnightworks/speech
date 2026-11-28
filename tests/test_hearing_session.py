"""C2's scheduling at its boundaries, with an injected clock and a level detector."""

import struct

import pytest

from speech_service.config import DEFAULT_LIVE_SILENCE_SECONDS
from speech_service.hearing_session import (
    MAX_TURN_SECONDS,
    HearingSession,
    SchedulerEvent,
    SpeechStart,
    SpeechSustain,
    SpeechSustainFrom,
    TurnEnd,
    TurnEndedBy,
)
from speech_service.pcm import BYTES_PER_SAMPLE
from speech_service.pcm import HEARING_SAMPLE_RATE as SAMPLE_RATE
from speech_service.voice_activity import (
    PRE_ROLL_SECONDS,
    START_WINDOWS,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
)
from tests.support.detector_support import (
    FOUR_TENTHS_MS,
    LOUD_LEVEL,
    THRESHOLD_WINDOWS,
    LevelDetector,
    ScriptedDetector,
    end_of_window,
    start_of_window,
)

BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE
SILENT_WINDOWS = round(DEFAULT_LIVE_SILENCE_SECONDS / WINDOW_SECONDS + 0.5)
PRE_ROLL_MS = int(PRE_ROLL_SECONDS * 1000)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _windows(count: int, level: int) -> bytes:
    return struct.pack("<h", level) * (count * WINDOW_SAMPLES)


def _seconds(seconds: float, level: int) -> bytes:
    return struct.pack("<h", level) * int(seconds * SAMPLE_RATE)


def _speech(count: int) -> bytes:
    return _windows(count, LOUD_LEVEL)


def _quiet(count: int) -> bytes:
    return _windows(count, 0)


def _session() -> tuple[HearingSession, _Clock]:
    clock = _Clock()
    return HearingSession(clock, LevelDetector(), DEFAULT_LIVE_SILENCE_SECONDS), clock


def _ends(events: list[SpeechStart | TurnEnd]) -> list[TurnEnd]:
    return [event for event in events if isinstance(event, TurnEnd)]


def test_speech_starts_a_turn_once_it_is_confirmed_at_the_first_window_of_it() -> None:
    """WIRE-70: a turn starts at the first window of the speech that Silero confirms."""
    session, _ = _session()
    session.add(_quiet(16))

    unconfirmed = session.add(_speech(START_WINDOWS - 1))
    confirmed = session.add(_speech(1))
    later = session.add(_speech(5))

    assert unconfirmed == []
    assert confirmed == [SpeechStart(audio_ms=round(16 * WINDOW_SECONDS * 1000))]
    assert later == []


def test_a_window_of_noise_between_speech_windows_restarts_the_confirmation() -> None:
    session, _ = _session()

    events = session.add(
        _speech(START_WINDOWS - 1) + _quiet(1) + _speech(START_WINDOWS - 1)
    )

    assert events == []


def test_a_turn_ends_once_the_silence_is_complete_and_not_a_window_before() -> None:
    """WIRE-66: a turn ends once its silence is complete and not a window before."""
    session, _ = _session()
    session.add(_speech(30))

    one_window_short = session.add(_quiet(SILENT_WINDOWS - 1))
    complete = session.add(_quiet(1))

    assert one_window_short == []
    (ended,) = _ends(complete)
    assert ended.ended_by is TurnEndedBy.SILENCE


def test_a_turn_ended_by_silence_names_that_silence_and_the_others_name_none() -> None:
    session, _ = _session()
    session.add(_speech(30))
    (by_silence,) = _ends(session.add(_quiet(SILENT_WINDOWS)))
    session.add(_speech(30))
    by_client = session.commit()

    assert by_silence.silence_ms == round(SILENT_WINDOWS * WINDOW_SECONDS * 1000)
    assert by_client is not None
    assert by_client.silence_ms == 0


def test_the_turn_is_silent_from_its_first_quiet_window_until_speech_returns() -> None:
    session, _ = _session()

    session.add(_speech(30))
    speaking = session.silent
    session.add(_quiet(1))
    waiting = session.silent
    session.add(_speech(1))
    resumed = session.silent
    session.add(_quiet(SILENT_WINDOWS))

    assert (speaking, waiting, resumed, session.silent) == (False, True, False, False)


def test_a_turn_ends_after_the_configured_silence_and_not_the_default() -> None:
    longer = DEFAULT_LIVE_SILENCE_SECONDS * 3
    session = HearingSession(_Clock(), LevelDetector(), longer)
    long_windows = round(longer / WINDOW_SECONDS + 0.5)
    session.add(_speech(30))

    after_the_default = session.add(_quiet(SILENT_WINDOWS))
    after_the_configured = session.add(_quiet(long_windows - SILENT_WINDOWS))

    assert _ends(after_the_default) == []
    assert len(_ends(after_the_configured)) == 1


def test_a_dip_shorter_than_the_silence_does_not_end_the_turn() -> None:
    session, _ = _session()

    events = session.add(_speech(30) + _quiet(SILENT_WINDOWS - 1) + _speech(30))

    assert _ends(events) == []


def test_a_turn_holds_its_voiced_seconds_and_not_its_silence() -> None:
    session, _ = _session()
    session.add(_speech(30) + _quiet(5) + _speech(10))

    (ended,) = _ends(session.add(_quiet(SILENT_WINDOWS)))

    assert ended.voiced_seconds == pytest.approx(40 * WINDOW_SECONDS)


@pytest.mark.parametrize("leading", [10, 30])
def test_a_turn_counts_the_voiced_windows_of_its_pre_roll_and_no_older_ones(
    leading: int,
) -> None:
    """Windows from the end threshold up are voiced, confirmed or not."""
    faint, speech = 0.4, 0.9
    probabilities = [faint] * leading + [speech] * 50 + [0.0] * SILENT_WINDOWS
    session = HearingSession(
        _Clock(), ScriptedDetector(probabilities), DEFAULT_LIVE_SILENCE_SECONDS
    )

    (ended,) = _ends(session.add(_quiet(len(probabilities))))

    pre_roll_windows = int(PRE_ROLL_SECONDS / WINDOW_SECONDS)
    assert ended.voiced_seconds == pytest.approx(
        (min(leading, pre_roll_windows) + 50) * WINDOW_SECONDS
    )


def test_a_turn_starts_a_pre_roll_before_its_speech_and_drops_older_audio() -> None:
    """WIRE-66: the audio of a turn starts a pre-roll before its speech."""
    session, _ = _session()
    session.add(_quiet(100))
    speech_ms = round(100 * WINDOW_SECONDS * 1000)

    (ended,) = _ends(session.add(_speech(30) + _quiet(SILENT_WINDOWS)))

    assert ended.start_ms == pytest.approx(speech_ms - PRE_ROLL_MS, abs=1)
    held = len(ended.pcm) / BYTES_PER_SECOND
    assert held == pytest.approx(
        PRE_ROLL_SECONDS + (30 + SILENT_WINDOWS) * WINDOW_SECONDS, abs=WINDOW_SECONDS
    )


def test_a_turn_after_a_gap_counts_its_place_in_the_session_audio() -> None:
    """WIRE-70: positions count the session's audio, the dropped audio included."""
    session, _ = _session()
    session.add(_speech(30) + _quiet(SILENT_WINDOWS))
    session.add(_quiet(200))

    (ended,) = _ends(session.add(_speech(30) + _quiet(SILENT_WINDOWS)))

    first_end_ms = round((30 + SILENT_WINDOWS) * WINDOW_SECONDS * 1000)
    speech_ms = first_end_ms + round(200 * WINDOW_SECONDS * 1000)
    assert ended.start_ms == pytest.approx(speech_ms - PRE_ROLL_MS, abs=1)
    assert ended.end_ms == pytest.approx(
        speech_ms + (30 + SILENT_WINDOWS) * WINDOW_SECONDS * 1000, abs=1
    )


def test_a_long_quiet_holds_no_more_than_a_pre_roll() -> None:
    session, _ = _session()
    session.add(_seconds(60.0, 0))

    held = session.commit()

    assert isinstance(held, TurnEnd)
    assert len(held.pcm) / BYTES_PER_SECOND <= PRE_ROLL_SECONDS + WINDOW_SECONDS
    assert held.voiced_seconds == 0.0


def test_audio_that_crosses_the_cap_ends_the_turn_at_the_cap_and_opens_the_next() -> (
    None
):
    """WIRE-66: audio that crosses the cap ends the turn at the cap."""
    session, _ = _session()

    events = session.add(_seconds(31.0, LOUD_LEVEL))

    (ended,) = _ends(events)
    assert len(ended.pcm) == round(MAX_TURN_SECONDS * BYTES_PER_SECOND)
    assert ended.end_ms - ended.start_ms == round(MAX_TURN_SECONDS * 1000)
    assert ended.ended_by is TurnEndedBy.CAP
    rest = session.commit()
    assert isinstance(rest, TurnEnd)
    assert rest.start_ms == ended.end_ms
    assert len(rest.pcm) / BYTES_PER_SECOND == pytest.approx(1.0, abs=WINDOW_SECONDS)


def test_a_chunking_of_the_audio_does_not_change_its_turns() -> None:
    audio = _quiet(40) + _speech(60) + _quiet(40) + _speech(20) + _quiet(40)
    whole, _ = _session()
    expected = whole.add(audio)

    for size in (1, 333, WINDOW_SAMPLES * 2, 1999):
        session, _ = _session()
        events: list[SpeechStart | TurnEnd] = []
        for start in range(0, len(audio), size):
            events += session.add(audio[start : start + size])
        assert events == expected


def test_no_partial_is_due_before_four_tenths_of_a_second_of_audio() -> None:
    session, _ = _session()
    session.add(_speech(START_WINDOWS))

    before = session.partial_due()
    session.add(_speech(12))

    assert before is None
    assert session.partial_due() is not None


def test_a_partial_carries_the_voiced_seconds_of_the_turn_so_far() -> None:
    session, _ = _session()
    session.add(_quiet(4) + _speech(20))

    partial = session.partial_due()

    assert partial is not None
    assert partial.voiced_seconds == pytest.approx(20 * WINDOW_SECONDS)


def test_a_partial_is_due_only_every_four_tenths_of_a_second() -> None:
    session, clock = _session()
    session.add(_speech(20))
    session.partial_started()
    session.add(_speech(3))

    clock.now = 0.39
    too_soon = session.partial_due()
    clock.now = 0.4

    assert too_soon is None
    assert session.partial_due() is not None


def test_a_partial_waits_for_new_audio() -> None:
    session, clock = _session()
    session.add(_speech(20))
    session.partial_started()

    clock.now = 10.0

    assert session.partial_due() is None


def test_a_partial_is_not_due_before_speech() -> None:
    session, _ = _session()
    session.add(_quiet(60))

    assert session.partial_due() is None


def test_a_commit_ends_the_open_turn_at_once_for_the_client() -> None:
    """WIRE-63: a commit ends the open turn at once for the client."""
    session, _ = _session()
    session.add(_speech(20))

    ended = session.commit()

    assert ended == TurnEnd(
        pcm=_speech(20),
        start_ms=0,
        end_ms=round(20 * WINDOW_SECONDS * 1000),
        voiced_seconds=pytest.approx(20 * WINDOW_SECONDS),
        ended_by=TurnEndedBy.CLIENT,
    )


def test_a_commit_with_no_audio_since_the_last_turn_ended_is_nothing() -> None:
    session, _ = _session()
    session.add(_speech(20))
    session.commit()

    assert session.commit() is None


def test_a_turn_ended_by_the_client_leaves_the_next_to_start_at_its_end() -> None:
    session, _ = _session()
    session.add(_speech(20))
    first = session.commit()
    session.add(_speech(20))

    second = session.commit()

    assert isinstance(first, TurnEnd)
    assert isinstance(second, TurnEnd)
    assert second.start_ms == first.end_ms


# One character is one 32 ms window: voiced speech, a faint voiced window that only
# the pre-roll can hold, or a window that is not voiced.
_WINDOW_PROBABILITY = {"v": 0.9, "f": 0.4, ".": 0.0}
SHORT_PAUSE = "." * 4
LONG_PAUSE = "." * 5
PRE_ROLL_WINDOWS = int(PRE_ROLL_SECONDS / WINDOW_SECONDS)


def _sustained_session(
    windows: str, sustained_ms: int | None, **session_settings: float
) -> HearingSession:
    """A session whose detector answers `windows`, asked for `sustained_ms`."""
    detector = ScriptedDetector(_WINDOW_PROBABILITY[window] for window in windows)
    session = HearingSession(
        _Clock(), detector, DEFAULT_LIVE_SILENCE_SECONDS, **session_settings
    )
    if sustained_ms is not None:
        session.sustain_after(sustained_ms)
    return session


def _heard(
    windows: str,
    sustained_ms: int | None = FOUR_TENTHS_MS,
    **session_settings: float,
) -> list[SpeechStart | SpeechSustain | TurnEnd]:
    """Every event of a session that is fed `windows` and asked for `sustained_ms`."""
    session = _sustained_session(windows, sustained_ms, **session_settings)
    return session.add(_quiet(len(windows)))


def _sustained_ends(
    windows: str,
    sustained_ms: int | None = FOUR_TENTHS_MS,
    **session_settings: float,
) -> list[int]:
    """The position of each `SpeechSustain` of `windows`, in milliseconds."""
    return [
        event.audio_ms
        for event in _heard(windows, sustained_ms, **session_settings)
        if isinstance(event, SpeechSustain)
    ]


@pytest.mark.parametrize(
    ("windows", "reached_at"),
    [
        pytest.param("v" * (THRESHOLD_WINDOWS - 1), None, id="one-window-short"),
        pytest.param(
            "." * 4 + "v" * THRESHOLD_WINDOWS,
            4 + THRESHOLD_WINDOWS - 1,
            id="exactly-the-threshold",
        ),
        pytest.param("." * 4 + "v" * 60, 4 + THRESHOLD_WINDOWS - 1, id="longer"),
        pytest.param(
            "v" * 8 + SHORT_PAUSE + "v" * 8,
            8 + len(SHORT_PAUSE) + (THRESHOLD_WINDOWS - 8) - 1,
            id="a-pause-within-the-bound-adds-nothing-and-restarts-nothing",
        ),
        pytest.param(
            "v" * 8 + LONG_PAUSE + "v" * 8, None, id="a-longer-pause-restarts-the-count"
        ),
        pytest.param(
            "v" * 8 + LONG_PAUSE + "v" * THRESHOLD_WINDOWS,
            8 + len(LONG_PAUSE) + THRESHOLD_WINDOWS - 1,
            id="a-voice-after-a-long-pause-reaches-the-threshold-on-its-own",
        ),
        pytest.param(
            "v" * 4 + SHORT_PAUSE + "v" * 4 + LONG_PAUSE + "v" * 12,
            None,
            id="the-count-before-a-long-pause-is-forgotten",
        ),
    ],
)
def test_one_event_comes_where_the_voice_since_the_last_long_pause_reaches_the_count(
    windows: str, reached_at: int | None
) -> None:
    """WIRE-171, WIRE-173: one event at the end of the window that reaches the count."""
    expected = [] if reached_at is None else [end_of_window(reached_at)]

    assert _sustained_ends(windows + "." * SILENT_WINDOWS) == expected


@pytest.mark.parametrize(
    ("pause_windows", "reached_at"),
    [
        pytest.param(3, None, id="a-pause-of-the-bound-restarts-the-count"),
        pytest.param(
            5,
            8 + len(SHORT_PAUSE) + (THRESHOLD_WINDOWS - 8) - 1,
            id="a-pause-short-of-the-bound-adds-nothing-and-restarts-nothing",
        ),
    ],
)
def test_the_pause_keyword_moves_the_bound_a_pause_restarts_the_count_at(
    pause_windows: int, reached_at: int | None
) -> None:
    """WIRE-173: the keyword sets the session's own pause bound."""
    windows = "v" * 8 + SHORT_PAUSE + "v" * 8 + "." * SILENT_WINDOWS
    expected = [] if reached_at is None else [end_of_window(reached_at)]

    heard = _sustained_ends(
        windows, pause_seconds=(pause_windows - 0.5) * WINDOW_SECONDS
    )

    assert heard == expected


def test_a_session_that_did_not_ask_hears_no_event() -> None:
    """WIRE-172: without the ask the scheduler raises no event."""
    assert _sustained_ends("v" * 60, sustained_ms=None) == []


def test_the_event_comes_after_the_start_and_before_the_end_of_its_turn() -> None:
    """WIRE-175: the event lies between its turn's start and its end."""
    events = _heard("." * 4 + "v" * 30 + "." * SILENT_WINDOWS)

    assert [type(event) for event in events] == [
        SpeechStart,
        SpeechSustain,
        TurnEnd,
    ]


@pytest.mark.parametrize(
    ("windows", "sustained_ms", "reached_at"),
    [
        pytest.param(
            "f" * PRE_ROLL_WINDOWS + "v" * START_WINDOWS,
            FOUR_TENTHS_MS,
            PRE_ROLL_WINDOWS + START_WINDOWS - 1,
            id="a-pre-roll-that-is-all-voiced-reaches-the-default-at-the-start",
        ),
        pytest.param(
            "f" * PRE_ROLL_WINDOWS + "v" * START_WINDOWS,
            200,
            PRE_ROLL_WINDOWS + START_WINDOWS - 1,
            id="a-duration-the-pre-roll-already-holds-is-reached-at-the-start",
        ),
        pytest.param(
            "f" * 3 + SHORT_PAUSE + "f" * 4 + "v" * START_WINDOWS,
            250,
            3 + len(SHORT_PAUSE) + 4 + START_WINDOWS - 1,
            id="a-short-pause-in-the-pre-roll-counts-on",
        ),
        pytest.param(
            "f" * 3 + LONG_PAUSE + "f" * 4 + "v" * START_WINDOWS,
            250,
            None,
            id="a-long-pause-in-the-pre-roll-counts-only-the-windows-since-it",
        ),
    ],
)
def test_the_pre_roll_windows_count_under_the_same_rule(
    windows: str, sustained_ms: int, reached_at: int | None
) -> None:
    """WIRE-173: pre-roll windows count since the last pause longer than the bound."""
    events = _heard(windows, sustained_ms)

    if reached_at is None:
        assert [type(event) for event in events] == [SpeechStart]
    else:
        assert events == [
            SpeechStart(audio_ms=events[0].audio_ms),
            SpeechSustain(audio_ms=end_of_window(reached_at)),
        ]


def test_a_voice_that_stops_first_sends_no_event_and_the_next_item_counts_afresh() -> (
    None
):
    """WIRE-173: silence ends an item before its count; the next item starts at zero."""
    first_short = "v" * 8 + "." * SILENT_WINDOWS
    second_short = "v" * 8 + "." * SILENT_WINDOWS

    assert _sustained_ends(first_short + second_short) == []


def test_each_item_gets_its_own_event() -> None:
    """WIRE-171, WIRE-175: a second item is sustained once on its own."""
    first = "v" * 20 + "." * SILENT_WINDOWS
    second = "v" * 20 + "." * SILENT_WINDOWS

    ends = _sustained_ends(first + second)

    second_start = len(first)
    assert ends == [
        end_of_window(THRESHOLD_WINDOWS - 1),
        end_of_window(second_start + THRESHOLD_WINDOWS - 1),
    ]


def test_a_duration_the_turn_cap_reaches_first_sends_no_event() -> None:
    """WIRE-172, WIRE-173: a capped item ends before a duration of the cap's length."""
    session, _ = _session()
    session.sustain_after(round(MAX_TURN_SECONDS * 1000))

    events = session.add(_seconds(MAX_TURN_SECONDS + 1, LOUD_LEVEL))

    assert not any(isinstance(event, SpeechSustain) for event in events)
    assert _ends(events)[0].ended_by is TurnEndedBy.CAP


@pytest.mark.parametrize("ends", ["commit", "cut"])
def test_a_client_commit_or_a_drain_cut_before_the_count_sends_no_event(
    ends: str,
) -> None:
    """WIRE-173: an item ended by the client or a drain before the count has none."""
    session, _ = _session()
    session.sustain_after(FOUR_TENTHS_MS)
    heard = session.add(_speech(THRESHOLD_WINDOWS - 1))

    ended = session.commit() if ends == "commit" else session.cut()
    later = session.add(_speech(THRESHOLD_WINDOWS - 1))

    assert not any(isinstance(event, SpeechSustain) for event in heard + later)
    assert isinstance(ended, TurnEnd)


def test_a_commit_after_the_event_leaves_the_event_standing() -> None:
    """WIRE-175: an event sent before the item ends stays sent."""
    session, _ = _session()
    session.sustain_after(FOUR_TENTHS_MS)

    heard = session.add(_speech(THRESHOLD_WINDOWS + 3))
    ended = session.commit()

    assert [event.audio_ms for event in heard if isinstance(event, SpeechSustain)] == [
        end_of_window(THRESHOLD_WINDOWS - 1)
    ]
    assert isinstance(ended, TurnEnd)


# The moment's cases count under a duration of 7 windows and a pause bound of 5.
MOMENT_DURATION_MS = 200
MOMENT_PAUSE_SECONDS = 0.15
ITEM = "." * 4 + "v" * 20 + "." * SILENT_WINDOWS


def _moment_session(windows: str) -> HearingSession:
    return _sustained_session(
        windows, MOMENT_DURATION_MS, pause_seconds=MOMENT_PAUSE_SECONDS
    )


def _counted_from(
    windows: str, from_ms: int, *, named_after: int = 0
) -> list[SchedulerEvent]:
    """Every event of `windows` when `from_ms` is named after `named_after` of them."""
    session = _moment_session(windows)
    heard = session.add(_quiet(named_after))
    session.sustain_from(from_ms)
    return heard + session.add(_quiet(len(windows) - named_after))


def _moments(events: list[SchedulerEvent]) -> list[SpeechSustainFrom]:
    return [event for event in events if isinstance(event, SpeechSustainFrom)]


@pytest.mark.parametrize(
    ("windows", "from_ms", "named_after", "reached_at"),
    [
        pytest.param(
            ITEM,
            start_of_window(10),
            0,
            16,
            id="a-moment-ahead-of-the-audio-waits-for-it",
        ),
        pytest.param(
            ITEM, start_of_window(10), 8, 16, id="a-moment-named-while-the-item-is-open"
        ),
        pytest.param(
            ITEM,
            start_of_window(9),
            0,
            15,
            id="a-window-that-begins-at-the-moment-counts",
        ),
        pytest.param(
            ITEM,
            start_of_window(9) + 1,
            0,
            16,
            id="a-window-straddling-the-moment-does-not-count",
        ),
        pytest.param(
            "v" * 5 + "." * 4 + "v" * 10 + "." * SILENT_WINDOWS,
            start_of_window(0),
            0,
            10,
            id="a-pause-within-the-bound-restarts-nothing",
        ),
        pytest.param(
            "v" * 10 + "." * 5 + "v" * 10 + "." * SILENT_WINDOWS,
            start_of_window(4),
            0,
            21,
            id="a-pause-past-the-bound-restarts-the-count-from-zero",
        ),
        pytest.param(
            ITEM, start_of_window(5), 0, 11, id="a-moment-inside-the-confirmation-run"
        ),
        pytest.param(
            ITEM,
            start_of_window(6),
            0,
            12,
            id="a-moment-at-the-window-that-confirms-the-run",
        ),
        pytest.param(
            ITEM,
            start_of_window(4),
            0,
            10,
            id="a-run-whose-first-window-begins-at-the-moment",
        ),
        pytest.param(
            "f" * 3 + "v" * 20 + "." * SILENT_WINDOWS,
            start_of_window(3),
            0,
            9,
            id="pre-roll-windows-before-the-moment-do-not-count",
        ),
        pytest.param(
            ITEM,
            start_of_window(4) - 1,
            0,
            None,
            id="an-item-that-begins-after-the-moment-gets-none",
        ),
        pytest.param(
            "v" * 10 + "." * SILENT_WINDOWS + "v" * 30 + "." * SILENT_WINDOWS,
            start_of_window(5),
            0,
            None,
            id="an-item-that-ends-before-the-count-gets-none-and-the-next-none",
        ),
        pytest.param(
            "v" * 10 + "." * SILENT_WINDOWS + "." * 4 + "v" * 30,
            end_of_window(10 + SILENT_WINDOWS - 1),
            0,
            None,
            id="an-item-that-ended-at-the-moment-is-not-its-item",
        ),
        pytest.param(
            "." * 30 + "v" * 20 + "." * SILENT_WINDOWS,
            start_of_window(10),
            0,
            None,
            id="no-item-at-the-moment",
        ),
        pytest.param(
            "v" * 2 + "." * 3 + "v" * 20 + "." * SILENT_WINDOWS,
            start_of_window(1),
            0,
            None,
            id="a-run-that-fails-confirmation-gets-nothing",
        ),
    ],
)
def test_one_event_comes_where_the_voice_from_the_moment_reaches_the_count(
    windows: str, from_ms: int, named_after: int, reached_at: int | None
) -> None:
    """WIRE-177: the windows that begin at or after the moment count, once."""
    heard = _counted_from(windows, from_ms, named_after=named_after)

    expected = (
        []
        if reached_at is None
        else [SpeechSustainFrom(from_ms=from_ms, audio_ms=end_of_window(reached_at))]
    )
    assert _moments(heard) == expected


def test_the_moments_event_follows_the_items_own_when_both_complete_in_one_window() -> (
    None
):
    """WIRE-177: the item's `SpeechSustain` comes before the moment's event."""
    heard = _counted_from(ITEM, start_of_window(4))

    assert [type(event) for event in heard if not isinstance(event, TurnEnd)] == [
        SpeechStart,
        SpeechSustain,
        SpeechSustainFrom,
    ]


def test_a_moment_equal_to_the_standing_one_changes_nothing() -> None:
    """WIRE-178: naming the standing moment again keeps its count and sends no more."""
    from_ms = start_of_window(10)
    session = _moment_session(ITEM)
    session.sustain_from(from_ms)
    heard = session.add(_quiet(12))
    session.sustain_from(from_ms)
    heard += session.add(_quiet(8))
    session.sustain_from(from_ms)
    heard += session.add(_quiet(len(ITEM) - 20))

    assert _moments(heard) == [
        SpeechSustainFrom(from_ms=from_ms, audio_ms=end_of_window(16))
    ]


def test_a_new_moment_replaces_the_standing_one_that_has_not_sent() -> None:
    """WIRE-178: no event of the older moment follows."""
    session = _moment_session(ITEM)
    session.sustain_from(start_of_window(8))
    heard = session.add(_quiet(11))
    session.sustain_from(start_of_window(12))
    heard += session.add(_quiet(len(ITEM) - 11))

    assert _moments(heard) == [
        SpeechSustainFrom(from_ms=start_of_window(12), audio_ms=end_of_window(18))
    ]


def test_a_new_moment_leaves_the_event_the_older_one_already_sent_standing() -> None:
    """WIRE-178: an event sent stands, and the new moment counts on its own."""
    session = _moment_session(ITEM)
    session.sustain_from(start_of_window(6))
    heard = session.add(_quiet(13))
    session.sustain_from(start_of_window(16))
    heard += session.add(_quiet(len(ITEM) - 13))

    assert _moments(heard) == [
        SpeechSustainFrom(from_ms=start_of_window(6), audio_ms=end_of_window(12)),
        SpeechSustainFrom(from_ms=start_of_window(16), audio_ms=end_of_window(22)),
    ]


def test_a_moment_serves_only_its_item_once_that_item_ended() -> None:
    """WIRE-178: after a commit ends the item, the next item gets no event."""
    session = _moment_session("v" * 30)
    session.sustain_from(start_of_window(6))
    heard = session.add(_quiet(8))
    session.commit()
    heard += session.add(_quiet(22))

    assert _moments(heard) == []


def test_an_item_after_a_commit_inside_a_window_begins_at_the_moment() -> None:
    """WIRE-177: an item whose start is in the moment's millisecond is not after it."""
    from_ms = start_of_window(10)
    session = _moment_session("v" * 30 + "." * SILENT_WINDOWS)
    session.add(_quiet(10) + b"\0" * 8 * BYTES_PER_SAMPLE)
    session.commit()
    session.sustain_from(from_ms)

    heard = session.add(_quiet(20 + SILENT_WINDOWS))

    assert [event.from_ms for event in _moments(heard)] == [from_ms]
