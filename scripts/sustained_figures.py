"""The pure figures of the sustained-speech probe: cuts, joins, trials and the pick.

Nothing here reads a file, a corpus or the voice model: the probe hands over
annotations and the speech probabilities of audio, and these functions cut the
excerpts, judge them, join pairs, replay every excerpt through the real
`HearingSession` for each cell of the grid, and tell which cell the pick rule
takes. A cell is a duration `n` and a restart run `m`, both in whole 32 ms
windows: the session is asked for `32 n - 16` milliseconds and restarts its count
after a pause of `(m - 0.5)` windows, so `m` unvoiced windows in a row restart it.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from speech_service.config import DEFAULT_LIVE_SILENCE_SECONDS
from speech_service.hearing_session import HearingSession, SpeechSustain
from speech_service.pcm import BYTES_PER_SAMPLE, HEARING_SAMPLE_RATE, LIVE_INPUT_RATE
from speech_service.voice_activity import WINDOW_SAMPLES, WINDOW_SECONDS, is_voiced

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

WINDOW_MS: Final = round(WINDOW_SECONDS * 1000)
WIRE_WINDOW_BYTES: Final = (
    WINDOW_SAMPLES * LIVE_INPUT_RATE // HEARING_SAMPLE_RATE * BYTES_PER_SAMPLE
)
CLOSING_WINDOWS: Final = math.ceil(DEFAULT_LIVE_SILENCE_SECONDS / WINDOW_SECONDS)

DURATION_WINDOWS: Final = range(9, 20)
RESTART_RUNS: Final = range(3, 10)
PICK_DURATION_WINDOWS: Final = range(10, 16)
PICK_RESTART_RUNS: Final = range(4, 7)
MEDIAN_DELAY_CAP: Final = 0.5
P95_DELAY_CAP: Final = 0.7
DISAGREEMENT_SIGMAS: Final = 2

LEAD_IN_SECONDS: Final = 0.5
MAX_SOUND_WINDOWS: Final = 31
ISOLATION_TOLERANCE_SECONDS: Final = 0.1
PAIR_GAP_WINDOWS: Final = range(2, 16)

BACKCHANNEL_MAX_WORDS: Final = 3
BACKCHANNEL_MAX_SECONDS: Final = 1.0
ALONE_SECONDS: Final = 1.0
ONSET_QUIET_SECONDS: Final = 1.0
ONSET_RUN_SECONDS: Final = 1.5
ONSET_GAP_SECONDS: Final = 0.5
EXCERPT_BEFORE_SECONDS: Final = 2.0
BACKCHANNEL_AFTER_SECONDS: Final = 1.0
ONSET_AFTER_SECONDS: Final = 1.5


@dataclass(frozen=True, order=True)
class Cell:
    """One duration and one restart run of the grid, in whole windows."""

    duration_windows: int
    restart_run: int

    @property
    def duration_ms(self) -> int:
        """What the session is asked for: `32 n - 16`, so 13 windows are 400 ms."""
        return WINDOW_MS * self.duration_windows - WINDOW_MS // 2

    @property
    def pause_seconds(self) -> float:
        """The pause bound the session gets: `(m - 0.5)` windows."""
        return (self.restart_run - 0.5) * WINDOW_SECONDS


GRID: Final = tuple(
    Cell(duration, run) for duration in DURATION_WINDOWS for run in RESTART_RUNS
)
RULED_CELL: Final = Cell(13, 5)


class Sound(StrEnum):
    """What a false-hold trial hears; the values are the corpora's own labels."""

    BACKCHANNEL = "backchannel"
    LAUGHTER = "laughter"
    COUGH = "cough"
    THROAT_CLEARING = "throatclearing"


class DelayCorpus(StrEnum):
    """Where an onset comes from."""

    AMI = "ami english"
    FLEURS = "fleurs german"


class DelayRow(StrEnum):
    """A row of the delay table; the overlapped row is shown and not constrained."""

    AMI = "ami english"
    AMI_OVERLAPPED = "ami english, overlapped"
    FLEURS = "fleurs german"


MINIMUM_KEPT: Final[Mapping[str, int]] = {
    Sound.BACKCHANNEL: 150,
    Sound.LAUGHTER: 60,
    Sound.COUGH: 60,
    Sound.THROAT_CLEARING: 60,
    DelayCorpus.AMI: 100,
    DelayCorpus.FLEURS: 200,
}


class Fate(StrEnum):
    """What becomes of one excerpt: heard, not heard (still a trial), or dropped."""

    HEARD = "heard"
    NOT_HEARD = "not heard"
    NO_LEAD_IN = "no lead-in"
    NOT_ISOLATED = "not isolated"
    TOO_LONG = "too long"

    @property
    def kept(self) -> bool:
        """Whether the excerpt is a trial."""
        return self in {Fate.HEARD, Fate.NOT_HEARD}


class ReplayDetector:
    """The probabilities of one excerpt as a session's detector, window by window.

    Like the real stream it keeps the samples short of a window, so any split of
    the audio into chunks gives the same windows.
    """

    def __init__(self, probabilities: Sequence[float]) -> None:
        """Answer `probabilities` in order, one for each whole window fed."""
        self._probabilities = probabilities
        self._samples = 0
        self._answered = 0

    def probabilities(self, pcm: bytes) -> list[float]:
        """The answers of the windows `pcm` completes."""
        self._samples += len(pcm) // BYTES_PER_SAMPLE
        complete = self._samples // WINDOW_SAMPLES
        answers = list(self._probabilities[self._answered : complete])
        self._answered = complete
        return answers


def first_sustain_ms(probabilities: Sequence[float], cell: Cell) -> int | None:
    """When the real session first raises `SpeechSustain` over these windows.

    The millisecond is the end of the window that completes the count, counted
    from the excerpt's first sample; `None` when the excerpt raises none.
    """
    session = HearingSession(
        lambda: 0.0,
        ReplayDetector(probabilities),
        DEFAULT_LIVE_SILENCE_SECONDS,
        pause_seconds=cell.pause_seconds,
    )
    session.sustain_after(cell.duration_ms)
    events = session.add(bytes(len(probabilities) * WINDOW_SAMPLES * BYTES_PER_SAMPLE))
    return next(
        (event.audio_ms for event in events if isinstance(event, SpeechSustain)), None
    )


def closed(probabilities: Sequence[float]) -> tuple[float, ...]:
    """The windows followed by the silence that ends a turn."""
    return (*probabilities, *(0.0,) * CLOSING_WINDOWS)


def voiced_indices(probabilities: Sequence[float]) -> list[int]:
    """The indices of the windows the service counts as voiced."""
    return [index for index, p in enumerate(probabilities) if is_voiced(p)]


def lies_outside(
    probabilities: Sequence[float], lower_seconds: float, upper_seconds: float
) -> bool:
    """Whether a voiced window has any part outside [lower, upper] seconds."""
    return any(
        index * WINDOW_SECONDS < lower_seconds
        or (index + 1) * WINDOW_SECONDS > upper_seconds
        for index in voiced_indices(probabilities)
    )


def fate_of(
    probabilities: Sequence[float],
    *,
    inside: tuple[float, float] | None = None,
    max_windows: int | None = None,
) -> Fate:
    """Whether an excerpt is heard, not heard, or dropped, and why.

    It is dropped, for the first reason that holds, for a voiced window outside
    `inside` (seconds from the excerpt's start), for less than `LEAD_IN_SECONDS`
    unvoiced before its first voiced window, or for a span from its first to its
    last voiced window longer than `max_windows`.
    """
    voiced = voiced_indices(probabilities)
    if not voiced:
        return Fate.NOT_HEARD
    if inside is not None and lies_outside(probabilities, *inside):
        return Fate.NOT_ISOLATED
    if voiced[0] * WINDOW_SECONDS < LEAD_IN_SECONDS:
        return Fate.NO_LEAD_IN
    if max_windows is not None and voiced[-1] - voiced[0] + 1 > max_windows:
        return Fate.TOO_LONG
    return Fate.HEARD


@dataclass(frozen=True)
class Timed:
    """One of a wearer's words (`spoken`) or vocal sounds, in seconds."""

    start: float
    end: float
    spoken: bool


@dataclass(frozen=True)
class Segment:
    """A stretch in which another speaker talks, in seconds."""

    start: float
    end: float


@dataclass(frozen=True)
class Backchannel:
    """A backchannel's annotated span in seconds, and whether another spoke in it."""

    start: float
    end: float
    overlapped: bool


@dataclass(frozen=True)
class Onset:
    """The start of a stretch of the wearer's speech, and whether another spoke."""

    at: float
    overlapped: bool


def backchannel_spans(
    timed: Sequence[Timed], acts: Iterable[range], others: Sequence[Segment]
) -> list[Backchannel]:
    """The backchannels among `acts` (index ranges into `timed`) that lie alone.

    A backchannel is one to three spoken words over more than 0 and at most 1.0 s;
    it lies alone when every other timed word and vocal sound of the wearer ends
    1.0 s before its start or begins 1.0 s after its end.
    """
    found = []
    for act in acts:
        words = [item for item in timed[act.start : act.stop] if item.spoken]
        if not 1 <= len(words) <= BACKCHANNEL_MAX_WORDS:
            continue
        start, end = words[0].start, words[-1].end
        if not 0 < end - start <= BACKCHANNEL_MAX_SECONDS:
            continue
        rest = (
            *timed[: act.start],
            *timed[act.stop :],
            *(item for item in timed[act.start : act.stop] if not item.spoken),
        )
        if all(
            item.end <= start - ALONE_SECONDS or item.start >= end + ALONE_SECONDS
            for item in rest
        ):
            found.append(Backchannel(start, end, _spoken_over(others, start, end)))
    return found


def speech_onsets(timed: Sequence[Timed], others: Sequence[Segment]) -> list[Onset]:
    """The words that start the wearer's speech after a quiet that is long enough.

    An onset is the first word after 1.0 s without any own word or vocal sound
    that continues for at least 1.5 s with no own gap of 0.5 s or more.
    """
    found = []
    previous_end = -math.inf
    for index, item in enumerate(timed):
        if (
            item.spoken
            and item.start - previous_end >= ONSET_QUIET_SECONDS
            and _continues(timed, index)
        ):
            found.append(
                Onset(item.start, _spoken_over(others, item.start, item.start))
            )
        previous_end = max(previous_end, item.end)
    return found


def _continues(timed: Sequence[Timed], index: int) -> bool:
    """Whether the speech from `timed[index]` goes on for 1.5 s without a long gap."""
    first = timed[index]
    reach = first.end
    for item in timed[index + 1 :]:
        if reach >= first.start + ONSET_RUN_SECONDS:
            break
        if item.start - reach >= ONSET_GAP_SECONDS:
            return False
        reach = max(reach, item.end)
    return reach >= first.start + ONSET_RUN_SECONDS


def _spoken_over(others: Sequence[Segment], start: float, end: float) -> bool:
    return any(other.start <= end and other.end >= start for other in others)


@dataclass(frozen=True)
class HeardClip:
    """A clip's 24 kHz audio and the speech probability of each of its windows."""

    pcm: bytes
    probabilities: tuple[float, ...]


def pair_gap_windows(index: int) -> int:
    """The silence of the pair of this index, every gap of the range in turn."""
    return PAIR_GAP_WINDOWS[index % len(PAIR_GAP_WINDOWS)]


def join_pair(first: HeardClip, second: HeardClip, gap_windows: int) -> bytes:
    """Both clips' speech in one 24 kHz audio, `gap_windows` of silence between them.

    The first clip runs from its start to its last voiced window, the second from
    its first to its last voiced window.
    """
    first_voiced = voiced_indices(first.probabilities)
    second_voiced = voiced_indices(second.probabilities)
    return (
        first.pcm[: (first_voiced[-1] + 1) * WIRE_WINDOW_BYTES]
        + bytes(gap_windows * WIRE_WINDOW_BYTES)
        + second.pcm[
            second_voiced[0] * WIRE_WINDOW_BYTES : (second_voiced[-1] + 1)
            * WIRE_WINDOW_BYTES
        ]
    )


def two_by_two[T](clips: Sequence[T]) -> list[tuple[T, T]]:
    """The clips taken two by two in order; an odd last clip pairs with nothing."""
    return list(zip(clips[0::2], clips[1::2], strict=False))


@dataclass(frozen=True)
class HoldTrial:
    """One false-hold trial: an excerpt or a pair, closed by a turn's silence."""

    sound: Sound
    paired: bool
    overlapped: bool
    probabilities: tuple[float, ...]

    @property
    def group(self) -> str:
        """The row it counts in: its sound, or its sound as a pair."""
        return f"{self.sound} pair" if self.paired else str(self.sound)


@dataclass(frozen=True)
class DelayTrial:
    """One onset: the excerpt's windows and the second its onset lies at."""

    corpus: DelayCorpus
    overlapped: bool
    probabilities: tuple[float, ...]
    onset_seconds: float

    @property
    def rows(self) -> tuple[DelayRow, ...]:
        """The delay rows it counts in."""
        if self.corpus is DelayCorpus.FLEURS:
            return (DelayRow.FLEURS,)
        if self.overlapped:
            return (DelayRow.AMI, DelayRow.AMI_OVERLAPPED)
        return (DelayRow.AMI,)


def delay_seconds(trial: DelayTrial, cell: Cell) -> float:
    """The seconds from the onset to the first `SpeechSustain`; infinite when none."""
    reached = first_sustain_ms(trial.probabilities, cell)
    if reached is None:
        return math.inf
    return reached / 1000 - trial.onset_seconds


@dataclass(frozen=True)
class CellFigures:
    """What one cell gives: whether each false-hold trial holds, each onset's delay."""

    cell: Cell
    holds: tuple[bool, ...]
    delays: Mapping[DelayRow, tuple[float, ...]]

    @property
    def hold_count(self) -> int:
        """The false holds of every trial of every class and pair."""
        return sum(self.holds)

    def meets_delay_cap(self) -> bool:
        """Whether each constrained delay row keeps its median and 95th percentile."""
        return all(
            statistics.median(self.delays[row]) <= MEDIAN_DELAY_CAP
            and percentile_95(self.delays[row]) <= P95_DELAY_CAP
            for row in (DelayRow.AMI, DelayRow.FLEURS)
        )


def percentile_95(values: Sequence[float]) -> float:
    """The 95th percentile by nearest rank."""
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def measure(
    cell: Cell, hold_trials: Sequence[HoldTrial], delay_trials: Sequence[DelayTrial]
) -> CellFigures:
    """Replay every trial through the real session under `cell`."""
    delays: dict[DelayRow, list[float]] = {row: [] for row in DelayRow}
    for trial in delay_trials:
        delay = delay_seconds(trial, cell)
        for row in trial.rows:
            delays[row].append(delay)
    return CellFigures(
        cell,
        tuple(
            first_sustain_ms(trial.probabilities, cell) is not None
            for trial in hold_trials
        ),
        {row: tuple(values) for row, values in delays.items()},
    )


@dataclass(frozen=True)
class Pick:
    """The cell the pick rule takes, or why it takes none.

    `cell` is `None` when no cell meets the delay cap on both rows. A cell
    outside the bound is a card for the operator, not a pick to build.
    """

    cell: Cell | None
    replaces_ruled: bool
    within_bound: bool


def pick(measured: Sequence[CellFigures]) -> Pick:
    """The pick rule: the lowest pooled share under the delay cap, the ruled pair kept.

    Among the cells that meet both delay caps the lowest pooled false-hold share
    wins, ties to the shorter duration and then the smaller restart run. It
    replaces the ruled cell only when the ruled cell misses the cap, or its extra
    false holds exceed the best's by more than twice the square root of the number
    of trials where the two disagree.
    """
    meeting = [figures for figures in measured if figures.meets_delay_cap()]
    if not meeting:
        return Pick(None, replaces_ruled=False, within_bound=False)
    best = min(meeting, key=lambda f: (f.hold_count, f.cell))
    ruled = next((f for f in meeting if f.cell == RULED_CELL), None)
    chosen = best if ruled is None or _outweighs(ruled, best) else ruled
    return Pick(
        chosen.cell,
        replaces_ruled=chosen.cell != RULED_CELL,
        within_bound=chosen.cell.duration_windows in PICK_DURATION_WINDOWS
        and chosen.cell.restart_run in PICK_RESTART_RUNS,
    )


def _outweighs(ruled: CellFigures, best: CellFigures) -> bool:
    disagreeing = sum(a != b for a, b in zip(ruled.holds, best.holds, strict=True))
    extra = ruled.hold_count - best.hold_count
    return extra > DISAGREEMENT_SIGMAS * math.sqrt(disagreeing)


class Tally:
    """What became of each excerpt of each group, whatever the cell."""

    def __init__(self) -> None:
        """Start with nothing counted."""
        self._fates: dict[str, Counter[Fate]] = {}

    def add(self, group: str, fate: Fate) -> None:
        """Count one excerpt of `group`."""
        self._fates.setdefault(group, Counter())[fate] += 1

    def kept(self, group: str) -> int:
        """The excerpts of `group` that are trials."""
        return sum(n for fate, n in self._fates.get(group, {}).items() if fate.kept)

    def missed(self, groups: Iterable[str]) -> list[str]:
        """Those of `groups` below their minimum."""
        return [group for group in groups if self.kept(group) < MINIMUM_KEPT[group]]

    def record(self) -> dict[str, dict[str, object]]:
        """Each group's trials, those not heard and those dropped by reason."""
        return {
            group: {
                "kept": self.kept(group),
                "not_heard": fates[Fate.NOT_HEARD],
                "dropped": {
                    fate.value: n for fate, n in fates.items() if not fate.kept
                },
            }
            for group, fates in self._fates.items()
        }


def _share(holds: int, trials: int) -> dict[str, float | int]:
    return {
        "holds": holds,
        "trials": trials,
        "share_percent": round(100 * holds / trials, 2) if trials else 0.0,
    }


def _finite(seconds: float) -> float | None:
    return None if math.isinf(seconds) else round(seconds, 3)


def _delay_record(delays: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(delays),
        "median": _finite(statistics.median(delays)),
        "p95": _finite(percentile_95(delays)),
        "infinite_share": round(sum(math.isinf(d) for d in delays) / len(delays), 3),
    }


def cell_record(
    figures: CellFigures, hold_trials: Sequence[HoldTrial]
) -> dict[str, object]:
    """One cell's row: false holds by group and label, pooled, and the delay rows.

    An infinite delay prints as `null`.
    """
    by_group: dict[str, list[bool]] = {}
    by_label: dict[str, list[bool]] = {}
    for trial, held in zip(hold_trials, figures.holds, strict=True):
        by_group.setdefault(trial.group, []).append(held)
        if trial.sound is Sound.BACKCHANNEL and not trial.paired:
            label = "overlapped" if trial.overlapped else "alone"
            by_label.setdefault(label, []).append(held)
    return {
        "duration_windows": figures.cell.duration_windows,
        "duration_ms": figures.cell.duration_ms,
        "restart_run": figures.cell.restart_run,
        "pause_seconds": round(figures.cell.pause_seconds, 3),
        "false_holds": {
            group: _share(sum(held), len(held)) for group, held in by_group.items()
        },
        "backchannel_by_label": {
            label: _share(sum(held), len(held)) for label, held in by_label.items()
        },
        "pooled": _share(figures.hold_count, len(figures.holds)),
        "delay": {
            row.value: _delay_record(delays)
            for row, delays in figures.delays.items()
            if delays
        },
    }
