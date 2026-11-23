"""Sound events: which sounds a model's scores name, and when one begins and ends."""

import threading
from typing import TYPE_CHECKING

import pytest

from speech_client.realtime_wire import SoundName
from speech_service.sound_events import (
    SoundModelError,
    SoundModelMissingError,
    SoundStream,
    SoundTaggers,
    SoundWindow,
    check_model_directory,
)
from tests.support.sound_support import (
    LABELS,
    FakeSoundModel,
    loaded_taggers,
    opener_of,
    scores_of,
    write_model_directory,
)

if TYPE_CHECKING:
    from pathlib import Path

SECOND = 16_000 * 2


def _audio(seconds: float) -> bytes:
    return bytes(int(seconds * SECOND) // 2 * 2)


def _stream(tmp_path: Path) -> SoundStream:
    taggers, _ = loaded_taggers(tmp_path)
    return taggers.open_stream()


def _hop(stream: SoundStream, levels: dict[str, float]) -> list[SoundName]:
    """Feed half a second, score the hop that is due, name the sounds it begins."""
    stream.add(_audio(0.5))
    window = stream.next_window()
    assert window is not None
    return [sound.name for sound in stream.scored(window, scores_of(levels))]


def _first_window(stream: SoundStream) -> SoundWindow:
    stream.add(_audio(2))
    window = stream.next_window()
    assert window is not None
    return window


@pytest.mark.parametrize(
    ("label", "sound"),
    [
        ("Laughter", SoundName.LAUGHTER),
        ("Giggle", SoundName.LAUGHTER),
        ("Snicker", SoundName.LAUGHTER),
        ("Belly laugh", SoundName.LAUGHTER),
        ("Cough", SoundName.COUGH),
        ("Clapping", SoundName.CLAPPING),
        ("Applause", SoundName.CLAPPING),
        ("Screaming", SoundName.SCREAMING),
    ],
)
def test_a_label_of_a_sound_names_that_sound(
    tmp_path: Path, label: str, sound: SoundName
) -> None:
    """WIRE-166: the labels of a sound fold into its name."""
    stream = _stream(tmp_path)
    window = _first_window(stream)

    heard = stream.scored(window, scores_of({label: 0.5}))

    assert [detected.name for detected in heard] == [sound]


@pytest.mark.parametrize("label", ["Throat clearing", "Sigh", "Speech"])
def test_a_label_outside_the_four_sounds_names_none(tmp_path: Path, label: str) -> None:
    stream = _stream(tmp_path)
    window = _first_window(stream)

    assert stream.scored(window, scores_of({label: 0.9})) == []


@pytest.mark.parametrize(
    ("level", "begins"), [(0.1, True), (0.099, False)], ids=["reaches", "below"]
)
def test_a_sound_begins_when_its_score_reaches_the_threshold(
    tmp_path: Path, level: float, *, begins: bool
) -> None:
    stream = _stream(tmp_path)
    window = _first_window(stream)

    heard = stream.scored(window, scores_of({"Cough": level}))

    assert bool(heard) is begins


def test_no_hop_is_due_before_two_seconds_of_audio(tmp_path: Path) -> None:
    """WIRE-167: no window is due before 2 s of audio."""
    stream = _stream(tmp_path)

    stream.add(_audio(1.99))

    assert stream.next_window() is None


def test_each_hop_is_two_seconds_hopped_by_half_a_second_on_the_session_clock(
    tmp_path: Path,
) -> None:
    stream = _stream(tmp_path)

    first = _first_window(stream)
    stream.scored(first, scores_of({}))
    stream.add(_audio(0.5))
    second = stream.next_window()

    assert second is not None
    assert (first.audio_start_ms, first.audio_end_ms) == (0, 2000)
    assert (second.audio_start_ms, second.audio_end_ms) == (500, 2500)
    assert first.samples.shape == second.samples.shape == (1, 32000)


def test_a_sound_event_carries_the_bounds_of_the_hop_that_heard_it(
    tmp_path: Path,
) -> None:
    stream = _stream(tmp_path)
    _hop_without_sound = _first_window(stream)
    stream.scored(_hop_without_sound, scores_of({}))

    stream.add(_audio(0.5))
    window = stream.next_window()
    assert window is not None
    (heard,) = stream.scored(window, scores_of({"Laughter": 0.4}))

    assert (heard.name, heard.audio_start_ms, heard.audio_end_ms) == (
        SoundName.LAUGHTER,
        500,
        2500,
    )
    assert heard.score == pytest.approx(0.4)


def test_several_due_hops_and_a_hop_in_flight_score_only_the_latest(
    tmp_path: Path,
) -> None:
    """WIRE-167: only the latest due hop is scored."""
    stream = _stream(tmp_path)

    stream.add(_audio(4))
    latest = stream.next_window()
    stream.add(_audio(1))
    while_in_flight = stream.next_window()
    assert latest is not None
    stream.scored(latest, scores_of({}))
    after = stream.next_window()

    assert (latest.audio_start_ms, latest.audio_end_ms) == (2000, 4000)
    assert while_in_flight is None
    assert after is not None
    assert (after.audio_start_ms, after.audio_end_ms) == (3000, 5000)
    stream.scored(after, scores_of({}))
    assert stream.next_window() is None


def test_a_sound_is_one_event_until_two_scored_hops_in_a_row_are_below(
    tmp_path: Path,
) -> None:
    """WIRE-167: a sound ends after two scored hops in a row below the threshold."""
    stream = _stream(tmp_path)
    window = _first_window(stream)
    began = stream.scored(window, scores_of({"Cough": 0.5}))

    after_a_gap_of_one = [
        _hop(stream, {}),
        _hop(stream, {"Cough": 0.5}),
    ]
    after_a_gap_of_two = [
        _hop(stream, {}),
        _hop(stream, {}),
        _hop(stream, {"Cough": 0.5}),
    ]

    assert [sound.name for sound in began] == [SoundName.COUGH]
    assert after_a_gap_of_one == [[], []]
    assert after_a_gap_of_two == [[], [], [SoundName.COUGH]]


def test_a_skipped_hop_neither_continues_nor_ends_a_sound(tmp_path: Path) -> None:
    """WIRE-167: a skipped hop counts neither way."""
    stream = _stream(tmp_path)
    window = _first_window(stream)
    stream.scored(window, scores_of({"Cough": 0.5}))
    _hop(stream, {})

    stream.add(_audio(3))
    skipping = stream.next_window()
    assert skipping is not None
    after_skips = stream.scored(skipping, scores_of({"Cough": 0.5}))

    assert after_skips == []

    # A quiet hop, skipped hops and a second quiet hop end the sound: the skips
    # neither continue it nor count as quiet themselves.
    ended = _stream(tmp_path)
    ended.scored(_first_window(ended), scores_of({"Cough": 0.5}))
    _hop(ended, {})
    ended.add(_audio(3))
    skipping = ended.next_window()
    assert skipping is not None
    ended.scored(skipping, scores_of({}))

    assert _hop(ended, {"Cough": 0.5}) == [SoundName.COUGH]


def test_two_sounds_in_one_window_are_two_events_in_a_fixed_order(
    tmp_path: Path,
) -> None:
    """WIRE-167: two sounds in one window are two events in a fixed order."""
    stream = _stream(tmp_path)
    window = _first_window(stream)

    heard = stream.scored(window, scores_of({"Screaming": 0.9, "Cough": 0.2}))

    assert [detected.name for detected in heard] == [
        SoundName.COUGH,
        SoundName.SCREAMING,
    ]


def test_a_directory_lacking_the_model_names_the_setting_the_build_and_the_switch(
    tmp_path: Path,
) -> None:
    """WIRE-169: a start without the model names the way out."""
    directory = write_model_directory(tmp_path / "model", skip="labels.json")

    with pytest.raises(SoundModelMissingError) as missing:
        check_model_directory(directory)

    message = str(missing.value)
    assert "SPEECH_SOUND_MODEL_DIRECTORY" in message
    assert f"scripts/build_sound_model.py {directory}" in message
    assert "SPEECH_SOUND_EVENTS=false" in message


def test_a_complete_directory_passes_the_check(tmp_path: Path) -> None:
    check_model_directory(write_model_directory(tmp_path / "model"))


def test_a_model_whose_labels_lack_a_label_of_a_sound_does_not_load(
    tmp_path: Path,
) -> None:
    """WIRE-169: a model whose labels lack a mapped one does not load."""
    labels = [label for label in LABELS if label != "Applause"]
    directory = write_model_directory(tmp_path, labels)
    taggers = SoundTaggers(directory, opener_of(FakeSoundModel(answer_labels=labels)))

    with pytest.raises(SoundModelError, match="Applause"):
        taggers.load()


@pytest.mark.parametrize(
    "model",
    [
        FakeSoundModel(input_shape=(1, 16000)),
        FakeSoundModel(input_shape=("batch", 32000)),
        FakeSoundModel(answer_labels=LABELS[:-1]),
    ],
    ids=["short-window-input", "symbolic-input", "fewer-score-columns"],
)
def test_a_model_of_another_shape_does_not_load(
    tmp_path: Path, model: FakeSoundModel
) -> None:
    """WIRE-169: a model of another shape does not load."""
    directory = write_model_directory(tmp_path)
    taggers = SoundTaggers(directory, opener_of(model))

    with pytest.raises(SoundModelError):
        taggers.load()


def test_a_loaded_model_scores_a_window_on_its_own_thread(tmp_path: Path) -> None:
    taggers, model = loaded_taggers(tmp_path)
    stream = taggers.open_stream()
    window = _first_window(stream)
    model.script({"Laughter": 0.7})

    scores = taggers.submit(window).result(timeout=5)

    assert [heard.name for heard in stream.scored(window, scores)] == [
        SoundName.LAUGHTER
    ]
    assert model.windows[0].shape == (1, 32000)
    assert model.thread_names[-1].startswith("sounds")
    assert model.thread_names[-1] != threading.current_thread().name


def test_a_closed_taggers_takes_no_window(tmp_path: Path) -> None:
    taggers, _ = loaded_taggers(tmp_path)
    window = _first_window(taggers.open_stream())

    taggers.close()

    with pytest.raises(RuntimeError):
        taggers.submit(window)
