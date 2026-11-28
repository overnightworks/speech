"""The speaking stage proof's figures hold for answers worked out by hand."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest

import hearing_figures
import speaking_figures as figures
from shipped_catalogue import PIPER_VOICES, first_language
from speech_client.realtime_wire import PcmFormat
from speech_client.wire import Language
from wav_audio import SILENCE_RMS

if TYPE_CHECKING:
    from collections.abc import Callable

    from speech_client.wire import VoiceId

RATE = 24000
LOUD = 8000
LIMIT_SECONDS = 0.5
SENTENCES = {Language.DE: "ein deutscher Satz", Language.EN: "an english sentence"}
SINE_PEAK_PER_RMS = 2**0.5
REFERENCE = "wie spät ist es und wann beginnt die nächste sitzung"


def wav(seconds: float, amplitude: int = LOUD) -> bytes:
    count = round(seconds * RATE)
    wave_form = amplitude * np.sin(2 * np.pi * 440 * np.arange(count) / RATE)
    return hearing_figures.wav_of(wave_form.astype("<i2").tobytes())


def answer(
    *,
    first: float = 0.2,
    total: float = 1.0,
    seconds: float = 2.0,
    amplitude: int = LOUD,
) -> tuple[bytes, float, float]:
    return wav(seconds, amplitude), first, total


def singles(seconds: float = 2.0) -> dict[VoiceId, figures.VoiceFigures]:
    return {
        voice: figures.VoiceFigures(wav(seconds), 0.2, 1.0, seconds, 0.5)
        for voice in PIPER_VOICES
    }


def german_and_english() -> tuple[VoiceId, VoiceId]:
    return figures.voice_pair(PIPER_VOICES)


def test_the_pair_is_a_german_and_an_english_voice_of_the_offer() -> None:
    german, english = german_and_english()

    assert (figures.language_of(german), figures.language_of(english)) == (
        Language.DE,
        Language.EN,
    )
    assert {german, english} <= set(PIPER_VOICES)


def test_each_voice_speaks_the_sentence_of_its_language() -> None:
    spoken: dict[VoiceId, str] = {}

    def speak(text: str, voice: VoiceId) -> tuple[bytes, float, float]:
        spoken[voice] = text
        return answer()

    medians = figures.prove_each_voice(
        speak, PIPER_VOICES, SENTENCES, limit=LIMIT_SECONDS
    )

    assert set(medians) == set(PIPER_VOICES)
    assert spoken == {voice: SENTENCES[first_language(voice)] for voice in PIPER_VOICES}


@pytest.mark.parametrize(
    ("spoken", "failure"),
    [
        (answer(amplitude=0), "silence"),
        (answer(seconds=0.5), "too little audio"),
        (answer(first=LIMIT_SECONDS + 0.1), "exceeded"),
    ],
    ids=["silent", "too short", "late first byte"],
)
def test_a_repetition_that_is_silent_short_or_late_fails_the_voice(
    spoken: tuple[bytes, float, float], failure: str
) -> None:
    with pytest.raises(SystemExit, match=failure):
        figures.prove_each_voice(
            lambda *_: spoken, PIPER_VOICES, SENTENCES, limit=LIMIT_SECONDS
        )


def amplitude_at(level_against_threshold: float) -> int:
    return round(
        SILENCE_RMS * level_against_threshold * SINE_PEAK_PER_RMS * PcmFormat.full_scale
    )


def prove_speak(heard: tuple[bytes, float, float]) -> figures.VoiceFigures:
    return figures.prove_speak_medians(
        lambda *_: heard,
        "text",
        PIPER_VOICES[0],
        limit=LIMIT_SECONDS,
        label="speak",
    )


@pytest.mark.parametrize(
    ("level_against_threshold", "silent"),
    [(0.9, True), (1.1, False)],
    ids=["below the silence level", "above the silence level"],
)
def test_audio_is_silent_only_below_the_silence_level(
    level_against_threshold: float, *, silent: bool
) -> None:
    heard = answer(amplitude=amplitude_at(level_against_threshold))

    assert figures.spoken_of(heard).is_silent is silent


def test_a_silent_repetition_is_refused() -> None:
    heard = answer(amplitude=amplitude_at(0.9))

    with pytest.raises(SystemExit, match="speak_rep1 produced silence"):
        prove_speak(heard)


def test_the_medians_of_a_voice_are_those_of_its_repetitions() -> None:
    firsts = iter([0.1, 0.5, 0.2, 0.4, 0.3])

    medians = figures.prove_speak_medians(
        lambda *_: answer(first=next(firsts)),
        "text",
        PIPER_VOICES[0],
        limit=LIMIT_SECONDS,
        label="speak",
    )

    assert medians.first_audio_seconds == pytest.approx(0.3)
    assert medians.real_time_factor == pytest.approx(0.5)


def test_two_voices_report_their_times_beside_their_own_single_runs() -> None:
    german, english = german_and_english()
    first_audio = {german: 0.4, english: 0.9}

    record = figures.prove_two_voices(
        lambda _text, voice: answer(first=first_audio[voice], total=1.5),
        (german, english),
        SENTENCES,
        singles(),
    )

    assert record[german.value] == {
        "single": {"first_audio_seconds": 0.2, "total_seconds": 1.0},
        "together": {"first_audio_seconds": 0.4, "total_seconds": 1.5},
    }
    assert (
        record[english.value]["together"]["first_audio_seconds"] == first_audio[english]
    )


@pytest.mark.parametrize(
    ("beside", "failure"),
    [
        (answer(amplitude=0), "silence"),
        (answer(seconds=1.0), "cut short"),
    ],
    ids=["silent", "cut short"],
)
def test_an_answer_beside_another_voice_that_is_silent_or_cut_short_fails(
    beside: tuple[bytes, float, float], failure: str
) -> None:
    voices = german_and_english()
    measured_alone = singles()

    with pytest.raises(SystemExit, match=failure):
        figures.prove_two_voices(lambda *_: beside, voices, SENTENCES, measured_alone)


EXTRA_WAIT_SECONDS = 2.0


def prove_concurrent(
    spoken: Callable[[str], tuple[bytes, float, float]],
) -> None:
    figures.prove_concurrent_speak(
        lambda text, _voice: spoken(text),
        PIPER_VOICES[0],
        ("one", "three"),
        limit=LIMIT_SECONDS,
        extra_wait_seconds=EXTRA_WAIT_SECONDS,
    )


def test_two_texts_spoken_at_once_pass_when_each_is_heard_in_time() -> None:
    prove_concurrent(
        lambda text: answer(
            first=LIMIT_SECONDS + EXTRA_WAIT_SECONDS, seconds=2.0 + len(text) / 10
        )
    )


@pytest.mark.parametrize(
    ("spoken", "failure"),
    [
        (answer(amplitude=0), "concurrent first produced silence"),
        (answer(seconds=0.2), "concurrent first produced too little audio"),
        (
            answer(first=LIMIT_SECONDS + EXTRA_WAIT_SECONDS + 0.1),
            "concurrent first first byte .* plus",
        ),
        (answer(), "identical bodies"),
    ],
    ids=["silent", "too short", "late beyond the serialized wait", "identical bodies"],
)
def test_two_texts_spoken_at_once_fail_on_a_silent_short_late_or_repeated_answer(
    spoken: tuple[bytes, float, float], failure: str
) -> None:
    with pytest.raises(SystemExit, match=failure):
        prove_concurrent(lambda _text: spoken)


def test_tasks_run_together_when_each_waits_for_the_other() -> None:
    meeting = threading.Barrier(2, timeout=30)

    def task(result: str) -> Callable[[], str]:
        def run() -> str:
            meeting.wait()
            return result

        return run

    assert figures.run_together([task("first"), task("second")]) == ["first", "second"]


def test_a_task_that_fails_ends_the_run_with_its_own_message() -> None:
    def fails() -> str:
        message = "the service refused"
        raise SystemExit(message)

    with pytest.raises(SystemExit, match="the service refused"):
        figures.run_together([fails, lambda: "fine"])


def test_every_task_completes_once_whatever_the_call_beside_it_takes() -> None:
    calls: list[str] = []

    outcome, results = figures.run_during(
        [lambda: calls.append("first") or "first", lambda: "second"],
        lambda: "heard",
    )

    assert outcome == "heard"
    assert {"first", "second"} <= set(results)
    assert calls


def test_a_task_repeats_until_the_call_beside_it_ends() -> None:
    second_call = threading.Event()
    calls: list[int] = []

    def task() -> int:
        calls.append(len(calls))
        if len(calls) >= 2:  # noqa: PLR2004 - the second call
            second_call.set()
        return len(calls)

    def during() -> str:
        assert second_call.wait(timeout=30)
        return "heard"

    outcome, results = figures.run_during([task], during)

    assert outcome == "heard"
    assert len(results) >= 2  # noqa: PLR2004 - the repeat


def test_a_loaded_hearing_model_raises_nothing_beside_speaking() -> None:
    figures.require_hearing_loaded("small", ready=True)


def test_a_hearing_model_that_did_not_load_ends_the_run_beside_speaking() -> None:
    with pytest.raises(SystemExit, match="hearing model small did not load"):
        figures.require_hearing_loaded("small", ready=False)


def test_a_task_that_fails_beside_the_call_ends_the_run() -> None:
    def fails() -> str:
        message = "the service refused"
        raise SystemExit(message)

    with pytest.raises(SystemExit, match="the service refused"):
        figures.run_during([fails], lambda: "heard")


def test_a_right_transcript_passes_and_a_wrong_one_fails() -> None:
    assert figures.require_heard("alone", REFERENCE, REFERENCE.upper()) == 0.0
    with pytest.raises(SystemExit, match="words wrong"):
        figures.require_heard("alone", REFERENCE, "ganz etwas anderes")


def test_hearing_beside_speaking_reports_both_latencies() -> None:
    record = figures.prove_hearing_beside_speaking(
        lambda: (2.5, REFERENCE),
        [answer],
        REFERENCE,
    )

    assert record["alone"] == {"seconds": 2.5, "word_error_rate": 0.0}
    assert record["beside_speaking"]["seconds"] == record["alone"]["seconds"]
    assert record["beside_speaking"]["speaks_completed"] >= 1


def test_a_wrong_transcript_beside_speaking_fails_the_proof() -> None:
    heard = iter([REFERENCE, "ganz etwas anderes"])

    with pytest.raises(SystemExit, match="beside speaking"):
        figures.prove_hearing_beside_speaking(
            lambda: (2.0, next(heard)), [answer], REFERENCE
        )


def test_a_silent_speak_beside_the_hearing_fails_the_proof() -> None:
    with pytest.raises(SystemExit, match="silence"):
        figures.prove_hearing_beside_speaking(
            lambda: (2.0, REFERENCE), [lambda: answer(amplitude=0)], REFERENCE
        )


def test_the_record_carries_the_device_and_each_voices_figures() -> None:
    record = figures.speaking_record("cpu", singles(), {"two": 1}, None)

    assert record["device"] == "cpu"
    assert set(record["voices"]) == {voice.value for voice in PIPER_VOICES}
    assert record["two_voices"] == {"two": 1}
    assert record["hearing"] is None
