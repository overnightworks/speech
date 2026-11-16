"""The speaking engines the runtime builds: the local Piper voice and the providers."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from speech_provider_contract import QWEN_MODEL_ID, QWEN_SAMPLE_RATE

from speech_client.wire import VOICES, SentenceMarks, SpeakingModel, VoiceId, Word
from speech_service import speaking as speaking_module
from speech_service.config import Settings
from speech_service.espeak_marks import EspeakEvent, EventKind, SentencePass
from speech_service.speaking import PiperSpeaking, gives_marks, speaking_for_voice
from speech_service.voices import PIPER_VOICES


@pytest.mark.parametrize("voice", PIPER_VOICES)
def test_every_piper_voice_constructs_its_own_local_engine(voice: VoiceId) -> None:
    engine = speaking_for_voice(Settings(), voice)

    assert isinstance(engine, PiperSpeaking)
    assert engine.model_name == voice.value


@pytest.mark.parametrize(
    "voice", [v for v in VoiceId if VOICES[v].model is SpeakingModel.QWEN]
)
def test_every_qwen_voice_constructs_the_complete_waveform_voice(
    voice: VoiceId,
) -> None:
    engine = speaking_for_voice(Settings(device="cuda"), voice)

    assert engine.model_name == QWEN_MODEL_ID
    assert engine.sample_rate == QWEN_SAMPLE_RATE


def test_piper_load_requires_local_onnx_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[object] = []

    class LocalVoice:
        config = SimpleNamespace(sample_rate=22_050)

    def load(path: object, *, include_alignments: bool = False) -> LocalVoice:
        loaded.append((path, include_alignments))
        return LocalVoice()

    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=load)),
    )
    engine = PiperSpeaking("local-voice", tmp_path)

    with pytest.raises(FileNotFoundError):
        engine.load()
    (tmp_path / "local-voice.onnx").touch()
    (tmp_path / "local-voice.onnx.json").touch()

    engine.load()

    assert loaded == [(tmp_path / "local-voice.onnx", True)]
    assert engine.ready is True


SENTENCE_SAMPLES = 400
SAMPLE_RATE = 1_000


def a_chunk(phonemes: list[str]) -> SimpleNamespace:
    """What Piper yields for one sentence of `phonemes`: 100 samples per token."""
    audio = np.zeros(SENTENCE_SAMPLES, dtype=np.float32)
    alignments = [SimpleNamespace(num_samples=100) for _ in range(len(phonemes) + 2)]
    return SimpleNamespace(
        audio_int16_bytes=bytes(2 * SENTENCE_SAMPLES),
        audio_float_array=audio,
        phonemes=phonemes,
        phoneme_alignments=alignments,
    )


class StubVoice:
    """A Piper voice that speaks two sentences and records how it was asked."""

    config = SimpleNamespace(sample_rate=SAMPLE_RATE, espeak_voice="de")

    def __init__(self) -> None:
        """Nothing was asked yet."""
        self.asked_for_alignments: list[bool] = []

    def synthesize(
        self, text: str, *, include_alignments: bool = False
    ) -> list[SimpleNamespace]:
        """Two sentences, `Ja` each."""
        del text
        self.asked_for_alignments.append(include_alignments)
        return [a_chunk(["j", "a", "."]), a_chunk(["j", "a", "."])]


class StubEvents:
    """The passes espeak would give for `Ja. Ja.`."""

    def read(self, text: str, espeak_voice: str, start: int) -> SentencePass:
        """The pass of the sentence at `start`."""
        assert (text, espeak_voice) == ("Ja. Ja.", "de")
        events = (
            EspeakEvent(EventKind.WORD, start),
            EspeakEvent(EventKind.PHONEME, start, "ja"),
        )
        return SentencePass(events, 5 if start == 1 else None)


def a_loaded_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    events: StubEvents | None,
) -> tuple[PiperSpeaking, StubVoice]:
    voice = StubVoice()
    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=lambda *_a, **_k: voice)),
    )
    monkeypatch.setattr(speaking_module, "load_espeak_events", lambda _voice: events)
    (tmp_path / "local-voice.onnx").touch()
    (tmp_path / "local-voice.onnx.json").touch()
    engine = PiperSpeaking("local-voice", tmp_path)
    engine.load()
    return engine, voice


@pytest.mark.parametrize("passes", [True, False])
def test_a_piper_voice_gives_word_marks_exactly_when_its_load_check_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, passes: bool
) -> None:
    """WIRE-122, WIRE-156: `gives_marks.words` is true only after the load check."""
    engine, _voice = a_loaded_engine(
        tmp_path, monkeypatch, StubEvents() if passes else None
    )

    assert gives_marks(engine).words is passes
    assert gives_marks(engine).sounds is False


def test_a_voice_that_is_not_loaded_gives_no_marks() -> None:
    engine = PiperSpeaking("local-voice", Path())

    assert gives_marks(engine).words is False


def test_a_piper_sentence_is_its_audio_then_its_words_on_one_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WIRE-153: each sentence's marks follow its audio; times continue."""
    engine, _voice = a_loaded_engine(tmp_path, monkeypatch, StubEvents())

    items = list(engine.marked_chunks("Ja. Ja.", "de"))

    audio_first, marks_first, audio_second, marks_second = items
    assert isinstance(audio_first, bytes)
    assert isinstance(audio_second, bytes)
    assert [mark.words for mark in (marks_first, marks_second)] == [
        (Word(text="Ja", start_char=0, end_char=2, start=0.1, end=0.3),),
        (Word(text="Ja", start_char=4, end_char=6, start=0.5, end=0.7),),
    ]


def test_a_piper_voice_whose_words_are_off_still_marks_each_sentence_as_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WIRE-152: one marks event per sentence; a kind not given is `null`."""
    engine, _voice = a_loaded_engine(tmp_path, monkeypatch, None)

    items = list(engine.marked_chunks("Ja. Ja.", "de"))

    assert [type(item) for item in items] == [bytes, SentenceMarks] * 2
    assert items[1] == items[3] == SentenceMarks()


def test_the_plain_audio_asks_piper_for_no_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audio path pays nothing for marks: no alignment, no espeak pass."""
    engine, voice = a_loaded_engine(tmp_path, monkeypatch, StubEvents())

    list(engine.pcm_chunks("Ja. Ja.", "de"))
    list(engine.marked_chunks("Ja. Ja.", "de"))

    assert voice.asked_for_alignments == [False, True]


def test_a_voice_that_is_not_loaded_cannot_speak_with_marks() -> None:
    engine = PiperSpeaking("local-voice", Path())

    with pytest.raises(RuntimeError):
        next(engine.marked_chunks("Ja.", "de"))
