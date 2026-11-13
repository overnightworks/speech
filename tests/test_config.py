"""Configuration comes from the environment."""

import io
import ipaddress
import wave
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from speech_client.wire import VoiceId
from speech_service.config import Settings
from speech_service.hearing_models import HearingModel
from speech_service.pcm import wav_header
from speech_service.voices import PIPER_VOICES
from tests.support.fakes_support import SPEAK_SAMPLE_RATE

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SPEECH_DEBUG", "true"),
        ("SPEECH_STATE_DIRECTORY", "relative/state"),
    ],
)
def test_a_stale_setting_in_the_environment_is_ignored(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    Settings()


def test_settings_read_speech_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """WIRE-107: settings read speech environment."""
    monkeypatch.setenv("SPEECH_DEVICE", "cpu")
    monkeypatch.setenv("SPEECH_VOICES", VoiceId.RYAN.value)

    settings = Settings()

    assert settings.device == "cpu"
    assert settings.voices == {VoiceId.RYAN}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SPEECH_BUSY_WAIT_SECONDS", "-1"),
        ("SPEECH_BUSY_WAIT_SECONDS", "nan"),
        ("SPEECH_BUSY_WAIT_SECONDS", "inf"),
        ("SPEECH_MAX_INPUT_CHARACTERS", "0"),
        ("SPEECH_MAX_INPUT_CHARACTERS", "-1"),
        ("SPEECH_MAX_WAITING_REQUESTS", "-1"),
        ("SPEECH_MAX_WAITING_REQUESTS", "1.5"),
        ("SPEECH_MAX_BODY_BYTES", "0"),
        ("SPEECH_MAX_BODY_BYTES", "many"),
        ("SPEECH_HEARING_MODEL", "gigantic"),
        ("SPEECH_HEARING_BUSY_WAIT_SECONDS", "-1"),
        ("SPEECH_HEARING_MAX_WAITING_REQUESTS", "-1"),
        ("SPEECH_HEARING_FINAL_WAIT_SECONDS", "0"),
        ("SPEECH_HEARING_FINAL_WAIT_SECONDS", "-1"),
        ("SPEECH_MAX_RECORDING_BYTES", "0"),
        ("SPEECH_MAX_RECORDING_SECONDS", "0"),
        ("SPEECH_MAX_RECORDING_SECONDS", "inf"),
    ],
)
def test_the_wire_settings_refuse_values_that_cannot_hold(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    """WIRE-108, WIRE-109, WIRE-110, WIRE-111, WIRE-120, WIRE-121: values are refused.

    A value that cannot hold is refused, naming the setting.
    """
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match=name.removeprefix("SPEECH_").lower()):
        Settings()


def test_the_wire_settings_accept_the_smallest_whole_limit_and_no_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WIRE-108, WIRE-109, WIRE-110, WIRE-111: the smallest whole limit is accepted."""
    monkeypatch.setenv("SPEECH_BUSY_WAIT_SECONDS", "0")
    monkeypatch.setenv("SPEECH_MAX_INPUT_CHARACTERS", "1")
    monkeypatch.setenv("SPEECH_MAX_WAITING_REQUESTS", "0")
    monkeypatch.setenv("SPEECH_MAX_BODY_BYTES", "1")

    settings = Settings()

    assert settings.busy_wait_seconds == 0
    assert settings.max_input_characters == 1
    assert settings.max_waiting_requests == 0
    assert settings.max_body_bytes == 1


def test_the_hearing_settings_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WIRE-117, WIRE-118, WIRE-120, WIRE-121: hearing settings are read."""
    monkeypatch.setenv("SPEECH_HEARING_MODEL", HearingModel.LARGE_V3.value)
    monkeypatch.setenv("SPEECH_HEARING_BUSY_WAIT_SECONDS", "0")
    monkeypatch.setenv("SPEECH_HEARING_MAX_WAITING_REQUESTS", "0")
    monkeypatch.setenv("SPEECH_HEARING_FINAL_WAIT_SECONDS", "0.5")
    monkeypatch.setenv("SPEECH_MAX_RECORDING_BYTES", "1")
    monkeypatch.setenv("SPEECH_MAX_RECORDING_SECONDS", "0.5")

    settings = Settings()

    assert settings.hearing_model is HearingModel.LARGE_V3
    assert settings.hearing_busy_wait_seconds == 0
    assert settings.hearing_max_waiting_requests == 0
    assert settings.hearing_final_wait_seconds == float("0.5")
    assert settings.max_recording_bytes == 1
    assert settings.max_recording_seconds == float("0.5")


def test_settings_default_uses_the_provider_hub_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider_cache = tmp_path / "provider-hub-cache"
    monkeypatch.delenv("SPEECH_HUGGINGFACE_CACHE", raising=False)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", provider_cache)

    assert Settings().huggingface_cache == provider_cache


def test_settings_preserves_an_explicit_huggingface_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit_cache = tmp_path / "operator-hub-cache"
    monkeypatch.setenv("SPEECH_HUGGINGFACE_CACHE", str(explicit_cache))

    assert Settings().huggingface_cache == explicit_cache


def test_settings_bind_to_a_loopback_tcp_port_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WIRE-01: settings bind to a loopback tcp port by default."""
    monkeypatch.delenv("SPEECH_HOST", raising=False)
    monkeypatch.delenv("SPEECH_PORT", raising=False)

    settings = Settings()

    assert ipaddress.ip_address(settings.host).is_loopback
    assert settings.port in range(1, 2**16)


def test_settings_read_the_bind_address_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WIRE-01: settings read the bind address from the environment."""
    all_interfaces = "0.0.0.0"  # noqa: S104 - the container form of the bind address
    monkeypatch.setenv("SPEECH_HOST", all_interfaces)
    monkeypatch.setenv("SPEECH_PORT", "65535")

    settings = Settings()

    assert (settings.host, settings.port) == (all_interfaces, 65535)


@pytest.mark.parametrize("port", ["0", "65536", "-1", "http", ""])
def test_settings_refuse_a_port_that_is_not_a_tcp_port(
    monkeypatch: pytest.MonkeyPatch, port: str
) -> None:
    """WIRE-01: settings refuse a port that is not a tcp port."""
    monkeypatch.setenv("SPEECH_PORT", port)

    with pytest.raises(ValidationError, match="port"):
        Settings()


def test_every_piper_voice_is_offered_by_default() -> None:
    """WIRE-107: every piper voice is offered by default."""
    assert Settings().voices == set(PIPER_VOICES)


@pytest.mark.parametrize("value", ["", " "])
def test_an_empty_voice_setting_offers_none(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """WIRE-107: an empty voice setting offers none."""
    monkeypatch.setenv("SPEECH_VOICES", value)

    assert Settings().voices == frozenset()


@pytest.mark.parametrize("voice", list(VoiceId))
def test_every_voice_of_the_catalogue_can_be_offered(
    monkeypatch: pytest.MonkeyPatch, voice: VoiceId
) -> None:
    """WIRE-107: every voice of the catalogue can be offered."""
    monkeypatch.setenv("SPEECH_VOICES", f" {voice.value} ")

    assert Settings().voices == {voice}


@pytest.mark.parametrize(
    "value",
    [
        "alloy",
        "piper",
        "qwen3-tts-1.7b",
        f"{VoiceId.RYAN.value},",
        f",{VoiceId.RYAN.value}",
        f"{VoiceId.RYAN.value},{VoiceId.RYAN.value}",
    ],
)
def test_an_unknown_empty_or_repeated_voice_id_fails_with_the_fixed_message(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """WIRE-107: an unknown, empty or repeated voice id fails with the fixed message."""
    monkeypatch.setenv("SPEECH_VOICES", value)

    with pytest.raises(ValueError, match="SPEECH_VOICES") as refused:
        Settings()

    assert value not in str(refused.value)


def test_a_relative_provider_root_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEECH_PROVIDER_ROOT", "relative/providers")

    with pytest.raises(ValueError, match="provider root"):
        Settings()


def test_provider_root_is_optional_for_the_default_piper_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPEECH_PROVIDER_ROOT", raising=False)
    assert Settings().provider_root is None


def test_streamed_wav_header_is_readable_by_the_wave_module() -> None:
    frames = b"\x01\x00" * 8
    with wave.open(io.BytesIO(wav_header(SPEAK_SAMPLE_RATE) + frames)) as reader:
        assert reader.getframerate() == SPEAK_SAMPLE_RATE
        assert reader.readframes(8) == frames


def test_sized_wav_header_keeps_its_exact_riff_size() -> None:
    header = wav_header(SPEAK_SAMPLE_RATE, 100)
    assert int.from_bytes(header[4:8], "little") == len(header) - 8 + 100


def test_wav_header_is_sixteen_bit_mono() -> None:
    header = wav_header(SPEAK_SAMPLE_RATE, 100)
    assert header[20:22] == (1).to_bytes(2, "little")
    assert header[22:24] == (1).to_bytes(2, "little")
    assert int.from_bytes(header[24:28], "little") == SPEAK_SAMPLE_RATE
    assert header[34:36] == (16).to_bytes(2, "little")
