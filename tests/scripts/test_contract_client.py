"""The proof scripts speak to a running service with minted tokens and nothing else."""

from __future__ import annotations

import asyncio
import http.server
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import jwt
import pytest

from shipped_catalogue import CATALOGUE, ON_DEMAND_VOICES, PIPER_VOICES, first_language
from speech_client.wire import (
    AUTHORIZATION_HEADER,
    CONTENT_TYPE_HEADER,
    WORKLOAD_TOKEN_HEADER,
    Health,
    Language,
    ResponseFormat,
    SpeechRequest,
    VoiceId,
)
from speech_service.pcm import wav_header
from speech_service.service import Runtime, RuntimeDependencies
from tests.support.fakes_support import SPEAK_SAMPLE_RATE, FakeSpeaking
from tests.support.identity_support import callers_trusting
from tests.support.script_support import script_module
from tests.support.speak_support import FailingSpeaking
from tests.support.speaking_health_support import health_offering
from tests.support.tcp_server_support import serving_over_tcp
from wav_audio import pcm_from_wav

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from proof_identity import ProofIdentity
    from proof_minting import Minting
    from speech_service.callers import SpeechCallers

ARGUMENT_ERROR_EXIT_CODE = 2
# The coverage run measures a child by the repository-relative paths of its config,
# so the child starts where the tests run.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLIENT = script_module("contract_client")
PROVE = script_module("prove_speech")


def contract_speak(base_url: str, text: str, voice: VoiceId, minting: Minting) -> bytes:
    return CLIENT.speak(base_url, text, voice, first_language(voice), minting)


HEARING = script_module("prove_hearing")
MARKS = script_module("speaking_marks")


def test_no_credential_holder_shows_a_key_or_token_when_represented(
    minting: Minting,
) -> None:
    pair = CLIENT.mint_pair(minting)

    shown = [repr(minting), repr(pair)]

    for text in shown:
        assert "-----BEGIN" not in text
        assert "eyJ" not in text


def test_both_proof_scripts_reach_a_running_speech_over_tcp(
    minting: Minting, identity: ProofIdentity
) -> None:
    async def scenario() -> tuple[object, bytes, bytes]:
        runtime = Runtime(
            {PIPER_VOICES[0]: FakeSpeaking()},
            memory_probe=lambda: 0,
            dependencies=RuntimeDependencies(catalogue=CATALOGUE),
        )
        callers = callers_trusting(identity.configuration_path, minting)
        async with serving_over_tcp(runtime, callers) as server:
            return (
                await asyncio.to_thread(PROVE.read_health, server.base_url),
                await asyncio.to_thread(
                    contract_speak, server.base_url, "Hallo", PIPER_VOICES[0], minting
                ),
                await asyncio.to_thread(
                    PROVE.speak_measured,
                    server.base_url,
                    "Hallo",
                    PIPER_VOICES[0],
                    minting,
                ),
            )

    health, contract_wav, measured = asyncio.run(scenario())

    assert isinstance(health, Health)
    assert pcm_from_wav(contract_wav)[0] == SPEAK_SAMPLE_RATE
    assert pcm_from_wav(measured[0])[0] == SPEAK_SAMPLE_RATE


def test_prove_points_the_service_at_the_repository_provider_environments(
    monkeypatch: pytest.MonkeyPatch, identity: ProofIdentity
) -> None:
    monkeypatch.delenv("SPEECH_PROVIDER_ROOT", raising=False)

    environment = PROVE.speech_environment(8080, PIPER_VOICES, identity)

    provider_root = Path(environment["SPEECH_PROVIDER_ROOT"])
    assert provider_root.is_absolute()
    assert (provider_root / "qwen" / "pyproject.toml").is_file()


@pytest.mark.parametrize(
    "offered",
    [
        frozenset(PIPER_VOICES),
        frozenset({ON_DEMAND_VOICES[0]}),
        frozenset({*PIPER_VOICES, ON_DEMAND_VOICES[0]}),
    ],
    ids=["piper", "gpu", "both"],
)
def test_prove_speaks_the_gpu_voice_the_offer_names_else_piper(
    offered: frozenset[VoiceId],
) -> None:
    on_demand = offered - set(PIPER_VOICES)

    assert PROVE.proof_voice(offered) is next(iter(on_demand), PROVE.PIPER_PROOF_VOICE)


def test_prove_keeps_a_provider_root_the_caller_set(
    monkeypatch: pytest.MonkeyPatch, identity: ProofIdentity
) -> None:
    monkeypatch.setenv("SPEECH_PROVIDER_ROOT", "/elsewhere")

    environment = PROVE.speech_environment(8080, PIPER_VOICES, identity)

    assert environment["SPEECH_PROVIDER_ROOT"] == "/elsewhere"


@pytest.mark.parametrize(
    ("exported", "started_with"),
    [(None, "false"), ("true", "true")],
    ids=["unset", "exported"],
)
def test_prove_starts_a_hearing_service_with_sounds_off_unless_exported(
    monkeypatch: pytest.MonkeyPatch,
    identity: ProofIdentity,
    exported: str | None,
    started_with: str,
) -> None:
    monkeypatch.delenv("SPEECH_SOUND_EVENTS", raising=False)
    if exported is not None:
        monkeypatch.setenv("SPEECH_SOUND_EVENTS", exported)

    environment = PROVE.speech_environment(
        8080, PIPER_VOICES, identity, hearing_model="small"
    )

    assert environment["SPEECH_SOUND_EVENTS"] == started_with


def test_prove_leaves_the_sound_switch_alone_without_a_hearing_model(
    monkeypatch: pytest.MonkeyPatch, identity: ProofIdentity
) -> None:
    monkeypatch.delenv("SPEECH_SOUND_EVENTS", raising=False)

    environment = PROVE.speech_environment(8080, PIPER_VOICES, identity)

    assert "SPEECH_SOUND_EVENTS" not in environment


def test_prove_defaults_to_the_card_proof(
    monkeypatch: pytest.MonkeyPatch, identity: ProofIdentity
) -> None:
    for name in ("SPEECH_DEVICE", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)

    environment = PROVE.speech_environment(8080, PIPER_VOICES, identity)

    assert environment["SPEECH_DEVICE"] == "cuda"
    assert "CUDA_VISIBLE_DEVICES" not in environment


def test_prove_takes_the_device_and_hides_the_card_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch, identity: ProofIdentity
) -> None:
    monkeypatch.setenv("SPEECH_DEVICE", "cpu")

    environment = PROVE.speech_environment(8080, PIPER_VOICES, identity)

    assert environment["SPEECH_DEVICE"] == "cpu"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


def test_peak_resident_memory_is_read_from_the_process_status_in_mebibytes() -> None:
    """Change: peak resident memory from the process status."""
    status = "Name:\tpython\nVmPeak:\t 9999 kB\nVmHWM:\t 2048000 kB\nVmRSS:\t 1 kB\n"

    assert PROVE.peak_rss_mib(status) == pytest.approx(2000.0)


def test_a_process_status_without_a_peak_resident_set_fails() -> None:
    with pytest.raises(SystemExit, match="no VmHWM"):
        PROVE.peak_rss_mib("Name:\tpython\nVmRSS:\t 1 kB\n")


@pytest.mark.parametrize("arguments", [["--help"], ["--measure"], ["extra"]])
def test_prove_exits_before_any_start_on_help_or_an_unknown_argument(
    arguments: list[str],
) -> None:
    """Never: --help and an unknown argument never start a proof."""
    with pytest.raises(SystemExit) as exit_info:
        PROVE.parse_arguments(arguments)

    assert (exit_info.value.code == 0) == (arguments == ["--help"])


def sse_speak(base_url: str, text: str, voice: VoiceId, minting: Minting) -> bytes:
    return MARKS.speak_sse(base_url, text, voice, ResponseFormat.WAV, minting)


def speak_against_the_service(
    speak: Callable[[str, str, str, Minting], object],
    minting: Minting,
    voice: FakeSpeaking,
    callers: SpeechCallers | None,
    failure: str,
) -> None:
    """Speak once to a TCP service; the call must end the proof with `failure`."""

    async def scenario() -> None:
        runtime = Runtime(
            {PIPER_VOICES[0]: voice},
            memory_probe=lambda: 0,
            dependencies=RuntimeDependencies(catalogue=CATALOGUE),
        )
        async with serving_over_tcp(runtime, callers) as server:
            with pytest.raises(SystemExit, match=failure):
                await asyncio.to_thread(
                    speak, server.base_url, "Hallo", PIPER_VOICES[0], minting
                )

    asyncio.run(scenario())


@pytest.mark.parametrize("chunks_before_failing", [0, 1])
@pytest.mark.parametrize(
    ("speak", "failure"),
    [
        pytest.param(contract_speak, "speak ended as Cut", id="contract client"),
        pytest.param(
            PROVE.speak_measured,
            "speak body ended before it was complete",
            id="measured speak",
        ),
        pytest.param(
            sse_speak, "sse speak body ended before its done event", id="sse speak"
        ),
    ],
)
def test_a_proof_client_rejects_a_body_the_service_aborted(
    minting: Minting,
    identity: ProofIdentity,
    speak: Callable[[str, str, str, Minting], object],
    failure: str,
    chunks_before_failing: int,
) -> None:
    speak_against_the_service(
        speak,
        minting,
        FailingSpeaking(chunks_before_failing=chunks_before_failing),
        callers_trusting(identity.configuration_path, minting),
        failure,
    )


@pytest.mark.parametrize(
    ("speak", "failure"),
    [
        pytest.param(contract_speak, "speak ended as Refused", id="contract client"),
        pytest.param(
            PROVE.speak_measured, "speak HTTP status: 401", id="measured speak"
        ),
        pytest.param(sse_speak, "sse speak HTTP status: 401", id="sse speak"),
    ],
)
def test_a_proof_client_rejects_a_service_that_does_not_trust_its_key(
    minting: Minting,
    speak: Callable[[str, str, str, Minting], object],
    failure: str,
) -> None:
    speak_against_the_service(speak, minting, FakeSpeaking(), None, failure)


SPOKEN_WAV = wav_header(SPEAK_SAMPLE_RATE) + b"\x00\x40" * 4
# The shortest head the client's own reader accepts, a format chunk of four bytes,
# which the scripts' WAV reader refuses as shorter than a header.
UNREADABLE_WAV = (
    b"RIFF\x00\x00\x00\x00WAVEfmt \x04\x00\x00\x00\x00\x00\x00\x00data\x00\x00\x00\x00"
)
SILENT_WAV = wav_header(SPEAK_SAMPLE_RATE) + b"\x00\x00" * 4


class _RecordedPost(http.server.BaseHTTPRequestHandler):
    """A request the recording server answered, with the body it carried."""

    body: bytes = b""


@contextmanager
def _recording_speech_service(
    health: Health | None = None, answer: bytes = SPOKEN_WAV
) -> Iterator[tuple[str, list[_RecordedPost]]]:
    """A local HTTP server that keeps its POSTs and serves `health`."""
    seen: list[_RecordedPost] = []

    class Handler(_RecordedPost):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            body = health.model_dump_json().encode() if health else b""
            self.send_response(200 if health else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            seen.append(self)
            self.body = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            if answer:
                self.wfile.write(b"%x\r\n%b\r\n" % (len(answer), answer))
            self.wfile.write(b"0\r\n\r\n")

        def log_message(self, *_: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    ("health", "answer", "failure"),
    [
        pytest.param(None, SPOKEN_WAV, "health ended as", id="no health"),
        pytest.param(
            health_offering(PIPER_VOICES[0], (Language.DE,), ready=False),
            SPOKEN_WAV,
            "the speaking model is not ready",
            id="speaking not ready",
        ),
        pytest.param(
            health_offering(PIPER_VOICES[0], (Language.DE,)),
            UNREADABLE_WAV,
            "not a WAV body",
            id="an answer that is no WAV",
        ),
        pytest.param(
            health_offering(PIPER_VOICES[0], (Language.DE,)),
            SILENT_WAV,
            "speak returned silence",
            id="a silent answer",
        ),
    ],
)
def test_the_contract_client_fails_on_a_service_that_is_not_ready_or_answers_badly(
    minting: Minting, health: Health | None, answer: bytes, failure: str
) -> None:
    with (
        _recording_speech_service(health, answer) as (base_url, _requests),
        pytest.raises(SystemExit, match=failure),
    ):
        CLIENT.run(base_url, PIPER_VOICES[0], minting)


def test_the_measured_speak_fails_on_an_empty_answer(minting: Minting) -> None:
    with (
        _recording_speech_service(answer=b"") as (base_url, _requests),
        pytest.raises(SystemExit, match="speak returned no bytes"),
    ):
        PROVE.speak_measured(base_url, "Hallo", PIPER_VOICES[0], minting)


def test_the_sse_speak_fails_on_an_answer_that_is_no_event_stream(
    minting: Minting,
) -> None:
    with (
        _recording_speech_service() as (base_url, _requests),
        pytest.raises(SystemExit, match="not a text/event-stream"),
    ):
        sse_speak(base_url, "Hallo", PIPER_VOICES[0], minting)


def test_the_contract_client_refuses_a_process_that_imported_the_service() -> None:
    with pytest.raises(SystemExit, match="imported speech_service"):
        CLIENT.main()


async def _contract_proof_against_a_service(
    minting: Minting, identity: ProofIdentity, *, trusting_caller: bool
) -> None:
    runtime = Runtime(
        {PIPER_VOICES[0]: FakeSpeaking()},
        memory_probe=lambda: 0,
        dependencies=RuntimeDependencies(catalogue=CATALOGUE),
    )
    callers = (
        callers_trusting(identity.configuration_path, minting)
        if trusting_caller
        else None
    )
    async with serving_over_tcp(runtime, callers) as server:
        await asyncio.to_thread(
            PROVE.prove_contract,
            server.base_url,
            dict(os.environ),
            PIPER_VOICES[0],
            identity.minting_path,
            REPOSITORY_ROOT,
        )


def test_the_contract_client_speaks_as_its_own_process_without_loading_the_service(
    minting: Minting,
    identity: ProofIdentity,
    capsys: pytest.CaptureFixture[str],
) -> None:
    asyncio.run(
        _contract_proof_against_a_service(minting, identity, trusting_caller=True)
    )

    assert "contract_speak" in capsys.readouterr().out


def test_a_contract_client_the_service_refuses_ends_the_proof_naming_its_exit_status(
    minting: Minting, identity: ProofIdentity
) -> None:
    refused_proof = _contract_proof_against_a_service(
        minting, identity, trusting_caller=False
    )

    with pytest.raises(SystemExit, match="contract client failed: 1"):
        asyncio.run(refused_proof)


def _unverified_subject(token: str) -> str:
    return jwt.decode(token, options={"verify_signature": False})["sub"]


@pytest.mark.parametrize("speak", [contract_speak, PROVE.speak_measured])
def test_each_speaking_script_sends_the_workload_token_and_the_persons_bearer(
    minting: Minting, speak: Callable[..., object]
) -> None:
    with _recording_speech_service() as (base_url, requests):
        speak(base_url, "Hallo", PIPER_VOICES[0], minting)

    (headers,) = (request.headers for request in requests)
    scheme, _, person = headers[AUTHORIZATION_HEADER].partition(" ")
    assert headers[CONTENT_TYPE_HEADER] == "application/json"
    assert scheme == "Bearer"
    assert _unverified_subject(headers[WORKLOAD_TOKEN_HEADER]) == (
        minting.workload_subject
    )
    assert _unverified_subject(person) == minting.person_subject


@pytest.mark.parametrize(
    "speak",
    [
        pytest.param(
            lambda url, voice, minting: contract_speak(url, "Hallo", voice, minting),
            id="contract client",
        ),
        pytest.param(
            lambda url, voice, minting: HEARING.speaks(url, minting, voice),
            id="hearing probe",
        ),
    ],
)
@pytest.mark.parametrize("voice", list(CATALOGUE))
def test_a_script_speaks_each_voice_in_a_language_the_service_accepts(
    minting: Minting, voice: VoiceId, speak: Callable[[str, VoiceId, Minting], object]
) -> None:
    with _recording_speech_service() as (base_url, requests):
        speak(base_url, voice, minting)

    (request,) = requests
    spoken = SpeechRequest.model_validate_json(request.body)
    assert spoken.spoken_language == first_language(voice)


@pytest.mark.parametrize(
    ("voice", "language"),
    [
        (voice, language)
        for voice in CATALOGUE
        for language in CATALOGUE[voice].languages
    ],
)
def test_the_measured_speak_names_the_language_it_is_given_else_the_voices_first(
    minting: Minting, voice: VoiceId, language: Language
) -> None:
    with _recording_speech_service() as (base_url, requests):
        PROVE.speak_measured(base_url, "Hallo", voice, minting, language=language)
        PROVE.speak_measured(base_url, "Hallo", voice, minting)

    named, default = (
        SpeechRequest.model_validate_json(request.body).spoken_language
        for request in requests
    )
    assert named == language
    assert default == first_language(voice)


def test_the_contract_client_speaks_the_language_health_names_first_for_the_voice(
    minting: Minting,
) -> None:
    voice = next(voice for voice in CATALOGUE if len(CATALOGUE[voice].languages) > 1)
    offered = tuple(reversed(CATALOGUE[voice].languages))

    with _recording_speech_service(health_offering(voice, offered)) as (
        base_url,
        requests,
    ):
        CLIENT.run(base_url, voice, minting)

    (request,) = requests
    assert SpeechRequest.model_validate_json(request.body).language == offered[0]


@pytest.mark.parametrize("voice", list(CATALOGUE))
def test_the_contract_client_speaks_the_voice_named_on_its_command_line(
    minting: Minting, identity: ProofIdentity, voice: VoiceId
) -> None:
    health = health_offering(voice, CATALOGUE[voice].languages)

    with _recording_speech_service(health) as (base_url, requests):
        options = CLIENT.parse_arguments(
            ["--minting-file", str(identity.minting_path), base_url, str(voice)]
        )
        CLIENT.run(base_url, options.voice, minting)

    (request,) = requests
    assert SpeechRequest.model_validate_json(request.body).voice == voice


def test_the_contract_client_refuses_a_voice_the_catalogue_does_not_name(
    identity: ProofIdentity, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        CLIENT.parse_arguments(
            ["--minting-file", str(identity.minting_path), "http://unused", "nobody"]
        )

    assert exit_info.value.code == ARGUMENT_ERROR_EXIT_CODE
    assert "invalid choice" in capsys.readouterr().err


def test_the_contract_client_command_passes_the_minting_file_path_and_no_secret(
    minting: Minting, identity: ProofIdentity
) -> None:
    command = PROVE.contract_client_command(
        "http://127.0.0.1:1", PIPER_VOICES[0], identity.minting_path
    )

    assert str(identity.minting_path) in command
    assert all(
        minting.private_key_pem not in part and "eyJ" not in part for part in command
    )
    options = CLIENT.parse_arguments(command[2:])
    assert options.minting_file == identity.minting_path
    assert options.voice == PIPER_VOICES[0]


def test_prove_hands_the_service_its_configuration_and_the_proofs_ca(
    identity: ProofIdentity,
) -> None:
    environment = PROVE.speech_environment(8080, PIPER_VOICES, identity)

    assert environment["SPEECH_CONFIG_FILE"] == str(identity.configuration_path)
    assert environment["SSL_CERT_FILE"] == str(identity.ca_path)


def test_prove_binds_the_service_to_the_loopback_port_it_picked(
    identity: ProofIdentity,
) -> None:
    port = PROVE.free_local_port()

    environment = PROVE.speech_environment(port, PIPER_VOICES, identity)

    assert environment["SPEECH_HOST"] == "127.0.0.1"
    assert environment["SPEECH_PORT"] == str(port)
