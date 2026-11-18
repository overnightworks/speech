"""The design script renders candidate voices from a description through a server."""

from __future__ import annotations

import io
import subprocess
import sys
import wave
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from speech_qwen import design_voice
from speech_qwen.server import Server

if TYPE_CHECKING:
    from collections.abc import Iterator

    from support.scripted_server import ScriptedSpeechServer

DESCRIPTION = "A warm, calm female voice with a slight smile"


def _wav(frames: int = 240) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24_000)
        writer.writeframes(bytes(2 * frames))
    return buffer.getvalue()


class _Downloaded:
    """A stand-in download that records what it was asked for."""

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self.calls: list[tuple[str, str]] = []

    def __call__(self, repo_id: str, revision: str, _cache: Path) -> str:
        self.calls.append((repo_id, revision))
        return str(self.checkpoint)


class _Started:
    """A stand-in server starter that hands back the scripted server."""

    def __init__(self, server: Server, download: _Downloaded) -> None:
        self.server = server
        self.download = download
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], _socket_path: Path) -> Server:
        self.commands.append(command)
        return self.server


@pytest.fixture
def started(
    speech_server: ScriptedSpeechServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_Started]:
    deploy_config = tmp_path / "package" / "vllm_omni" / "deploy" / "qwen3_tts.yaml"
    deploy_config.parent.mkdir(parents=True)
    deploy_config.write_text("stages: []\n", encoding="utf-8")
    deploy_config.parents[1].joinpath("__init__.py").write_text("raise SystemExit\n")
    monkeypatch.syspath_prepend(str(tmp_path / "package"))
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    server = Server(process, speech_server.server.socket_path)
    yield _Started(server, _Downloaded(tmp_path / "checkpoint"))
    process.kill()
    process.wait()


def _design(
    started: _Started, destination: Path, *, language: str = "en", count: int = 3
) -> list[Path]:
    return design_voice.design_voices(
        DESCRIPTION,
        language,
        count,
        destination,
        destination.parent / "cache",
        start=started,
        download=started.download,
    )


def test_each_candidate_is_one_non_streaming_voice_design_request(
    speech_server: ScriptedSpeechServer, started: _Started, tmp_path: Path
) -> None:
    speech_server.parts = [_wav()]

    candidates = 2
    _design(started, tmp_path / "out", language="de", count=candidates)

    assert len(speech_server.requests) == candidates
    request = speech_server.requests[0]
    assert request["task_type"] == "VoiceDesign"
    assert request["instructions"] == DESCRIPTION
    assert request["language"] == "German"
    assert request["response_format"] == "wav"
    assert not request.get("stream")
    assert isinstance(request["input"], str)
    assert "voice" not in request


def test_the_destination_holds_one_valid_wav_and_its_exact_transcript_per_candidate(
    speech_server: ScriptedSpeechServer, started: _Started, tmp_path: Path
) -> None:
    speech_server.parts = [_wav()]
    destination = tmp_path / "out"

    candidates = 3
    audio_files = _design(started, destination, count=candidates)

    assert len(audio_files) == candidates
    assert {path.parent for path in audio_files} == {destination}
    for audio in audio_files:
        with wave.open(str(audio)) as reader:
            assert reader.getnframes() > 0
        transcript = audio.with_suffix(".txt").read_text(encoding="utf-8")
        assert transcript == speech_server.requests[0]["input"]


def test_the_server_is_stopped_once_the_candidates_are_written(
    speech_server: ScriptedSpeechServer, started: _Started, tmp_path: Path
) -> None:
    speech_server.parts = [_wav()]

    _design(started, tmp_path / "out", count=1)

    assert started.server.process is not None
    assert started.server.process.poll() is not None


def test_the_server_runs_the_downloaded_pinned_voice_design_snapshot(
    speech_server: ScriptedSpeechServer, started: _Started, tmp_path: Path
) -> None:
    speech_server.parts = [_wav()]

    _design(started, tmp_path / "out", count=1)

    ((command,),) = [started.commands]
    assert command[1:3] == ["serve", str(started.download.checkpoint)]
    assert started.download.calls == [
        (design_voice.DESIGN_MODEL_ID, design_voice.DESIGN_REVISION)
    ]


def test_a_refusal_fails_loud_and_leaves_no_candidate(
    speech_server: ScriptedSpeechServer, started: _Started, tmp_path: Path
) -> None:
    speech_server.status = HTTPStatus.BAD_REQUEST
    destination = tmp_path / "out"

    with pytest.raises(design_voice.DesignVoiceError):
        _design(started, destination)

    assert list(destination.glob("candidate-*")) == []
    assert started.server.process.poll() is not None


@pytest.mark.parametrize("body", [b"not audio", _wav(frames=0)])
def test_a_body_that_is_no_audio_fails_loud(
    speech_server: ScriptedSpeechServer,
    started: _Started,
    tmp_path: Path,
    body: bytes,
) -> None:
    speech_server.parts = [body]

    with pytest.raises(design_voice.DesignVoiceError):
        _design(started, tmp_path / "out")


def test_an_unsupported_language_fails_before_any_download(
    started: _Started, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="language"):
        _design(started, tmp_path / "out", language="fr")

    assert started.download.calls == []


@pytest.mark.parametrize("count", [0, -1])
def test_a_count_below_one_fails_before_any_download(
    started: _Started, tmp_path: Path, count: int
) -> None:
    with pytest.raises(ValueError, match="count"):
        _design(started, tmp_path / "out", count=count)

    assert started.download.calls == []


def test_a_candidate_already_in_the_destination_is_never_overwritten(
    speech_server: ScriptedSpeechServer, started: _Started, tmp_path: Path
) -> None:
    speech_server.parts = [_wav()]
    destination = tmp_path / "out"
    _design(started, destination, count=1)

    with pytest.raises(FileExistsError):
        _design(started, destination, count=1)


@pytest.mark.parametrize("requested", ["../escape", "/etc/elsewhere", "a/../../b"])
def test_a_destination_outside_the_working_directory_is_refused(
    tmp_path: Path, requested: str
) -> None:
    with pytest.raises(ValueError, match="outside the working directory"):
        design_voice.confined_destination(Path(requested), tmp_path)


def test_a_destination_inside_the_working_directory_resolves_under_it(
    tmp_path: Path,
) -> None:
    assert design_voice.confined_destination(Path("out/sub"), tmp_path) == (
        tmp_path.resolve() / "out" / "sub"
    )
