"""The design script renders candidate voices from a description through a server."""

from __future__ import annotations

import io
import subprocess
import sys
import threading
import wave
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from speech_provider_contract import QWEN_ARTIFACTS

from speech_qwen import design_voice
from speech_qwen.server import WATCH_THREAD_NAME, Server

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from support.scripted_server import ScriptedSpeechServer

_FAKE_SERVER = Path(__file__).parent / "support" / "fake_vllm_omni.py"
DESCRIPTION = "A warm, calm female voice with a slight smile"


def _wav(frames: int = 240) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24_000)
        writer.writeframes(bytes(2 * frames))
    return buffer.getvalue()


class _Snapshots:
    """The Hub library's download, replaced by one that records its request."""

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self.requests: list[dict[str, object]] = []

    def __call__(self, **request: object) -> str:
        self.requests.append(request)
        return str(self.checkpoint)


class _Started:
    """A stand-in server starter that hands back the scripted server."""

    def __init__(self, server: Server, snapshots: _Snapshots) -> None:
        self.server = server
        self.snapshots = snapshots
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
    snapshots = _Snapshots(tmp_path / "checkpoint")
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshots)
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    server = Server(process, speech_server.server.socket_path)
    yield _Started(server, snapshots)
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
    assert command[1:3] == ["serve", str(started.snapshots.checkpoint)]
    assert started.snapshots.requests == [
        {
            "repo_id": design_voice.DESIGN_MODEL_ID,
            "repo_type": "model",
            "revision": design_voice.DESIGN_REVISION,
            "allow_patterns": QWEN_ARTIFACTS,
            "cache_dir": tmp_path / "cache",
        }
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

    assert started.snapshots.requests == []


@pytest.mark.parametrize("count", [0, -1])
def test_a_count_below_one_fails_before_any_download(
    started: _Started, tmp_path: Path, count: int
) -> None:
    with pytest.raises(ValueError, match="count"):
        _design(started, tmp_path / "out", count=count)

    assert started.snapshots.requests == []


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


def _run_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *options: str
) -> list[tuple[str, str, int, Path, Path]]:
    """Run the console script on `options`; the calls its `design_voices` got."""
    calls: list[tuple[str, str, int, Path, Path]] = []

    def designed(
        description: str, language: str, count: int, destination: Path, cache: Path
    ) -> list[Path]:
        calls.append((description, language, count, destination, cache))
        return [destination / "candidate-01.wav", destination / "candidate-02.wav"]

    monkeypatch.setattr(design_voice, "design_voices", designed)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "speech-qwen-design-voice",
            "--description",
            DESCRIPTION,
            "--cache",
            "hub",
            *options,
        ],
    )
    design_voice.main()
    return calls


def test_the_console_script_prints_each_designed_path_on_its_own_line(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    [call] = _run_main(
        monkeypatch,
        tmp_path,
        "--language",
        "de",
        "--count",
        "2",
        "--destination",
        "out",
    )

    destination = tmp_path.resolve() / "out"
    assert call[1:4] == ("de", 2, destination)
    assert capsys.readouterr().out.splitlines() == [
        str(destination / "candidate-01.wav"),
        str(destination / "candidate-02.wav"),
    ]


@pytest.mark.parametrize(
    ("options", "refusal"),
    [
        (["--language", "en", "--destination", "../escape"], ValueError),
        (["--language", "fr", "--destination", "out"], SystemExit),
    ],
    ids=["destination outside the working directory", "language outside de and en"],
)
def test_the_console_script_refuses_what_the_design_refuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    options: list[str],
    refusal: type[BaseException],
) -> None:
    with pytest.raises(refusal):
        _run_main(monkeypatch, tmp_path, *options)


def test_the_designs_stop_of_its_own_server_ends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ends_seen: list[object] = []
    real_start_server = design_voice.start_server

    def recording_start_server(
        command: Sequence[str],
        socket_path: Path,
        *,
        when_server_ends: Callable[[], None],
    ) -> Server:
        return real_start_server(
            command,
            socket_path,
            when_server_ends=lambda: (when_server_ends(), ends_seen.append(None))[1],
        )

    monkeypatch.setattr(design_voice, "start_server", recording_start_server)
    socket_path = tmp_path / "server.sock"
    command = [sys.executable, str(_FAKE_SERVER), "healthy", "--uds", str(socket_path)]

    server = design_voice.start_until_stopped(command, socket_path)
    connection = server.connect(5)
    connection.request("GET", "/quit")
    connection.getresponse().read()
    connection.close()
    server.process.wait()
    for thread in threading.enumerate():
        if thread.name == WATCH_THREAD_NAME:
            thread.join()

    assert ends_seen == [None]
