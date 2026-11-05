"""A Piper voice is fetched at its pin, checked against its hashes, and placed whole."""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from speech_runtime_service import fetch_voice
from speech_runtime_service.config import Settings
from speech_runtime_service.piper_voices import (
    PINS,
    PinnedFile,
    PinnedVoice,
    VoiceHashMismatchError,
    VoicePin,
    fetch_voice_files,
)
from speech_runtime_service.voices import (
    DEFAULT_SPEAKING_MODEL,
    VoiceId,
    missing_artifacts,
    piper_voice_paths,
)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass
class Scenario:
    """A pinned voice of scenario bytes, and a downloader that serves them."""

    voice: PinnedVoice
    revision: str
    model_bytes: bytes
    metadata_bytes: bytes
    calls: list[dict[str, object]] = field(default_factory=list)
    served: dict[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Serve both files of the voice."""
        self.served = {
            f"a/b/{self.voice}.onnx": self.model_bytes,
            f"a/b/{self.voice}.onnx.json": self.metadata_bytes,
        }

    def tamper(self, filename: str, content: bytes) -> None:
        """Serve `content` for the voice file `filename` instead."""
        self.served[f"a/b/{filename}"] = content

    def __call__(self, repo_id: str, filename: str, **options: object) -> str:
        """Write the scenario bytes of `filename` under `local_dir`, as the hub does."""
        self.calls.append({"repo_id": repo_id, "filename": filename, **options})
        target = Path(str(options["local_dir"])) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.served[filename])
        return str(target)


@pytest.fixture
def scenario(monkeypatch: pytest.MonkeyPatch) -> Scenario:
    voice = PinnedVoice.THORSTEN
    made = Scenario(voice, "1" * 40, b"scenario model", b'{"scenario": "metadata"}')
    monkeypatch.setitem(
        PINS,
        voice,
        VoicePin(
            "scenario/voices",
            made.revision,
            PinnedFile(f"a/b/{voice}.onnx", sha256(made.model_bytes)),
            PinnedFile(f"a/b/{voice}.onnx.json", sha256(made.metadata_bytes)),
        ),
    )
    return made


def test_a_voice_is_fetched_at_its_revision_without_token_or_user_cache(
    scenario: Scenario, tmp_path: Path
) -> None:
    onnx = fetch_voice_files(scenario.voice, tmp_path, scenario)

    assert onnx.read_bytes() == scenario.model_bytes
    assert {call["repo_id"] for call in scenario.calls} == {"scenario/voices"}
    assert {call["revision"] for call in scenario.calls} == {scenario.revision}
    assert {call["token"] for call in scenario.calls} == {False}
    assert {call["force_download"] for call in scenario.calls} == {True}


def test_a_fetched_voice_is_found_by_the_service_under_the_names_it_reads(
    scenario: Scenario, tmp_path: Path
) -> None:
    fetch_voice_files(scenario.voice, tmp_path, scenario)

    onnx, metadata = piper_voice_paths(tmp_path, scenario.voice.value)
    assert onnx.read_bytes() == scenario.model_bytes
    assert metadata.read_bytes() == scenario.metadata_bytes
    assert missing_artifacts(Settings(voice_cache=tmp_path), VoiceId.PIPER) == ()
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [onnx.name, metadata.name]
    )


@pytest.mark.parametrize("filename", ["onnx", "onnx.json"])
def test_a_wrong_hash_fails_naming_the_file_and_leaves_nothing_at_the_final_path(
    scenario: Scenario, tmp_path: Path, filename: str
) -> None:
    scenario.tamper(f"{scenario.voice}.{filename}", b"tampered")

    with pytest.raises(VoiceHashMismatchError, match=re.escape(f".{filename} is")):
        fetch_voice_files(scenario.voice, tmp_path, scenario)

    assert list(tmp_path.iterdir()) == []


def test_a_present_matching_voice_is_not_fetched_again(
    scenario: Scenario, tmp_path: Path
) -> None:
    fetch_voice_files(scenario.voice, tmp_path, scenario)
    scenario.calls.clear()

    fetch_voice_files(scenario.voice, tmp_path, scenario)

    assert scenario.calls == []


def test_a_damaged_voice_is_fetched_again(scenario: Scenario, tmp_path: Path) -> None:
    onnx = fetch_voice_files(scenario.voice, tmp_path, scenario)
    onnx.write_bytes(b"damaged")

    fetch_voice_files(scenario.voice, tmp_path, scenario)

    assert onnx.read_bytes() == scenario.model_bytes


def test_a_damaged_file_whose_refetch_fails_is_gone_from_the_final_path(
    scenario: Scenario, tmp_path: Path
) -> None:
    onnx = fetch_voice_files(scenario.voice, tmp_path, scenario)
    onnx.write_bytes(b"damaged")
    scenario.tamper(f"{scenario.voice}.onnx.json", b"tampered")

    with pytest.raises(VoiceHashMismatchError):
        fetch_voice_files(scenario.voice, tmp_path, scenario)

    assert not onnx.exists()


def test_the_command_prints_where_the_voice_went_in_the_configured_cache(
    scenario: Scenario,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SPEECH_VOICE_CACHE", str(tmp_path))

    fetch_voice.main([scenario.voice.value], scenario)

    assert capsys.readouterr().out.strip() == str(
        piper_voice_paths(tmp_path, scenario.voice.value)[0]
    )


def test_the_command_refuses_a_voice_it_does_not_know() -> None:
    with pytest.raises(SystemExit) as stopped:
        fetch_voice.main(["nobody-medium"])

    assert stopped.value.code != 0


def test_the_services_default_voice_is_pinned() -> None:
    assert DEFAULT_SPEAKING_MODEL in {voice.value for voice in PINS}


def test_every_voice_is_pinned_in_full_under_the_names_the_service_reads() -> None:
    assert set(PINS) == set(PinnedVoice)
    for voice, pin in PINS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", pin.revision)
        assert re.fullmatch(r"[0-9a-f]{64}", pin.model.sha256)
        assert re.fullmatch(r"[0-9a-f]{64}", pin.metadata.sha256)
        onnx, metadata = piper_voice_paths(Path("voices"), voice.value)
        assert pin.model.path.rsplit("/", 1)[1] == onnx.name
        assert pin.metadata.path.rsplit("/", 1)[1] == metadata.name
