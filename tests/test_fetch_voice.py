"""A voice is fetched at its pin: Piper checked and whole, a GPU voice into the Hub."""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from speech_provider_contract import (
    QWEN_ARTIFACTS,
    QWEN_MODEL_ID,
    QWEN_REVISION,
)

from speech_client.wire import VoiceId
from speech_service import fetch_voice
from speech_service.config import Settings
from speech_service.piper_voices import (
    PINS,
    PinnedFile,
    VoiceHashMismatchError,
    VoicePin,
    fetch_voice_files,
)
from speech_service.voices import (
    PIPER_VOICES,
    missing_artifacts,
    piper_voice_paths,
)
from tests.support.hear_support import write_snapshot


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass
class Scenario:
    """The pinned Piper voices of scenario bytes, and a downloader that serves them."""

    voices: tuple[VoiceId, ...]
    revision: str
    model_bytes: bytes
    metadata_bytes: bytes
    calls: list[dict[str, object]] = field(default_factory=list)
    served: dict[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Serve both files of every voice."""
        for voice in self.voices:
            self.served[f"a/b/{voice}.onnx"] = self.model_bytes
            self.served[f"a/b/{voice}.onnx.json"] = self.metadata_bytes

    @property
    def voice(self) -> VoiceId:
        """The first of the voices."""
        return self.voices[0]

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
    made = Scenario(
        PIPER_VOICES, "1" * 40, b"scenario model", b'{"scenario": "metadata"}'
    )
    for voice in made.voices:
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
    assert missing_artifacts(Settings(voice_cache=tmp_path), scenario.voice) == ()
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


def test_the_command_without_an_argument_fetches_every_offered_piper_voice(
    scenario: Scenario,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SPEECH_VOICE_CACHE", str(tmp_path))

    fetch_voice.main([], scenario)

    assert capsys.readouterr().out.split() == [
        str(piper_voice_paths(tmp_path, voice.value)[0]) for voice in PIPER_VOICES
    ]
    settings = Settings(voice_cache=tmp_path)
    assert [missing_artifacts(settings, voice) for voice in PIPER_VOICES] == [()] * len(
        PIPER_VOICES
    )


def test_the_command_fetches_only_the_voices_the_setting_offers(
    scenario: Scenario,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPEECH_VOICE_CACHE", str(tmp_path))
    monkeypatch.setenv("SPEECH_VOICES", scenario.voices[1].value)

    fetch_voice.main([], scenario)

    assert {call["filename"] for call in scenario.calls} == {
        f"a/b/{scenario.voices[1]}.onnx",
        f"a/b/{scenario.voices[1]}.onnx.json",
    }


def test_an_empty_offer_fetches_nothing_and_says_nothing(
    scenario: Scenario,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SPEECH_VOICE_CACHE", str(tmp_path))
    monkeypatch.setenv("SPEECH_VOICES", "")

    fetch_voice.main([], scenario)

    assert (capsys.readouterr().out, scenario.calls) == ("", [])


@dataclass
class SnapshotScenario:
    """A Hub snapshot downloader that writes the files it is asked for."""

    calls: list[dict[str, object]] = field(default_factory=list)

    def __call__(self, repo_id: str, **options: object) -> str:
        """Note the call, then write the snapshot as the Hub cache keeps it."""
        self.calls.append({"repo_id": repo_id, **options})
        return write_snapshot(
            repo_id,
            revision=str(options["revision"]),
            cache_dir=Path(str(options["cache_dir"])),
            allow_patterns=list(options["allow_patterns"]),
        )


def test_an_offered_gpu_voice_is_fetched_once_into_the_configured_hub_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SPEECH_HUGGINGFACE_CACHE", str(tmp_path))
    monkeypatch.setenv("SPEECH_VOICES", VoiceId.RYAN.value)
    snapshot = SnapshotScenario()

    fetch_voice.main([], download_snapshot=snapshot)

    printed = Path(capsys.readouterr().out.strip())
    assert sorted(path.name for path in printed.iterdir()) == sorted(
        {name.split("/", 1)[0] for name in QWEN_ARTIFACTS}
    )
    missing = missing_artifacts(Settings(huggingface_cache=tmp_path), VoiceId.RYAN)
    assert not set(missing) & set(QWEN_ARTIFACTS)
    assert [
        (call["repo_id"], call["revision"], call["token"]) for call in snapshot.calls
    ] == [(QWEN_MODEL_ID, QWEN_REVISION, False)]


def test_the_command_takes_no_argument() -> None:
    with pytest.raises(SystemExit) as stopped:
        fetch_voice.main([VoiceId.THORSTEN.value])

    assert stopped.value.code != 0


def test_every_piper_voice_is_pinned_in_full_under_the_names_the_service_reads() -> (
    None
):
    assert set(PINS) == set(PIPER_VOICES)
    for voice, pin in PINS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", pin.revision)
        assert re.fullmatch(r"[0-9a-f]{64}", pin.model.sha256)
        assert re.fullmatch(r"[0-9a-f]{64}", pin.metadata.sha256)
        onnx, metadata = piper_voice_paths(Path("voices"), voice.value)
        assert pin.model.path.rsplit("/", 1)[1] == onnx.name
        assert pin.metadata.path.rsplit("/", 1)[1] == metadata.name
