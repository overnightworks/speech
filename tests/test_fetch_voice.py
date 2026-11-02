"""A Piper voice is fetched at its pin, checked against its hashes, and placed whole."""

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from speech_runtime_service import fetch_voice
from speech_runtime_service.piper_voices import (
    PINS,
    PinnedFile,
    PinnedVoice,
    VoiceHashMismatchError,
    VoicePin,
    fetch_voice_files,
)

if TYPE_CHECKING:
    from pathlib import Path

VOICE = PinnedVoice.THORSTEN
REVISION = "1" * 40
MODEL_BYTES = b"scenario model"
METADATA_BYTES = b'{"scenario": "metadata"}'


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass
class Downloader:
    """Writes scenario bytes by the file name its url ends in; keeps the urls."""

    contents: dict[str, bytes]
    urls: list[str]

    def __call__(self, url: str, destination: Path) -> None:
        """Write the bytes of the file `url` names to `destination`."""
        self.urls.append(url)
        destination.write_bytes(self.contents[url.rsplit("/", 1)[1]])


@pytest.fixture(autouse=True)
def scenario_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        PINS,
        VOICE,
        VoicePin(
            "scenario/voices",
            REVISION,
            (
                PinnedFile(f"a/b/{VOICE}.onnx", sha256(MODEL_BYTES)),
                PinnedFile(f"a/b/{VOICE}.onnx.json", sha256(METADATA_BYTES)),
            ),
        ),
    )


def downloader(model: bytes = MODEL_BYTES) -> Downloader:
    return Downloader(
        {f"{VOICE}.onnx": model, f"{VOICE}.onnx.json": METADATA_BYTES}, []
    )


def test_a_voice_is_fetched_at_its_revision_under_the_names_the_service_reads(
    tmp_path: Path,
) -> None:
    download = downloader()

    onnx = fetch_voice_files(VOICE, tmp_path, download)

    assert onnx == tmp_path / f"{VOICE.value}.onnx"
    assert onnx.read_bytes() == MODEL_BYTES
    assert (tmp_path / f"{VOICE.value}.onnx.json").read_bytes() == METADATA_BYTES
    assert all(f"scenario/voices/resolve/{REVISION}/" in url for url in download.urls)


def test_a_wrong_hash_fails_naming_the_file_and_leaves_nothing_at_the_final_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(VoiceHashMismatchError, match=rf"{VOICE}\.onnx\b"):
        fetch_voice_files(VOICE, tmp_path, downloader(b"tampered"))

    assert list(tmp_path.iterdir()) == []


def test_a_present_matching_voice_is_not_fetched_again(tmp_path: Path) -> None:
    fetch_voice_files(VOICE, tmp_path, downloader())
    again = downloader()

    fetch_voice_files(VOICE, tmp_path, again)

    assert again.urls == []


def test_a_present_voice_that_no_longer_matches_is_fetched_again(
    tmp_path: Path,
) -> None:
    onnx = fetch_voice_files(VOICE, tmp_path, downloader())
    onnx.write_bytes(b"damaged")

    fetch_voice_files(VOICE, tmp_path, downloader())

    assert onnx.read_bytes() == MODEL_BYTES


def test_the_command_prints_where_the_voice_went_in_the_configured_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("SPEECH_VOICE_CACHE", str(tmp_path))

    fetch_voice.main([VOICE.value], downloader())

    assert capsys.readouterr().out.strip() == str(tmp_path / f"{VOICE.value}.onnx")


def test_the_command_refuses_a_voice_it_does_not_know() -> None:
    with pytest.raises(SystemExit) as stopped:
        fetch_voice.main(["nobody-medium"])

    assert stopped.value.code != 0


def test_every_voice_is_pinned_to_a_full_revision_and_full_hashes() -> None:
    assert set(PINS) == set(PinnedVoice)
    for pin in PINS.values():
        assert re.fullmatch(r"[0-9a-f]{40}", pin.revision)
        assert {f.path.rsplit(".", 1)[1] for f in pin.files} == {"onnx", "json"}
        assert all(re.fullmatch(r"[0-9a-f]{64}", f.sha256) for f in pin.files)
