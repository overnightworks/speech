"""The catalogue is one file: the shipped one loads, others bend names and pins."""

from typing import TYPE_CHECKING

import pytest

from shipped_catalogue import CATALOGUE, CATALOGUE_FILE
from speech_client.wire import VOICES, VoiceId
from speech_service.catalogue import CatalogueError, load_catalogue
from speech_service.config import Settings
from speech_service.service import Runtime
from tests.support.catalogue_support import (
    piper_table,
    qwen_table,
    write_catalogue,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_the_shipped_catalogue_loads_every_voice_of_the_wire() -> None:
    assert load_catalogue(CATALOGUE_FILE).keys() == set(VoiceId)


def test_the_client_catalogue_and_the_shipped_one_agree_through_the_wire_model() -> (
    None
):
    shipped = {
        voice: (
            entry.engine.wire_model,
            entry.name,
            entry.languages,
            entry.engine.takes_instructions,
        )
        for voice, entry in CATALOGUE.items()
    }
    client = {
        voice: (entry.model, entry.name, entry.languages, entry.takes_instructions)
        for voice, entry in VOICES.items()
    }

    assert shipped == client


def test_a_catalogue_whose_known_ids_carry_other_names_is_shown_in_health(
    tmp_path: Path,
) -> None:
    path = write_catalogue(
        tmp_path,
        piper_table(name="Scenario Thorsten", languages='["en"]'),
        qwen_table(name="Scenario Ryan"),
    )
    settings = Settings(
        catalogue_file=path, voices=frozenset({VoiceId.THORSTEN, VoiceId.RYAN})
    )

    voices = Runtime.from_settings(settings).health().speaking.voices

    assert {
        voice: (reported.name, [*reported.languages])
        for voice, reported in voices.items()
    } == {
        VoiceId.THORSTEN.value: ("Scenario Thorsten", ["en"]),
        VoiceId.RYAN.value: ("Scenario Ryan", ["de", "en"]),
    }


def test_an_offered_voice_the_catalogue_lacks_stops_the_start(tmp_path: Path) -> None:
    path = write_catalogue(tmp_path, piper_table())
    settings = Settings(catalogue_file=path, voices=frozenset({VoiceId.RYAN}))

    with pytest.raises(CatalogueError, match=VoiceId.RYAN.value):
        Runtime.from_settings(settings)


@pytest.mark.parametrize(
    ("table", "field"),
    [
        (piper_table(engine="espeak"), "voices.de_DE-thorsten-medium.*engine"),
        (piper_table(model_sha256="xyz"), "pin.model.sha256"),
        (piper_table(metadata_sha256="B" * 64), "pin.metadata.sha256"),
        (piper_table(revision="main"), "pin.revision"),
        (piper_table(languages="[]"), "languages"),
        (piper_table(languages='["fr"]'), "languages"),
        (piper_table(name=""), "name"),
        (piper_table(voice="not-a-voice"), "voices.not-a-voice.*id"),
        (qwen_table().replace('"Ryan"', '"Nobody"', 1), "pin.speaker"),
        (piper_table() + piper_table(), "de_DE-thorsten-medium"),
    ],
)
def test_a_malformed_entry_fails_loud_naming_the_field(
    tmp_path: Path, table: str, field: str
) -> None:
    path = write_catalogue(tmp_path, table)

    with pytest.raises(CatalogueError, match=field):
        load_catalogue(path)


def test_an_unreadable_catalogue_fails_loud_naming_the_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.toml"

    with pytest.raises(CatalogueError, match=r"missing\.toml"):
        load_catalogue(path)


def test_a_catalogue_that_is_not_toml_fails_loud_naming_the_file(
    tmp_path: Path,
) -> None:
    path = write_catalogue(tmp_path, "voices = [")

    with pytest.raises(CatalogueError, match=r"catalogue\.toml"):
        load_catalogue(path)


def test_the_catalogue_file_must_be_absolute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEECH_CATALOGUE_FILE", "deploy/catalogue.toml")

    with pytest.raises(ValueError, match=r"catalogue file must be absolute"):
        Settings()


def test_an_unset_catalogue_file_stops_the_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPEECH_CATALOGUE_FILE")

    with pytest.raises(CatalogueError, match="SPEECH_CATALOGUE_FILE"):
        Runtime.from_settings(Settings())
