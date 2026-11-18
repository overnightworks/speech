"""Fixture catalogue files: known ids that carry other names and pins."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

PIPER_REPOSITORY = "scenario/voices"
PIPER_REVISION = "1" * 40
MODEL_SHA256 = "a" * 64
METADATA_SHA256 = "b" * 64


def piper_table(  # noqa: PLR0913 - one keyword per field a scenario may bend
    voice: str = "de_DE-thorsten-medium",
    *,
    engine: str = "piper",
    name: str = "Scenario Thorsten",
    languages: str = '["de"]',
    revision: str = PIPER_REVISION,
    model_sha256: str = MODEL_SHA256,
    metadata_sha256: str = METADATA_SHA256,
) -> str:
    """A `[voices.<voice>]` table of a Piper voice; each argument may be bent."""
    return f"""
[voices.{voice}]
engine = "{engine}"
name = "{name}"
languages = {languages}

[voices.{voice}.pin]
repository = "{PIPER_REPOSITORY}"
revision = "{revision}"

[voices.{voice}.pin.model]
path = "a/b/{voice}.onnx"
sha256 = "{model_sha256}"

[voices.{voice}.pin.metadata]
path = "a/b/{voice}.onnx.json"
sha256 = "{metadata_sha256}"
"""


def qwen_table(voice: str = "ryan", *, name: str = "Scenario Ryan") -> str:
    """A `[voices.<voice>]` table of the Qwen custom voice Ryan."""
    return f"""
[voices.{voice}]
engine = "qwen3-tts-customvoice"
name = "{name}"
languages = ["de", "en"]

[voices.{voice}.pin]
speaker = "Ryan"
"""


def write_catalogue(directory: Path, *tables: str) -> Path:
    """The catalogue file made of `tables`, written in `directory`."""
    path = directory / "catalogue.toml"
    path.write_text("".join(tables))
    return path
