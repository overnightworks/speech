"""Every `SPEECH_` variable the repository names is a setting or a declared own one."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Final

import pytest

from speech_service.config import Settings, setting_variable

ROOT: Final = Path(__file__).resolve().parents[1]
CHECKED_DIRECTORIES: Final = ("src", "scripts", "tests", "packages", "docs")
CHECKED_SUFFIXES: Final = frozenset(
    {".py", ".md", ".ts", ".js", ".json", ".yaml", ".toml"}
)
IGNORED_PARTS: Final = frozenset({"node_modules", "dist", "__pycache__", ".venv"})
VARIABLE: Final = re.compile(r"(?<![A-Za-z0-9_])SPEECH_[A-Z0-9_]+")

# Variables that are no `Settings` field, each with the file that owns it.
OWN_VARIABLES: Final = {
    "SPEECH_PROOF_PROJECT": "scripts/prove_image.py",
    "SPEECH_PROOF_UNDER_PROBE_RUN": "scripts/prove_speech.py",
    "SPEECH_BROWSER_COVERAGE_DIR": "tests/support/page_coverage_support.py",
    # Variables the service no longer reads; the tests that set one prove it is ignored.
    "SPEECH_DEBUG": "tests/test_config.py",
    "SPEECH_STATE_DIRECTORY": "tests/test_config.py",
    # A caller's token the provider's environment must not carry.
    "SPEECH_PERSON_TOKEN": "tests/test_provider_process_environment.py",
}


def _variables_named_in(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".py":
        return set(VARIABLE.findall(text))
    return {
        name
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        for name in VARIABLE.findall(node.value)
    }


def _checked_files() -> list[Path]:
    return [
        path
        for directory in CHECKED_DIRECTORIES
        for path in (ROOT / directory).rglob("*")
        if path.suffix in CHECKED_SUFFIXES and not IGNORED_PARTS & set(path.parts)
    ]


def test_every_speech_variable_named_is_a_setting_or_declared() -> None:
    known = {setting_variable(field) for field in Settings.model_fields} | set(
        OWN_VARIABLES
    )

    unknown = {
        str(path.relative_to(ROOT)): sorted(named - known)
        for path in _checked_files()
        if (named := _variables_named_in(path)) - known
    }

    assert not unknown


@pytest.mark.parametrize("owner", sorted(set(OWN_VARIABLES.values())))
def test_a_declared_variable_is_named_by_its_owner(owner: str) -> None:
    declared = {name for name, file in OWN_VARIABLES.items() if file == owner}

    assert declared <= _variables_named_in(ROOT / owner)


def test_a_name_that_is_no_setting_has_no_variable() -> None:
    with pytest.raises(ValueError, match="no_such_setting"):
        setting_variable("no_such_setting")


def test_a_setting_maps_to_its_prefixed_upper_case_variable() -> None:
    assert setting_variable("hearing_model") == "SPEECH_HEARING_MODEL"
