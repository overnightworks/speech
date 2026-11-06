"""A fake `docker` on PATH that answers each command by the words it contains."""

from __future__ import annotations

import os
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pytest


class DockerOnPath:
    """The Docker client as a script: it keeps each command and prints an answer."""

    def __init__(
        self,
        directory: Path,
        monkeypatch: pytest.MonkeyPatch,
        answers: Mapping[str, str],
    ) -> None:
        """Put the fake first on PATH; a command containing a key prints its answer.

        The first key that a command's words contain wins; any other command
        prints nothing.
        """
        self._log = directory / "docker.log"
        cases = "\n".join(
            f"  *{shlex.quote(words)}*) printf '%s\\n' {shlex.quote(answer)} ;;"
            for words, answer in answers.items()
        )
        script = directory / "docker"
        script.write_text(
            f'#!/bin/sh\necho "$*" >> {shlex.quote(str(self._log))}\n'
            f'case "$*" in\n{cases}\nesac\n'
        )
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")

    @property
    def commands(self) -> list[str]:
        """Every command run so far, as the words after `docker`."""
        if not self._log.exists():
            return []
        return self._log.read_text().splitlines()
