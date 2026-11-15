"""The process logs through logline and holds Piper's input-carrying loggers back."""

import json
import os
import subprocess
import sys

_MAIN_CHILD = """
import logging
from unittest import mock

from speech_service import __main__

with mock.patch.object(__main__, "serve", new=mock.Mock(return_value=None)), \\
        mock.patch.object(__main__.asyncio, "run", return_value=0):
    try:
        __main__.main()
    except SystemExit:
        pass
    logging.getLogger("piper.voice").debug("text=%s", "SPOKEN-SENTENCE")
    logging.getLogger("piper.phoneme_ids").warning("missing %s", "SPOKEN-PHONEMES")
    logging.getLogger("speech_service.service").info("service.line")
"""


def _lines_of_main() -> list[dict[str, object]]:
    environment = {**os.environ, "SPEECH_DEBUG": "true", "LOG_LEVEL": "debug"}
    completed = subprocess.run(
        [sys.executable, "-c", _MAIN_CHILD],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return [json.loads(line) for line in completed.stdout.splitlines()]


def test_the_process_logs_json_lines_from_the_service_named_speech() -> None:
    lines = _lines_of_main()

    started = next(line for line in lines if line["event"] == "process.started")
    assert started["service"] == "speech"
    assert started["level"] == "info"


def test_logging_drops_piper_input_and_keeps_the_service_lines() -> None:
    lines = _lines_of_main()

    assert any(line["event"] == "service.line" for line in lines)
    assert "SPOKEN" not in json.dumps(lines)
