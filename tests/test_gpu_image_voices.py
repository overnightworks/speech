"""The gpu image offers every voice of the catalogue."""

from __future__ import annotations

import re

from speech_client.wire import VoiceId
from tests.support.browser_support import REPOSITORY


def test_the_gpu_image_offers_every_catalogue_voice() -> None:
    dockerfile = (REPOSITORY / "Dockerfile").read_text()
    offered = set(
        re.search(r"^ARG GPU_VOICES=(\S+)$", dockerfile, re.MULTILINE)[1].split(",")
    )
    catalogue = {voice.value for voice in VoiceId}

    assert (sorted(catalogue - offered), sorted(offered - catalogue)) == ([], [])
