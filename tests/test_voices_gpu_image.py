"""The gpu image offers exactly the voices of the catalogue."""

from __future__ import annotations

import re

from speech_client.wire import VoiceId
from tests.support.browser_support import REPOSITORY


def test_the_gpu_image_offers_exactly_the_catalogue_voices() -> None:
    dockerfile = (REPOSITORY / "Dockerfile").read_text()
    offered = set(
        re.search(r"^ARG GPU_VOICES=(\S+)$", dockerfile, re.MULTILINE)[1].split(",")
    )
    catalogue = {voice.value for voice in VoiceId}

    assert {
        "missing_from_image": sorted(catalogue - offered),
        "unknown_to_catalogue": sorted(offered - catalogue),
    } == {"missing_from_image": [], "unknown_to_catalogue": []}
