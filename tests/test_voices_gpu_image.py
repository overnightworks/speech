"""The gpu image offers the catalogue voices; each stage that needs it holds it."""

import re

from shipped_catalogue import CATALOGUE
from tests.support.browser_support import REPOSITORY

_DOCKERFILE = (REPOSITORY / "Dockerfile").read_text()
_STAGES = re.split(r"^FROM ", _DOCKERFILE, flags=re.MULTILINE)[1:]


def test_the_gpu_image_offers_exactly_the_catalogue_voices() -> None:
    offered = set(
        re.search(r"^ARG GPU_VOICES=(\S+)$", _DOCKERFILE, re.MULTILINE)[1].split(",")
    )
    catalogue = {voice.value for voice in CATALOGUE}

    assert {
        "missing_from_image": sorted(catalogue - offered),
        "unknown_to_catalogue": sorted(offered - catalogue),
    } == {"missing_from_image": [], "unknown_to_catalogue": []}


def test_each_stage_that_fetches_or_serves_holds_the_file_the_setting_names() -> None:
    needing = [
        stage
        for stage in _STAGES
        if "speech-fetch-voice" in stage or 'CMD ["speech-service"]' in stage
    ]
    named = [re.search(r"SPEECH_CATALOGUE_FILE=(\S+)", stage) for stage in needing]

    assert needing
    assert all(
        match and f"COPY deploy/catalogue.toml {match[1]}\n" in stage
        for match, stage in zip(named, needing, strict=True)
    )
