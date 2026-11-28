"""The image proof's verdicts that only a broken image or stack would reach."""

from __future__ import annotations

from dataclasses import replace
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest

import image_judgement as judged
from image_stack import Reply, SoundOutcome
from prove_image import BUILD_LIMIT_SECONDS
from speech_client.realtime_wire import SoundName
from speech_client.utterances import HeardSound

if TYPE_CHECKING:
    from collections.abc import Callable

KEY_SERVER = "keys.example:443"
PUBLIC = "public.example:443"
SEALED = judged.SealedState(
    health_status="healthy",
    read_only_root=True,
    user="1000",
    speaking_ready=True,
    root_writable=False,
)
STARTED = ("container-id", "2026-11-28T10:00:00Z", 0)


def test_a_build_past_its_limit_fails_the_proof_naming_the_limit() -> None:
    judged.require_built_within(BUILD_LIMIT_SECONDS, BUILD_LIMIT_SECONDS)

    with pytest.raises(SystemExit, match=f"past {BUILD_LIMIT_SECONDS} s"):
        judged.require_built_within(BUILD_LIMIT_SECONDS + 1, BUILD_LIMIT_SECONDS)


def test_a_sealed_container_passes() -> None:
    judged.require_sealed_container(SEALED)


@pytest.mark.parametrize(
    ("change", "fault"),
    [
        pytest.param({"health_status": "starting"}, "it is starting", id="not healthy"),
        pytest.param({"read_only_root": False}, "not read-only", id="writable mount"),
        pytest.param({"user": "0"}, "runs as root", id="root user"),
        pytest.param(
            {"speaking_ready": False}, "speaking is not ready", id="not ready"
        ),
        pytest.param({"root_writable": True}, "root is writable", id="writes the root"),
    ],
)
def test_a_container_that_is_not_sealed_fails_the_proof_naming_its_fault(
    change: dict[str, object], fault: str
) -> None:
    with pytest.raises(SystemExit, match=fault):
        judged.require_sealed_container(replace(SEALED, **change))


def test_a_reach_of_the_key_server_alone_passes() -> None:
    judged.require_reach_of_key_server_alone(
        {KEY_SERVER: True, PUBLIC: False}, KEY_SERVER
    )


@pytest.mark.parametrize(
    ("reach", "fault"),
    [
        pytest.param(
            {KEY_SERVER: True, PUBLIC: True},
            f"beyond.*{PUBLIC}",
            id="a public host answers",
        ),
        pytest.param(
            {KEY_SERVER: False, PUBLIC: False},
            f"{KEY_SERVER} does not answer",
            id="the key server does not answer",
        ),
    ],
)
def test_a_reach_beyond_the_key_server_alone_fails_the_proof(
    reach: dict[str, bool], fault: str
) -> None:
    with pytest.raises(SystemExit, match=fault):
        judged.require_reach_of_key_server_alone(reach, KEY_SERVER)


def test_a_health_not_ready_while_the_key_server_is_away_fails_the_proof() -> None:
    judged.require_health_ready_without_key_server({"speaking": {"ready": True}})

    with pytest.raises(SystemExit, match="does not stay honest"):
        judged.require_health_ready_without_key_server({"speaking": {"ready": False}})


def test_a_healthcheck_that_passes_without_a_hearing_model_fails_the_proof() -> None:
    judged.require_healthcheck_fails_without_model(healthcheck_passes=False)

    with pytest.raises(SystemExit, match="passes without a hearing model"):
        judged.require_healthcheck_fails_without_model(healthcheck_passes=True)


def test_a_log_that_lacks_the_setting_it_stopped_for_fails_the_proof() -> None:
    judged.require_log_names_setting("error: HEARING_MODEL unset", "HEARING_MODEL")

    with pytest.raises(SystemExit, match="HEARING_MODEL"):
        judged.require_log_names_setting("started", "HEARING_MODEL")


def test_a_teardown_that_left_something_fails_the_proof_naming_it() -> None:
    judged.require_teardown_clean([])

    with pytest.raises(SystemExit, match="left something: compose down failed: busy"):
        judged.require_teardown_clean(["compose down failed: busy"])


def test_a_container_that_restarted_under_the_offered_gpu_voice_fails_the_proof() -> (
    None
):
    judged.require_container_unrestarted(STARTED, STARTED)

    with pytest.raises(SystemExit, match="restarted"):
        judged.require_container_unrestarted(STARTED, (*STARTED[:2], 1))


def test_an_admitted_speak_that_is_refused_fails_naming_its_status() -> None:
    judged.require_admitted_speak(Reply(HTTPStatus.OK, b""))

    with pytest.raises(SystemExit, match="answered 403"):
        judged.require_admitted_speak(Reply(403, b""))


@pytest.mark.parametrize(
    "judgement",
    [
        pytest.param(
            # The error fires before the clip is judged: the length is any value.
            lambda outcome: judged.require_one_laughter(outcome, 3000),
            id="asked",
        ),
        pytest.param(judged.require_no_sounds, id="not asked"),
    ],
)
def test_a_sound_turn_that_ended_in_an_error_fails_the_proof_naming_it(
    judgement: Callable[[SoundOutcome], None],
) -> None:
    ended = SoundOutcome((HeardSound(SoundName.LAUGHTER, 0, 1000),), "Refused")

    with pytest.raises(SystemExit, match="ended in Refused"):
        judgement(ended)


def test_a_wav_with_no_mark_to_cut_fails_loud_instead_of_passing_the_negative() -> None:
    unmarked = b"RIFF\0\0\0\0WAVE"

    with pytest.raises(SystemExit, match="no INFO chunk"):
        judged.without_mark(unmarked)
