"""The caller check against tokens a real Keycloak issues.

The `keycloak-token-shapes` CI job serves `tests/fixtures/keycloak/realm.json`
and runs these tests with `-m keycloak`; the default run deselects them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
)

from speech_client.wire import (
    WORKLOAD_TOKEN_HEADER,
    RefusalCode,
    speech_request_headers,
)
from tests.support.identity_support import MINTING
from tests.support.keycloak_support import (
    AGENT_CLIENT,
    APP_CLIENT,
    AUDIENCE_SCOPE,
    OTHER_CLIENT,
    ROLELESS_CLIENT,
    UNSCOPED_CLIENT,
    WORKLOAD_ROLE,
    audiences_of,
    claims_of,
    exchanged_token,
    keycloak_callers,
    person_token,
    token_response,
    workload_token,
)
from tests.support.speak_support import (
    assert_refusal,
    info_comment_before_data,
    post_speech,
    speak_app,
    speak_body,
)

if TYPE_CHECKING:
    from httpx2 import Response as ClientResponse

pytestmark = pytest.mark.keycloak

SERVICE_GROUP = MINTING.service_group
APP_GROUP = MINTING.app_group


def test_a_person_token_with_the_scope_carries_the_audience_and_groups() -> None:
    """WIRE-13, WIRE-15: a person token with the scope carries audience and groups."""
    claims = claims_of(person_token("ann"))

    assert MINTING.audience in audiences_of(claims)
    assert {SERVICE_GROUP, APP_GROUP} <= set(claims["groups"])


def test_a_workload_token_with_the_scope_carries_the_audience_and_the_role() -> None:
    """WIRE-13, WIRE-14: a workload token with the scope carries audience and role."""
    claims = claims_of(workload_token(APP_CLIENT))

    assert MINTING.audience in audiences_of(claims)
    assert WORKLOAD_ROLE in claims["realm_access"]["roles"]


def test_a_workload_token_asked_without_the_scope_carries_no_audience() -> None:
    """WIRE-13: a workload token asked without the scope carries no audience."""
    claims = claims_of(workload_token(APP_CLIENT, with_audience=False))

    assert MINTING.audience not in audiences_of(claims)


def test_a_client_without_the_scope_is_refused_when_it_asks_for_it() -> None:
    """WIRE-13: a client without the scope is refused when it asks for it."""
    response = token_response(
        {
            "grant_type": "client_credentials",
            "client_id": UNSCOPED_CLIENT,
            "client_secret": f"{UNSCOPED_CLIENT}-secret",
            "scope": AUDIENCE_SCOPE,
        }
    )

    assert response.status_code == HTTP_400_BAD_REQUEST
    assert response.json()["error"] == "invalid_scope"


@pytest.fixture(scope="module")
def wire() -> TestClient:
    return TestClient(speak_app(callers=keycloak_callers()))


def speak(wire: TestClient, workload: str, person: str | None) -> ClientResponse:
    if person is None:
        headers = {WORKLOAD_TOKEN_HEADER: workload}
    else:
        headers = speech_request_headers(workload_token=workload, person_token=person)
    return post_speech(wire, speak_body(), headers)


def test_the_app_speaks_for_a_real_pair(wire: TestClient) -> None:
    """WIRE-12, WIRE-13, WIRE-14, WIRE-15: the app speaks for a real pair."""
    response = speak(wire, workload_token(APP_CLIENT), person_token("ann"))

    assert response.status_code == HTTP_200_OK
    assert info_comment_before_data(response.content) is not None


def test_a_person_in_a_subgroup_of_the_app_group_is_served(wire: TestClient) -> None:
    """WIRE-15: a person in a subgroup of the app group is served."""
    response = speak(wire, workload_token(APP_CLIENT), person_token("dot"))

    assert response.status_code == HTTP_200_OK


def test_the_real_workload_token_alone_is_unauthorized(wire: TestClient) -> None:
    """WIRE-20, WIRE-97: the real workload token alone is unauthorized."""
    response = speak(wire, workload_token(APP_CLIENT), None)

    assert_refusal(response, HTTP_401_UNAUTHORIZED, RefusalCode.UNAUTHORIZED, None)


def test_a_workload_asked_without_the_scope_is_unauthorized_beside_a_valid_person(
    wire: TestClient,
) -> None:
    """WIRE-13, WIRE-97: a workload asked without the scope is unauthorized."""
    workload = workload_token(APP_CLIENT, with_audience=False)

    response = speak(wire, workload, person_token("ann"))

    assert_refusal(response, HTTP_401_UNAUTHORIZED, RefusalCode.UNAUTHORIZED, None)


def test_a_workload_whose_service_account_lacks_the_role_is_forbidden(
    wire: TestClient,
) -> None:
    """WIRE-14, WIRE-98: a workload without the role is forbidden."""
    response = speak(wire, workload_token(ROLELESS_CLIENT), person_token("ann"))

    assert_refusal(response, HTTP_403_FORBIDDEN, RefusalCode.FORBIDDEN, None)


def test_a_person_token_issued_to_another_client_is_forbidden(wire: TestClient) -> None:
    """WIRE-15, WIRE-98: a person token issued to another client is forbidden."""
    response = speak(
        wire, workload_token(APP_CLIENT), person_token("ann", client=OTHER_CLIENT)
    )

    assert_refusal(response, HTTP_403_FORBIDDEN, RefusalCode.FORBIDDEN, None)


@pytest.mark.parametrize(
    "user",
    [
        pytest.param("cy", id="outside the app group"),
        pytest.param("ben", id="outside the service group"),
        pytest.param("eli", id="only in a subgroup of the service group"),
    ],
)
def test_a_person_missing_the_app_or_the_service_group_is_forbidden(
    wire: TestClient, user: str
) -> None:
    """WIRE-15, WIRE-98: a person missing the app or the service group is forbidden."""
    response = speak(wire, workload_token(APP_CLIENT), person_token(user))

    assert_refusal(response, HTTP_403_FORBIDDEN, RefusalCode.FORBIDDEN, None)


def assert_exchanged_for_the_app(subject_token: str, exchanged: str) -> None:
    subject_claims = claims_of(subject_token)
    claims = claims_of(exchanged)

    assert claims["azp"] == APP_CLIENT
    assert claims["sub"] == subject_claims["sub"]
    assert MINTING.audience in audiences_of(claims)
    assert {SERVICE_GROUP, APP_GROUP} <= set(claims["groups"])


def test_an_agent_token_exchanged_by_its_app_speaks(wire: TestClient) -> None:
    """WIRE-15: an agent's token exchanged by its app speaks."""
    agent = person_token("mozart", client=AGENT_CLIENT, with_audience=False)
    exchanged = exchanged_token(agent)

    assert_exchanged_for_the_app(agent, exchanged)
    response = speak(wire, workload_token(APP_CLIENT), exchanged)

    assert response.status_code == HTTP_200_OK
    assert info_comment_before_data(response.content) is not None


def test_an_agent_service_account_token_exchanged_by_its_app_speaks(
    wire: TestClient,
) -> None:
    """WIRE-15: a service account's token exchanged by its app speaks."""
    agent = workload_token(AGENT_CLIENT, with_audience=False)
    exchanged = exchanged_token(agent)

    assert_exchanged_for_the_app(agent, exchanged)
    response = speak(wire, workload_token(APP_CLIENT), exchanged)

    assert response.status_code == HTTP_200_OK


def test_an_exchanged_agent_outside_the_service_group_is_forbidden(
    wire: TestClient,
) -> None:
    """WIRE-15, WIRE-98: an exchanged agent outside the service group is forbidden."""
    agent = person_token("salieri", client=AGENT_CLIENT, with_audience=False)

    response = speak(wire, workload_token(APP_CLIENT), exchanged_token(agent))

    assert_refusal(response, HTTP_403_FORBIDDEN, RefusalCode.FORBIDDEN, None)


def test_an_unexchanged_agent_token_is_forbidden(wire: TestClient) -> None:
    """WIRE-15, WIRE-98: an agent token whose `azp` is its own client is forbidden."""
    agent = person_token("mozart", client=AGENT_CLIENT)

    response = speak(wire, workload_token(APP_CLIENT), agent)

    assert_refusal(response, HTTP_403_FORBIDDEN, RefusalCode.FORBIDDEN, None)
