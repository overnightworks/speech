"""WS /v1/realtime?intent=transcription: who may upgrade, what is refused, a session."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocketDisconnect

from speech_client.wire import (
    AUTHENTICATE_HEADER,
    REALTIME_PATH,
    RETRY_AFTER_HEADER,
    RefusalCode,
    token_headers,
)
from speech_service.config import Settings
from speech_service.control import create_control_app
from tests.support.hear_support import (
    HEARD_TEXT,
    WAIT_SECONDS,
    FakeHearing,
    hear_app,
    hear_runtime,
    hearing_gate_is_free,
    multipart_body,
    post_recording,
    recording,
)
from tests.support.identity_support import (
    APP,
    OTHER_MINTING,
    altered,
    callers_of_two_apps,
    valid_headers,
    valid_pair,
)
from tests.support.live_support import append, audio, commit, update
from tests.support.speak_support import assert_refusal_body

URL = f"{REALTIME_PATH}?intent=transcription"
COMPLETED = "conversation.item.input_audio_transcription.completed"


def _client(engine: FakeHearing | None = None, **settings: float) -> TestClient:
    return TestClient(hear_app(hear_runtime(engine or FakeHearing()), **settings))


def _denied(
    client: TestClient, url: str = URL, headers: dict[str, str] | None = None
) -> WebSocketDenialResponse:
    sent = valid_headers() if headers is None else headers
    connection = client.websocket_connect(url, headers=sent)
    with pytest.raises(WebSocketDenialResponse) as denial:
        connection.__enter__()
    return denial.value


def _assert_denied(
    denial: WebSocketDenialResponse, status: int, code: RefusalCode, param: str | None
) -> None:
    assert_refusal_body(denial.status_code, denial.json(), status, code, param)


def test_a_session_hears_a_committed_turn_and_ends_when_the_client_leaves() -> None:
    """WIRE-03, WIRE-48, WIRE-55, WIRE-56, WIRE-63: a session hears a committed turn."""
    with _client().websocket_connect(URL, headers=valid_headers()) as socket:
        assert socket.receive_json()["type"] == "session.created"
        socket.send_text(update("en"))
        assert socket.receive_json()["type"] == "session.updated"
        socket.send_text(append(audio(0.5)))
        socket.send_text(commit())
        events = [socket.receive_json() for _ in range(4)]

    assert [event["type"] for event in events] == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.committed",
        "conversation.item.created",
        COMPLETED,
    ]
    assert events[-1]["transcript"] == HEARD_TEXT


def test_a_registered_app_with_no_hearing_key_opens_a_live_session() -> None:
    """WIRE-48, WIRE-114: every registered app opens a session without a hearing key."""
    callers = callers_of_two_apps()
    client = TestClient(
        create_control_app(Settings(), hear_runtime(FakeHearing()), callers)
    )

    with client.websocket_connect(
        URL, headers=valid_headers(minting=OTHER_MINTING)
    ) as socket:
        assert socket.receive_json()["type"] == "session.created"


@pytest.mark.parametrize(
    ("setup", "param"),
    [
        pytest.param(update("fr"), "language", id="language-fr"),
        pytest.param(
            update(model="speech-runtime"), "model", id="model-speech-runtime"
        ),
    ],
)
def test_a_setup_outside_the_one_transcription_shape_ends_the_session_with_an_error(
    setup: str, param: str
) -> None:
    """WIRE-56, WIRE-57, WIRE-58, WIRE-77: a setup outside the shape is refused."""
    with _client().websocket_connect(URL, headers=valid_headers()) as socket:
        socket.receive_json()
        socket.send_text(setup)
        error = socket.receive_json()
        assert error["type"] == "error"
        with pytest.raises(WebSocketDisconnect):
            socket.receive_json()

    assert error["error"]["code"] == RefusalCode.INVALID_REQUEST
    assert error["error"]["param"] == f"session.audio.input.transcription.{param}"


def test_a_caller_without_tokens_is_told_so_before_anything_else() -> None:
    """WIRE-50, WIRE-51, WIRE-97: a caller without tokens is told so first."""
    runtime = hear_runtime(None)
    assert runtime.hearing.open_session(1)
    client = TestClient(hear_app(runtime, max_hearing_sessions=1))

    denial = _denied(client, headers={})

    _assert_denied(denial, HTTP_401_UNAUTHORIZED, RefusalCode.UNAUTHORIZED, None)
    assert denial.headers[AUTHENTICATE_HEADER] == "Bearer"


def test_a_person_outside_the_service_group_is_forbidden() -> None:
    """WIRE-51, WIRE-98: a person outside the service group is forbidden."""
    pair = valid_pair()
    headers = token_headers(
        workload_token=pair.workload,
        person_token=altered(pair.person, groups=[f"/apps/{APP}"]),
    )

    denial = _denied(_client(), headers=headers)

    _assert_denied(denial, HTTP_403_FORBIDDEN, RefusalCode.FORBIDDEN, None)


@pytest.mark.parametrize(
    ("url", "status", "code", "param"),
    [
        (
            "/v2/realtime?intent=transcription",
            HTTP_400_BAD_REQUEST,
            RefusalCode.UNSUPPORTED_VERSION,
            None,
        ),
        ("/v1/elsewhere", HTTP_404_NOT_FOUND, RefusalCode.NOT_FOUND, None),
        (
            REALTIME_PATH,
            HTTP_422_UNPROCESSABLE_CONTENT,
            RefusalCode.INVALID_REQUEST,
            "intent",
        ),
        (
            f"{REALTIME_PATH}?intent=conversation",
            HTTP_422_UNPROCESSABLE_CONTENT,
            RefusalCode.INVALID_REQUEST,
            "intent",
        ),
    ],
    ids=["wrong-version", "no-such-path", "no-intent", "other-intent"],
)
def test_an_upgrade_that_is_not_the_one_route_is_denied(
    url: str, status: int, code: RefusalCode, param: str | None
) -> None:
    """WIRE-51, WIRE-53, WIRE-96: an upgrade that is not the one route is denied."""
    _assert_denied(_denied(_client(), url), status, code, param)


def test_hearing_that_is_unavailable_denies_the_upgrade() -> None:
    """WIRE-51, WIRE-103: hearing that is unavailable denies the upgrade."""
    client = TestClient(hear_app(hear_runtime(None)))

    _assert_denied(
        _denied(client),
        HTTP_503_SERVICE_UNAVAILABLE,
        RefusalCode.UNAVAILABLE,
        None,
    )


def test_a_session_beyond_the_bound_is_denied_as_busy_until_one_leaves() -> None:
    """WIRE-52, WIRE-119: a session beyond the bound is denied as busy."""
    client = _client(max_hearing_sessions=1)

    with client.websocket_connect(URL, headers=valid_headers()) as first:
        first.receive_json()
        denial = _denied(client)
    with client.websocket_connect(URL, headers=valid_headers()) as again:
        reopened = again.receive_json()

    _assert_denied(denial, HTTP_429_TOO_MANY_REQUESTS, RefusalCode.BUSY, None)
    assert int(denial.headers[RETRY_AFTER_HEADER]) >= 1
    assert reopened["type"] == "session.created"


def test_a_recording_is_heard_while_a_live_session_sits_idle() -> None:
    """WIRE-54: an idle session holds its place only, never the decode."""
    engine = FakeHearing()
    client = _client(engine, hearing_busy_wait_seconds=0)

    with client.websocket_connect(URL, headers=valid_headers()) as idle:
        idle.receive_json()
        response = post_recording(client, multipart_body(file=recording()))

    assert response.status_code == HTTP_200_OK
    assert response.json() == {"text": HEARD_TEXT}
    assert len(engine.heard) == 1


def _upgrade_scope() -> dict[str, object]:
    return {
        "type": "websocket",
        "path": REALTIME_PATH,
        "query_string": b"intent=transcription",
        "headers": [
            (name.lower().encode(), value.encode())
            for name, value in valid_headers().items()
        ],
        "subprotocols": [],
    }


def test_a_session_cancelled_twice_frees_its_place_and_hears_no_final() -> None:
    engine = FakeHearing()
    runtime = hear_runtime(engine)
    app = hear_app(runtime, max_hearing_sessions=1)
    messages = [
        {"type": "websocket.connect"},
        {"type": "websocket.receive", "text": append(audio(0.5))},
        {"type": "websocket.receive", "text": commit()},
    ]

    async def scenario() -> bool:
        inbox: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        for message in messages:
            inbox.put_nowait(message)
        accepted = asyncio.Event()
        waiting_for_the_client = asyncio.Event()
        serving: asyncio.Task[None] | None = None

        async def receive() -> dict[str, object]:
            if inbox.empty():
                waiting_for_the_client.set()
            try:
                return await inbox.get()
            except asyncio.CancelledError:
                assert serving is not None
                asyncio.get_running_loop().call_soon(serving.cancel)
                raise

        async def send(message: dict[str, object]) -> None:
            if message["type"] == "websocket.accept":
                accepted.set()

        running_decode = runtime.hearing.gate.try_acquire()
        assert running_decode is not None
        serving = asyncio.create_task(app(_upgrade_scope(), receive, send))
        async with asyncio.timeout(WAIT_SECONDS):
            await accepted.wait()
            await waiting_for_the_client.wait()
        serving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await serving
        running_decode.close()
        return hearing_gate_is_free(runtime)

    gate_was_free_for_the_next_job = asyncio.run(scenario())

    assert runtime.hearing.open_session(1) is not None
    assert gate_was_free_for_the_next_job
