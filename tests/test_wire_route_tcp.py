"""The speech routes over one runtime answer on a real TCP listener."""

import asyncio
import base64
import io
import json
import os
import struct
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, NamedTuple
from urllib.parse import urlsplit

import av
import httpx2
import pytest
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_404_NOT_FOUND,
)
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from speech_runtime_client.wire import (
    AUTHORIZATION_HEADER,
    REALTIME_PATH,
    SPEECH_PATH,
    RefusalCode,
    ResponseFormat,
)
from speech_runtime_service.pcm import (
    AI_MARK,
    BYTES_PER_SAMPLE,
    WAV_HEADER_BYTES,
    pcm_from_wav,
)
from speech_runtime_service.service import Runtime
from tests.support.fakes_support import (
    GATE_TIMEOUT_SECONDS,
    SPEAK_SAMPLE_RATE,
    FakeSpeaking,
    GatedSpeaking,
    sine_pcm,
)
from tests.support.identity_support import valid_headers
from tests.support.speak_support import (
    FailingSpeaking,
    assert_refusal,
    assert_refusal_body,
    gate_is_free,
    speak_body,
    synthesis_gate,
)
from tests.support.tcp_server_support import TcpServer, serving_over_tcp

if TYPE_CHECKING:
    import threading
    from collections.abc import AsyncIterator, Mapping

_SPEAK_BODY = speak_body()
_OPUS_BODY = speak_body(response_format=ResponseFormat.OPUS)
_BODIES = {ResponseFormat.WAV: _SPEAK_BODY, ResponseFormat.OPUS: _OPUS_BODY}
_OGG_END_OF_STREAM = 0x04
_OGG_FRAME_SECONDS = 0.02
_OGG_RATE = 48_000
_OGG_HEADER_PAGES = 2
_LONG_CHUNK_SECONDS = 0.6


class _LongVoice(GatedSpeaking):
    """A gated voice whose first chunk is long enough to fill Opus pages."""

    def __init__(
        self, *, hold_last_chunk: bool = True, sample_rate: int = SPEAK_SAMPLE_RATE
    ) -> None:
        super().__init__(hold_last_chunk=hold_last_chunk)
        self.sample_rate = sample_rate
        self.release_first_step.set()
        self.audio = sine_pcm(2 * _LONG_CHUNK_SECONDS, rate=sample_rate)


class _OggPage(NamedTuple):
    flags: int
    granule: int
    payload: bytes


def _ogg_pages(data: bytes) -> list[_OggPage]:
    """The whole pages in `data`, read from the bytes."""
    pages = []
    position = 0
    while position + 27 <= len(data):
        assert data[position : position + 4] == b"OggS"
        flags, granule = struct.unpack_from("<xBq", data, position + 4)
        segments = data[position + 26]
        header_end = position + 27 + segments
        if header_end > len(data):
            break
        size = sum(data[position + 27 : header_end])
        if header_end + size > len(data):
            break
        pages.append(_OggPage(flags, granule, data[header_end : header_end + size]))
        position = header_end + size
    return pages


def _opus_pre_skip(pages: list[_OggPage]) -> int:
    head = pages[0].payload
    assert head[:8] == b"OpusHead"
    return struct.unpack_from("<H", head, 10)[0]


def _opus_comments(tags: bytes) -> list[str]:
    assert tags[:8] == b"OpusTags"
    (vendor_size,) = struct.unpack_from("<I", tags, 8)
    position = 12 + vendor_size
    (count,) = struct.unpack_from("<I", tags, position)
    position += 4
    comments = []
    for _ in range(count):
        (size,) = struct.unpack_from("<I", tags, position)
        comments.append(tags[position + 4 : position + 4 + size].decode())
        position += 4 + size
    return comments


def _decoded_opus(body: bytes) -> tuple[int, int, float]:
    """Channels, rate and seconds of the audio in an Ogg Opus `body`."""
    with av.open(io.BytesIO(body)) as container:
        stream = container.streams.audio[0]
        samples = sum(
            frame.samples for frame in container.decode(stream) if frame.samples
        )
        return stream.codec_context.channels, stream.rate, samples / stream.rate


@asynccontextmanager
async def _gates_released(voice: GatedSpeaking) -> AsyncIterator[None]:
    """Open the gates on exit, so a failed assertion still lets the server stop."""
    try:
        yield
    finally:
        voice.release_all()


def _client(
    server: TcpServer, headers: Mapping[str, str] | None = None
) -> httpx2.AsyncClient:
    """A client on the listener; `headers` default to a valid pair of tokens."""
    return httpx2.AsyncClient(
        base_url=server.base_url,
        timeout=GATE_TIMEOUT_SECONDS,
        headers=valid_headers() if headers is None else headers,
    )


def _address(server: TcpServer) -> tuple[str, int]:
    address = urlsplit(server.base_url)
    return address.hostname or "", address.port or 0


def _gated_runtime(voice: FakeSpeaking | GatedSpeaking) -> Runtime:
    return Runtime(voice, memory_probe=lambda: 0)


async def _first_chunk(chunks: AsyncIterator[bytes]) -> bytes:
    first = b""
    while len(first) <= WAV_HEADER_BYTES:
        first += await asyncio.wait_for(anext(chunks), timeout=GATE_TIMEOUT_SECONDS)
    return first


async def _wait_for(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, GATE_TIMEOUT_SECONDS)


def _assert_whole_wav(status_code: int, body: bytes, voice: GatedSpeaking) -> None:
    assert status_code == HTTP_200_OK
    rate, pcm = pcm_from_wav(body)
    assert rate == SPEAK_SAMPLE_RATE
    assert pcm == voice.audio


def _assert_whole_answer(
    response_format: ResponseFormat, body: bytes, audio: bytes
) -> None:
    """The body is a finished answer: the voice's audio in WAV, an end page in Opus."""
    if response_format is ResponseFormat.WAV:
        assert pcm_from_wav(body) == (SPEAK_SAMPLE_RATE, audio)
    else:
        assert _ogg_pages(body)[-1].flags & _OGG_END_OF_STREAM


def test_tcp_streams_speak_while_the_voice_is_still_synthesizing() -> None:
    """WIRE-01, WIRE-88: tcp streams speak while the voice is still synthesizing."""

    async def scenario() -> tuple[GatedSpeaking, int, bytes, bytes]:
        voice = GatedSpeaking(hold_last_chunk=True)
        voice.release_first_step.set()
        async with (
            serving_over_tcp(_gated_runtime(voice)) as server,
            _client(server) as client,
            _gates_released(voice),
            client.stream("POST", SPEECH_PATH, content=_SPEAK_BODY) as response,
        ):
            assert response.headers["content-type"] == "audio/wav"
            chunks = response.aiter_bytes()
            first = await _first_chunk(chunks)
            assert not voice.release_last_chunk.is_set()
            voice.release_last_chunk.set()
            rest = b"".join([chunk async for chunk in chunks])
        return voice, response.status_code, first, first + rest

    voice, status_code, first, whole = asyncio.run(scenario())

    first_rate, first_pcm = pcm_from_wav(first)
    assert first_rate == SPEAK_SAMPLE_RATE
    assert 0 < len(first_pcm) < len(voice.audio)
    _assert_whole_wav(status_code, whole, voice)


def _audio_page_arrived(received: bytes) -> bool:
    pages = _ogg_pages(received)
    return len(pages) > _OGG_HEADER_PAGES and any(
        page.granule > _opus_pre_skip(pages) for page in pages[_OGG_HEADER_PAGES:]
    )


def test_tcp_streams_opus_pages_while_the_voice_is_still_synthesizing() -> None:
    """WIRE-01, WIRE-133: audio pages leave before the voice's last chunk does."""

    async def scenario() -> tuple[httpx2.Response, bool, bytes]:
        voice = _LongVoice()
        async with (
            serving_over_tcp(_gated_runtime(voice)) as server,
            _client(server) as client,
            _gates_released(voice),
            client.stream("POST", SPEECH_PATH, content=_OPUS_BODY) as response,
        ):
            chunks = response.aiter_bytes()
            received = b""
            while not _audio_page_arrived(received):
                received += await asyncio.wait_for(
                    anext(chunks), timeout=GATE_TIMEOUT_SECONDS
                )
            arrived_early = not voice.release_last_chunk.is_set()
            voice.release_last_chunk.set()
            rest = b"".join([chunk async for chunk in chunks])
        return response, arrived_early, received + rest

    response, arrived_early, whole = asyncio.run(scenario())

    assert response.status_code == HTTP_200_OK
    assert response.headers["content-type"] == ResponseFormat.OPUS.media_type
    assert "content-length" not in response.headers
    assert arrived_early
    assert _ogg_pages(whole)[-1].flags & _OGG_END_OF_STREAM


def _opus_answer(voice: _LongVoice) -> httpx2.Response:
    async def scenario() -> httpx2.Response:
        async with (
            serving_over_tcp(_gated_runtime(voice)) as server,
            _client(server) as client,
        ):
            return await client.post(SPEECH_PATH, content=_OPUS_BODY)

    return asyncio.run(scenario())


@pytest.mark.parametrize("voice_rate", [SPEAK_SAMPLE_RATE, 24_000])
def test_tcp_opus_is_mono_48_khz_and_lasts_as_long_as_the_voice_spoke(
    voice_rate: int,
) -> None:
    """WIRE-133: the Ogg Opus decodes to mono at 48 kHz, the voice's whole length."""
    voice = _LongVoice(hold_last_chunk=False, sample_rate=voice_rate)

    answer = _opus_answer(voice)

    channels, rate, seconds = _decoded_opus(answer.content)
    spoken = len(voice.audio) / BYTES_PER_SAMPLE / voice_rate
    assert (channels, rate) == (1, _OGG_RATE)
    assert seconds == pytest.approx(spoken, abs=_OGG_FRAME_SECONDS)


def test_tcp_opus_carries_the_ai_mark_in_its_raw_opus_tags() -> None:
    """WIRE-134: the OpusTags packet holds DESCRIPTION= with the AI mark."""
    answer = _opus_answer(_LongVoice(hold_last_chunk=False))

    tags = _ogg_pages(answer.content)[1].payload
    assert f"DESCRIPTION={AI_MARK}" in _opus_comments(tags)


@pytest.mark.parametrize("response_format", list(ResponseFormat))
def test_tcp_speak_disconnect_frees_the_voice_for_the_next_caller(
    response_format: ResponseFormat,
) -> None:
    """WIRE-132: a speak caller that has gone leaves its voice free."""

    async def scenario() -> tuple[GatedSpeaking, httpx2.Response]:
        voice = GatedSpeaking(hold_last_chunk=True)
        voice.release_first_step.set()
        async with (
            serving_over_tcp(_gated_runtime(voice)) as server,
            _client(server) as client,
            _gates_released(voice),
        ):
            async with client.stream(
                "POST", SPEECH_PATH, content=_BODIES[response_format]
            ) as leaving:
                await _first_chunk(leaving.aiter_bytes())
                await _wait_for(voice.entered_last_step)
            await asyncio.wait_for(
                server.speak_disconnect_received.wait(), timeout=GATE_TIMEOUT_SECONDS
            )
            voice.release_last_chunk.set()
            return voice, await client.post(
                SPEECH_PATH, content=_BODIES[response_format]
            )

    voice, later = asyncio.run(scenario())

    assert later.status_code == HTTP_200_OK
    _assert_whole_answer(response_format, later.content, voice.audio)


def test_a_speak_holds_the_one_synthesis_gate() -> None:
    """WIRE-21: a speak holds the one synthesis gate."""

    async def scenario() -> tuple[GatedSpeaking, bool, httpx2.Response, bool]:
        voice = GatedSpeaking()
        runtime = _gated_runtime(voice)
        async with (
            serving_over_tcp(runtime) as server,
            _client(server) as socket_client,
            _gates_released(voice),
        ):
            talk = asyncio.create_task(
                socket_client.post(SPEECH_PATH, content=_SPEAK_BODY)
            )
            await _wait_for(voice.entered_first_step)
            held = synthesis_gate(runtime).try_acquire() is None
            voice.release_first_step.set()
            talked = await asyncio.wait_for(talk, timeout=GATE_TIMEOUT_SECONDS)
            return voice, held, talked, gate_is_free(runtime)

    voice, held, talked, free = asyncio.run(scenario())

    assert held
    assert voice.most_concurrent_syntheses == 1
    _assert_whole_wav(talked.status_code, talked.content, voice)
    assert free


@pytest.mark.parametrize("response_format", list(ResponseFormat))
@pytest.mark.parametrize("chunks_before_failing", [0, 1])
def test_tcp_speak_failure_aborts_the_body_and_the_next_request_is_whole(
    chunks_before_failing: int, response_format: ResponseFormat
) -> None:
    """WIRE-90: tcp speak failure aborts the body and the next request is whole."""
    body = _BODIES[response_format]

    async def scenario() -> tuple[bool, httpx2.Response]:
        voice = FailingSpeaking(chunks_before_failing=chunks_before_failing)
        async with (
            serving_over_tcp(_gated_runtime(voice)) as server,
            _client(server) as client,
        ):
            try:
                await client.post(SPEECH_PATH, content=body)
            except httpx2.RemoteProtocolError:
                aborted = True
            else:
                aborted = False
            return aborted, await client.post(SPEECH_PATH, content=body)

    aborted, later = asyncio.run(scenario())

    assert aborted
    assert later.status_code == HTTP_200_OK
    voice_audio = b"".join(FakeSpeaking().pcm_chunks("", ""))
    _assert_whole_answer(response_format, later.content, voice_audio)


def test_tcp_opus_failure_after_audio_pages_sends_no_end_page() -> None:
    """WIRE-90: a failed Opus stream is cut off without the end page."""

    async def scenario() -> tuple[bool, bytes, httpx2.Response]:
        received = bytearray()
        async with (
            serving_over_tcp(
                _gated_runtime(
                    FailingSpeaking(
                        chunks_before_failing=1, chunk_seconds=_LONG_CHUNK_SECONDS
                    )
                )
            ) as server,
            _client(server) as client,
        ):
            try:
                async with client.stream(
                    "POST", SPEECH_PATH, content=_OPUS_BODY
                ) as response:
                    async for chunk in response.aiter_bytes():
                        received += chunk
            except httpx2.RemoteProtocolError:
                aborted = True
            else:
                aborted = False
            return (
                aborted,
                bytes(received),
                await client.post(SPEECH_PATH, content=_OPUS_BODY),
            )

    aborted, received, later = asyncio.run(scenario())

    assert aborted
    assert _audio_page_arrived(received)
    assert not any(page.flags & _OGG_END_OF_STREAM for page in _ogg_pages(received))
    assert later.status_code == HTTP_200_OK


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_tcp_serves_no_api_documentation(path: str) -> None:
    """WIRE-131: no API documentation is served."""

    async def scenario() -> httpx2.Response:
        async with (
            serving_over_tcp(_gated_runtime(GatedSpeaking())) as server,
            _client(server) as client,
        ):
            return await client.get(path)

    assert asyncio.run(scenario()).status_code == HTTP_404_NOT_FOUND


def test_tcp_refuses_a_speak_without_the_persons_token() -> None:
    """WIRE-20, WIRE-97: tcp refuses a speak without the persons token."""

    async def scenario() -> tuple[httpx2.Response, httpx2.Response]:
        headers = valid_headers()
        del headers[AUTHORIZATION_HEADER]
        async with (
            serving_over_tcp(_gated_runtime(FakeSpeaking())) as server,
            _client(server, headers) as client,
            _client(server) as admitted,
        ):
            return (
                await client.post(SPEECH_PATH, content=_SPEAK_BODY),
                await admitted.post(SPEECH_PATH, content=_SPEAK_BODY),
            )

    refused, served = asyncio.run(scenario())

    assert refused.status_code == HTTP_401_UNAUTHORIZED
    assert served.status_code == HTTP_200_OK


@pytest.mark.parametrize(
    ("method", "path", "status", "code"),
    [
        ("GET", "/health", HTTP_200_OK, None),
        ("POST", SPEECH_PATH, HTTP_401_UNAUTHORIZED, RefusalCode.UNAUTHORIZED),
        ("POST", "/v1/other", HTTP_404_NOT_FOUND, RefusalCode.NOT_FOUND),
        (
            "POST",
            "/v2/audio/speech",
            HTTP_400_BAD_REQUEST,
            RefusalCode.UNSUPPORTED_VERSION,
        ),
        ("GET", "/voices", HTTP_404_NOT_FOUND, None),
        ("GET", "/hear", HTTP_404_NOT_FOUND, None),
    ],
)
def test_tcp_answers_without_tokens_only_health_and_the_wires_plain_refusals(
    method: str, path: str, status: int, code: RefusalCode | None
) -> None:
    """WIRE-95, WIRE-96, WIRE-122, WIRE-130: only health and plain refusals answer."""

    async def scenario() -> httpx2.Response:
        async with (
            serving_over_tcp(_gated_runtime(FakeSpeaking())) as server,
            _client(server, {}) as client,
        ):
            return await client.request(method, path, content=_SPEAK_BODY)

    response = asyncio.run(scenario())

    if code is None:
        assert response.status_code == status
    else:
        assert_refusal(response, status, code, None)


def test_tcp_refuses_a_websocket_upgrade_off_the_wire() -> None:
    """WIRE-130: tcp refuses a websocket upgrade off the wire."""
    upgrade = (
        b"GET /hear HTTP/1.1\r\nHost: speech\r\nUpgrade: websocket\r\n"
        b"Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
        b"Sec-WebSocket-Key: " + base64.b64encode(os.urandom(16)) + b"\r\n\r\n"
    )

    async def scenario() -> bytes:
        async with serving_over_tcp(_gated_runtime(FakeSpeaking())) as server:
            host, port = _address(server)
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(upgrade)
            status_line = await asyncio.wait_for(
                reader.readline(), timeout=GATE_TIMEOUT_SECONDS
            )
            writer.close()
            return status_line

    status_line = asyncio.run(scenario())

    assert status_line.split()[1] != b"101"


def test_tcp_denies_a_realtime_upgrade_without_tokens_in_the_error_form() -> None:
    """WIRE-51, WIRE-97: a realtime upgrade without tokens is denied as 401."""

    async def scenario() -> tuple[int, dict[str, dict[str, object]]]:
        async with serving_over_tcp(_gated_runtime(FakeSpeaking())) as server:
            host, port = _address(server)
            url = f"ws://{host}:{port}{REALTIME_PATH}?intent=transcription"
            with pytest.raises(InvalidStatus) as denied:
                await connect(url)
            response = denied.value.response
            return response.status_code, json.loads(response.body)

    status, body = asyncio.run(scenario())

    assert_refusal_body(
        status, body, HTTP_401_UNAUTHORIZED, RefusalCode.UNAUTHORIZED, None
    )
