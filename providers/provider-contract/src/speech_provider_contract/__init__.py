"""The narrow binary protocol and worker loop shared by local providers.

It also owns the payload-free failure renderer the service's `log_failure` uses.
"""

import contextlib
import errno
import functools
import inspect
import logging
import os
import queue
import re
import struct
import sys
import threading
import traceback
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Final, TypeGuard

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

logger = logging.getLogger(__name__)

QWEN_SAMPLE_RATE = 24_000
# The only device the Qwen adapter loads on, whatever device hearing uses.
QWEN_DEVICE = "cuda"
QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"
QWEN_ARTIFACTS = (
    ".gitattributes",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
)


class QwenSpeaker(StrEnum):
    """The closed custom-voice values shared across the process boundary."""

    VIVIAN = "Vivian"
    SERENA = "Serena"
    UNCLE_FU = "Uncle_Fu"
    DYLAN = "Dylan"
    ERIC = "Eric"
    RYAN = "Ryan"
    AIDEN = "Aiden"
    ONO_ANNA = "Ono_Anna"
    SOHEE = "Sohee"


MAX_FRAME_LENGTH = 1_048_585
FRAME_OVERHEAD = 9
MIN_PCM_BYTES = 2
MAX_PCM_BYTES = 1_048_576
MAX_INSTRUCTION_BYTES = 2_000
MAX_TEXT_BYTES = 16_384
# The language code, the instructions' byte length, the instructions, the text.
MIN_SYNTHESIS_BYTES = 4
MAX_SYNTHESIS_BYTES = 3 + MAX_INSTRUCTION_BYTES + MAX_TEXT_BYTES
SYNTHESIS_LANGUAGES: Final = {"de": 1, "en": 2}
_INSTRUCTION_LENGTH = struct.Struct("!H")
MAX_FAILURE_CAUSE_BYTES = 4096
_CAUSE_LINKS = 4
_CAUSE_FRAMES = 6
_CAUSE_LINE_BYTES = 256
_CAUSE_LABEL_BYTES = 128
_CAUSE_BYTES = re.compile(rb"[\n\x20-\x7e]*")
_UNPRINTABLE = re.compile(r"[^\x20-\x7e]")
# The C descriptors read an exception's own slots without running its code.
_EXCEPTION_CAUSE = BaseException.__dict__["__cause__"]
_EXCEPTION_CONTEXT = BaseException.__dict__["__context__"]
_EXCEPTION_TRACEBACK = BaseException.__dict__["__traceback__"]
_OS_ERROR_ERRNO = OSError.__dict__["errno"]
_HEADER = struct.Struct("!IBQ")
_LENGTH = struct.Struct("!I")
_READY = struct.Struct("!HI")


class FrameKind(IntEnum):
    """The seven directions and terminal states of the provider protocol."""

    READY = 1
    SYNTHESIZE = 2
    CANCEL = 3
    PCM = 4
    COMPLETE = 5
    CANCELLED = 6
    FAILED = 7


class ProviderFailure(IntEnum):
    """The sole sanitized provider failure vocabulary."""

    PROVIDER_FAILURE = 1


class ProtocolError(ValueError):
    """A malformed private provider frame."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One framed provider message after its length prefix."""

    kind: FrameKind
    request_id: int
    payload: bytes


class FrameDecoder:
    """Incrementally decode complete frames without blocking on partial input."""

    def __init__(self) -> None:
        """Start with no partial frame bytes."""
        self._buffer = bytearray()

    def feed(self, data: bytes) -> None:
        """Append bytes after rejecting an invalid announced length promptly."""
        self._buffer.extend(data)
        self._validate_announced_length()

    def next_frame(self) -> Frame | None:
        """Return one complete frame, retaining any partial or following frame."""
        if len(self._buffer) < _LENGTH.size:
            return None
        (length,) = _LENGTH.unpack_from(self._buffer)
        _validate_frame_length(length)
        total = _LENGTH.size + length
        if len(self._buffer) < total:
            return None
        encoded = bytes(self._buffer[:total])
        del self._buffer[:total]
        return read_frame(BytesIO(encoded))

    def finish(self) -> None:
        """Reject EOF when it leaves an incomplete frame."""
        if self._buffer:
            raise ProtocolError

    def _validate_announced_length(self) -> None:
        if len(self._buffer) >= _LENGTH.size:
            (length,) = _LENGTH.unpack_from(self._buffer)
            _validate_frame_length(length)


def write_frame(stream: BinaryIO | None, frame: Frame) -> None:
    """Validate and write one complete network-order frame."""
    if stream is None:
        raise OSError
    _validate(frame)
    stream.write(
        _HEADER.pack(FRAME_OVERHEAD + len(frame.payload), frame.kind, frame.request_id)
    )
    stream.write(frame.payload)
    stream.flush()


def read_frame(stream: BinaryIO | None) -> Frame:
    """Read and validate one complete network-order frame."""
    if stream is None:
        raise EOFError
    length, raw_kind, request_id = _HEADER.unpack(_read_exact(stream, _HEADER.size))
    _validate_frame_length(length)
    try:
        kind = FrameKind(raw_kind)
    except ValueError as error:
        raise ProtocolError from error
    frame = Frame(kind, request_id, _read_exact(stream, length - FRAME_OVERHEAD))
    _validate(frame)
    return frame


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = bytearray()
    while len(value) < size:
        chunk = stream.read(size - len(value))
        if not chunk:
            raise EOFError
        value.extend(chunk)
    return bytes(value)


def _validate_frame_length(length: int) -> None:
    if length < FRAME_OVERHEAD or length > MAX_FRAME_LENGTH:
        raise ProtocolError


def _validate(frame: Frame) -> None:
    if frame.request_id < 0 or frame.request_id >= 1 << 64:
        raise ProtocolError
    payload = frame.payload
    if frame.kind is FrameKind.READY:
        valid = frame.request_id == 0 and _ready_sample_rate(payload) is not None
    elif frame.kind is FrameKind.SYNTHESIZE:
        valid = frame.request_id != 0 and _valid_synthesis(payload)
    elif frame.kind in {FrameKind.CANCEL, FrameKind.COMPLETE, FrameKind.CANCELLED}:
        valid = frame.request_id != 0 and not payload
    elif frame.kind is FrameKind.PCM:
        valid = frame.request_id != 0 and _valid_pcm(payload)
    else:
        valid = _valid_failure(frame.request_id, payload)
    if not valid:
        raise ProtocolError


@dataclass(frozen=True, slots=True)
class Synthesis:
    """What one SYNTHESIZE frame asks: the text, its language code, how to speak it."""

    language: str
    text: str
    instructions: str | None = None


def encode_synthesis(synthesis: Synthesis) -> bytes:
    """Encode `synthesis`, whose language is a key of `SYNTHESIS_LANGUAGES`."""
    instructions = (synthesis.instructions or "").encode("utf-8")
    return (
        bytes([SYNTHESIS_LANGUAGES[synthesis.language]])
        + _INSTRUCTION_LENGTH.pack(len(instructions))
        + instructions
        + synthesis.text.encode("utf-8")
    )


def decode_synthesis(payload: bytes) -> Synthesis:
    """Decode a SYNTHESIZE payload; ProtocolError when it is none."""
    if not MIN_SYNTHESIS_BYTES <= len(payload) <= MAX_SYNTHESIS_BYTES:
        raise ProtocolError
    languages = {code: language for language, code in SYNTHESIS_LANGUAGES.items()}
    language = languages.get(payload[0])
    (instruction_bytes,) = _INSTRUCTION_LENGTH.unpack_from(payload, 1)
    text_start = 3 + instruction_bytes
    if (
        language is None
        or instruction_bytes > MAX_INSTRUCTION_BYTES
        or not 0 < len(payload) - text_start <= MAX_TEXT_BYTES
    ):
        raise ProtocolError
    try:
        instructions = payload[3:text_start].decode("utf-8")
        text = payload[text_start:].decode("utf-8")
    except UnicodeDecodeError:
        raise ProtocolError from None
    return Synthesis(language, text, instructions or None)


def _valid_synthesis(payload: bytes) -> bool:
    try:
        decode_synthesis(payload)
    except ProtocolError:
        return False
    return True


def _valid_pcm(payload: bytes) -> bool:
    return MIN_PCM_BYTES <= len(payload) <= MAX_PCM_BYTES and len(payload) % 2 == 0


def _valid_failure(request_id: int, payload: bytes) -> bool:
    if payload[:1] != bytes([ProviderFailure.PROVIDER_FAILURE]):
        return False
    cause = payload[1:]
    return not cause or (
        request_id != 0
        and len(cause) <= MAX_FAILURE_CAUSE_BYTES
        and _CAUSE_BYTES.fullmatch(cause) is not None
    )


@dataclass(frozen=True, slots=True)
class ProviderFunctions:
    """Provider callbacks behind the shared command loop."""

    load: Callable[[str, Path], object]
    synthesize: Callable[[object, str, str, str | None], Iterable[bytes]]
    prepare: Callable[[object], None] | None = None


def _load_model_and_signal_ready(
    *,
    device: str,
    cache: str,
    stream: BinaryIO,
    sample_rate: int,
    functions: ProviderFunctions,
) -> object | None:
    """Load and prepare the model, then announce readiness; None on failure."""
    try:
        model = functions.load(device, Path(cache))
        if functions.prepare is not None:
            functions.prepare(model)
        _discard_stderr()
        write_frame(stream, Frame(FrameKind.READY, 0, ready_payload(sample_rate)))
    except Exception:
        logger.exception("provider failed to load or prepare the model")
        _failed(stream, 0)
        return None
    return model


def _discard_stderr() -> None:
    """Stop the service-owned startup file from receiving text after READY."""
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
    finally:
        os.close(null)


def serve_provider(
    *,
    device: str,
    cache: str,
    protocol_stdout: int,
    sample_rate: int,
    functions: ProviderFunctions,
) -> int:
    """Load once, optionally prepare, then process serialized commands."""
    stream = os.fdopen(protocol_stdout, "wb", buffering=0)
    model = _load_model_and_signal_ready(
        device=device,
        cache=cache,
        stream=stream,
        sample_rate=sample_rate,
        functions=functions,
    )
    try:
        if model is not None:
            commands: queue.Queue[Frame | BaseException] = queue.Queue()
            reader_stopped = threading.Event()
            reader = threading.Thread(
                target=_read_commands,
                args=(commands, reader_stopped),
                name="speech-provider-input",
                daemon=True,
            )
            reader.start()
            try:
                serve_commands(commands, stream, functions.synthesize, model)
            finally:
                if reader_stopped.is_set():
                    reader.join()
    finally:
        stream.close()
    return 1


def _read_commands(
    commands: queue.Queue[Frame | BaseException], stopped: threading.Event
) -> None:
    try:
        while True:
            commands.put(read_frame(sys.stdin.buffer))
    except (EOFError, ProtocolError, OSError) as error:
        stopped.set()
        commands.put(error)


def serve_commands(
    commands: queue.Queue[Frame | BaseException],
    stream: BinaryIO,
    synthesize: Callable[[object, str, str, str | None], Iterable[bytes]],
    model: object,
) -> None:
    """Serve validated commands until input or synthesis fails."""
    previous_terminal = 0
    while True:
        command = commands.get()
        if isinstance(command, BaseException):
            _failed(stream, 0)
            return
        if command.kind is FrameKind.CANCEL and command.request_id == previous_terminal:
            continue
        if (
            command.kind is not FrameKind.SYNTHESIZE
            or command.request_id <= previous_terminal
        ):
            _failed(stream, 0)
            return
        terminal = _synthesize_request(
            commands,
            stream,
            functools.partial(synthesize, model),
            command,
            previous_terminal,
        )
        if terminal is None:
            return
        write_frame(stream, Frame(terminal, command.request_id, b""))
        previous_terminal = command.request_id


def _synthesize_request(
    commands: queue.Queue[Frame | BaseException],
    stream: BinaryIO,
    synthesize: Callable[[str, str, str | None], Iterable[bytes]],
    command: Frame,
    previous_terminal: int,
) -> FrameKind | None:
    """Stream one request; return its terminal kind, or None after FAILED."""
    request_id = command.request_id
    synthesis = decode_synthesis(command.payload)
    iterator: Iterable[bytes] | None = None
    try:
        cancelled = False
        iterator = iter(
            synthesize(synthesis.text, synthesis.language, synthesis.instructions)
        )
        for pcm in iterator:
            cancelled = _consume_cancellation(commands, request_id, previous_terminal)
            if cancelled:
                break
            write_frame(stream, Frame(FrameKind.PCM, request_id, pcm))
        if not cancelled:
            cancelled = _consume_cancellation(commands, request_id, previous_terminal)
        terminal = FrameKind.CANCELLED if cancelled else FrameKind.COMPLETE
    except Exception as error:  # noqa: BLE001 - the cause crosses in the FAILED frame
        _fail_request(stream, request_id, iterator, error)
        return None
    try:
        _close_iterator(iterator)
    except Exception as error:  # noqa: BLE001 - the cause crosses in the FAILED frame
        _fail_request(stream, request_id, None, error)
        return None
    return terminal


def _consume_cancellation(
    commands: queue.Queue[Frame | BaseException],
    request_id: int,
    previous_terminal: int,
) -> bool:
    cancelled = False
    while True:
        try:
            command = commands.get_nowait()
        except queue.Empty:
            return cancelled
        if isinstance(command, BaseException):
            raise command
        if command.kind is not FrameKind.CANCEL:
            raise ProtocolError
        if command.request_id == previous_terminal:
            continue
        if command.request_id != request_id:
            raise ProtocolError
        cancelled = True


def ready_payload(sample_rate: int) -> bytes:
    """Encode the version-one worker declaration for one native PCM rate."""
    if sample_rate <= 0 or sample_rate >= 1 << 32:
        raise ValueError
    return _READY.pack(1, sample_rate)


def _ready_sample_rate(payload: bytes) -> int | None:
    if len(payload) != _READY.size:
        return None
    version, sample_rate = _READY.unpack(payload)
    return sample_rate if version == 1 and sample_rate > 0 else None


def ready_sample_rate(payload: bytes) -> int:
    """Return a validated version-one provider rate."""
    sample_rate = _ready_sample_rate(payload)
    if sample_rate is None:
        raise ProtocolError
    return sample_rate


def _close_iterator(iterator: Iterable[bytes] | None) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def _fail_request(
    stream: BinaryIO,
    request_id: int,
    iterator: Iterable[bytes] | None,
    error: BaseException,
) -> None:
    with contextlib.suppress(Exception):
        _close_iterator(iterator)
    _failed(stream, request_id, error)


def _failed(
    stream: BinaryIO, request_id: int, error: BaseException | None = None
) -> None:
    payload = bytes([ProviderFailure.PROVIDER_FAILURE])
    if error is not None:
        payload += payload_free_cause(error)
    with contextlib.suppress(OSError, ProtocolError):
        write_frame(stream, Frame(FrameKind.FAILED, request_id, payload))


@dataclass(frozen=True, slots=True)
class FailureLink:
    """One chained exception as it may be logged: its label and its code frames.

    Each frame is its location line, then its source line when one is known.
    """

    label: str
    frames: tuple[tuple[str, ...], ...]


def failure_links(error: BaseException) -> tuple[FailureLink, ...]:
    """Render each chained exception, outermost first, without what it carries.

    A message, argument, note, filename or local can hold the text a person
    spoke or heard, so only the type, the errno, an int error code and the code
    frames are read, each without running the exception's own code. Every
    character outside printable ASCII is escaped. This never raises.
    """
    links: list[FailureLink] = []
    seen: set[int] = set()
    link: BaseException | None = error
    while link is not None and id(link) not in seen:
        seen.add(id(link))
        links.append(FailureLink(_label(link), _frames(link)))
        link = _next_link(link)
    return tuple(links)


def payload_free_cause(error: BaseException) -> bytes:
    """Encode a failure's chain for its FAILED frame, nearest the root cause first.

    The first line names the links kept nearest the root cause; each kept
    link's label and innermost frames follow, root cause first, in whole lines
    within `MAX_FAILURE_CAUSE_BYTES`. A cause that cannot be built is empty.
    """
    return _guarded(lambda: _bounded_cause(failure_links(error)).encode("ascii"), b"")


def failure_cause(payload: bytes) -> str:
    """Return the cause a validated FAILED payload carries; empty when it has none."""
    return payload[1:].decode("ascii")


def _bounded_cause(links: tuple[FailureLink, ...]) -> str:
    kept = links[-_CAUSE_LINKS:][::-1]
    dropped = len(links) - len(kept)
    cause = " -> ".join(link.label[:_CAUSE_LABEL_BYTES] for link in kept)
    if dropped:
        cause += f" (+{dropped} outer)"
    for line in _block_lines(kept):
        if len(cause) + 1 + len(line) > MAX_FAILURE_CAUSE_BYTES:
            break
        cause += "\n" + line
    return cause


def _block_lines(links: tuple[FailureLink, ...]) -> Iterator[str]:
    for link in links:
        yield link.label[:_CAUSE_LINE_BYTES]
        for frame in link.frames[-_CAUSE_FRAMES:]:
            yield from (line[:_CAUSE_LINE_BYTES] for line in frame)


def _next_link(link: BaseException) -> BaseException | None:
    cause = _guarded(lambda: _EXCEPTION_CAUSE.__get__(link), None)
    if cause is not None:
        return cause
    return _guarded(lambda: _EXCEPTION_CONTEXT.__get__(link), None)


def _label(link: BaseException) -> str:
    kind = type(link)
    module = _exact_str(_guarded(lambda: kind.__module__, None))
    name = _exact_str(_guarded(lambda: kind.__qualname__, None))
    return _escaped(f"{module}.{name}{_errno_label(link)}{_error_code_label(link)}")


def _errno_label(link: BaseException) -> str:
    number = _guarded(lambda: _OS_ERROR_ERRNO.__get__(link), None)
    if not _is_bounded_int(number):
        return ""
    return f"[{errno.errorcode.get(number, f'errno {number}')}]"


def _error_code_label(link: BaseException) -> str:
    code = _guarded(lambda: inspect.getattr_static(link, "error_code", None), None)
    return f" error_code {code}" if _is_bounded_int(code) else ""


def _frames(link: BaseException) -> tuple[tuple[str, ...], ...]:
    summaries = _guarded(
        lambda: traceback.extract_tb(_EXCEPTION_TRACEBACK.__get__(link)), []
    )
    return tuple(_frame_lines(summary) for summary in summaries)


def _frame_lines(summary: traceback.FrameSummary) -> tuple[str, ...]:
    line = summary.lineno if type(summary.lineno) is int else "?"
    location = _escaped(f'  File "{summary.filename}", line {line}, in {summary.name}')
    source = _exact_str(_guarded(lambda: summary.line, None), "")
    return (location, _escaped(f"    {source}")) if source else (location,)


def _exact_str(value: object, fallback: str = "?") -> str:
    return value if type(value) is str else fallback


def _is_bounded_int(value: object) -> TypeGuard[int]:
    return type(value) is int and 0 <= value < 1 << 32


def _escaped(text: str) -> str:
    return _UNPRINTABLE.sub(
        lambda match: match.group().encode("unicode_escape").decode("ascii"), text
    )


def _guarded[T](read: Callable[[], T], fallback: T) -> T:
    try:
        return read()
    except Exception:  # noqa: BLE001 - a failure's rendering never fails; a broken read renders its fallback
        return fallback
