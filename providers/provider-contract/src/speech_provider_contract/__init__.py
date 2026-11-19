"""The Qwen pins the service and its Qwen environment share.

It also owns the payload-free failure renderer the service's `log_failure` uses.
"""

from __future__ import annotations

import errno
import inspect
import re
import traceback
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypeGuard

if TYPE_CHECKING:
    from collections.abc import Callable

QWEN_SAMPLE_RATE = 24_000
QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"
QWEN_LANGUAGE_NAMES = {"de": "German", "en": "English"}
"""The languages Qwen speaks, by the name its server takes."""
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


_UNPRINTABLE = re.compile(r"[^\x20-\x7e]")
# The C descriptors read an exception's own slots without running its code.
_EXCEPTION_CAUSE = BaseException.__dict__["__cause__"]
_EXCEPTION_CONTEXT = BaseException.__dict__["__context__"]
_EXCEPTION_TRACEBACK = BaseException.__dict__["__traceback__"]
_OS_ERROR_ERRNO = OSError.__dict__["errno"]


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
