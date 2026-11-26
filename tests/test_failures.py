"""A logged failure names its chain and frames, never what may carry a sentence."""

from __future__ import annotations

import errno
import logging

import pytest

from speech_service.failures import (
    PayloadFreeError,
    PayloadFreeFailure,
    SpeakingFailedError,
    failure_fields,
    payload_free_chain,
)

_SENTENCE = "Ein Satz, den nur der Aufrufer kennt"


def _raise_at_depth(depth: int, error: Exception, cause: Exception | None) -> None:
    if depth:
        _raise_at_depth(depth - 1, error, cause)
    raise error from cause


def _raised(error: Exception, cause: Exception | None = None) -> Exception:
    try:
        _raise_at_depth(0, error, cause)
    except Exception as raised:  # noqa: BLE001 - the raised link is the test's subject
        return raised
    pytest.fail("the error was not raised")


def _logged(caplog: pytest.LogCaptureFixture, error: Exception) -> str:
    logger = logging.getLogger("tests.failures")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        logger.error(
            "speaking.failed",
            extra=failure_fields(error),
            exc_info=payload_free_chain(error),
        )
    (record,) = caplog.records
    assert not hasattr(record, "failure_chain")
    assert record.getMessage() == "speaking.failed"
    assert record.error_type == type(error).__name__
    return _rendered_exception(record)


def _rendered_exception(record: logging.LogRecord) -> str:
    """The chain note logline renders under the message-free exception's type."""
    rendered = logging.Formatter().formatException(record.exc_info)  # type: ignore[arg-type]
    return rendered.split("\n", 1)[1]


def test_every_link_and_every_frame_of_a_deep_chain_is_logged_outermost_first(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error: Exception | None = None
    for kind in (KeyError, ValueError, TypeError, LookupError, RuntimeError):
        try:
            _raise_at_depth(40, kind(_SENTENCE), error)
        except Exception as raised:  # noqa: BLE001 - each kind becomes the next link
            error = raised
    assert error is not None

    message = _logged(caplog, error)

    lines = message.split("\n")
    labels = [line for line in lines if line.startswith("builtins.")]
    assert labels == [
        "builtins.RuntimeError",
        "builtins.LookupError",
        "builtins.TypeError",
        "builtins.ValueError",
        "builtins.KeyError",
    ]
    assert message.count(", in _raise_at_depth\n") == 5 * 41
    assert _SENTENCE not in message


@pytest.mark.parametrize(
    ("error", "label"),
    [
        (OSError(errno.EROFS, _SENTENCE, _SENTENCE), "builtins.OSError[EROFS]"),
        (OSError(4_000_000, _SENTENCE), "builtins.OSError[errno 4000000]"),
        (OSError(_SENTENCE, "x"), "builtins.OSError"),
    ],
    ids=["named", "unknown", "sentence"],
)
def test_an_os_error_is_labelled_by_its_errno_alone(
    caplog: pytest.LogCaptureFixture, error: OSError, label: str
) -> None:
    lines = _logged(caplog, _raised(error)).split("\n")

    assert lines[0] == label
    assert _SENTENCE not in "\n".join(lines)


@pytest.mark.parametrize(
    ("error_code", "suffix"),
    [(209, " error_code 209"), (2**32, ""), (-1, ""), (_SENTENCE, "")],
    ids=["int", "too-large", "negative", "sentence"],
)
def test_an_error_code_is_labelled_only_when_it_is_a_bounded_int(
    caplog: pytest.LogCaptureFixture, error_code: object, suffix: str
) -> None:
    error = RuntimeError(_SENTENCE)
    error.error_code = error_code  # type: ignore[attr-defined]

    lines = _logged(caplog, _raised(error)).split("\n")

    assert lines[0] == f"builtins.RuntimeError{suffix}"


def test_control_characters_in_a_type_name_are_escaped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    kind = type("Broken\tLine\nName\xe9", (Exception,), {})

    message = _logged(caplog, _raised(kind(_SENTENCE)))

    lines = message.split("\n")
    assert lines[0].endswith(".Broken\\tLine\\nName\\xe9")
    assert all(" " <= character <= "~" for line in lines for character in line)


def test_a_stopping_layer_logs_the_note_a_payload_free_error_carries_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("tests.failures")
    note = "ModuleNotFoundError: No module named 'vllm_omni'"
    error = PayloadFreeError("provider failed")
    error.add_note(note)

    with (
        caplog.at_level(logging.WARNING, logger=logger.name),
        pytest.raises(SpeakingFailedError),
        PayloadFreeFailure(logger, "speaking.failed", reraise_as=SpeakingFailedError),
    ):
        raise error

    assert caplog.text.count(note) == 1
    assert all(not hasattr(record, "failure_chain") for record in caplog.records)


def test_a_stopping_layer_never_logs_the_notes_of_an_ordinary_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("tests.failures")
    error = RuntimeError("x")
    error.add_note(_SENTENCE)

    with (
        caplog.at_level(logging.WARNING, logger=logger.name),
        PayloadFreeFailure(logger, "speaking.failed"),
    ):
        raise error

    assert _SENTENCE not in caplog.text


def test_a_failure_logs_the_chain_once_as_a_message_free_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("tests.failures")
    error = _raised(KeyError(_SENTENCE))

    with caplog.at_level(logging.ERROR, logger=logger.name):
        logger.error(
            "speaking.failed",
            extra=failure_fields(error),
            exc_info=payload_free_chain(error),
        )

    (record,) = caplog.records
    assert isinstance(record.exc_info, tuple)
    assert isinstance(record.exc_info[1], PayloadFreeError)
    formatted = logging.Formatter().format(record)
    assert formatted.count("builtins.KeyError") == 1
    assert _SENTENCE not in formatted
    assert _SENTENCE not in vars(record).get("exc_text", "")
    assert record.error_type == "KeyError"
