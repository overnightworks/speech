"""A logged failure names its chain and frames, never what may carry a sentence."""

from __future__ import annotations

import errno
import logging

import pytest

from speech_service.failures import log_failure

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
        log_failure(logger, "speaking failed", error)
    (record,) = caplog.records
    assert record.exc_text is None
    return record.getMessage()


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
    assert lines[0] == "speaking failed"
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

    assert lines[1] == label
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

    assert lines[1] == f"builtins.RuntimeError{suffix}"


def test_control_characters_in_a_type_name_are_escaped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    kind = type("Broken\tLine\nName\xe9", (Exception,), {})

    message = _logged(caplog, _raised(kind(_SENTENCE)))

    lines = message.split("\n")
    assert lines[1].endswith(".Broken\\tLine\\nName\\xe9")
    assert all(" " <= character <= "~" for line in lines for character in line)
