"""Failures that are logged without the payload that may have caused them."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from speech_provider_contract import failure_links

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Self


class SpeakingFailedError(Exception):
    """A speaking stream failed; the cause is logged, never carried."""

    def __init__(self) -> None:
        """Carry only the fixed message."""
        super().__init__("speaking failed")


class PayloadFreeError(RuntimeError):
    """An error whose message and notes never hold what a person spoke or heard."""


def log_failure(
    logger: logging.Logger,
    event: str,
    error: BaseException,
    *,
    level: int = logging.ERROR,
    **fields: str,
) -> None:
    """Log `event` with `fields`, the error class and its rendered chain.

    A failure that a timed step's own line ends as failed passes `level` below
    ERROR, so the failure is one error line, the step's.

    An exception message, argument or local variable can hold the text a person
    spoke or heard, so none of them is ever logged. A `PayloadFreeError` is the
    exception: its message and notes are payload-free by construction, so the
    logger renders it through `exc_info`, once.
    """
    lines: list[str] = []
    for link in failure_links(error):
        lines.append(link.label)
        lines.extend(line for frame in link.frames for line in frame)
    logger.log(
        level,
        event,
        extra={
            **fields,
            "error_type": type(error).__name__,
            "failure_chain": "\n".join(lines),
        },
        exc_info=error if isinstance(error, PayloadFreeError) else None,
    )


class PayloadFreeFailure:
    """Log an Exception that leaves the block by type and frames, never its message.

    BaseException (cancellation, generator close) and the `expected` exceptions
    pass through untouched. Any other Exception is logged and then either
    raised as `reraise_as` without its cause, or swallowed and recorded in
    `failed` so the caller can close with a fixed reason.
    """

    def __init__(
        self,
        logger: logging.Logger,
        event: str,
        *,
        expected: tuple[type[Exception], ...] = (),
        level: int = logging.ERROR,
        reraise_as: type[Exception] | None = None,
    ) -> None:
        """Name where and how a failure is logged and what becomes of it."""
        self._logger = logger
        self._event = event
        self._expected = expected
        self._level = level
        self._reraise_as = reraise_as
        self.failed = False

    def __enter__(self) -> Self:
        """Guard the block."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Log an unexpected Exception; swallow it unless it is re-raised."""
        del exception_type, traceback
        if not isinstance(error, Exception) or isinstance(error, self._expected):
            return False
        log_failure(
            self._logger,
            self._event,
            error,
            level=self._level,
        )
        self.failed = True
        if self._reraise_as is not None:
            raise self._reraise_as from None
        return True
