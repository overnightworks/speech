"""One live hearing session: its turns, its items' terminals and how it ends.

The session is the one owner of every item's terminal event. All its state moves
under one lock, so a message, a poll and a decode's result never interleave; a
decode runs in the hearing worker and its result comes back as a task.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from logline import StepTimer, current_traceparent, unit_trace
from starlette.status import WS_1000_NORMAL_CLOSURE

from speech_client.realtime_wire import (
    AudioAppend,
    AudioCommit,
    AudioCommitted,
    ClientEventError,
    DurationUsage,
    ItemCreated,
    SessionCreated,
    SessionUpdate,
    SessionUpdated,
    SoundDetected,
    SpeechStarted,
    SpeechStopped,
    TranscriptCompleted,
    TranscriptDelta,
    TranscriptFailed,
    TranscriptionSession,
    TranscriptLanguage,
    UserMessageItem,
    error_event,
    item_failure,
    new_item_id,
    parse_client_event,
)
from speech_client.wire import DRAIN_CLOSE_CODE, Language, RefusalCode
from speech_service.failures import log_failure
from speech_service.hearing_session import (
    POLL_WAIT_SECONDS,
    HearingSession,
    SpeechStart,
    TurnEnd,
    TurnEndedBy,
)
from speech_service.pcm import BYTES_PER_SAMPLE, HEARING_SAMPLE_RATE, Resampler

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Sequence
    from concurrent.futures import Future

    from pydantic import BaseModel

    from speech_service.admission_gate import Lease
    from speech_service.hearing import Heard, Hearing, HearingEngine
    from speech_service.sound_events import Scores, SoundStream, SoundWindow

logger = logging.getLogger(__name__)

__all__ = ["POLL_WAIT_SECONDS", "LiveSession", "poll_forever"]

_PCM_BYTES_PER_SECOND: Final = HEARING_SAMPLE_RATE * BYTES_PER_SAMPLE
_MAX_ENDED_TURNS: Final = 2


@dataclass(slots=True)
class _Final:
    """A turn's audio to decode.

    `hearing` times it from the turn's end to the transcript. The final runs in its
    own task, so it carries the turn's `traceparent` to re-enter.
    """

    item_id: str
    pcm: bytes
    voiced_seconds: float
    deadline: float
    hearing: StepTimer
    traceparent: str | None
    emitted: asyncio.Event = field(default_factory=asyncio.Event)


def _samples_of(encoded: str) -> bytes | None:
    """The PCM16 in `encoded`, or `None` for invalid base64 or an odd byte count.

    Non-ASCII text and malformed padding both raise `ValueError`.
    """
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError:
        return None
    return None if len(raw) % BYTES_PER_SAMPLE else raw


class LiveSession:
    """The state machine of one Realtime transcription session."""

    def __init__(
        self,
        hearing: Hearing,
        *,
        final_wait_seconds: float,
        silence_seconds: float,
        send: Callable[[str], Awaitable[None]],
        close: Callable[[int], Awaitable[None]],
    ) -> None:
        """Hear through `hearing`; send events with `send` and end with `close(code)`.

        A final may wait `final_wait_seconds` from its turn's end; a turn ends after
        `silence_seconds` without speech. Deadlines and
        the partials' pace follow the clock of the hearing's gate.
        """
        self._hearing = hearing
        self._final_wait_seconds = final_wait_seconds
        self._send = send
        self._close = close
        self._clock = hearing.gate.clock
        self._scheduler = HearingSession(
            self._clock.now, hearing.open_detector(), silence_seconds
        )
        self._resampler = Resampler()
        self._sounds: SoundStream | None = None
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._languages = (Language.DE,)
        self._configured = False
        self._audio_seen = False
        self._ended = False
        self._open_item: str | None = None
        self._last_item: str | None = None
        self._unfinished: dict[str, None] = {}
        self._turn_number = 0
        self._previous_words: list[str] = []
        self._sent_words = 0
        self._silence_wait: StepTimer | None = None
        self._finals: deque[_Final] = deque()

    async def start(self) -> None:
        """Send `session.created`, the first event of every session."""
        await self._emit(
            SessionCreated(session=TranscriptionSession(type="transcription"))
        )

    async def on_message(self, text: str | None) -> None:
        """Take one client message; `None` is a binary frame, which is refused."""
        async with self._lock:
            if self._ended:
                return
            try:
                event = parse_client_event(text or "")
            except ClientEventError as error:
                await self._end_locked(
                    RefusalCode.INVALID_REQUEST, error.param, error.event_id
                )
                return
            if isinstance(event, SessionUpdate):
                await self._on_update(event)
            elif isinstance(event, AudioAppend):
                await self._on_append(event)
            else:
                await self._on_commit(event)

    async def on_tick(self) -> None:
        """Decode a partial when due; a draining hearing ends the session at a turn."""
        async with self._lock:
            if self._ended:
                return
            if self._hearing.draining:
                await self._drain()
                if self._ended:
                    return
            await self._decode_partial()

    async def _drain(self) -> None:
        """Cut the open turn when the drain's turn bound passed; close when none is.

        The close is `1012` with no `error`: nothing failed, and every announced
        item has completed. A turn or final still unfinished at the drain's bound
        fails its items, then the session closes `1012` with an `error`.
        """
        self._scheduler.stop_opening_turns()
        if self._scheduler.turn_open and self._hearing.drain_turn_over():
            turn = self._scheduler.cut()
            if turn is not None:
                await self._apply([turn])
        if self._ended:
            return
        if not self._scheduler.turn_open and not self._finals:
            self._ended = True
            self._cancel_decodes()
            await self._close(DRAIN_CLOSE_CODE)
        elif self._hearing.drain_left(self._final_wait_seconds) == 0:
            await self._end_locked(RefusalCode.UNAVAILABLE, close_code=DRAIN_CLOSE_CODE)

    async def on_disconnect(self) -> None:
        """The client left: cancel what is outstanding and send nothing."""
        self._ended = True
        self._cancel_decodes()
        await self.settled()

    async def settled(self) -> None:
        """Wait until every decode task has finished or been cancelled."""
        while pending := [task for task in self._tasks if not task.done()]:
            await asyncio.wait(pending)

    async def _decode_partial(self) -> None:
        open_turn = self._scheduler.partial_due()
        if open_turn is None:
            return
        lease = self._hearing.gate.try_acquire()
        if lease is None:
            return
        try:
            future = self._hearing.submit(
                lease, self._transcription(open_turn.pcm, open_turn.voiced_seconds)
            )
        except RuntimeError:
            await self._end_locked(RefusalCode.UNAVAILABLE)
            return
        self._scheduler.partial_started()
        task = self._spawn(self._run_partial(future, self._turn_number))
        task.add_done_callback(lambda _done: future.cancel())

    async def _start_sound_hop(self) -> None:
        """Score the latest due hop, unless one is in flight or the session ended."""
        window = (
            None if self._sounds is None or self._ended else self._sounds.next_window()
        )
        if window is None:
            return
        try:
            future = self._hearing.submit_sound(window)
        except RuntimeError:
            await self._end_locked(RefusalCode.UNAVAILABLE)
            return
        task = self._spawn(self._run_sound_hop(future, window))
        task.add_done_callback(lambda _done: future.cancel())

    async def _run_sound_hop(self, future: Future[Scores], window: SoundWindow) -> None:
        try:
            scores = await asyncio.wrap_future(future)
        except Exception as error:  # noqa: BLE001 - logged by type and never by message
            log_failure(logger, "sound.failed", error)
            await self._end(RefusalCode.UNAVAILABLE)
            return
        async with self._lock:
            if self._ended or self._sounds is None:
                return
            for sound in self._sounds.scored(window, scores):
                logger.info(
                    "sound.detected",
                    extra={
                        "sound": sound.name.value,
                        "score": sound.score,
                        "audio_start_ms": sound.audio_start_ms,
                    },
                )
                await self._emit(
                    SoundDetected(
                        sound=sound.name,
                        audio_start_ms=sound.audio_start_ms,
                        audio_end_ms=sound.audio_end_ms,
                    )
                )
                if self._ended:
                    return
            await self._start_sound_hop()

    async def _on_update(self, event: SessionUpdate) -> None:
        if self._configured or self._audio_seen:
            await self._end_locked(
                RefusalCode.INVALID_REQUEST, "session", event.event_id
            )
            return
        self._configured = True
        self._languages = event.session.audio.input.transcription.allowed
        session = event.session
        if session.audio.input.sound_detection is not None:
            self._sounds = self._hearing.open_sound_stream()
            if self._sounds is None:
                session = session.without_sound_detection()
        await self._emit(SessionUpdated(session=session))

    async def _on_append(self, event: AudioAppend) -> None:
        raw = _samples_of(event.audio)
        if raw is None:
            await self._end_locked(RefusalCode.INVALID_REQUEST, "audio", event.event_id)
            return
        self._audio_seen = True
        if self._hearing.draining:
            self._scheduler.stop_opening_turns()
        pcm = self._resampler.feed(raw)
        events = self._scheduler.add(pcm)
        self._watch_silence(events)
        await self._apply(events)
        if self._sounds is not None:
            self._sounds.add(pcm)
            await self._start_sound_hop()

    def _watch_silence(self, events: Sequence[SpeechStart | TurnEnd]) -> None:
        """Time the wait for the open turn's end from the first silent audio on."""
        if any(isinstance(event, TurnEnd) for event in events):
            return
        if not self._scheduler.silent:
            self._silence_wait = None
        elif self._silence_wait is None:
            self._silence_wait = StepTimer(
                "turn_end_wait.ended",
                logger,
                item_id=self._open_item_id(),
                ended_by=TurnEndedBy.SILENCE.value,
            )

    async def _on_commit(self, event: AudioCommit) -> None:
        if self._hearing.draining:
            self._scheduler.stop_opening_turns()
        turn = self._scheduler.commit()
        if turn is None and self._hearing.draining:
            return
        if turn is None:
            await self._end_locked(RefusalCode.INVALID_REQUEST, None, event.event_id)
            return
        await self._apply([turn])

    async def _apply(self, events: Sequence[SpeechStart | TurnEnd]) -> None:
        """Settle every ended turn of `events` before the first send suspends."""
        outgoing: list[BaseModel] = []
        backlog = False
        for event in events:
            if isinstance(event, SpeechStart):
                outgoing.append(
                    SpeechStarted(
                        item_id=self._open_item_id(), audio_start_ms=event.audio_ms
                    )
                )
                continue
            ended, backlog = self._end_turn(event)
            outgoing += ended
            if backlog:
                break
        for message in outgoing:
            if self._ended:
                return
            await self._emit(message)
        if backlog and not self._ended:
            await self._end_locked(RefusalCode.BUSY)

    def _open_item_id(self) -> str:
        if self._open_item is None:
            self._open_item = new_item_id()
            self._unfinished[self._open_item] = None
        return self._open_item

    def _end_turn(self, turn: TurnEnd) -> tuple[list[BaseModel], bool]:
        """Queue the turn's final; return what to send and whether the queue is full."""
        deadline = self._clock.now() + self._final_wait_seconds
        item_id = self._open_item_id()
        self._open_item = None
        self._turn_number += 1
        self._previous_words = []
        self._sent_words = 0
        with unit_trace(None):
            self._end_the_wait(item_id, turn.ended_by)
            traceparent = current_traceparent()
        backlog = len(self._finals) == _MAX_ENDED_TURNS
        if not backlog:
            hearing = StepTimer(
                "hearing.ended",
                logger,
                item_id=item_id,
                audio_end_ms=turn.end_ms,
                silence_ms=turn.silence_ms,
            )
            self._queue_final(
                _Final(
                    item_id,
                    turn.pcm,
                    turn.voiced_seconds,
                    deadline,
                    hearing,
                    traceparent,
                )
            )
        previous, self._last_item = self._last_item, item_id
        outgoing: list[BaseModel] = []
        if turn.ended_by is not TurnEndedBy.CLIENT:
            outgoing.append(SpeechStopped(item_id=item_id, audio_end_ms=turn.end_ms))
        outgoing.append(AudioCommitted(item_id=item_id, previous_item_id=previous))
        outgoing.append(
            ItemCreated(previous_item_id=previous, item=UserMessageItem(id=item_id))
        )
        return outgoing, backlog

    def _end_the_wait(self, item_id: str, ended_by: TurnEndedBy) -> None:
        """Write the line of the wait that ended the turn, in the turn's trace."""
        waiting, self._silence_wait = self._silence_wait, None
        if waiting is None or ended_by is not TurnEndedBy.SILENCE:
            waiting = StepTimer(
                "turn_end_wait.ended", logger, item_id=item_id, ended_by=ended_by.value
            )
        waiting.end()

    def _queue_final(self, final: _Final) -> None:
        ahead = self._finals[-1] if self._finals else None
        self._finals.append(final)
        self._spawn(self._run_final(final, ahead))

    async def _run_final(self, final: _Final, ahead: _Final | None) -> None:
        refusal: RefusalCode | None = None
        decode_failure: Exception | None = None
        with unit_trace(final.traceparent):
            try:
                async with final.hearing:
                    lease = await self._hearing.gate.acquire_final(final.deadline)
                    if lease is None:
                        refusal = RefusalCode.BUSY
                        final.hearing.refuse(refusal.value)
                    else:
                        try:
                            heard = await self._decode(final, lease)
                        except Exception as error:
                            decode_failure = error
                            raise
                        await self._deliver(final, heard, ahead)
            except Exception as error:
                if error is not decode_failure:
                    raise
                log_failure(logger, "hearing.failed", error, level=logging.WARNING)
                refusal = RefusalCode.UNAVAILABLE
        if refusal is not None:
            await self._end(refusal)

    async def _decode(self, final: _Final, lease: Lease) -> Heard:
        future = self._hearing.submit(
            lease, self._transcription(final.pcm, final.voiced_seconds)
        )
        return await asyncio.wrap_future(future)

    async def _deliver(self, final: _Final, heard: Heard, ahead: _Final | None) -> None:
        if ahead is not None:
            await ahead.emitted.wait()
        async with self._lock:
            await self._complete(final, heard)

    async def _complete(self, final: _Final, heard: Heard) -> None:
        seconds = len(final.pcm) / _PCM_BYTES_PER_SECOND
        del self._unfinished[final.item_id]
        await self._emit(
            TranscriptCompleted(
                item_id=final.item_id,
                transcript=heard.text,
                usage=DurationUsage(seconds=seconds),
                languages=(TranscriptLanguage(code=heard.language),),
                traceparent=final.traceparent,
            )
        )
        self._finals.popleft()
        final.emitted.set()
        logger.info(
            "hearing.measured",
            extra={
                "item_id": final.item_id,
                "audio_ms": round(seconds * 1000),
                "voiced_ms": round(final.voiced_seconds * 1000),
                "no_speech_prob": heard.no_speech_prob,
                "avg_logprob": heard.avg_logprob,
            },
        )

    async def _run_partial(self, future: Future[Heard], turn_number: int) -> None:
        try:
            heard = await asyncio.wrap_future(future)
        except Exception as error:  # noqa: BLE001 - logged by type and never by message
            log_failure(logger, "hearing.failed", error)
            await self._end(RefusalCode.UNAVAILABLE)
            return
        async with self._lock:
            if self._ended or turn_number != self._turn_number:
                return
            await self._emit_agreed(heard.text)

    async def _emit_agreed(self, text: str) -> None:
        words = text.split()
        agreed = 0
        for before, now in zip(self._previous_words, words, strict=False):
            if before != now:
                break
            agreed += 1
        self._previous_words = words
        if agreed <= self._sent_words:
            return
        piece = " ".join(words[self._sent_words : agreed])
        delta = f" {piece}" if self._sent_words else piece
        self._sent_words = agreed
        await self._emit(TranscriptDelta(item_id=self._open_item_id(), delta=delta))

    def _transcription(
        self, pcm: bytes, voiced_seconds: float
    ) -> Callable[[HearingEngine], Heard]:
        languages = self._languages

        def job(engine: HearingEngine) -> Heard:
            return engine.transcribe(pcm, languages, voiced_seconds)

        return job

    async def _end(self, code: RefusalCode) -> None:
        async with self._lock:
            await self._end_locked(code)

    async def _end_locked(
        self,
        code: RefusalCode,
        param: str | None = None,
        event_id: str | None = None,
        *,
        close_code: int = WS_1000_NORMAL_CLOSURE,
    ) -> None:
        """Fail every announced unfinished item, then send `error`, then close."""
        logger.info("session.refused", extra={"reason": code.value})
        self._ended = True
        self._cancel_decodes()
        failure = item_failure(code)
        for item_id in self._unfinished:
            await self._emit(TranscriptFailed(item_id=item_id, error=failure))
        self._unfinished.clear()
        await self._emit(error_event(code, param, event_id))
        await self._close(close_code)

    def _cancel_decodes(self) -> None:
        for task in self._tasks:
            if task is not asyncio.current_task():
                task.cancel()

    def _spawn(self, work: Coroutine[object, object, None]) -> asyncio.Task[None]:
        # Eager, so a final stands in the gate's queue before any release can pass it.
        task = asyncio.create_task(work, eager_start=True)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _emit(self, event: BaseModel) -> None:
        await self._send(event.model_dump_json())


async def poll_forever(
    session: LiveSession,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Poll `session` every C2 interval until cancelled."""
    while True:
        await sleep(POLL_WAIT_SECONDS)
        await session.on_tick()
