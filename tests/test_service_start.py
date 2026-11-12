"""The runtime starts its Piper voices or fails loudly, and shutdown wins races."""

import threading
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from speech_client.wire import VoiceId
from speech_service.clock import LoopClock
from speech_service.hearing import Hearing
from speech_service.service import Runtime, RuntimeDependencies, SpeakingEngine
from speech_service.voices import PIPER_VOICES
from tests.support.fakes_support import FakeSpeaking
from tests.support.hear_support import FakeHearing
from tests.support.speak_support import app_over, post_speech, speak_body

if TYPE_CHECKING:
    from collections.abc import Callable

WAIT_SECONDS = 5


class _ClosableSpeaking(FakeSpeaking):
    def __init__(self, *, ready: bool = True, fail: bool = False) -> None:
        super().__init__(ready=ready, fail=fail)
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _loading(engine: SpeakingEngine) -> Runtime:
    return Runtime(
        {VoiceId.THORSTEN: None},
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: engine,
            missing_artifacts=lambda _voice: (),
        ),
    )


def test_a_voice_whose_artifacts_are_missing_stops_the_start_before_building_it() -> (
    None
):
    built: list[VoiceId] = []

    def build(voice: VoiceId) -> SpeakingEngine:
        built.append(voice)
        return FakeSpeaking()

    runtime = Runtime(
        {VoiceId.THORSTEN: None},
        dependencies=RuntimeDependencies(
            engine_factory=build,
            missing_artifacts=lambda _voice: ("voice.onnx",),
        ),
    )

    with pytest.raises(RuntimeError, match=VoiceId.THORSTEN.value):
        runtime.load()

    assert built == []
    assert runtime.loading is False
    assert runtime.capture(VoiceId.THORSTEN) is None


def test_a_runtime_without_an_engine_factory_refuses_to_start() -> None:
    runtime = Runtime({VoiceId.THORSTEN: None})

    with pytest.raises(RuntimeError, match="no engine factory"):
        runtime.load()


def test_a_voice_that_fails_to_load_names_the_voice_and_is_closed() -> None:
    engine = _ClosableSpeaking(fail=True)
    runtime = _loading(engine)

    with pytest.raises(RuntimeError, match=f"{VoiceId.THORSTEN.value} failed to load"):
        runtime.load()

    assert engine.closed == 1
    assert runtime.loading is False
    assert runtime.capture(VoiceId.THORSTEN) is None


def test_an_engine_that_cannot_be_closed_after_a_failed_load_ends_the_service() -> None:
    fatal = threading.Event()

    class StuckSpeaking(_ClosableSpeaking):
        def close(self) -> None:
            raise OSError

    runtime = _loading(StuckSpeaking(fail=True))
    runtime.set_fatal_callback(fatal.set)

    with pytest.raises(RuntimeError, match="failed to load"):
        runtime.load()

    assert fatal.is_set()


def test_the_started_engine_is_the_one_piper_speaks_with() -> None:
    engine = _ClosableSpeaking(ready=False)
    runtime = _loading(engine)

    runtime.load()

    assert runtime.capture(VoiceId.THORSTEN) is engine


def test_shutdown_winning_before_pending_registration_never_loads_or_publishes() -> (
    None
):
    constructed = threading.Event()
    release = threading.Event()

    class DeferredSpeaking(_ClosableSpeaking):
        loaded = False

        def load(self) -> None:
            self.loaded = True

    engine = DeferredSpeaking(ready=False)

    def build(_voice: VoiceId) -> DeferredSpeaking:
        constructed.set()
        assert release.wait(timeout=WAIT_SECONDS)
        return engine

    runtime = Runtime(
        {VoiceId.THORSTEN: None},
        dependencies=RuntimeDependencies(
            engine_factory=build, missing_artifacts=lambda _voice: ()
        ),
    )
    starting = threading.Thread(target=runtime.load)
    starting.start()
    assert constructed.wait(timeout=WAIT_SECONDS)

    runtime.begin_shutdown()
    runtime.close()
    release.set()
    starting.join(timeout=WAIT_SECONDS)

    assert not starting.is_alive()
    assert engine.loaded is False
    assert engine.closed == 1
    assert runtime.capture(VoiceId.THORSTEN) is None


def test_shutdown_during_the_load_closes_the_engine_once_and_publishes_nothing() -> (
    None
):
    loading = threading.Event()
    release = threading.Event()

    class SlowSpeaking(_ClosableSpeaking):
        def load(self) -> None:
            loading.set()
            assert release.wait(timeout=WAIT_SECONDS)
            self.ready = True

    engine = SlowSpeaking(ready=False)
    runtime = _loading(engine)
    starting = threading.Thread(target=runtime.load)
    starting.start()
    assert loading.wait(timeout=WAIT_SECONDS)

    runtime.begin_shutdown()
    release.set()
    starting.join(timeout=WAIT_SECONDS)
    runtime.close()

    assert not starting.is_alive()
    assert engine.closed == 1
    assert runtime.capture(VoiceId.THORSTEN) is None


@pytest.mark.parametrize("surface", ["health", "wire"])
def test_a_voice_that_stops_being_ready_is_closed_once_and_not_replaced(
    surface: str,
) -> None:
    engine = _ClosableSpeaking()
    runtime = _loading(engine)
    runtime.load()
    engine.ready = False

    if surface == "health":
        runtime.health()
    else:
        refused = post_speech(TestClient(app_over(runtime)), speak_body())
        assert refused.status_code == HTTP_503_SERVICE_UNAVAILABLE

    assert runtime.capture(VoiceId.THORSTEN) is None
    assert engine.closed == 1


class _RecordedSpeaking(_ClosableSpeaking):
    """Speaking of `voice` that writes its load and its close into `events`."""

    def __init__(
        self, events: list[str], voice: VoiceId, *, fail_close: bool = False
    ) -> None:
        super().__init__(ready=False)
        self._events = events
        self._voice = voice
        self._fail_close = fail_close

    def load(self) -> None:
        super().load()
        self._events.append(f"{self._voice.value} loaded")

    def close(self) -> None:
        self._events.append(f"{self._voice.value} closed")
        if self._fail_close:
            raise OSError


class _RecordedHearing(FakeHearing):
    """Hearing that writes its load and its close into the shared `events`."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def load(self) -> None:
        self._events.append("hearing loaded")

    def close(self) -> None:
        self._events.append("hearing closed")


def _offering(
    voices: dict[VoiceId, SpeakingEngine | None],
    events: list[str],
    build: Callable[[VoiceId], SpeakingEngine],
) -> Runtime:
    return Runtime(
        voices,
        memory_probe=lambda: 0,
        dependencies=RuntimeDependencies(
            engine_factory=build, missing_artifacts=lambda _voice: ()
        ),
        hearing=Hearing(_RecordedHearing(events), LoopClock()),
    )


def test_the_start_loads_the_piper_voices_in_catalogue_order_then_hearing() -> None:
    events: list[str] = []
    runtime = _offering(
        dict.fromkeys(reversed(VoiceId)),
        events,
        lambda voice: _RecordedSpeaking(events, voice),
    )

    runtime.load()

    assert events == [
        *(f"{voice.value} loaded" for voice in PIPER_VOICES),
        "hearing loaded",
    ]


def test_shutdown_during_the_start_load_leaves_later_voices_and_hearing_unloaded() -> (
    None
):
    events: list[str] = []
    built: list[VoiceId] = []
    loading = threading.Event()
    release = threading.Event()

    class SlowSpeaking(_RecordedSpeaking):
        def load(self) -> None:
            loading.set()
            assert release.wait(timeout=WAIT_SECONDS)
            super().load()

    def build(voice: VoiceId) -> SpeakingEngine:
        built.append(voice)
        return SlowSpeaking(events, voice)

    runtime = _offering({VoiceId.THORSTEN: None, VoiceId.NORMAN: None}, events, build)
    starting = threading.Thread(target=runtime.load)
    starting.start()
    assert loading.wait(timeout=WAIT_SECONDS)

    runtime.begin_shutdown()
    release.set()
    starting.join(timeout=WAIT_SECONDS)

    assert not starting.is_alive()
    assert built == [VoiceId.THORSTEN]
    assert "hearing loaded" not in events


def test_a_failing_voice_leaves_the_later_ones_unbuilt_and_closes_the_earlier() -> None:
    events: list[str] = []
    built: dict[VoiceId, SpeakingEngine] = {}

    def build(voice: VoiceId) -> SpeakingEngine:
        built[voice] = _ClosableSpeaking(fail=voice is VoiceId.NORMAN)
        return built[voice]

    runtime = _offering(
        {VoiceId.THORSTEN: None, VoiceId.NORMAN: None, VoiceId.CORI: None},
        events,
        build,
    )

    with pytest.raises(RuntimeError, match=f"{VoiceId.NORMAN.value} failed to load"):
        runtime.load()
    runtime.close()

    assert list(built) == [VoiceId.THORSTEN, VoiceId.NORMAN]
    assert [engine.closed for engine in built.values()] == [1, 1]
    assert "hearing loaded" not in events
    assert runtime.loading is False


def test_shutdown_before_the_start_load_leaves_the_start_voice_unbuilt() -> None:
    events: list[str] = []
    built: list[VoiceId] = []

    def build(voice: VoiceId) -> SpeakingEngine:
        built.append(voice)
        return _RecordedSpeaking(events, voice)

    runtime = _offering({VoiceId.THORSTEN: None}, events, build)
    runtime.begin_shutdown()

    runtime.load()

    assert built == []
    assert events == []
    assert runtime.loading is False


def test_a_runtime_without_a_start_voice_is_ready_once_started_until_shutdown() -> None:
    runtime = Runtime(
        {VoiceId.RYAN: None},
        memory_probe=lambda: 0,
        dependencies=RuntimeDependencies(
            engine_factory=lambda _voice: FakeSpeaking(),
            missing_artifacts=lambda _voice: (),
        ),
    )
    runtime.begin_loading()
    while_loading = runtime.health().speaking.ready

    runtime.load()
    started = runtime.health().speaking.ready
    runtime.begin_shutdown()
    after_shutdown = runtime.health().speaking.ready

    assert (while_loading, started, after_shutdown) == (False, True, False)


def test_close_closes_hearing_after_the_slots_even_when_a_slot_fails_to_close() -> None:
    events: list[str] = []
    runtime = Runtime(
        {
            VoiceId.THORSTEN: _RecordedSpeaking(
                events, VoiceId.THORSTEN, fail_close=True
            )
        },
        memory_probe=lambda: 0,
        hearing=Hearing(_RecordedHearing(events), LoopClock()),
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        runtime.close()

    assert events == [f"{VoiceId.THORSTEN.value} closed", "hearing closed"]
