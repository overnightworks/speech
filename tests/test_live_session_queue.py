"""A live session's finals in the gate: order, priority over file jobs, deadlines."""

import asyncio

from tests.support.clock_support import FakeClock
from tests.support.hear_support import FakeHearing, loaded
from tests.support.live_support import (
    COMPLETED,
    WAIT_SECONDS,
    FakeSocket,
    StallingSocket,
    append,
    audio,
    commit_a_turn,
    end_is,
    started,
)

FINALS = 2
THREE_FINALS = 3
CAP_PLUS_SECONDS = 31


def test_finals_of_several_sessions_are_served_in_arrival_order_before_file_jobs() -> (
    None
):
    """WIRE-84: finals of several sessions are served in arrival order."""

    async def scenario() -> tuple[list[int], int]:
        engine = FakeHearing()
        hearing = loaded(engine, FakeClock())
        running = hearing.gate.try_acquire()
        assert running is not None
        file_job = asyncio.create_task(
            hearing.gate.acquire(hearing.gate.clock.now() + WAIT_SECONDS, 1),
            eager_start=True,
        )
        sessions = [(await started(hearing)) for _ in range(2)]
        for (session, _), seconds in zip(sessions, (0.3, 0.4), strict=True):
            await commit_a_turn(session, seconds)
        running.close()
        for _, socket in sessions:
            await socket.until(lambda socket=socket: bool(socket.of_type(COMPLETED)))
        file_lease = await file_job
        assert file_lease is not None
        heard_before_the_file_job = len(engine.heard)
        file_lease.close()
        return [size for size, _ in engine.heard], heard_before_the_file_job

    sizes, heard_before_the_file_job = asyncio.run(scenario())

    assert sizes[0] < sizes[1]
    assert heard_before_the_file_job == len(sizes)


def test_a_second_ended_turn_decodes_before_a_waiting_file_job_and_follows() -> None:
    """WIRE-79, WIRE-84: a second ended turn decodes before a waiting recording."""

    async def scenario() -> tuple[FakeSocket, int]:
        engine = FakeHearing(hold=True)
        clock = FakeClock()
        hearing = loaded(engine, clock)
        session, socket = await started(hearing)
        await commit_a_turn(session, 0.3)
        assert await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        heard_when_the_file_job_runs: list[int] = []

        async def file_job() -> None:
            lease = await hearing.gate.acquire(clock.now() + WAIT_SECONDS, 1)
            assert lease is not None
            heard_when_the_file_job_runs.append(len(engine.heard))
            lease.close()

        file = asyncio.create_task(file_job(), eager_start=True)
        await commit_a_turn(session, 0.4)
        engine.release.set()
        await socket.until(lambda: len(socket.of_type(COMPLETED)) == FINALS)
        await file
        return socket, heard_when_the_file_job_runs[0]

    socket, heard_when_the_file_job_runs = asyncio.run(scenario())

    committed = socket.of_type("input_audio_buffer.committed")
    assert heard_when_the_file_job_runs == FINALS
    assert [done["item_id"] for done in socket.of_type(COMPLETED)] == [
        turn["item_id"] for turn in committed
    ]


def test_a_stalled_send_of_the_first_completed_does_not_hold_the_second_decode() -> (
    None
):
    """WIRE-84: a stalled send does not hold the second decode."""

    async def scenario() -> tuple[FakeSocket, int]:
        engine = FakeHearing(hold=True)
        clock = FakeClock()
        hearing = loaded(engine, clock)
        socket = StallingSocket(COMPLETED)
        session, _ = await started(hearing, socket=socket)
        await commit_a_turn(session)
        assert await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        await commit_a_turn(session)
        engine.release.set()
        await socket.stalled.wait()
        lease = await asyncio.wait_for(
            hearing.gate.acquire(clock.now() + WAIT_SECONDS, 1), WAIT_SECONDS
        )
        assert lease is not None
        heard_when_the_gate_is_free = len(engine.heard)
        lease.close()
        socket.proceed.set()
        await socket.until(lambda: len(socket.of_type(COMPLETED)) == FINALS)
        return socket, heard_when_the_gate_is_free

    socket, heard_when_the_gate_is_free = asyncio.run(scenario())

    assert heard_when_the_gate_is_free == FINALS
    assert len(socket.of_type(COMPLETED)) == FINALS


def test_a_stalled_send_at_the_second_turns_end_lets_nothing_pass_its_decode() -> None:
    """WIRE-84: a stalled send at the second turns end lets nothing pass its decode."""

    async def scenario() -> tuple[list[int], int]:
        engine = FakeHearing(hold=True)
        clock = FakeClock()
        hearing = loaded(engine, clock)
        socket = StallingSocket("input_audio_buffer.committed", occurrence=2)
        session, _ = await started(hearing, socket=socket)
        other, other_socket = await started(hearing)
        await commit_a_turn(session, 0.3)
        assert await asyncio.to_thread(engine.entered.wait, WAIT_SECONDS)
        heard_when_the_file_job_runs: list[int] = []

        async def file_job() -> None:
            lease = await hearing.gate.acquire(clock.now() + WAIT_SECONDS, 1)
            assert lease is not None
            heard_when_the_file_job_runs.append(len(engine.heard))
            lease.close()

        file = asyncio.create_task(file_job(), eager_start=True)
        second = asyncio.create_task(commit_a_turn(session, 0.4))
        await socket.stalled.wait()
        await commit_a_turn(other, 0.5)
        engine.release.set()
        await asyncio.wait_for(file, WAIT_SECONDS)
        socket.proceed.set()
        await second
        await other_socket.until(lambda: bool(other_socket.of_type(COMPLETED)))
        return [size for size, _ in engine.heard], heard_when_the_file_job_runs[0]

    sizes, heard_when_the_file_job_runs = asyncio.run(scenario())

    assert sizes == sorted(sizes)
    assert heard_when_the_file_job_runs == THREE_FINALS


def test_a_capped_turn_behind_a_stalled_send_queues_before_a_file_job() -> None:
    """WIRE-66, WIRE-84: a capped turn queues before a waiting recording."""

    async def scenario() -> int:
        engine = FakeHearing()
        clock = FakeClock()
        hearing = loaded(engine, clock)
        running = hearing.gate.try_acquire()
        assert running is not None
        file = asyncio.create_task(
            hearing.gate.acquire(clock.now() + WAIT_SECONDS, 1), eager_start=True
        )
        socket = StallingSocket("input_audio_buffer.speech_started")
        session, _ = await started(hearing, socket=socket)
        append_past_the_cap = asyncio.create_task(
            session.on_message(append(audio(CAP_PLUS_SECONDS)))
        )
        await socket.stalled.wait()
        running.close()
        file_lease = await asyncio.wait_for(file, WAIT_SECONDS)
        assert file_lease is not None
        heard_when_the_file_job_runs = len(engine.heard)
        file_lease.close()
        socket.proceed.set()
        await append_past_the_cap
        await session.on_disconnect()
        return heard_when_the_file_job_runs

    assert asyncio.run(scenario()) == 1


def test_a_final_whose_turn_end_send_stalls_past_its_wait_is_never_heard() -> None:
    """WIRE-85: a final whose turn end send stalls past its wait is never heard."""

    async def scenario() -> tuple[FakeSocket, FakeHearing]:
        engine = FakeHearing()
        clock = FakeClock()
        hearing = loaded(engine, clock)
        running = hearing.gate.try_acquire()
        assert running is not None
        socket = StallingSocket("input_audio_buffer.committed")
        session, _ = await started(
            hearing, final_wait_seconds=WAIT_SECONDS, socket=socket
        )
        commit_turn = asyncio.create_task(commit_a_turn(session))
        await socket.stalled.wait()
        clock.advance(2 * WAIT_SECONDS)
        socket.proceed.set()
        await commit_turn
        await socket.until_closed()
        running.close()
        return socket, engine

    socket, engine = asyncio.run(scenario())

    assert end_is(socket, "busy", failed=1)
    assert engine.heard == []


def test_a_turn_that_ends_stands_in_the_gates_queue_before_a_waiting_file_job() -> None:
    """WIRE-84: a turn that ends stands in the gates queue before a waiting file job."""

    async def scenario() -> tuple[bool, bool]:
        engine = FakeHearing()
        clock = FakeClock()
        hearing = loaded(engine, clock)
        running = hearing.gate.try_acquire()
        assert running is not None
        file = asyncio.create_task(
            hearing.gate.acquire(clock.now() + WAIT_SECONDS, 1), eager_start=True
        )
        session, socket = await started(hearing)
        await commit_a_turn(session)
        running.close()
        await socket.until(lambda: bool(socket.of_type(COMPLETED)))
        file_lease = await asyncio.wait_for(file, WAIT_SECONDS)
        return file_lease is not None, len(engine.heard) == 1

    file_served, final_was_heard_first = asyncio.run(scenario())

    assert file_served
    assert final_was_heard_first


def test_a_waiting_turn_is_heard_after_the_outstanding_final() -> None:
    """WIRE-79, WIRE-84: a waiting turn is heard after the outstanding final."""

    async def scenario() -> FakeSocket:
        engine = FakeHearing(hold=True)
        session, socket = await started(loaded(engine, FakeClock()))
        await commit_a_turn(session)
        await commit_a_turn(session)
        announced = len(socket.of_type("input_audio_buffer.committed"))
        engine.release.set()
        await socket.until(lambda: len(socket.of_type(COMPLETED)) == announced)
        await session.settled()
        return socket

    socket = asyncio.run(scenario())

    committed = socket.of_type("input_audio_buffer.committed")
    assert [done["item_id"] for done in socket.of_type(COMPLETED)] == [
        turn["item_id"] for turn in committed
    ]
