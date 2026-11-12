"""Stage proof of hearing: live and from a file, with numbers, on one device.

Starts the speech service once per hearing model and hears a German and an
English question and a long turn, live over a WebSocket and from a file. On
`cuda` it also measures the card's peak memory and starts the service on a card
a second process holds. It writes one JSON record and prints its stage lines.
It never downloads a hearing model and fetches only the English Piper voice
at its pin, never starts or stops anything but its own processes, and takes the
probe-stack lock as `prove_speech` does. Its clips are synthetic: German from the
service's speak route, English from a Piper voice.
"""

# ruff: noqa: T201 - the proof prints its stage lines as the command line output

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

import httpx2

import hearing_figures as figures
from hearing_card import (
    MEMORY_SAMPLE_SECONDS,
    MemorySampler,
    card_is_freed,
    held_card,
    used_mib,
)
from hearing_client import (
    Service,
    hear_file,
    hear_live,
    hear_upgrade,
    post_recording,
    refusal_code,
)
from hearing_figures import LiveTurn
from proof_identity import Minting, load_minting, mint_pair, proof_identity
from prove_speech import (
    LOCK_HELD_ENV,
    LOCK_PATH,
    LOOPBACK_HOST,
    READY_TIMEOUT_SECONDS,
    free_local_port,
    nvidia_memory,
    prove_under_lock,
    read_health,
    require_running,
    run_under_lock,
    speak_measured,
    speech_environment,
    speech_server,
    wait_for_load,
)
from speech_client.hearing import Refused
from speech_client.wire import (
    SPEECH_PATH,
    Health,
    HearingLanguage,
    ResponseFormat,
    speech_request_body,
    speech_request_headers,
)
from speech_service.config import load_settings
from speech_service.hearing_models import (
    HearingModel,
    HearingModelMissingError,
    model_directory,
)
from speech_service.pcm import pcm_from_wav
from speech_service.piper_voices import PinnedVoice, fetch_voice_files
from speech_service.voices import GPU_VOICES, RESIDENT_VOICE, VoiceId

if TYPE_CHECKING:
    from collections.abc import (
        Awaitable,
        Callable,
        Collection,
        Iterator,
        Mapping,
        Sequence,
    )
    from contextlib import AbstractContextManager

    from piper import PiperVoice

    from speech_client.health import HealthOutcome
    from speech_client.hearing import HearingOutcome


THIS_SCRIPT = Path(__file__).resolve()
GERMAN_QUESTION = "Wie spät ist es und wann beginnt die nächste Sitzung?"
ENGLISH_QUESTION = "What time is it and when does the next session begin?"
GERMAN_LONG_TURN = (
    "Willkommen zu unserem Vortrag über die Planung von Veranstaltungen. "
    "Zuerst betrachten wir, wie viele Gäste in den Saal passen und welche "
    "Technik wir für die Übertragung brauchen. Danach sprechen wir über den "
    "Zeitplan des Tages, über die Pausen und über das Essen am Mittag. "
    "Zum Schluss klären wir, wer die Gäste am Eingang begrüßt und wer ihnen "
    "den Weg zu den Räumen zeigt."
)
ENGLISH_LONG_TURN = (
    "Welcome to our talk about planning events. "
    "First we look at how many guests fit into the hall and which equipment "
    "we need for the broadcast. After that we talk about the schedule of the "
    "day, about the breaks and about lunch at noon. "
    "Finally we decide who greets the guests at the entrance and who shows "
    "them the way to the rooms."
)
CARD_STAGES = ("gpu_memory", "taken_card")
PROBE_TEXT = "Bereit."
SPEAK_WAIT_SECONDS = 30.0
POLL_SECONDS = 0.5


@dataclass(frozen=True)
class Language:
    """A language the proof hears, with the sentences it is heard against."""

    code: HearingLanguage
    question: str
    long_turn: str


LANGUAGES = (
    Language(HearingLanguage.DE, GERMAN_QUESTION, GERMAN_LONG_TURN),
    Language(HearingLanguage.EN, ENGLISH_QUESTION, ENGLISH_LONG_TURN),
)


BOTH_LANGUAGES = tuple(language.code for language in LANGUAGES)


@dataclass(frozen=True)
class Clip:
    """A language's two clips at the wire's 24 kHz PCM16 mono."""

    question: bytes
    long_turn: bytes


def spoken_texts() -> list[str]:
    """Every sentence the proof speaks or hears, which no service log may hold."""
    return [
        PROBE_TEXT,
        *(
            text
            for language in LANGUAGES
            for text in (language.question, language.long_turn)
        ),
    ]


# -- waiting for the service --------------------------------------------------


def _speaks(base_url: str, minting: Minting, voice: str) -> bool:
    """True once the speak route answers, which it does only after loading ends."""
    pair = mint_pair(minting)
    try:
        with httpx2.Client(base_url=base_url, timeout=SPEAK_WAIT_SECONDS) as client:
            response = client.post(
                SPEECH_PATH,
                content=speech_request_body(PROBE_TEXT, voice, ResponseFormat.WAV),
                headers=speech_request_headers(
                    workload_token=pair.workload, person_token=pair.person
                ),
            )
    except httpx2.HTTPError:
        return False
    return response.status_code == HTTPStatus.OK


def _loaded(health: Health, speaks: Callable[[], bool]) -> bool:
    return health.hearing.ready or (health.speaking.ready and speaks())


@dataclass(frozen=True)
class LoadWait:
    """What a wait for the service's load asks and how it tells the time."""

    speaks: Callable[[], bool]
    health_of: Callable[[str], HealthOutcome] = read_health
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic


def wait_loaded(
    base_url: str, exit_status: Callable[[], int | None], wait: LoadWait
) -> Health:
    """Wait until the service finished loading and return its health then.

    Hearing ready is the end of a good load. A load that found no room on the
    card ends with speaking served and hearing not ready, which the speak route
    shows by answering once loading is over. A server that exited ends the wait.
    """
    deadline = wait.monotonic() + READY_TIMEOUT_SECONDS
    last: Health | None = None
    while wait.monotonic() < deadline:
        require_running(exit_status, last)
        answer = wait.health_of(base_url)
        if isinstance(answer, Health):
            last = answer
            if _loaded(answer, wait.speaks):
                fresh = wait.health_of(base_url)
                if isinstance(fresh, Health):
                    return fresh
        wait.sleep(POLL_SECONDS)
    detail = "" if last is None else f": {last}"
    message = (
        f"the service did not finish loading within {READY_TIMEOUT_SECONDS}s{detail}"
    )
    raise SystemExit(message)


@contextlib.contextmanager
def hearing_service(
    device: str, hearing_model: str, voice: VoiceId
) -> Iterator[Service]:
    """Run the service on `device` with the model, speaking `voice`, on its own."""
    directory = Path(tempfile.mkdtemp(prefix="speech-"))
    port = free_local_port()
    base_url = f"http://{LOOPBACK_HOST}:{port}"
    try:
        with proof_identity(directory) as identity:
            minting = load_minting(identity.minting_path)
            environment = speech_environment(
                port, voice, identity, device, hearing_model
            )
            wait_ready = functools.partial(
                wait_loaded,
                wait=LoadWait(
                    speaks=functools.partial(_speaks, base_url, minting, voice.value)
                ),
            )
            with speech_server(
                base_url,
                environment,
                directory,
                wait_ready=wait_ready,
            ) as running:
                running.spoken_texts.extend(spoken_texts())
                yield Service(base_url, minting, running)
    finally:
        shutil.rmtree(directory)


@dataclass(frozen=True)
class Ports:
    """What the card stages do outside the proof, from holding to sampling it."""

    hold: Callable[[], AbstractContextManager[int]] = held_card
    serve: Callable[..., AbstractContextManager[Service]] = hearing_service
    post: Callable[..., httpx2.Response] = post_recording
    upgrade: Callable[[Service], Awaitable[HearingOutcome | None]] = hear_upgrade
    speak: Callable[..., tuple[bytes, float, float]] = speak_measured
    live: Callable[..., Awaitable[LiveTurn]] = hear_live
    hear: Callable[..., tuple[float, str]] = hear_file
    freed: Callable[[int], bool] = card_is_freed
    sample: Callable[[], AbstractContextManager[MemorySampler]] = MemorySampler


PORTS = Ports()


# -- clips --------------------------------------------------------------------


def german_clip(service: Service, text: str) -> bytes:
    """German speech from the service's speak route as 24 kHz PCM16.

    Each sentence is its own synthesis with its edge silence cut, so the
    sentences join without the pause the route puts between them.
    """
    sentences = []
    for sentence in re.split(r"(?<=\.) ", text):
        wav, _, _ = speak_measured(
            service.base_url,
            sentence,
            RESIDENT_VOICE.value,
            service.minting,
        )
        rate, pcm = pcm_from_wav(wav)
        sentences.append(figures.speech_only(figures.to_wire_rate(pcm, rate)))
    return b"".join(sentences)


def english_voice_path(directory: Path) -> Path:
    """The English Piper voice in `directory`, fetched there once if absent."""
    return fetch_voice_files(PinnedVoice.LESSAC, directory)


def english_clip(voice: PiperVoice, text: str) -> bytes:
    """English speech from a Piper voice as 24 kHz PCM16."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        voice.synthesize_wav(text, writer)
    buffer.seek(0)
    with wave.open(buffer) as reader:
        return figures.to_wire_rate(
            reader.readframes(reader.getnframes()), reader.getframerate()
        )


def make_clips(
    service: Service, english_directory: Path
) -> dict[HearingLanguage, Clip]:
    """Both languages' clips; a long turn must be one turn the service hears whole."""
    from piper import PiperVoice

    voice = PiperVoice.load(english_voice_path(english_directory))
    clips = {
        HearingLanguage.DE: Clip(
            german_clip(service, GERMAN_QUESTION),
            german_clip(service, GERMAN_LONG_TURN),
        ),
        HearingLanguage.EN: Clip(
            english_clip(voice, ENGLISH_QUESTION),
            english_clip(voice, ENGLISH_LONG_TURN),
        ),
    }
    for language, clip in clips.items():
        seconds = figures.check_continuous_turn(clip.long_turn)
        print("long_turn_clip", language.value, f"seconds={seconds:.1f}")
    return clips


# -- the figures of one language ----------------------------------------------


def _long_turn_record(
    service: Service,
    language: Language,
    clip: Clip,
    ports: Ports,
    runs: int,
) -> dict[str, object]:
    """The long turn's one- and two-language runs, taken in turn, and one file run."""
    single: list[LiveTurn] = []
    two: list[LiveTurn] = []
    for _ in range(runs):
        single.append(
            asyncio.run(ports.live(service, (language.code,), clip.long_turn))
        )
        two.append(asyncio.run(ports.live(service, BOTH_LANGUAGES, clip.long_turn)))
    file_result = ports.hear(service, language.code, clip.long_turn)
    return figures.long_turn_record(language.long_turn, single, two, file_result)


def measure_language(
    service: Service,
    language: Language,
    clip: Clip,
    runs: int,
    ports: Ports = PORTS,
) -> dict[str, object]:
    """One language's short question, live and from a file, and its long turn."""
    ports.hear(service, language.code, clip.question)
    live = [
        asyncio.run(ports.live(service, (language.code,), clip.question))
        for _ in range(runs)
    ]
    two = [
        asyncio.run(ports.live(service, BOTH_LANGUAGES, clip.question))
        for _ in range(runs)
    ]
    files = [ports.hear(service, language.code, clip.question) for _ in range(runs)]
    print("question", language.code.value, f"runs={runs}")
    return {
        "clip_sha256": {
            "question": hashlib.sha256(clip.question).hexdigest(),
            "long_turn": hashlib.sha256(clip.long_turn).hexdigest(),
        },
        "question": {
            "reference": language.question,
            "live": figures.live_record(language.question, live),
            "two_languages": figures.language_step_record(live, two),
            "file": figures.file_record(language.question, files),
        },
        "long_turn": _long_turn_record(service, language, clip, ports, runs),
    }


def measure_models(
    arguments: argparse.Namespace,
    models: dict[str, object],
    save: Callable[[], None],
) -> dict[HearingLanguage, Clip]:
    """Fill `models` with each hearing model's figures as each language ends.

    `save` writes the record after each language and each model, so a proof killed
    outright keeps what it measured. Returns the clips.
    """
    clips: dict[HearingLanguage, Clip] | None = None
    for model in arguments.hearing_models:
        with hearing_service(arguments.device, model, RESIDENT_VOICE) as service:
            if clips is None:
                clips = make_clips(service, arguments.english_voice_directory)
            print(
                "hearing_ready",
                model,
                f"load_to_ready_s={service.running.ready_seconds:.1f}",
            )
            languages: dict[str, object] = {}
            entry: dict[str, object] = {
                "status": figures.StageStatus.UNFINISHED,
                "load_to_ready_s": service.running.ready_seconds,
                "languages": languages,
            }
            models[model] = entry
            for language in LANGUAGES:
                languages[language.code.value] = measure_language(
                    service, language, clips[language.code], arguments.runs
                )
                save()
            entry["status"] = figures.StageStatus.MEASURED
            save()
    if clips is None:
        message = "no hearing model was given"
        raise SystemExit(message)
    return clips


# -- the card -----------------------------------------------------------------


def measure_peak(
    device: str,
    hearing_model: str,
    voice: VoiceId,
    warm_clip: bytes,
    ports: Ports = PORTS,
) -> dict[str, object]:
    """The card's peak over its baseline from service start, a synthesis, a long decode.

    The peak is the highest of the samples and the warmed resident reading. It is a
    sampled maximum: a spike shorter than the sampling period can fall between two
    samples, so the record names that period beside the peak.

    The voice has synthesised before memory is read, as speaking and hearing run
    side by side and the allocations of synthesis belong to the peak.
    """
    baseline = used_mib()
    with (
        ports.sample() as sampler,
        ports.serve(device, hearing_model, voice) as service,
    ):
        ports.speak(service.base_url, GERMAN_QUESTION, voice.value, service.minting)
        ports.hear(service, HearingLanguage.DE, warm_clip)
        resident = used_mib()
    return {
        "baseline_mib": baseline,
        "peak_over_baseline_mib": figures.peak_over_baseline(
            baseline, [*sampler.samples, resident]
        ),
        "peak_sampled_every_seconds": MEMORY_SAMPLE_SECONDS,
        "warmed_resident_over_baseline_mib": resident - baseline,
    }


def measure_memory(
    device: str,
    hearing_models: Sequence[str],
    clips: dict[HearingLanguage, Clip],
    memory: dict[str, object],
    offered: Collection[VoiceId],
) -> None:
    """Fill `memory` with each model's peak, alone and with each `offered` GPU voice."""
    measured = [voice for voice in GPU_VOICES if voice in offered]
    warm_clip = clips[HearingLanguage.DE].long_turn
    memory["voices_not_offered"] = [
        voice.value for voice in GPU_VOICES if voice not in offered
    ]
    models: dict[str, object] = {}
    memory["models"] = models
    for model in hearing_models:
        peaks: dict[str, object] = {}
        models[model] = peaks
        peaks["hearing_alone"] = measure_peak(device, model, RESIDENT_VOICE, warm_clip)
        with_voice: dict[str, object] = {}
        peaks["with_voice"] = with_voice
        for voice in measured:
            with_voice[voice.value] = measure_peak(device, model, voice, warm_clip)


def prove_taken_card(
    device: str,
    hearing_model: str,
    clips: dict[HearingLanguage, Clip],
    ports: Ports = PORTS,
) -> dict[str, object]:
    """Start the service on a card a second process holds; cell 30 says what follows."""
    baseline = used_mib()
    german = clips[HearingLanguage.DE]
    with ports.hold() as held_mib:
        used_while_held = used_mib()
        with ports.serve(device, hearing_model, RESIDENT_VOICE) as service:
            health = service.running.health
            file_answer = ports.post(service, HearingLanguage.DE, german.question)
            live = asyncio.run(ports.upgrade(service))
            wav, _, _ = ports.speak(
                service.base_url,
                GERMAN_QUESTION,
                RESIDENT_VOICE.value,
                service.minting,
            )
    freed = ports.freed(baseline)
    file_code = refusal_code(file_answer.status_code, file_answer.content)
    live_status, live_code = (
        (live.status, live.code) if isinstance(live, Refused) else (None, None)
    )
    failures = figures.taken_card_failures(
        figures.TakenCardObservation(
            speaking_ready=health.speaking.ready,
            hearing_ready=health.hearing.ready,
            file_status=file_answer.status_code,
            file_code=file_code,
            live_status=live_status,
            live_code=live_code,
            spoke_bytes=len(wav),
        )
    )
    return {
        "hearing_model": hearing_model,
        "held_mib": held_mib,
        "used_while_held_mib": used_while_held,
        "health": health.model_dump(),
        "refusals": {
            "file": {"status": file_answer.status_code, "code": file_code},
            "live": {
                "status": live_status,
                "code": live_code,
            }
            | (
                {}
                if live is None or isinstance(live, Refused)
                else {"outcome": repr(live)}
            ),
        },
        "freed_afterwards": freed,
        "failures": failures + ([] if freed else ["the card was not freed"]),
    }


# -- the proof ------------------------------------------------------------------


def _runs(text: str) -> int:
    runs = int(text)
    if runs < figures.MIN_RUNS:
        message = f"a median needs at least {figures.MIN_RUNS} runs"
        raise argparse.ArgumentTypeError(message)
    return runs


def _record_path(text: str) -> Path:
    path = Path(text)
    if not path.parent.is_dir():
        message = f"the record's directory {path.parent} does not exist"
        raise argparse.ArgumentTypeError(message)
    return path


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    """Read the command line before anything starts; `--help` and typos exit here."""
    parser = argparse.ArgumentParser(
        description="Stage proof of hearing under the probe-stack lock.",
        allow_abbrev=False,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--hearing-model",
        dest="hearing_models",
        action="append",
        type=HearingModel,
        choices=list(HearingModel),
        required=True,
        help="a hearing model to measure; give it once per model",
    )
    parser.add_argument("--record", type=_record_path, required=True)
    parser.add_argument(
        "--english-voice-directory",
        type=Path,
        required=True,
        help="where the English Piper voice is kept; fetched there when absent",
    )
    parser.add_argument("--runs", type=_runs, default=figures.MIN_RUNS)
    return parser.parse_args(arguments)


def require_cached_models(models: Sequence[HearingModel]) -> None:
    """Fail before anything starts when a hearing model is not in the cache."""
    cache = load_settings().huggingface_cache
    for model in models:
        try:
            model_directory(model, cache)
        except HearingModelMissingError as missing:
            raise SystemExit(str(missing)) from None


def required_models(arguments: argparse.Namespace) -> list[str]:
    """Every catalogue model on `cuda`, where the item asks for both; else the given."""
    if arguments.device == "cuda":
        return [str(model) for model in HearingModel]
    return [str(model) for model in arguments.hearing_models]


def card_line(device: str) -> str:
    """The card's memory line on `cuda`; the card is not asked about otherwise."""
    return nvidia_memory() if device == "cuda" else "not applicable"


def run_stage(
    stage: dict[str, object],
    measure: Callable[[dict[str, object]], None],
    save: Callable[[], None],
) -> None:
    """Measure an unfinished `stage` into itself; a failure leaves what it measured."""
    if stage["status"] == figures.StageStatus.UNFINISHED:
        measure(stage)
        stage["status"] = figures.StageStatus.MEASURED
        save()


def write_record(path: Path, record: dict[str, object]) -> None:
    """Replace the record file whole, so a kill never leaves half a record."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(record, indent=2), encoding="utf-8")
    partial.replace(path)


def recorded_failures(stages: Mapping[str, dict[str, object]]) -> list[str]:
    """The failures the taken-card stage has written into the record so far."""
    return stages["taken_card"].get("failures", [])


def record_writer(
    arguments: argparse.Namespace,
    record: dict[str, object],
) -> Callable[[], None]:
    """A save that recomputes `not_measured` and `complete` before each write."""
    models = record["models"]
    stages = {name: record[name] for name in CARD_STAGES}

    def save() -> None:
        unmeasured = figures.not_measured(required_models(arguments), models, stages)
        record["not_measured"] = unmeasured
        record["complete"] = not recorded_failures(stages) and not unmeasured
        write_record(arguments.record, record)

    return save


def prove(
    arguments: argparse.Namespace,
    wait_for_calm: Callable[[], None] = wait_for_load,
) -> None:
    """Measure every model and card stage into the record, then judge it."""
    require_cached_models(arguments.hearing_models)
    wait_for_calm()
    models: dict[str, object] = {}
    stages = {
        name: figures.open_stage(arguments.device, "cuda") for name in CARD_STAGES
    }
    record: dict[str, object] = {
        "device": arguments.device,
        "runs": arguments.runs,
        "clips": (
            "synthetic: German from the service's speak route, English from the"
            f" Piper voice {PinnedVoice.LESSAC.value}"
        ),
        "gpu_before": card_line(arguments.device),
        "models": models,
        **stages,
    }
    save = record_writer(arguments, record)
    save()
    try:
        clips = measure_models(arguments, models, save)
        run_stage(
            stages["gpu_memory"],
            lambda stage: measure_memory(
                arguments.device,
                arguments.hearing_models,
                clips,
                stage,
                load_settings().gpu_voices,
            ),
            save,
        )
        run_stage(
            stages["taken_card"],
            lambda stage: stage.update(
                prove_taken_card(arguments.device, arguments.hearing_models[0], clips)
            ),
            save,
        )
    finally:
        save()
        print("record", arguments.record)
        print("gpu_released", card_line(arguments.device))
    if failures := recorded_failures(stages):
        raise SystemExit("the taken card differs from cell 30: " + "; ".join(failures))
    if unmeasured := record["not_measured"]:
        raise SystemExit("not measured: " + "; ".join(unmeasured))
    print("PROOF_OK")


def main() -> None:
    """Run the stage proof, taking the probe-stack lock through the fleet form."""
    arguments = parse_arguments(sys.argv[1:])
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if LOCK_HELD_ENV not in os.environ:
        print("waiting_for_lock", LOCK_PATH)
        raise SystemExit(run_under_lock(sys.argv[1:], script=THIS_SCRIPT))
    prove_under_lock(os.environ, functools.partial(prove, arguments))


if __name__ == "__main__":
    main()
