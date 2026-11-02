"""Stage proof of hearing: live and from a file, with numbers, on one device.

Starts the speech service once per hearing model and hears a German and an
English question and a long turn, live over a WebSocket and from a file. On
`cuda` it also measures the card's peak memory and starts the service on a card
a second process holds. It writes one JSON record and prints its stage lines.
It never downloads a hearing model and fetches only the English Piper voice,
never starts or stops anything but its own processes, and takes the probe-stack
lock as `prove_speech` does. Its clips are synthetic: German from the service's
speak route, English from a Piper voice.
"""

# ruff: noqa: T201 - the proof prints its stage lines as the command line output

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import io
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

import httpx2
from speech_runtime_provider_contract import (
    MAGPIE_MODEL_ID,
    QWEN_MODEL_ID,
    VOXCPM_MODEL_ID,
)

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
    UpgradeRefusal,
    hear_file,
    hear_live,
    post_recording,
    refusal_code,
    upgrade_refusal,
)
from hearing_figures import LiveTurn, WordErrors
from proof_identity import Minting, load_minting, mint_pair, proof_identity
from prove_speech import (
    LOCK_HELD_ENV,
    LOCK_PATH,
    LOOPBACK_HOST,
    READY_TIMEOUT_SECONDS,
    free_local_port,
    get_json,
    nvidia_memory,
    prove_under_lock,
    run_under_lock,
    speak_measured,
    speech_environment,
    speech_server,
    voice_id,
    wait_for_load,
)
from speech_runtime_client.wire import (
    SPEECH_PATH,
    HearingLanguage,
    speech_request_body,
    speech_request_headers,
)
from speech_runtime_service.config import load_settings
from speech_runtime_service.hearing_models import (
    HearingModel,
    HearingModelMissingError,
    model_directory,
)
from speech_runtime_service.pcm import pcm_from_wav
from speech_runtime_service.voices import (
    CHATTERBOX_SPEAKING_MODEL,
    DEFAULT_SPEAKING_MODEL,
    installed_voice_ids,
    voice_for_model,
)

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

    from speech_runtime_service.voices import VoiceId


THIS_SCRIPT = Path(__file__).resolve()
ENGLISH_VOICE = "en_US-lessac-medium"
GPU_SPEAKING_MODELS = (
    CHATTERBOX_SPEAKING_MODEL,
    QWEN_MODEL_ID,
    VOXCPM_MODEL_ID,
    MAGPIE_MODEL_ID,
)
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
                content=speech_request_body(PROBE_TEXT, voice),
                headers=speech_request_headers(
                    workload_token=pair.workload, person_token=pair.person
                ),
            )
    except httpx2.HTTPError:
        return False
    return response.status_code == HTTPStatus.OK


def _loaded(health: dict[str, object], speaks: Callable[[], bool]) -> bool:
    hearing = health["hearing"]
    speaking = health["speaking"]
    if isinstance(hearing, dict) and hearing["ready"] is True:
        return True
    return isinstance(speaking, dict) and speaking["ready"] is True and speaks()


def wait_loaded(
    base_url: str,
    speaks: Callable[[], bool],
    health_of: Callable[[str, str], dict[str, object]] = get_json,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Wait until the service finished loading and return its health then.

    Hearing ready is the end of a good load. A load that found no room on the
    card ends with speaking served and hearing not ready, which the speak route
    shows by answering once loading is over.
    """
    deadline = clock() + READY_TIMEOUT_SECONDS
    last: dict[str, object] | None = None
    while clock() < deadline:
        try:
            last = health_of(base_url, "/health")
            if _loaded(last, speaks):
                return health_of(base_url, "/health")
        except httpx2.HTTPError, json.JSONDecodeError:
            pass
        sleep(POLL_SECONDS)
    detail = "" if last is None else f": {last}"
    message = (
        f"the service did not finish loading within {READY_TIMEOUT_SECONDS}s{detail}"
    )
    raise SystemExit(message)


@contextlib.contextmanager
def hearing_service(
    device: str, hearing_model: str, speaking_model: str
) -> Iterator[Service]:
    """Run the service on `device` with the models, in a directory of its own."""
    directory = Path(tempfile.mkdtemp(prefix="speech-"))
    port = free_local_port()
    base_url = f"http://{LOOPBACK_HOST}:{port}"
    voice = voice_id(speaking_model)
    try:
        with proof_identity(directory, voice) as identity:
            minting = load_minting(identity.minting_path)
            environment = speech_environment(
                port, speaking_model, identity, device, hearing_model
            )
            wait_ready = functools.partial(
                wait_loaded,
                speaks=functools.partial(_speaks, base_url, minting, voice),
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
    upgrade: Callable[[Service], Awaitable[UpgradeRefusal | None]] = upgrade_refusal
    speak: Callable[..., tuple[bytes, float, float]] = speak_measured
    live: Callable[..., Awaitable[LiveTurn]] = hear_live
    hear: Callable[..., tuple[float, str]] = hear_file
    freed: Callable[[int], bool] = card_is_freed
    sample: Callable[[], AbstractContextManager[MemorySampler]] = MemorySampler


PORTS = Ports()


# -- clips --------------------------------------------------------------------


def german_clip(service: Service, text: str) -> bytes:
    """German speech from the service's speak route as 24 kHz PCM16."""
    wav, _, _ = speak_measured(
        service.base_url,
        text,
        voice_id(DEFAULT_SPEAKING_MODEL),
        service.minting,
    )
    rate, pcm = pcm_from_wav(wav)
    return figures.to_wire_rate(pcm, rate)


def english_voice_path(directory: Path) -> Path:
    """The English Piper voice in `directory`, fetched there once if absent."""
    from piper.download_voices import download_voice

    onnx = directory / f"{ENGLISH_VOICE}.onnx"
    if not onnx.is_file():
        print("fetching_english_voice", ENGLISH_VOICE, directory)
        directory.mkdir(parents=True, exist_ok=True)
        download_voice(ENGLISH_VOICE, directory)
    return onnx


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


def _errors_of(reference: str, heard: Sequence[str]) -> WordErrors:
    return functools.reduce(
        WordErrors.plus, (figures.word_errors(reference, text) for text in heard)
    )


def _live_record(reference: str, turns: Sequence[LiveTurn]) -> dict[str, object]:
    timings = [figures.timing_of(turn) for turn in turns]
    heard = [turn.transcript for turn in turns]
    return {
        "runs": len(turns),
        "median": figures.timing_record(figures.median_timing(timings)),
        "runs_with_partial": figures.partial_runs(timings),
        "timings": [figures.timing_record(timing) for timing in timings],
        "word_errors": figures.error_record(_errors_of(reference, heard)),
        "heard": sorted(set(heard)),
    }


def _file_record(
    reference: str, results: Sequence[tuple[float, str]]
) -> dict[str, object]:
    seconds = [elapsed for elapsed, _ in results]
    heard = [text for _, text in results]
    return {
        "runs": len(results),
        "median_seconds": statistics.median(seconds),
        "seconds": seconds,
        "word_errors": figures.error_record(_errors_of(reference, heard)),
        "heard": sorted(set(heard)),
    }


def _long_turn_record(
    service: Service,
    language: Language,
    clip: Clip,
    ports: Ports,
) -> dict[str, object]:
    turn = asyncio.run(ports.live(service, language.code, clip.long_turn))
    seconds, text = ports.hear(service, language.code, clip.long_turn)
    return {
        "clip_seconds": turn.question_seconds,
        "live": {
            "growing_deltas": figures.count_growing_deltas(turn.deltas),
            "delta_seconds": list(turn.delta_seconds),
            "timing": figures.timing_record(figures.timing_of(turn)),
            "word_errors": figures.error_record(
                figures.word_errors(language.long_turn, turn.transcript)
            ),
            "heard": turn.transcript,
        },
        "file": {
            "seconds": seconds,
            "word_errors": figures.error_record(
                figures.word_errors(language.long_turn, text)
            ),
            "heard": text,
        },
    }


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
        asyncio.run(ports.live(service, language.code, clip.question))
        for _ in range(runs)
    ]
    files = [ports.hear(service, language.code, clip.question) for _ in range(runs)]
    print("question", language.code.value, f"runs={runs}")
    return {
        "question": {
            "reference": language.question,
            "live": _live_record(language.question, live),
            "file": _file_record(language.question, files),
        },
        "long_turn": _long_turn_record(service, language, clip, ports),
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
        with hearing_service(
            arguments.device, model, DEFAULT_SPEAKING_MODEL
        ) as service:
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
    speaking_model: str,
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
        ports.serve(device, hearing_model, speaking_model) as service,
    ):
        ports.speak(
            service.base_url, GERMAN_QUESTION, voice_id(speaking_model), service.minting
        )
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
    installed: Collection[VoiceId],
) -> None:
    """Fill `memory` with each model's peak, alone and with each `installed` voice."""
    cached = [m for m in GPU_SPEAKING_MODELS if voice_for_model(m) in installed]
    warm_clip = clips[HearingLanguage.DE].long_turn
    memory["voices_not_in_cache"] = [
        voice_for_model(m).value for m in GPU_SPEAKING_MODELS if m not in cached
    ]
    models: dict[str, object] = {}
    memory["models"] = models
    for model in hearing_models:
        peaks: dict[str, object] = {}
        models[model] = peaks
        peaks["hearing_alone"] = measure_peak(
            device, model, DEFAULT_SPEAKING_MODEL, warm_clip
        )
        with_voice: dict[str, object] = {}
        peaks["with_voice"] = with_voice
        for speaking in cached:
            with_voice[voice_for_model(speaking).value] = measure_peak(
                device, model, speaking, warm_clip
            )


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
        with ports.serve(device, hearing_model, DEFAULT_SPEAKING_MODEL) as service:
            health = service.running.health
            file_answer = ports.post(service, HearingLanguage.DE, german.question)
            live = asyncio.run(ports.upgrade(service))
            wav, _, _ = ports.speak(
                service.base_url,
                GERMAN_QUESTION,
                voice_id(DEFAULT_SPEAKING_MODEL),
                service.minting,
            )
    freed = ports.freed(baseline)
    speaking, hearing = health["speaking"], health["hearing"]
    failures = figures.taken_card_failures(
        figures.TakenCardObservation(
            speaking_ready=isinstance(speaking, dict) and speaking["ready"] is True,
            hearing_ready=isinstance(hearing, dict) and hearing["ready"] is True,
            file_status=file_answer.status_code,
            file_code=refusal_code(file_answer.status_code, file_answer.content),
            live_status=None if live is None else live.status,
            live_code=None if live is None else live.code,
            spoke_bytes=len(wav),
        )
    )
    return {
        "hearing_model": hearing_model,
        "held_mib": held_mib,
        "used_while_held_mib": used_while_held,
        "health": health,
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
            f" Piper voice {ENGLISH_VOICE}"
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
                installed_voice_ids(load_settings()),
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
