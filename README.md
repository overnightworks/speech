# Speech runtime

Audience: the operator starting this process on the home server, and
Presentator, which calls it.

One process holds the German Piper voice resident and offers `GET /health`,
the speak wire and a route that hears a recording, over HTTP on a TCP address
(`SPEECH_HOST`, `SPEECH_PORT`). The caller owns the address. Beside Piper it
offers each GPU voice `SPEECH_GPU_VOICES` names (today Qwen3-TTS 0.6B), loaded
on its first request from weights that are already cached and kept until the
process ends. A GPU voice that fails to load or whose worker dies is refused
until a later request loads it again, while Piper and hearing keep serving. When
`SPEECH_HEARING_MODEL` names a model, the process also holds a faster-whisper
model resident, loaded from local files only. The service stores no choice of
its own.

The code was extracted, with its history, from `speech/` in
[overnightworks/presentator](https://github.com/overnightworks/presentator).

## Models

| Role | Hugging Face | Licence | Why this one |
| --- | --- | --- | --- |
| Speaking (default) | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) voice `de_DE-thorsten-medium` | Voice recordings CC0; ONNX weights under the piper-voices MIT repo. The `piper-tts` runtime is `GPL-3.0-or-later`. The HTTP boundary keeps it out of `src/speech_runtime_service` while the speech program itself carries the GPL obligations. | German is documented and not in question. #70 measured this voice on this 3090 at 0.23 s to first audio on CPU. `thorsten-high` on the same sentence took 1.21 s through this service, over the one-second first-audio target, so medium is the default. The voice does not compete with the listener for the card. |
| Speaking (optional) | [`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice), snapshot `85e237c12c027371202489a0ec509ded67b5e4b5` | Apache-2.0 (`qwen-tts` 0.1.1) | German and English through one configured built-in preset, initially Ryan. It returns a complete 24 kHz waveform. The isolated provider always launches on `cuda`, as `cuda:0`, uses bfloat16 and package-default attention, and reads only the pinned local snapshot. Runtime quality and latency remain unmeasured. |

Hearing, optional, `SPEECH_HEARING_MODEL`: faster-whisper (MIT) with the CTranslate2 conversions of OpenAI's Whisper (MIT) in
[`Systran/faster-whisper-large-v3`](https://huggingface.co/Systran/faster-whisper-large-v3)
(`large-v3`, the reference model) and
[`Systran/faster-whisper-small`](https://huggingface.co/Systran/faster-whisper-small)
(`small`, small enough for a CPU proof and not recommended where the meaning of
what is heard matters: see the numbers below), each pinned to a commit. German and
English. The model is never downloaded by the service: `speech-runtime-fetch-hearing-model <model>`
fetches the pinned files into `SPEECH_HUGGINGFACE_CACHE` once, and the start loads them from there only.

Checked and not chosen:

- **Kokoro / Kokoro-German** (`Tundragoon/Kokoro-German`, Apache-2.0). Official Kokoro has no documented German. The community German fine-tune warns it is undertrained.
- **XTTS-v2** and maintained forks. Weights under the Coqui Public Model License (non-commercial). ADR 0004 already excluded them.
- **Orpheus** (`Thorsten-Voice/tv-orpheus-v1`, Apache-2.0 on the fine-tune; some GGUF conversions inherit Llama 3.2). A 3B SNAC talker, roughly 8 GB resident. First audio for a short sentence is the LLM's first codec frame, not a VITS chunk; not measured on this card.

## Card

Piper runs on the CPU. Qwen's source
integration makes no residency or latency claim before the target-machine proof.

Measured on this RTX 3090 (24 GB), one German sentence (14 words) and one
English sentence (15 words), `scripts/prove_speech.py`, 5 repetitions each, median
reported, 08.09.2026.

Piper default:

| What | Number |
| --- | --- |
| Time to first byte of `/speak` (wire, German, median of 5) | 0.216 s (Piper allows the first byte when the sentence is done; limit 0.5 s) |
| Time to first byte (English, median of 5) | 0.210 s |
| Real-time factor (German / English, median of 5) | 25.4× / 24.2× |
| WAV RMS | not silence on every repetition |
| Speaking footprint | 0 MiB GPU (CPU / ONNX) |

`scripts/prove_speech.py` fails the run the moment any single repetition
misses its first-byte limit; a passing run is the proof that every repetition
held, not just the medians. Qwen does not stream, so its first byte follows the
whole sentence, and its first request also waits for the load: its limit is half
again G0's load and first audio together, 6.9 s and 8.44 s measured in process,
which left out the HTTP hop and the container (`scripts/speech_figures.py`).
`scripts/prove_speech.py` writes the German WAV of the run it
just proved to `german.wav` in a private temporary directory the script prints for a listening judgment.

Hearing on the GPU needs `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`, which the `gpu` extra installs (`uv sync --locked --extra gpu`) and the service preloads at start. Without the extra, a `cuda` device leaves hearing unavailable and speaking stays up.

Weights stay in:

- Piper voices: `~/.cache/piper/` (`speech-runtime-fetch-voice <voice>`, pinned revision and SHA-256 in `piper_voices.py`)
- Qwen: `~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice` (`speech-runtime-fetch-voice qwen3-tts-0.6b` into `SPEECH_HUGGINGFACE_CACHE`, pinned snapshot `85e237c12c027371202489a0ec509ded67b5e4b5`)

## Contract

- `GET /health` answers without a token, also while a model is still loading. Its shape is in [`docs/wire.md`](docs/wire.md), Health.
- `POST /v1/audio/speech` → chunked `audio/wav`, German, with the AI mark. It needs a workload token and the person's token. The request, the tokens, the refusals and the settings that shape them are in [`docs/wire.md`](docs/wire.md).
- `POST /v1/audio/transcriptions` → `{"text"}`, for an app whose table says `hear = true`, in German or English (`language`, `de` when absent). The request, the limits and the refusals are in [`docs/wire.md`](docs/wire.md).
- `WS /v1/realtime?intent=transcription` → OpenAI's Realtime transcription events, for an app whose table says `hear = true`: PCM16 at 24 kHz in, provisional deltas and one terminal event per item out. The events, their order and the refusals are in [`docs/wire.md`](docs/wire.md).
- `/health`, `/v1/audio/speech`, `/v1/audio/transcriptions` and `/v1/realtime` answer on `SPEECH_HOST:SPEECH_PORT` and
  nothing else is mounted (no `/docs`, no `/openapi.json`, no voice
  administration). `python -m speech_runtime_service --help` (or the
  `speech-runtime-service` command) prints usage; any other argument is a usage
  error.

The service never writes spoken or heard text to its log. A speaking failure is logged with its exception types and stack frames, never with an exception message. A provider that ends before it is ready has seen no request, so its exit status and the tail of its stderr go to the log.

## Settings

All `SPEECH_*`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SPEECH_DEVICE` | `cuda` | Hearing's device, `cuda` or `cpu`. It moves no voice: Piper stays on the CPU, and Qwen always launches on `cuda`, the first card it sees (`cuda:0`). |
| `SPEECH_GPU_VOICES` | empty | The GPU voices offered beside Piper, as voice ids separated by commas: `qwen3-tts-0.6b`. Piper is always offered and loads at start; its weights must be there or the process exits. A GPU voice loads on its first request; when its weights or worker are missing, the log names the missing files and its requests answer 503 `unavailable` until they are there. `piper`, an unknown id or an empty item ends the process, naming the setting |
| `SPEECH_VOICE_CACHE` | `~/.cache/piper` | Where Piper ONNX files are kept; `speech-runtime-fetch-voice <voice>` fetches a voice there at its pinned revision, checked against its hashes (the service never fetches) |
| `SPEECH_HUGGINGFACE_CACHE` | Provider Hub cache (normally `~/.cache/huggingface/hub`) | The shared local Hub cache for provider loading and the artifact check |
| `SPEECH_QWEN_SPEAKER` | `Ryan` | One of Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, or Sohee |
| `SPEECH_HOST` | `127.0.0.1` | Address the listener binds. A container sets `0.0.0.0` so the app network reaches it |
| `SPEECH_PORT` | `8080` | TCP port the listener binds; a whole number from 1 to 65535. An address that cannot be bound ends the process before any model work |
| `SPEECH_BUSY_WAIT_SECONDS` | `5.0` | How long a speech request waits for its voice's one synthesis before it answers 429 `busy`, each voice apart; a request that waited for its voice to load counts from the load's end. At least 0, where 0 refuses at once |
| `SPEECH_MAX_WAITING_REQUESTS` | `3` | How many speech requests may wait for each voice's one synthesis; a request beyond them answers 429 `busy` at once. A whole number of at least 0, where 0 allows no waiting |
| `SPEECH_MAX_BODY_BYTES` | `32768` | The largest speech request body; a larger one answers 413 `too_large` before it is parsed. A whole number of at least 1 |
| `SPEECH_MAX_INPUT_CHARACTERS` | `4096` | The longest `input` a speech request may carry; a whole number of at least 1 |
| `SPEECH_HEARING_MODEL` | none | The hearing model the process loads at start, on `SPEECH_DEVICE`: `large-v3` (recommended) or `small`. None: hearing answers 503 `unavailable`. A model that was not fetched stops the start naming this setting (fetch it with `speech-runtime-fetch-hearing-model <model>`); a card that is missing, taken or too small, or a `cuda` device without the `gpu` extra, leaves hearing unavailable in `/health` while speaking stays up |
| `SPEECH_HEARING_BUSY_WAIT_SECONDS` | `30.0` | How long a recording waits for the one decode before it answers 429 `busy`; at least 0, where 0 refuses at once |
| `SPEECH_MAX_HEARING_SESSIONS` | `1` | How many live hearing sessions run at once; one more is refused at the upgrade with 429 `busy`. A whole number of at least 1 |
| `SPEECH_HEARING_FINAL_WAIT_SECONDS` | `60.0` | How long a live turn's final may wait for the one decode, counted from the turn's end; past it the session ends with `busy`. Above 0 |
| `SPEECH_HEARING_MAX_WAITING_REQUESTS` | `3` | How many recordings may wait for the one decode; a request beyond them answers 429 `busy` at once. A whole number of at least 0, where 0 allows no waiting |
| `SPEECH_MAX_RECORDING_BYTES` | `26214400` | The largest transcription request body; a larger one answers 413 `too_large` before it is parsed. A whole number of at least 1 |
| `SPEECH_MAX_RECORDING_SECONDS` | `300` | The longest recording; a longer one answers 413 `too_large` and is not transcribed. Above 0 |
| `SPEECH_CONFIG_FILE` | none, required | Absolute path of the TOML file that names the issuer, the service group, the apps whose tokens the routes trust, the voices each app may name and whether it may hear; the process does not start without it ([`docs/wire.md`](docs/wire.md)) |

When `SPEECH_HUGGINGFACE_CACHE` is absent, `HF_HUB_CACHE` and `HF_HOME` influence the provider default. An explicit `SPEECH_HUGGINGFACE_CACHE` value is already the Hub-cache root.

One instance serves one card, on one address: hearing on the device
`SPEECH_DEVICE` names and a GPU voice on the first card it sees. The
platform starts one instance per card. The service holds no lock per card.

At start the process loads Piper, then hearing; until both are done, every speech request answers 503 `unavailable`. If Piper cannot be loaded the process exits and names the model. A missing hearing resource leaves hearing unavailable and the process up. A GPU voice loads once, on its first request, and its waiting requests share that load; nothing unloads it.

## Install

Three distributions are released together at one version, as the Git tag
`vX.Y.Z`, and go to no registry. A host installs each by its Git pin:

```text
overnightworks-speech-runtime-client @ git+https://github.com/overnightworks/speech-runtime@vX.Y.Z#subdirectory=packages/client
overnightworks-speech-runtime-provider-contract @ git+https://github.com/overnightworks/speech-runtime@vX.Y.Z#subdirectory=providers/provider-contract
overnightworks-speech-runtime-service @ git+https://github.com/overnightworks/speech-runtime@vX.Y.Z
```

The service is an application, not a library: its pin installs the CPU path
(Piper); GPU hearing needs the pin with `[gpu]`. Every GPU voice runs from its own provider environment.
`scripts/check_release.py` holds the released distributions at one version, and the Release
workflow runs it, and the pins' installs, before it creates the tag.

## Run

From the repository root:

```sh
uv sync --group dev
uv run speech-runtime-service
```

For GPU hearing, sync with `uv sync --locked --extra gpu --group dev`.

### As a container

`Dockerfile` and `compose.yaml` run the service with the Piper voice and
hearing with `small` (`int8`, four threads) on the CPU, built from the locks:
`uv sync --locked --no-dev` installs the CPU path, and the voice
`de_DE-thorsten-medium` (checked against its hashes) and the `small` model are
fetched at build, each at its pinned revision, so the container fetches nothing
at start or restart. The image also carries Qwen, which it offers only when
`SPEECH_GPU_VOICES` names `qwen3-tts-0.6b`: its environment under
`SPEECH_PROVIDER_ROOT` (`/opt/providers`), installed `--locked` from
`providers/qwen/uv.lock` with wheels only except `sox` and the two local
projects, and its weights at their pinned snapshot in the Hub cache beside
hearing's model. That adds about 9 GB to the image. uv's and the Hub's
downloads go into BuildKit cache mounts, which outlive a build, so a build that
hal's 1800 s cap kills leaves the next one only what it had not fetched yet;
the image takes a copy of the Hub cache, so after a pin moves on a host whose
build cache still holds the old revision, the image carries both until that
cache is pruned. `compose.yaml` reserves the host's one card (`gpus`, one
`nvidia` device, no memory budget, which is hal's to declare), so every start
needs the nvidia runtime, Piper alone included; the card is touched only when a
GPU voice's first request loads it. `deploy/speech.toml` is
copied into the image and holds the issuer, the key set's internal address and
the callers, presentator with `hear` and hopin without, each of them allowed
`piper` and `qwen3-tts-0.6b`; a release tag fixes it. The
service joins no default network, only the three external, internal networks
`hal_speech_issuer`, `hal_speech_hopin` and `hal_speech_presentator`, which hal
creates; without them `docker compose up` fails and nothing runs with egress.
The root is read-only, the user is not root, and the identity issuer's CA is
mounted from `~/.secrets/step-ca/root_ca.crt`. The healthcheck requires
`speaking.ready` and `hearing.ready`, because `/health` answers while the models
load and the baked model makes a hearing that did not load a defect of the build.
Starlette spools an upload above 1 MiB to `/tmp`; the service parses the form
only once a request holds the decode, so one spool exists at a time, up to
`SPEECH_MAX_RECORDING_BYTES`, while a waiting request holds its body in memory,
not on `/tmp`.

`scripts/prove_image.py` builds that image and starts the real `compose.yaml`
on isolated networks with a test CA, a key-set server and a caller container.
It speaks as the baked table's callers, hears a recording and a live turn as the
app that may hear, refuses the one that may not, checks the refusals, a visible
refusal with the key server stopped, the absence of egress, a restart and the
negatives that make each check able to fail. By default it shows Qwen refused
`unavailable` with `param` `voice` and no worker started; offered through
`tests/fixtures/image/compose.gpu-voice.yaml`, Qwen loads on its first request
and speaks, and the proof prints its load, first audio, real-time factor and
peak card memory beside G0's in-process figures, while Piper and hearing serve
and the container keeps running. It ends with `PROOF_OK`. It takes the probe
lock as the stage proof does, and removes its containers, networks and image on
every exit, keeping the build caches.

A cold build alone may take most of one lock run on this host, so the proof
runs in stages, each its own lock run and each named with `--stage`:

- `cold-build` times a build with no cache against hal's 1800 s cap; BuildKit
  then also gives each step an empty build cache, so it fetches everything;
- `rebuild` times a second build from the layers and build caches the first
  left, as hal's next build runs, against half of the cap, which leaves room for
  what a build the cap killed had not fetched yet and for this host's spread,
  then runs the cells above;
- `card` needs the card: by default no process of the service computes on it
  while Piper speaks and hearing hears; while `scripts/hold_card.py` holds it,
  a Qwen request is refused `unavailable` with `param` `voice` and Piper and
  hearing serve; once it is free, the next request loads Qwen and speaks.

`--no-card` says the machine has neither the card nor the nvidia runtime; only
then does the proof lay `tests/fixtures/image/compose.no-card.yaml`, which drops
the reservation, and offered Qwen then finds no CUDA device on each request.
Without it, a machine that lacks either fails the run before the build, naming
what it lacks. CI runs `cold-build` and then `rebuild` with `--no-card` as the
job `Speech: image`; the host runs all three without it:

```sh
uv run python scripts/prove_image.py --stage cold-build
uv run python scripts/prove_image.py --stage rebuild
uv run python scripts/prove_image.py --stage card
```

### Stage proofs

Stage proof (checks load, one heavy step at a time). The script runs itself as
`flock -o -w 3600 /tmp/probe-stack.lock setpriv --pdeathsig TERM timeout -k 30 1800 …`, so it waits at most
an hour for `/tmp/probe-stack.lock` and fails the run, saying the lock is busy,
when that wait ends; the load wait and the proof then share the 30-minute cap,
whose TERM stops the speech server and its providers and removes the private
directory. If its caller dies, `setpriv` hands `timeout` a TERM at once, so
the lock stays held only while the stack tears down; a host without `setpriv`
fails the run, naming it. It speaks the GPU voice `SPEECH_GPU_VOICES` names, else Piper, and points
`SPEECH_PROVIDER_ROOT` at `providers` unless the caller set it, so each
provider needs its virtual environment first (`uv sync --locked` inside
`providers/<provider>`). `SPEECH_DEVICE` (default `cuda`) passes through; on
`cpu` the card is hidden from speech with an empty `CUDA_VISIBLE_DEVICES`. Weights must already be cached, because
`HF_HUB_OFFLINE=1` is the default and resolves `refs/main`:

```sh
uv run python scripts/prove_speech.py
```

The proof serves its own throwaway key set over local HTTPS and speaks with
fresh tokens minted from it; nothing of it outlives the run.

Every run reports load to ready, the card's compute pids at health ready and
the peak RSS of the speech process, and ends with `PROOF_OK` when health and
speaking held. On the CPU with Piper:

```sh
SPEECH_DEVICE=cpu OMP_NUM_THREADS=4 uv run python scripts/prove_speech.py
```

Hearing stage proof, under the same lock form. It hears a German and an English
question and one long turn per language, live over the Realtime WebSocket and
from a file, with the medians of at least five runs, the word errors against the
known sentence and the growing deltas of the long turn. On `cuda` it also
records the card's peak memory per hearing model, alone and with each GPU voice
`SPEECH_GPU_VOICES` names, and starts the service on a card a second process holds.
On `cpu` those two stages are recorded as not applicable. The clips are
synthetic: German from the service's speak route, English from a Piper voice
fetched into `--english-voice-directory` when absent. It never downloads a
hearing model (fetch it with `speech-runtime-fetch-hearing-model`) and never
starts or stops anything but its own processes, so the caller makes the card
free first. It writes the JSON record to `--record` after each language, model
and stage ends, and `not_measured` names what is missing (on `cuda` both hearing
models, every stage not finished, every median too few runs produced, every long
turn with too few growing deltas or without a timing, and every empty heard
text); only a record with an empty list is `complete`:

```sh
CUDA_VISIBLE_DEVICES= uv run python scripts/prove_hearing.py --device cpu \
  --hearing-model small --record record.json --english-voice-directory ~/.cache/piper-en
```

### Hearing on the GPU: numbers and budget

Hearing runs on the GPU with `large-v3`. The numbers are the medians of seven
runs of one short question per language on the RTX 3090 with a quiet machine
(load average about 1), through the hearing stage proof, with synthetic, clean
clips. Seconds; the first partial is counted from the start of the question and
the final from its end. A negative time means the partial arrived while the
question was still being spoken. The end is the end of the clip; the silence
at the clip's own tail lies before that end and counts toward the 0.7 s the
service waits for before it ends a turn, so a final can come sooner than 0.7 s
after the end. The first partial needs two agreeing decodes.

| Model | Language | First partial | Final | File route |
| --- | --- | --- | --- | --- |
| `large-v3` | German | 1.38 (−1.14 from the end) | 0.73 | 0.27 |
| `large-v3` | English | 1.36 (−1.44) | 1.01 | 0.26 |
| `small` | German | 1.26 (−1.26) | 0.58 | 0.14 |
| `small` | English | 1.25 (−1.56) | 0.89 | 0.13 |

The reference numbers are 6.8 s to the first partial and 13.9 s to the final
on the CPU, and 1.1 s and 9.6 s on the GPU, the GPU pair from a single
measurement; all of them count from the end of the question. In that sense
`large-v3` gives its first partial 1.14 to 1.44 s before the end, against 1.1 s
after it, and its final 0.73 to 1.01 s after the end, against 9.6 s.

A 16 to 19 s turn on the GPU gives 22 to 26 growing deltas about 0.6 s
apart. On the same card under a machine load of about 18 the final took 0.9 to
1.2 s and the file route 0.4 to 0.7 s.

Both models heard all four clips without a word error in the complete run. In
another run `small` heard the German question as "Wie geht es und wann beginnt
die nächste Sitzung?", 14 word errors of 70 and a different question, so a clean
run is no proof for it; clips are drawn anew in each proof. That is why
`large-v3` is the model to use.

Declared budget, GPU memory over the card's idle use of about 0.7 to 0.9 GiB.
The sampled peak is the highest of the card's use taken every 0.25 s over the
warmed service, over the card's use before the service started, the highest of
two runs (a quiet machine and a load of about 18). The budget is that peak plus
5 %, rounded up to 256 MiB; it holds for both runs, with a margin of 6 to 9 %
over the peak (30 % for `small` alone):

| Next to hearing | `large-v3` budget (sampled peak) | `small` budget (sampled peak) |
| --- | --- | --- |
| Piper (CPU) | 4608 MiB (4282) | 1280 MiB (983) |
| Qwen3-TTS 0.6B | 7424 MiB (6832) | 3840 MiB (3593) |

Speaking and hearing were measured one after the other,
never at the same moment, so the table is the memory the two hold loaded and
not the peak of both working at once.

When a second process holds the card, the service starts, `/health` names
hearing not ready, hearing answers 503 `unavailable` and speaking on the CPU
voice stays up.
