# Speech

The home server's speech service: it speaks text in the voices of a closed
catalogue and hears recordings and live audio for the apps that call it, Presentator among them. The operator
starts and configures it; an app developer reads the wire.

One process holds the offered Piper voices resident and offers `GET /health`, the
speak route, a route that hears a recording and a live hearing WebSocket, over
HTTP on a TCP address. Beside them it offers each GPU voice the operator names
(today Qwen3-TTS 0.6B's `ryan`), loaded on its first request. When a hearing model is
named, it also holds a faster-whisper model resident. Speaking and hearing need
the app's workload token and the person's token; `/health` needs none. The
service stores no choice of its own.

The code was extracted, with its history, from `speech/` in
[overnightworks/presentator](https://github.com/overnightworks/presentator).

## Quick start

From the repository root, with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
uv run speech-fetch-voice
SPEECH_CONFIG_FILE=/absolute/path/to/speech.toml uv run speech-service
```

The first command installs the CPU path (Piper). The second fetches the voices
`SPEECH_VOICES` offers (the five Piper voices by default) at their pinned revisions; the service never downloads. The third starts the
service on `127.0.0.1:8080`; it refuses to start without the caller file.
[`deploy/speech.toml`](deploy/speech.toml) is the file the container image
carries and shows its keys. `uv run speech-service --help` prints usage.

`uv run python scripts/prove_speech.py` starts the service against a throwaway
key set, speaks as a registered app and ends with `PROOF_OK`; on a host without
a card, run it as `SPEECH_DEVICE=cpu uv run python scripts/prove_speech.py`.

- **Hearing.** Fetch a model, then name it: `uv run speech-fetch-hearing-model large-v3`,
  then `SPEECH_HEARING_MODEL=large-v3`. On the GPU, sync with
  `uv sync --locked --extra gpu` as well, or hearing stays unavailable.
- **A GPU voice.** Name it beside the Piper voices in `SPEECH_VOICES`, for example
  `SPEECH_VOICES=de_DE-thorsten-medium,de_DE-kerstin-medium,en_US-norman-medium,en_US-ljspeech-medium,en_GB-cori-medium,ryan`,
  fetch with `uv run speech-fetch-voice` and run `uv sync --locked` inside
  `providers/qwen`. It always runs on `cuda`, so the card must be free.
- **A container.** `Dockerfile` and `compose.yaml` build and run the service on
  the CPU with the five Piper voices and the `small` hearing model,
  fetched at build. The image carries no GPU voice; see
  [Delivery](docs/ARCHITECTURE.md#delivery).

## Models

| Role | Model | Licence |
| --- | --- | --- |
| Speaking, Piper | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices), `de_DE-thorsten-medium` | Recordings CC0; ONNX weights under the piper-voices MIT repository; the `piper-tts` runtime is `GPL-3.0-or-later` |
| Speaking, Piper, German | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices), `de_DE-kerstin-medium` | CC0 for the dataset (OHF-Voice voice-datasets) per the model card; fine-tuned from the Hi-Fi TTS base model, CC BY 4.0 (NVIDIA), which needs credit |
| Speaking, Piper, English | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices), `en_US-norman-medium`, `en_US-ljspeech-medium`, `en_GB-cori-medium` | Public domain per each voice's model card (LibriVox for norman and cori, LJ Speech for ljspeech) |
| Speaking, optional | [`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice), German and English, the built-in speaker `ryan` | Apache-2.0 |
| Hearing, optional | [`Systran/faster-whisper-large-v3`](https://huggingface.co/Systran/faster-whisper-large-v3) (recommended) and [`Systran/faster-whisper-small`](https://huggingface.co/Systran/faster-whisper-small), German and English | MIT |

A speak request names a `model`, the engine, and a `voice` of it, and a `language` where the voice speaks several (`ryan`: `de` or `en`):

| `voice` | `model` | Name | Language |
| --- | --- | --- | --- |
| `de_DE-thorsten-medium` | `piper` | Thorsten | German |
| `de_DE-kerstin-medium` | `piper` | Kerstin | German |
| `en_US-norman-medium` | `piper` | Norman | English |
| `en_US-ljspeech-medium` | `piper` | LJ Speech | English |
| `en_GB-cori-medium` | `piper` | Cori | English |
| `ryan` | `qwen3-tts-0.6b` | Ryan | German and English, named by the request's `language` |

Each model is pinned to a revision. `small` fits a CPU and the image, but can
mishear a question, so `large-v3` is the model to use where the meaning counts.

## Settings

Every setting is read from `SPEECH_*` in the environment. The wire's settings
(the offered voices, the waits and limits, the hearing model and its limits, the
caller file) are in [`docs/wire.md`](docs/wire.md), Settings that shape the wire.
The process's own:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SPEECH_HOST` | `127.0.0.1` | Address the listener binds; a container sets `0.0.0.0` |
| `SPEECH_PORT` | `8080` | TCP port, 1 to 65535 |
| `SPEECH_DEVICE` | `cuda` | Hearing's device, `cuda` or `cpu`; Piper stays on the CPU and Qwen always uses `cuda` |
| `SPEECH_VOICE_CACHE` | `~/.cache/piper` | Where Piper's files are kept and fetched to |
| `SPEECH_HUGGINGFACE_CACHE` | the Hugging Face Hub cache | Where hearing's and Qwen's files are kept and fetched to |
| `SPEECH_PROVIDER_ROOT` | none | Directory of the provider environments, `providers` in a checkout; a GPU voice needs it |

## Install as a dependency

Three Python distributions and the browser package are released together at one
version, as the Git tag `vX.Y.Z`, and go to no registry. A host installs each
distribution by its Git pin:

```text
overnightworks-speech-client @ git+https://github.com/overnightworks/speech@vX.Y.Z#subdirectory=packages/client
overnightworks-speech-provider-contract @ git+https://github.com/overnightworks/speech@vX.Y.Z#subdirectory=providers/provider-contract
overnightworks-speech-service @ git+https://github.com/overnightworks/speech@vX.Y.Z
```

The client carries the wire's types, the bounded speaking call
(`speech_client.speak`), the bounded health fetch
(`speech_client.health`), the live hearing session
(`speech_client.hearing`), the relay of a browser's audio frames to that session
(`speech_client.relay`) and the finished utterances an app's server hears from
a browser (`speech_client.utterances`). The service is an application: its pin
installs the CPU path, and GPU hearing needs the pin with `[gpu]`.

The browser package `overnightworks-speech-browser` is a release file of the
tag, `overnightworks-speech-browser-X.Y.Z.tgz`, at the client's version. A host
pins its URL, and the release notes give the `sha512-` integrity that pnpm
writes into the lockfile:

```text
https://github.com/overnightworks/speech/releases/download/vX.Y.Z/overnightworks-speech-browser-X.Y.Z.tgz
```

The package gives an app's page two things. `startCapture` streams the app's
own microphone stream to the app's relay route as PCM16 at 24 kHz, and hands the
app, through an optional `onMessage`, each message its own route writes on that
socket. `createPlayer`
plays sounds one after another, each handed over as bytes or as the URL of the
app's own route, with stop, mute and close, and reports each sound's start and
end once. The app keeps the routes and the tokens, and none of the tokens reach the
browser; the switches are the app's own controls on its page.

`scripts/check_release.py` holds the distributions and the browser package at
one version.

## Release

The operator releases by dispatching a workflow with a version; his yes covers
whatever `main` holds at the dispatch, and release tags are his:

```bash
gh workflow run release.yml -f version=YYYY.MM.DD.N
gh workflow run release-distributions.yml -f version=X.Y.Z
```

`release.yml` releases the app: it builds, attests and pushes the image to GHCR,
then sets the dated tag `vYYYY.MM.DD.N`. `release-distributions.yml` releases the
three distributions and the browser package at one `X.Y.Z`, and uploads the
package's tarball as a release file. Both run quality's release gate; its refusals
and the path itself are in
[quality's `docs/release.md`](https://github.com/overnightworks/quality/blob/2d2b68bf3916b34f6f2c72a540c1e3d8a947b750/docs/release.md).

## Proofs

Each proof re-runs itself under `probe-run`, which waits for the machine to be
free and caps the run; the proof uses a free port and its own container and
project names, and cleans up what it started:

| Command | What it proves |
| --- | --- |
| `uv run python scripts/prove_speech.py` | Health and speaking: every offered Piper voice in its language, each repetition inside its first-byte limit; two voices at once; with a hearing model named, a recording heard while they speak |
| `uv run python scripts/prove_hearing.py --device cpu --hearing-model small --record record.json --english-voice-directory ~/.cache/piper-en` | Hearing, live and from a file, with timings, word errors and, on `cuda`, the card's memory |
| `uv run python scripts/prove_image.py` | The built image under the real `compose.yaml`, on isolated networks (CI job `Speech: image`) |

The hearing proof never downloads a hearing model and never stops anything but
its own processes. It names what it did not measure in `not_measured`; only a
record with an empty list is `complete`.

## Documentation

- [`docs/wire.md`](docs/wire.md): the speak, hearing and health wire, its refusals and settings, for an app developer.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): the parts, the trust, the delivery and the reasons, for an engineer changing the service.
- [`AGENTS.md`](AGENTS.md): this repository's commands and checks, for agents.

Licence: [MIT](LICENSE).
