# The speech wire

Audience: an app developer who speaks, or has audio heard, through this
service.

**WIRE-01** One request speaks one text: `POST /v1/audio/speech` on the TCP address
`SPEECH_HOST:SPEECH_PORT`. It is a strict subset of
OpenAI's speech endpoint, so an OpenAI-shaped client can build the body.

**WIRE-02** One request hears one recording: `POST /v1/audio/transcriptions`, a strict
subset of OpenAI's transcription endpoint, described under
[Hearing a recording](#hearing-a-recording).

**WIRE-03** Live audio is heard over the
Realtime transcription WebSocket `/v1/realtime?intent=transcription`, described
under [Hearing live](#hearing-live).

`speech_runtime_client.wire` (the `overnightworks-speech-runtime-client`
distribution) owns the paths, the request types, the transcription answer, the
Realtime events and the refusal list; build the body and read the refusals through it.
`speech_runtime_client.speak` makes the speaking call itself, within a total deadline and a
byte cap, and answers with a typed outcome; its module documents them.
`speech_runtime_client.health` fetches `GET /health` (WIRE-122) within a deadline and a
cap and answers with a typed outcome; the outcomes the calls share live in
`speech_runtime_client.outcomes`. `speech_runtime_client.hearing` runs the live hearing
session: audio in, typed events out, one closed outcome when it ends, within a total deadline.

## Request

**WIRE-04** A JSON object with exactly these fields, all required, all strict JSON types:

| ID | Field | Accepted value |
|---|---|---|
| WIRE-05 | `model` | `speech-runtime`. Any other value is refused, so a request naming another model is never answered by this one. |
| WIRE-06 | `input` | Text with at least one letter or digit, at most `SPEECH_MAX_INPUT_CHARACTERS` characters (default 4096; the operator may raise it above OpenAI's 4096). |
| WIRE-07 | `voice` | The id of an offered voice: `piper`, always offered, or `qwen3-tts-0.6b` when `SPEECH_GPU_VOICES` names it. |
| WIRE-09 | `voice` | A GPU voice that is not loaded yet loads for this request, which waits for the load. |
| WIRE-10 | `response_format` | `wav` or `opus`. OpenAI defaults an absent field to `mp3`, so it is required here. |

The service speaks German only. **WIRE-11** There is no `language` field, and any field
not listed here (`speed`, `instructions`, `stream_format`, `language`, ...) is
refused.

## Headers

**WIRE-12** A speak request carries the app's workload token in `Speech-Workload-Token`
and the person's token as `Authorization: Bearer <token>`.
`speech_request_headers` in `speech_runtime_client.wire` builds them with the
content type, and `wire.py` owns the header names. The service answers only
when both tokens are valid and belong together:

- **WIRE-13** Both come from the configured issuer, name the service in `aud`, and are
  unexpired. The app requests the service's audience scope in its workload
  grant and at the person's sign-in.
- **WIRE-14** The workload token carries the configured realm role and comes from the app's
  registered client (`azp` and `client_id`).
- **WIRE-15** The person's token has the same `azp`, a different subject, membership of
  `/apps/<app>` or a group below it, and membership of exactly the configured
  service group (no subgroup of it).

An agent with its own account is the person: its app exchanges the agent's token for one
of its own client through Keycloak's standard token exchange, and that token is the
person's token above.

**WIRE-16** The service checks the headers before it reads the body, and **WIRE-17** the wait of
`busy` starts only after the tokens are accepted, or, for a request that
waited for its voice to load, when that load ends. **WIRE-19** An unknown voice is
`invalid_request` with `param` `voice`. **WIRE-20** Calls an app makes as itself,
without a person, are not offered in version 1.

## Admission

**WIRE-21** Each voice and hearing have one scarce resource, the one synthesis of that
voice and the one decode a process runs at a time, each under its own limits; a
request is refused only by the limits of its own resource. For speaking, **WIRE-22** a
request that finds its voice's synthesis taken may wait, within two limits that
hold for each voice apart, and anything beyond them is refused before any work
starts:

- **WIRE-23** `SPEECH_MAX_WAITING_REQUESTS` places for requests that wait for the synthesis.
  A request that finds the synthesis taken and every place taken is refused at
  once with `busy` and `Retry-After`.
- **WIRE-24** `SPEECH_BUSY_WAIT_SECONDS`: how long a waiting request waits. When it runs out,
  the answer is the same `busy` with `Retry-After`. **WIRE-25** Requests waiting for a GPU
  voice's load hold no place, and their wait starts when the load ends.
- **WIRE-26** `SPEECH_MAX_BODY_BYTES`: the largest request body. The service stops reading
  at the limit, whatever `Content-Length` says, and answers `too_large` without
  parsing the body. **WIRE-27** It answers only a caller whose tokens are accepted.

**WIRE-28** A refused request holds no place, so the next one is served.
**WIRE-132** A speak request whose caller has gone frees its voice for the next request.

## Hearing a recording

`POST /v1/audio/transcriptions` answers the words of one recording. **WIRE-29** It carries
the same two headers as a speak request and **WIRE-30** is checked in this order, each step
before the next: the tokens, whether hearing is
available, the body's size, the decode's admission, the form, the recording's
length, the decoding itself.

**WIRE-31** The body is `multipart/form-data` with exactly these fields:

| ID | Field | Accepted value |
|---|---|---|
| WIRE-32 | `file` | The recording, required, in any container and codec PyAV reads (WAV, MP3, Ogg, FLAC, ...). |
| WIRE-33 | `model` | `speech-runtime`, required. |
| WIRE-34 | `language` | `de` or `en`; `de` when absent. Any other value is refused. |
| WIRE-35 | `response_format` | `json`; `json` when absent. |

**WIRE-36** Any other field (`prompt`, `temperature`, `timestamp_granularities[]`, ...) is
refused. **WIRE-37** The answer is `200` with `{"text": "..."}`; **WIRE-38** the same recording gives
the same text, because the service decodes greedily, at temperature 0.

**WIRE-40** Hearing is `unavailable` while no model is configured
(`SPEECH_HEARING_MODEL`), while it loads, when its load found no CUDA device or
no room on the card, and when a decode fails.

The limits, each answered without decoding more than it must:

- **WIRE-41** `SPEECH_MAX_RECORDING_BYTES` bounds the request body. The service stops
  reading at the limit, whatever `Content-Length` says, and answers `too_large`
  before it parses the form.
- **WIRE-42** `SPEECH_MAX_RECORDING_SECONDS` bounds the recording's length. The decoder
  stops at the limit and the answer is `too_large` with `param` `file`; a
  recording longer than the limit is never transcribed in part.
- **WIRE-43** `SPEECH_HEARING_MAX_WAITING_REQUESTS` places for requests that wait for the
  one decode, and `SPEECH_HEARING_BUSY_WAIT_SECONDS` for how long they wait. A
  request that finds the decode taken and every place taken is refused at once
  with `busy` and `Retry-After`; so is one whose wait runs out. **WIRE-44** The decode is
  freed only when it has finished, even if the caller has gone.

**WIRE-45** Audio that PyAV cannot read, or that holds no sound, is `invalid_request` with
`param` `file`. **WIRE-46** A body that is not a multipart form, or has no `file`, is
`invalid_request` with `param` `file` as well. **WIRE-47** The form is parsed only once the
request holds the decode, so one upload is spooled to disk at a time: while the
service is busy, a malformed form is answered `busy` like any other request, and
`invalid_request` once it reaches the decode.

## Hearing live

**WIRE-48** `WS /v1/realtime?intent=transcription` hears audio as it arrives. **WIRE-49** It speaks the
events of OpenAI's Realtime transcription session that this service applies, and
nothing else. **WIRE-50** Both tokens travel in the upgrade request's headers, as on the
other routes; a client sends them from a server, never from a browser.

**Upgrade.** **WIRE-51** A refusal before the session is an HTTP answer in the refusal form
below, in this order: the tokens, the
`intent` (`invalid_request`, `param` `intent`), whether hearing is available
(`unavailable`), the session bound. **WIRE-52** `SPEECH_MAX_HEARING_SESSIONS` (default `1`)
sessions run at once; one more is `busy` with `Retry-After`. **WIRE-53** A wrong version is
`unsupported_version`. **WIRE-54** An idle session holds its place only, never the decode.

**Setup.** **WIRE-55** The service sends `session.created` with its default session: PCM16
mono at 24 kHz, `de`, the model `speech-runtime`, server turn detection. **WIRE-56** The
client may send one `session.update` with `session.type` `transcription` before
its first append; the service answers `session.updated` with the applied session.
**WIRE-57** Accepted: the format absent or exactly `{"type": "audio/pcm", "rate": 24000}`;
`transcription.model` absent or `speech-runtime`; `transcription.language`
absent, `de` or `en`; `turn_detection` absent or `{"type": "server_vad"}`. **WIRE-58** Anything
else ends the session with an `error`, among it a second update, an update after
audio, `null` turn detection, `languages`, and every field the service does not
apply (prompt, keywords, noise reduction, include).

**Client events.** **WIRE-59** Only `session.update`, `input_audio_buffer.append` (base64
PCM16 at 24 kHz) and
`input_audio_buffer.commit`. **WIRE-60** Invalid base64 or an odd byte count ends the session. **WIRE-61** Any other event type, `input_audio_buffer.clear`
included, is a protocol violation. **WIRE-62** A client `event_id` is optional; the service
echoes it only in an `error` about that event. **WIRE-63** A commit ends the open turn at
once; **WIRE-64** a commit with no audio since the last turn ended is a protocol violation.
**WIRE-65** With server turn detection a client need not commit.

**Turns.** **WIRE-66** The service ends a turn after speech followed by 0.5 s of silence, or
at 30 s of audio; the rest of the audio opens the next turn. **WIRE-67** A session holds at
most two ended turns, both queued to be heard from their turn's end, and the
open turn, so at most 90 s of audio.

**Server events**, in the order one turn produces them:

| ID | Event | When |
|---|---|---|
| WIRE-68 | `session.created` | first, on every session |
| WIRE-69 | `session.updated` | after a valid `session.update` |
| WIRE-70 | `input_audio_buffer.speech_started` (`item_id`, `audio_start_ms`) | speech first heard in the turn |
| WIRE-71 | `conversation.item.input_audio_transcription.delta` (`item_id`, `content_index` 0, `delta`) | zero or more, before the commit: the words two consecutive decodes agree on. Provisional. |
| WIRE-72 | `input_audio_buffer.speech_stopped` (`item_id`, `audio_end_ms`) | silence or the cap ended the turn; not sent after a client commit |
| WIRE-73 | `input_audio_buffer.committed` (`item_id`, `previous_item_id`) | the turn is closed; `previous_item_id` is `null` for the first |
| WIRE-74 | `conversation.item.created` (`item`, `previous_item_id`) | right after `committed`; the item is a user message with one `input_audio` part whose `transcript` is `null` |
| WIRE-75 | `conversation.item.input_audio_transcription.completed` (`item_id`, `content_index` 0, `transcript`, `usage`) | the authoritative text, possibly empty, with `usage` `{"type": "duration", "seconds": ...}`; it replaces the deltas |
| WIRE-76 | `conversation.item.input_audio_transcription.failed` (`item_id`, `content_index` 0, `error`) | only as a session ends |
| WIRE-77 | `error` (`error.type`, `error.code`, `error.message`, `error.param`, `error.event_id`) | the session ends after it |

**WIRE-78** Every server event has a fresh `event_id`. Correlate by `item_id`: OpenAI does not
order completions across items; **WIRE-79** here the second ended turn's `completed` waits
for the first's, so they come in turn order, but do not depend on it.

**Every announced item ends once.** **WIRE-80** An item is announced by its `speech_started`,
its first delta or its `committed`. While connected it ends with exactly one
`completed` or one `failed`. **WIRE-81** When the service ends a session (a decode fails or
the engine is lost, a final waited longer than `SPEECH_HEARING_FINAL_WAIT_SECONDS`,
a third ended turn while two are outstanding, or a protocol violation) it sends one `failed`
per announced unfinished item, then one `error`, then closes. **WIRE-82** The `error`'s `code`
is `invalid_request` (`param` the field at fault), `busy` or `unavailable`. **WIRE-83** A client
that disconnects gets nothing.

**Finals first.** **WIRE-84** A turn's final waits for the one decode ahead of every waiting
recording, behind earlier finals. **WIRE-85** Its deadline runs from the turn's end for
`SPEECH_HEARING_FINAL_WAIT_SECONDS` (default `60`, above 0), the time behind the
session's own earlier final included; a final past it is never started, even on a
free decode, and the session ends `busy`. **WIRE-86** A partial that finds the decode taken is
skipped. **WIRE-87** The decode is freed only when it has finished, even if the client has gone.

## Response

**WIRE-88** For `wav`, `200` with `Content-Type: audio/wav`, chunked: 16-bit mono PCM at the voice's
native rate, which is in the WAV header. **WIRE-89** Every WAV carries an AI mark, a RIFF
`LIST`/`INFO` `ICMT` chunk before `data`. **WIRE-90** The status and headers leave before
synthesis, so a synthesis failure cannot change the status: the server aborts
the connection without the chunked terminator. Read the body to its end and
treat a missing end as a failed request; never use partial audio.

**WIRE-133** For `opus`, `200` with `Content-Type: audio/ogg`, chunked: Ogg Opus
(RFC 7845), mono at 48 kHz whatever the voice's rate, its pages leaving as the voice
speaks. RFC 7845 names `audio/ogg`; OpenAI lists `audio/opus` among its answer types
and names no container, so the media type deviates from OpenAI's. **WIRE-134** Every
Ogg Opus carries the AI mark as `DESCRIPTION=<mark>` in its OpusTags.

## Refusals

**WIRE-91** Every refusal is `Content-Type: application/json` in OpenAI's error form,
`{"error": {"message", "type", "param", "code"}}`. **WIRE-92** `message` is a fixed
sentence per code and never echoes the request. **WIRE-93** `type` is
`invalid_request_error` for a 4xx and `server_error` for a 5xx. **WIRE-94** `param` is one
of `file`, `model`, `input`, `voice`, `language`, `response_format`, `intent`, or `null`
when no request field is at fault (an unknown field, a body that is not a JSON
object). Apps read `code`. With several faults in one request, which field `param`
names is not part of the contract.

| ID | `code` | Status | When | Header |
|---|---|---|---|---|
| WIRE-95 | `unsupported_version` | 400 | any method or upgrade on `/v{n}` or `/v{n}/...` with `n` other than 1 | |
| WIRE-96 | `not_found` | 404 | any other path, method or upgrade under version 1 | |
| WIRE-97 | `unauthorized` | 401 | a token is missing, malformed, expired, for another issuer or audience, badly signed, or lacks a subject or a valid group claim | `WWW-Authenticate`: `Bearer` |
| WIRE-98 | `forbidden` | 403 | both tokens are valid but do not permit the request: not a workload token, an unregistered app, tokens of different clients or of one subject, a person outside the app group or the service group | |
| WIRE-100 | `invalid_request` | 422 | the body is not a JSON object, a field is missing, unknown or mistyped, a value is not accepted, or `input` is empty, without a letter or digit, or over the limit; for a recording: the form or `file` is missing, a field is unknown or not accepted, or the audio cannot be decoded (`param` `file`) | |
| WIRE-101 | `too_large` | 413 | the body is over `SPEECH_MAX_BODY_BYTES` (a recording: `SPEECH_MAX_RECORDING_BYTES`), or the recording is longer than `SPEECH_MAX_RECORDING_SECONDS` (`param` `file`) | |
| WIRE-102 | `busy` | 429 | the one synthesis (a recording: the one decode) is taken and its waiting places are taken (answered at once), or it is not free within its wait: `SPEECH_MAX_WAITING_REQUESTS` and `SPEECH_BUSY_WAIT_SECONDS`, for a recording `SPEECH_HEARING_MAX_WAITING_REQUESTS` and `SPEECH_HEARING_BUSY_WAIT_SECONDS` | `Retry-After`: the wait in whole seconds, at least 1 |
| WIRE-103 | `unavailable` | 503 | the tokens cannot be checked because the issuer's key set is unreachable and no cached key serves, or the named voice cannot speak: it is not offered, the start load still runs, its load failed or found its files missing, its worker died while the request waited, or the service is shutting down (`param` `voice`); for a recording: hearing is not loaded or configured, or its decode failed | |

**WIRE-104** Apps read the `busy` wait with `busy_wait(status, headers)` in
`speech_runtime_client.wire`: it returns the wait in whole seconds, `None` for any
other status, and fails loud on a `busy` without whole seconds, at least 1, in `Retry-After`.

**WIRE-105** `n` is read as a decimal integer, so `/v01/...` is version 1. **WIRE-106** A path that is no
version path (`/v-1/...`, `/vx/...`, `/speak`) gets the app's default 404 with
`{"detail": "Not Found"}`, outside the wire.

## Settings that shape the wire

- **WIRE-107** `SPEECH_GPU_VOICES` (voice ids separated by commas, none when absent): the
  GPU voices offered beside Piper.
- **WIRE-108** `SPEECH_BUSY_WAIT_SECONDS` (default `5.0`, at least 0): how long a request
  waits for its voice's synthesis before `busy`.
- **WIRE-109** `SPEECH_MAX_WAITING_REQUESTS` (default `3`, at least 0, where 0 allows no
  waiting): how many requests wait for each voice's synthesis.
- **WIRE-110** `SPEECH_MAX_BODY_BYTES` (default `32768`, at least 1): the largest body. It
  holds a 4096-character `input` even when every character is escaped as a
  six-byte `\uXXXX`, plus the other fields.
- **WIRE-111** `SPEECH_MAX_INPUT_CHARACTERS` (default `4096`, at least 1): the longest
  `input`.
- **WIRE-112** `SPEECH_CONFIG_FILE` (absolute path, required): a TOML file that says whom the
  service trusts. The service does not start without a valid file. **WIRE-113** The
  `[identity]` table holds `issuer`, `jwks_url` (https; it may carry a
  credential and is never printed), `audience`, `workload_role`, `group_claim`,
  `service_group` (a group path such as `/services/speech-service`). **WIRE-114** Each `[apps.<name>]` table, at least one,
  holds `workload_client`, the client id of that app's workload token; a client
  belongs to one app, and the app group `/apps/<name>` follows from the name.
  Every registered app may name every offered voice and may hear. **WIRE-116** Unknown keys are refused.
- **WIRE-117** `SPEECH_HEARING_MODEL` (`large-v3` or `small`, none when absent): the hearing
  model the process loads at start. With none, hearing is `unavailable`.
- **WIRE-118** `SPEECH_HEARING_BUSY_WAIT_SECONDS` (default `30.0`, at least 0) and
  `SPEECH_HEARING_MAX_WAITING_REQUESTS` (default `3`, at least 0): the
  decode's wait and its waiting places, as for speaking.
- **WIRE-119** `SPEECH_MAX_HEARING_SESSIONS` (default `1`, at least 1): live sessions at
  once; one more is refused at the upgrade as `busy`.
- **WIRE-120** `SPEECH_HEARING_FINAL_WAIT_SECONDS` (default `60`, above 0): how long a live
  final may wait for the decode, from its turn's end.
- **WIRE-121** `SPEECH_MAX_RECORDING_BYTES` (default 25 MiB, at least 1) and
  `SPEECH_MAX_RECORDING_SECONDS` (default `300`, above 0): the largest request
  body and the longest recording.

## Health

**WIRE-122** `GET /health` answers without a token, so a supervisor can poll it:
`{"speaking": {"ready", "voices": {"<id>": {"model", "ready", "sample_rate"}}}, "hearing": {"model", "ready"}, "card_memory_mb"}`.
**WIRE-123** It answers while the models are still loading, with `ready` false.
**WIRE-124** `speaking.ready` is the speaking service's own readiness: whether Piper can
speak. **WIRE-125** `speaking.voices` lists the offered voices by id; a voice reports its
catalogue `model` whether it is loaded or not, and its WAV `sample_rate` (Piper's
0 until it is loaded). **WIRE-126** A voice's `ready` is true exactly when a request for it
would be spoken now: never during the start load or once shutdown began, and
never for a GPU voice that is not loaded, failed, or whose worker died.
**WIRE-127** The service builds the answer through `Health` in `speech_runtime_client.wire`,
which ignores keys it does not know. **WIRE-128** `hearing.model` is the configured hearing
model, or `unavailable` when none is configured; **WIRE-129** `hearing.ready` is false while
it loads and for good when its load found no CUDA device or no room on the card,
which leaves speaking up.

## Not offered

**WIRE-130** Version 1 has no voice administration: `/v1/voices` answers
`not_found`, and the service mounts no `/hear` and no `/voices*` route at all.
**WIRE-131** The service serves no API documentation: `/docs`, `/redoc` and
`/openapi.json` answer `404`.
