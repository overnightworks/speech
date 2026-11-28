# The speech wire

Audience: an app developer who speaks, or has audio heard, through this
service.

- WIRE-01: One request speaks one text: `POST /v1/audio/speech` on the TCP address
  `SPEECH_HOST:SPEECH_PORT`. It is a strict subset of
  OpenAI's speech endpoint, so an OpenAI-shaped client can build the body.

- WIRE-02: One request hears one recording: `POST /v1/audio/transcriptions`, a strict
  subset of OpenAI's transcription endpoint, described under
  [Hearing a recording](#hearing-a-recording).

- WIRE-03: Live audio is heard over the
  Realtime transcription WebSocket `/v1/realtime?intent=transcription`, described
  under [Hearing live](#hearing-live).

`speech_client.wire` (the `overnightworks-speech-client`
distribution) owns the paths, the request types, the transcription answer and the
refusal list, and `speech_client.realtime_wire` the Realtime session and events;
build the body and read the refusals through them.
`speech_client.speak` makes the speaking call itself, within a total deadline and a
byte cap, and answers with a typed outcome; its module documents them. `speak` and
`speech_request_body` take `language`; one the voice lacks, or none for a voice of
several languages, raises the request's validation error before the network.
`speak_stream` and `speak_stream_with_preference` take `stream_format` `sse`
(WIRE-158) and then also yield each sentence's `SentenceMarks` after its audio;
a stream that ends without `done` ends `Cut`. Their items are unchanged without it.
`speech_client.voice_preference` speaks with an ordered preference of voices instead of
one: it reads health, speaks with the first voice that `speaking.voices` offers as
`ready` in the requested language, sends `instructions` only to a voice that
`takes_instructions` (WIRE-142) and says in its outcome which voice spoke and whether
the instructions were left out. Its `speak_stream_with_preference` streams the audio
as `speak_stream` does, after a `VoiceChosen` item; `speak_with_preference` is that
stream read to its end. A speak answered `unavailable` before its first chunk (the
service switched voices mid-call) reads health again and speaks with the next voice;
after a chunk a failure ends the stream and no other voice is tried. With none offered
the outcome is `unavailable` for `voice`. It adds nothing to the wire.
`speech_client.health` fetches `GET /health` (WIRE-122) within a deadline and a
cap and answers with a typed outcome; the outcomes the calls share live in
`speech_client.outcomes`. `speech_client.hearing` runs the live hearing
session: audio in, typed events out, one closed outcome when it ends, within a total
deadline and an opening bound; a service that does not open the session in time ends
it with `Unreachable`. It takes its tokens from a token source (an async callable
returning `SessionTokens`, or `None`), which it awaits before every opening, so each
opening is an upgrade (WIRE-140) with current tokens: `None` ends the hearing with
`NoToken`, and an exception of the source is raised from leaving the session, with
nothing more opened. Once one opening has succeeded, a session that closes with `1012`
(`DRAIN_CLOSE_CODE`, WIRE-146) or is cut after its `session.created`, having sent no
`error` and no failed item, is reopened at the same origin with the same
`session.update`; any other end ends the hearing as before. `RECONNECT_BOUND` bounds
the reopening (attempts, a pause between them, a total time), counted from the end of
the open session and reset by a session that answered, one that sent an event about the
audio or a transcript after its `session.created` (a repeated `session.created` and
`session.updated` do not answer); a refusal `unavailable` or `busy`, `Unreachable`, a
cut before `session.created`, an expired opening bound and a session that ended without
answering each cost one attempt. A session that ended without answering does not restart
the total time, which runs from the end of the last session that answered, or of the
first session when none did, less the time the sessions since stayed open; no timer of
it runs while a session is open, so a reopened session that stays open, silent or not,
runs to the hearing deadline. Any other refusal
or `Malformed` ends the hearing with it, and a bound that runs out ends it with
`ReconnectFailed`. Audio is held under the caller's byte limit meanwhile
(`NEEDED_HOLD_BYTES` covers the opening bound and a whole reconnect), a chunk leaves the
held audio when its write begins and is never written again, and audio is written only
after a session's `session.created`. The positions of `speech_started` and
`speech_stopped` lie on one audio timeline across sessions: the
service's position plus the audio of every chunk whose write began on an earlier
session, in whole milliseconds, rounded down, lost audio included. Before the new
session's events `events()` yields `Reconnected` (a client-side record, no wire event):
its cause (`DRAIN` or `CUT`) and the span of audio, on that timeline, that may have gone
unheard, from the end of the last turn the old session delivered to the new session's
first position. The new session's items start afresh.
A `SustainMoments` handle passed to `hear_utterances` names a moment (WIRE-176) at
`now()`, which returns the position of the audio handed to the hearing so far on that
timeline, whole milliseconds rounded down, and `None` while no hearing runs; the library
delivers only the event of the newest moment (WIRE-177, WIRE-178), also across a
reconnect, and writes the newest moment on a new session when its position is at or
above that session's start.
`speech_client.relay` relays a browser's binary audio frames to such a session for an
app's server and hands the app the typed events, never the browser; it ends the hop
when the browser leaves, breaks a frame rule or the app ends it, and closes the
browser with a code of its own (`NoToken` closes it in order, `ReconnectFailed` as a
service failure). A text frame of the page's own reaches the app's `on_text` unread,
bounded like every frame; without `on_text` it breaks the frame rule as `TEXT_FRAME`.
`speech_client.utterances` builds the session on top
of the relay and hands the app finished utterances (text, language, audio
positions, the item's id and the turn's `traceparent`, WIRE-150), optionally the text so far, a `Reconnected` (it forgets the old session's
unfinished items), or the code of a failed item or of an `error` event
(`ServiceFailed`).

## Request

- WIRE-04: A JSON object with exactly these fields, all required but `language`, `instructions` and `stream_format`, all strict JSON types:

| ID | Field | Accepted value |
|---|---|---|
| WIRE-05 | `model` | `piper` or `qwen3-tts-1.7b`, the engine of `voice`. Any other value is refused, so a request naming another model is never answered by this one. |
| WIRE-06 | `input` | Text with at least one letter or digit, at most `SPEECH_MAX_INPUT_CHARACTERS` characters (default 4096; the operator may raise it above OpenAI's 4096). |
| WIRE-07 | `voice` | A catalogue voice of `model` as a string: `de_DE-thorsten-medium`, `de_DE-kerstin-medium`, `en_US-norman-medium`, `en_US-ljspeech-medium`, `en_GB-cori-medium` under `piper`; `ryan` under `qwen3-tts-1.7b`. `speaking.voices` in health lists those offered. OpenAI's object form is refused. |
| WIRE-09 | `voice` | A GPU voice that is not loaded yet loads for this request, which waits for the load; with WIRE-143 it is loaded at start, so none is. |
| WIRE-10 | `response_format` | `wav` or `opus`. OpenAI defaults an absent field to `mp3`, so it is required here. |
| WIRE-138 | `language` | An extension to OpenAI's request, which has no such field. `de` or `en`, the codes of WIRE-34; `null` counts as not named. A voice of one language speaks it, and the field is optional but, when named, must be that language. A voice of several languages (`ryan`: `de`, `en`) speaks the language named, and the field is required. A value other than `de` or `en`, or a language the voice does not speak, or no language for a voice of several, is refused. |
| WIRE-141 | `instructions` | OpenAI's field of that name: how to speak the text, for example its tone. A string with at least one letter and at most 500 characters; `null` counts as not named, and a request without it speaks as before. Otherwise it is refused. The voice speaks the text in the manner the instructions name. |
| WIRE-151 | `stream_format` | OpenAI's field of that name: `audio` (the default, when absent) answers with the audio itself as WIRE-88 and WIRE-133 say, byte for byte as without the field; `sse` answers with the event stream below. Any other value is refused. |
| WIRE-142 | `instructions` | `speaking.voices` in health lists for each voice whether it `takes_instructions`: `ryan` does, the Piper voices do not. A request that names instructions for a voice that takes none is refused, never ignored. |

- WIRE-11: Each voice speaks the languages `speaking.voices` in health lists for it, and
  any field not listed here (`speed`, ...) is refused.
- WIRE-139: The request is checked in the order `model`, `voice`, `language`,
  `instructions`: a refused `language` or `instructions` is `invalid_request` with `param`
  `language` or `instructions` unless an earlier field is refused first, which names that
  field, and it is answered before any `unavailable` (WIRE-103).

## Headers

- WIRE-12: A speak request carries the app's workload token in `Speech-Workload-Token`
  and the person's token as `Authorization: Bearer <token>`.
  `speech_request_headers` in `speech_client.wire` builds them with the
  content type, and `wire.py` owns the header names.
- WIRE-149: A speak request and a
  realtime upgrade may also carry a W3C `traceparent` header; the service writes the lines
  of that request or session in its trace, and a missing or malformed value starts a trace
  of its own. `speak`, `speak_stream`, `HearingSession` and `hear_utterances` take
  `traceparent=` and send it. Without one, `speak` and `speak_stream` carry the trace running
  at the call through `logline`'s hook once the host has called `logline.configure`, and
  none otherwise; `HearingSession` and `hear_utterances` send `current_traceparent()`
  themselves. The header parents only a session's own lines, never a turn
  (WIRE-150). The service answers only when both tokens are valid and belong together:

- WIRE-13: Both come from the configured issuer, name the service in `aud`, and are
  unexpired. The app requests the service's audience scope in its workload
  grant and at the person's sign-in.
- WIRE-14: The workload token carries the configured realm role and comes from the app's
  registered client (`azp` and `client_id`).
- WIRE-15: The person's token has the same `azp`, a different subject, membership of
  `/apps/<app>` or a group below it, and membership of exactly the configured
  service group (no subgroup of it).

An agent with its own account is the person: its app exchanges the agent's token for one
of its own client through Keycloak's standard token exchange, and that token is the
person's token above.

- WIRE-16: The service checks the headers before it reads the body, and
- WIRE-17: the wait of
  `busy` starts only after the tokens are accepted, or, for a request that
  waited for its voice to load, when that load ends.
- WIRE-19: An unknown voice, or a voice of another model, is
  `invalid_request` with `param` `voice`; a catalogue voice that is not offered is
  `unavailable` (WIRE-103).
- WIRE-137: An unknown model is `invalid_request` with
  `param` `model`; `speech` names hearing's model and is refused here.
- WIRE-20: Calls an app makes as itself,
  without a person, are not offered in version 1.

## Admission

- WIRE-21: Each voice and hearing have one scarce resource, the one synthesis of that
  voice and the one decode a process runs at a time, each under its own limits; a
  request is refused only by the limits of its own resource. For speaking,
- WIRE-22: a
  request that finds its voice's synthesis taken may wait, within two limits that
  hold for each voice apart, and anything beyond them is refused before any work
  starts:

- WIRE-23: `SPEECH_MAX_WAITING_REQUESTS` places for requests that wait for the synthesis.
  A request that finds the synthesis taken and every place taken is refused at
  once with `busy` and `Retry-After`.
- WIRE-24: `SPEECH_BUSY_WAIT_SECONDS`: how long a waiting request waits. When it runs out,
  the answer is the same `busy` with `Retry-After`.
- WIRE-25: Requests waiting for a GPU
  voice's load hold no place, and their wait starts when the load ends.
- WIRE-26: `SPEECH_MAX_BODY_BYTES`: the largest request body. The service stops reading
  at the limit, whatever `Content-Length` says, and answers `too_large` without
  parsing the body.
- WIRE-27: It answers only a caller whose tokens are accepted.

- WIRE-28: A refused request holds no place, so the next one is served.
- WIRE-132: A speak request whose caller has gone frees its voice for the next request.

## Hearing a recording

`POST /v1/audio/transcriptions` answers the words of one recording.

- WIRE-29: It carries
  the same two headers as a speak request and
- WIRE-30: is checked in this order, each step
  before the next: the tokens, whether hearing is
  available, the body's size, the decode's admission, the form, the recording's
  length, the decoding itself.

- WIRE-31: The body is `multipart/form-data` with exactly these fields:

| ID | Field | Accepted value |
|---|---|---|
| WIRE-32 | `file` | The recording, required, in any container and codec PyAV reads (WAV, MP3, Ogg, FLAC, ...). |
| WIRE-33 | `model` | `speech`, required. |
| WIRE-34 | `language` | `de` or `en`; `de` when absent. Any other value is refused. |
| WIRE-35 | `response_format` | `json`; `json` when absent. |

- WIRE-36: Any other field (`prompt`, `temperature`, `timestamp_granularities[]`, ...) is
  refused.
- WIRE-37: The answer is `200` with `{"text": "..."}`;
- WIRE-38: the same recording gives
  the same text, because the service decodes greedily, at temperature 0.

- WIRE-40: Hearing is `unavailable` while no model is configured
  (`SPEECH_HEARING_MODEL`), while it loads, when its load found no CUDA device or
  no room on the card, and when a decode fails.

The limits, each answered without decoding more than it must:

- WIRE-41: `SPEECH_MAX_RECORDING_BYTES` bounds the request body. The service stops
  reading at the limit, whatever `Content-Length` says, and answers `too_large`
  before it parses the form.
- WIRE-42: `SPEECH_MAX_RECORDING_SECONDS` bounds the recording's length. The decoder
  stops at the limit and the answer is `too_large` with `param` `file`; a
  recording longer than the limit is never transcribed in part.
- WIRE-43: `SPEECH_HEARING_MAX_WAITING_REQUESTS` places for requests that wait for the
  one decode, and `SPEECH_HEARING_BUSY_WAIT_SECONDS` for how long they wait. A
  request that finds the decode taken and every place taken is refused at once
  with `busy` and `Retry-After`; so is one whose wait runs out.
- WIRE-44: The decode is
  freed only when it has finished, even if the caller has gone.
- WIRE-160: A recording is kept in memory only while its request holds one of
  `1 + SPEECH_HEARING_MAX_WAITING_REQUESTS` places, which it takes when it arrives.
  Without a place the service still counts the body's bytes, so `too_large` is
  answered first (WIRE-30), and then answers `busy` with `Retry-After`: the
  recording is never heard, even if a place frees while it arrives.
- WIRE-161: A place is freed when its request is refused, when its caller leaves
  before its decode has begun and when its decode has returned. A decode that has
  begun keeps its place until it returns, even if the caller has gone (WIRE-44).
- WIRE-162: An upload still arriving `SPEECH_HEARING_BUSY_WAIT_SECONDS` after the
  request arrived is `busy`, even on an idle service, and its place goes to the next
  request. A request not admitted by then gives its place back.

- WIRE-45: Audio that PyAV cannot read, or that holds no sound, is `invalid_request` with
  `param` `file`.
- WIRE-46: A body that is not a multipart form, or has no `file`, is
  `invalid_request` with `param` `file` as well.
- WIRE-47: The form is parsed only once the
  request holds the decode, so one upload is spooled to disk at a time: while the
  service is busy, a malformed form is answered `busy` like any other request, and
  `invalid_request` once it reaches the decode.

## Hearing live

- WIRE-48: `WS /v1/realtime?intent=transcription` hears audio as it arrives.
- WIRE-49: It speaks the
  events of OpenAI's Realtime transcription session that this service applies, and
  events of its own, sound detection (WIRE-163) and sustained speech (WIRE-171,
  WIRE-177), and one client message of the sustained ask's own, the moment (WIRE-176).
- WIRE-50: Both tokens travel in the upgrade request's headers, as on the
  other routes; a client sends them from a server, never from a browser.
- WIRE-140: The tokens are checked once, at the upgrade; a session runs on after either
  token expires until it ends, so the app ends a hearing when its right to hear ends. A
  client that opens another session after a drain presents tokens again: each opening is
  a new upgrade.

**Upgrade.**

- WIRE-51: A refusal before the session is an HTTP answer in the refusal form
  below, in this order: the tokens, the
  `intent` (`invalid_request`, `param` `intent`), whether hearing is available
  (`unavailable`, which the drain makes true), the session bound.
- WIRE-52: `SPEECH_MAX_HEARING_SESSIONS` (default `1`)
  sessions run at once; one more is `busy` with `Retry-After`.
- WIRE-53: A wrong version is
  `unsupported_version`.
- WIRE-54: An idle session holds its place only, never the decode.

**Setup.**

- WIRE-55: The service sends `session.created` with its default session: PCM16
  mono at 24 kHz, `de`, the model `speech`, server turn detection.
- WIRE-56: The
  client may send one `session.update` with `session.type` `transcription` before
  its first append; the service answers `session.updated` with the applied session.
- WIRE-57: Accepted: the format absent or exactly `{"type": "audio/pcm", "rate": 24000}`;
  `transcription.model` absent or `speech`; `transcription.language`
  absent, `de` or `en`; `transcription.languages` absent or a non-empty array of
  distinct `de` or `en`; `turn_detection` absent or `{"type": "server_vad"}`;
  `sound_detection` absent or `{"type": "audioset"}` (WIRE-164); `sustained_speech` absent,
  `{}` or `{"duration_ms": <integer from 1 to 30000>}` (WIRE-172).
- WIRE-58: Anything
  else ends the session with an `error`, among it a second update, an update after
  audio, `null` turn detection, `language` and `languages` together, an empty or
  repeating `languages`, a code the service does not hear (`param` ends with the
  element's index, such as `...transcription.languages.1`), an explicit `null` for
  either field, an explicit `null` or any other `type` for `sound_detection`, an explicit
  `null` for `sustained_speech`, a `duration_ms` that is `null`, not an integer or outside
  1 to 30000, or a key it does not know (`param` `session.audio.input.sustained_speech`
  with `.duration_ms` or the key's name), and every field the service does not apply
  (prompt, keywords, noise reduction, include).

**Languages.**

- WIRE-135: `transcription.languages` is a strict, ordered set; its
  first element is the main language. A `languages` update is echoed by `session.updated`
  with `languages` and without `language`; a `language` update, or one that names neither
  field, is echoed with `language` and without `languages`. The service hears each ended
  turn, and each partial, in exactly one language of the set: the language the identifier
  names, or the main language when it does not know. No language outside the set is ever
  chosen. A set of one language, like `language`, is decoded in that language with no
  language step. This deviates from OpenAI, where `languages` is a hint: here it is a
  strict set.

**Client events.**

- WIRE-59: Only `session.update`, `input_audio_buffer.append` (base64
  PCM16 at 24 kHz),
  `input_audio_buffer.commit` and `input_audio_buffer.sustain_from` (WIRE-176).
- WIRE-60: Invalid base64 or an odd byte count ends the session.
- WIRE-61: Any other event type, `input_audio_buffer.clear`
  included, is a protocol violation.
- WIRE-62: A client `event_id` is optional; the service
  echoes it only in an `error` about that event.
- WIRE-63: A commit ends the open turn at
  once, and with no turn open it ends one of the audio the service still holds: the last
  0.35 s up to the newest 32 ms window it has judged, plus the samples short of the next
  window, or, while one or two speech windows wait to be confirmed, the audio from 0.35 s
  before the first of them (WIRE-66);
- WIRE-64: a commit with no audio since the last
  turn ended is a protocol violation.
- WIRE-65: With server turn detection a client need not commit.

**Turns.**

- WIRE-66: Speech is what Silero VAD hears as a voice, whatever its loudness
  or distance: any voice, a television's included, is speech, and noise is not. A turn opens
  once three consecutive 32 ms windows are speech and decodes from 0.35 s before the first
  of them. The service ends it after speech followed by the silence of WIRE-148 (0.5 s by
  default) without speech, or at 30 s of audio, pre-roll included; the rest of the audio opens the next turn. Between turns the
  service drops all audio older than that pre-roll.
- WIRE-67: A session holds at most two
  ended turns, both queued to be heard from their turn's end, and the open turn, so at most
  90 s of audio. The wire carries no confidence value; the service logs each finished
  utterance's audio and voiced milliseconds, no-speech probability and average log probability
  on a `hearing.measured` line, never its text. Each turn also writes `turn_end_wait.ended`
  (the wait that ended it) and `hearing.ended` (from its end to the transcript, with the
  `audio_end_ms` it ended at and the `silence_ms` that ended it), and each speak request writes
  exactly one `speak.ended` (with its `outcome`, which is `cancelled` when the client's leave or a shutdown ends the request at any point, and with `voice`, `first_output_ms`, `stream_format` and the counts `marks_words` and `marks_sounds` of what it marked, never a word; each of these fields is present when known, so a request that ends before its body is read has no `voice`). `first_output_ms` counts from the request's arrival to when the service rendered the first audio, not to when the client read it. All lines are in the shape of logline.

  **Server events**, in the order one turn produces them:

| ID | Event | When |
|---|---|---|
| WIRE-68 | `session.created` | first, on every session |
| WIRE-69 | `session.updated` | after a valid `session.update` |
| WIRE-70 | `input_audio_buffer.speech_started` (`item_id`, `audio_start_ms`) | speech confirmed in the turn (WIRE-66); the position counts as below |
| WIRE-71 | `conversation.item.input_audio_transcription.delta` (`item_id`, `content_index` 0, `delta`) | zero or more, before the commit: the words two consecutive decodes agree on. Provisional; a delta names no language, and a partial may be heard in another language than its final. |
| WIRE-72 | `input_audio_buffer.speech_stopped` (`item_id`, `audio_end_ms`) | silence or the cap ended the turn, at the position counted as below; not sent after a client commit |
| WIRE-73 | `input_audio_buffer.committed` (`item_id`, `previous_item_id`) | the turn is closed; `previous_item_id` is `null` for the first |
| WIRE-74 | `conversation.item.created` (`item`, `previous_item_id`) | right after `committed`; the item is a user message with one `input_audio` part whose `transcript` is `null` |
| WIRE-75 | `conversation.item.input_audio_transcription.completed` (`item_id`, `content_index` 0, `transcript`, `usage`, `languages`, `traceparent`) | the authoritative text, possibly empty, with `usage` `{"type": "duration", "seconds": ...}`, `languages` (WIRE-136) and the turn's `traceparent` (WIRE-150); it replaces the deltas |
| WIRE-76 | `conversation.item.input_audio_transcription.failed` (`item_id`, `content_index` 0, `error`) | only as a session ends |
| WIRE-77 | `error` (`error.type`, `error.code`, `error.message`, `error.param`, `error.event_id`) | the session ends after it |
| WIRE-163 | `input_audio_buffer.sound_detected` (`sound`, `audio_start_ms`, `audio_end_ms`) | a sound began, in a session that asked (below); it has no `item_id` |
| WIRE-171 | `input_audio_buffer.speech_sustained` (`item_id`, `audio_end_ms`) | once per item, in a session that asked (WIRE-172), when the item's voice has lasted the session's duration (WIRE-173): after its `speech_started`, before its `speech_stopped` and `committed` (WIRE-175); `audio_end_ms` counts as below |
| WIRE-177 | `input_audio_buffer.speech_sustained_from` (`item_id`, `from_ms`, `audio_end_ms`) | once per item and moment, for the item whose `audio_start_ms` is at or before `from_ms` and which had not ended before it, when its voiced windows that begin at or after `from_ms` reach the session's duration, counted as WIRE-173 counts, a pause past the bound restarting them; after the item's `speech_sustained`. No event for an item that ends first. |

`audio_start_ms` and `audio_end_ms` (WIRE-70, WIRE-72, WIRE-171) count milliseconds of the audio
the service received in this session, from its first byte, in the service's 16 kHz time,
the dropped audio between turns included. They depend on neither the wall clock nor when
an event is sent. Turns are ordered, not contiguous: a turn's end is the position of the
last audio it holds in whole milliseconds, and the next turn starts at or after that end.
`audio_start_ms` is where the first of the speech windows that opened the turn starts,
so the audio the turn decodes begins up to 0.35 s earlier. `audio_end_ms` is the end of
all audio in the turn: it includes the silence that ended it (WIRE-66), or lies 30 s
after the turn's pre-roll starts. `usage.seconds` (WIRE-75) is the length of that
audio, pre-roll included.

**Sounds.**

- WIRE-164: A session asks for sounds with `session.audio.input.sound_detection`
  `{"type": "audioset"}` in its `session.update`, and only a session that asked gets
  `input_audio_buffer.sound_detected`; without the ask every event and echo is as it would
  be without sounds.
- WIRE-165: `session.updated` echoes the field when it was asked for and
  sound events are on, and leaves it out, without an error, when they are off
  (`SPEECH_SOUND_EVENTS`, WIRE-169); `session.created` never carries it.
- WIRE-166: `sound` is
  one of `Laughter`, `Cough`, `Clapping` and `Screaming`, the AudioSet display names; throat
  clearing and sighing are not sent. The service folds the AudioSet classes Laughter, Giggle,
  Snicker and Belly laugh into `Laughter`, and Clapping and Applause into `Clapping`; Applause is
  a class of its own in AudioSet, not a child of Clapping, so that fold is this service's own
  mapping.
- WIRE-167: The service scores the session's audio in windows of 2 s, hopped by
  0.5 s on the session clock of WIRE-70, the first window once 2 s have arrived. A sound is sent
  at the first window in which its highest score reaches 0.1, with `audio_start_ms` and
  `audio_end_ms` the window's bounds, so `audio_start_ms` is a lower bound of where the sound
  starts. It ends after two scored windows in a row below 0.1, and then a new one may begin; a
  window the service skipped because an earlier one was still being scored, or because several
  fell due together, counts neither way. Two sounds in one window are sent in the order above.
  The wire carries no score (WIRE-67); the service writes one `sound.detected` line for each event
  and none for a window.
- WIRE-168: Sounds never delay a receive or a decode. A sound
  that fails to be scored ends the session `unavailable`, as a failed partial does (WIRE-81): the
  client was told sounds are on, so losing them silently would read as "no sound". After an
  `error` or the close no sound event follows. During a drain (WIRE-146) sounds are sent until
  the `1012` close, which they never delay.

- WIRE-170: `completed.transcript` carries no leading or trailing whitespace, and a turn
  with no words is `""`. The service trims every transcript it sends, and the client refuses
  a `completed` event that breaks the rule as a malformed server event. Deltas are provisional
  and follow no such rule (WIRE-71).

- WIRE-136: `completed.languages` is `[{"code": "<code>"}]` with exactly one element, the
  language the turn was decoded in: also for a session of one language, for a turn heard in
  the main language because the identifier did not know, and for an empty transcript. It is
  never `[]`, which deviates from OpenAI, whose `languages` may come back empty. `failed` and
  `error` carry no language.

- WIRE-150: `completed.traceparent` is the W3C `traceparent` of the trace the service
  started at the turn's end and wrote the turn's `turn_end_wait.ended` and `hearing.ended`
  lines under: a turn has no client message to carry a trace, so the service sends its own,
  a trace of its own for each turn and never the upgrade's (WIRE-149). A client reads it as
  optional: a service from before the field sends none. An app that runs its handling of
  the turn under `logline.unit_trace(traceparent)` writes its lines in the turn's trace;
  `speech_client.utterances` hands it over with the `item_id` in the `Utterance`.

- WIRE-78: Every server event has a fresh `event_id`. Correlate by `item_id`: OpenAI does not
  order completions across items;
- WIRE-79: here the second ended turn's `completed` waits
  for the first's, so they come in turn order, but do not depend on it.

**Sustained speech.**

- WIRE-172: A session asks for it with `session.audio.input.sustained_speech`
  `{"duration_ms": <integer>}` in its `session.update`, and only a session that asked gets
  `input_audio_buffer.speech_sustained` and `input_audio_buffer.speech_sustained_from`;
  without the ask every event and echo is as it would
  be without it. `duration_ms` is an integer from 1 to 30000, the turn cap (WIRE-66); left
  out, it is the service's default, 400 ms. The duration is fixed for the session (WIRE-58).
  An app holds a speaking agent on this event instead of on `speech_started`, so a short
  "mhm", cough or laugh does not stop it.
- WIRE-173: The count is the item's voiced 32 ms windows since its last pause longer than
  0.15 s, the pre-roll's windows included (WIRE-66). A window is voiced as WIRE-66 hears a
  voice, a faint one included; a pause is a run of consecutive windows that are not voiced.
  A run of up to 4 windows (128 ms) adds nothing and restarts nothing; a run of 5 (160 ms) or
  more restarts the count. The count reaches `duration_ms` at the window that completes
  `ceil(duration_ms / 32)` windows, 13 for 400 ms, and the event comes then, once per item:
  `audio_end_ms` is that window's end, or the end of the window that opened the turn when
  the pre-roll already holds the count, and the event follows `speech_started` in the same
  batch. No event is sent for an item that ends before the count: silence, the 30 s cap
  (which a duration of 30000 ms never beats), a client commit or a drain cut. An event
  sent stands, whatever the item hears afterwards. The 0.15 s is the service's own value,
  never the session's.
- WIRE-174: `session.updated` echoes the field with the duration the session applies,
  `{"duration_ms": <integer>}`, as asked or the default when the ask leaves it out;
  `session.created` never carries it.
- WIRE-175: Each event follows its item's `speech_started` and precedes its `speech_stopped`
  and `committed`, `speech_sustained_from` after `speech_sustained`; their order against
  deltas is free. Every announced item still ends
  once (WIRE-80). Each is sent while a drain runs (WIRE-146), until the `1012` close;
  after an `error`, a `failed` or the close none follows. Each is counted as audio arrives,
  so it never delays a receive or a decode.
- WIRE-176: A session that asked for sustained speech names a moment with
  `input_audio_buffer.sustain_from` (`from_ms`, an integer, milliseconds as WIRE-70 counts
  them). A session that did not ask refuses the type as WIRE-61 does, whatever its fields. A
  `from_ms` below the whole milliseconds of the audio appended so far, absent, `null`, not an
  integer, below 0, or beside a key the service does not know ends the session
  `invalid_request` with `param` `from_ms` or the key's name. A position ahead of the audio
  appended waits for the audio.
- WIRE-178: A session holds one moment. A new `sustain_from` replaces it as it is received:
  no event of the older moment follows, and one already sent stands. A moment equal to the
  standing one changes nothing while no audio was appended after it; once audio was, the
  equal moment is behind and WIRE-176 refuses it.

**Every announced item ends once.**

- WIRE-80: An item is announced by its `speech_started`,
  its first delta or its `committed`. While connected it ends with exactly one
  `completed` or one `failed`.
- WIRE-81: When the service ends a session (a decode fails or
  the engine is lost, a final waited longer than `SPEECH_HEARING_FINAL_WAIT_SECONDS`,
  a third ended turn while two are outstanding, or a protocol violation) it sends one `failed`
  per announced unfinished item, then one `error`, then closes.
- WIRE-82: The `error`'s `code`
  is `invalid_request` (`param` the field at fault), `busy` or `unavailable`.
- WIRE-83: A client
  that disconnects gets nothing.

**Finals first.**

- WIRE-84: A turn's final waits for the one decode ahead of every waiting
  recording, behind earlier finals.
- WIRE-85: Its deadline runs from the turn's end for
  `SPEECH_HEARING_FINAL_WAIT_SECONDS` (default `60`, above 0), the time behind the
  session's own earlier final included; a final past it is never started, even on a
  free decode, and the session ends `busy`.
- WIRE-86: A partial that finds the decode taken is
  skipped.
- WIRE-87: The decode is freed only when it has finished, even if the client has gone.

**Drain.**

- WIRE-144: On the stop signal (SIGTERM or SIGINT) the service reports not ready at
  once, for speaking and hearing (WIRE-124, WIRE-129), and refuses a new session, speak request
  or recording as `unavailable` (WIRE-103).
- WIRE-145: A speak stream or recording already
  running finishes whole; a speak request still waiting for the engine is refused.
- WIRE-146: Each live session keeps taking audio until its open turn ends, by silence or
  the cap, or by the drain: an open turn ends itself 30 s after the drain began. Audio after
  that turn's end opens no new turn. The session delivers that turn's `completed` and every
  final ahead of it, then closes with `1012` (Service Restart, `DRAIN_CLOSE_CODE` in
  `speech_client.wire`) and no `error`, since nothing failed. A session with no open turn closes with `1012`
  at its next poll (0.3 s). A client reconnects with a new session; its items start afresh.
- WIRE-147: The drain is bounded by 30 s plus `SPEECH_HEARING_FINAL_WAIT_SECONDS`
  (WIRE-120): a final waits at most that long, as WIRE-85 says. The service then closes its
  engines and exits 0; whatever still runs at the bound is cut. A session with an item still
  unfinished at the bound fails it, sends `error` `unavailable` and closes `1012`.

## Response

- WIRE-88: For `wav`, `200` with `Content-Type: audio/wav`, chunked: 16-bit mono PCM at the voice's
  native rate, which is in the WAV header.
- WIRE-89: Every WAV carries an AI mark, a RIFF
  `LIST`/`INFO` `ICMT` chunk before `data`.
- WIRE-90: The status and headers leave before
  synthesis, so a synthesis failure cannot change the status: the server aborts
  the connection without the chunked terminator. Read the body to its end and
  treat a missing end as a failed request; never use partial audio, except to
  play it live: a player may play the chunks as they arrive, and a body that
  breaks off stops the playback where it broke and ends it failed.

- WIRE-133: For `opus`, `200` with `Content-Type: audio/ogg`, chunked: Ogg Opus
  (RFC 7845), mono at 48 kHz whatever the voice's rate, its pages leaving as the voice
  speaks. RFC 7845 names `audio/ogg`; OpenAI lists `audio/opus` among its answer types
  and names no container, so the media type deviates from OpenAI's.
- WIRE-134: Every
  Ogg Opus carries the AI mark as `DESCRIPTION=<mark>` in its OpusTags.

## Event stream

- WIRE-158: With `stream_format` `sse` the answer is `200` with `Content-Type: text/event-stream`, in the
  same chunked way. Each event is one line `data: <JSON object>` and a blank line; the object's `type` names
  it, and a reader ignores a field it does not know. There are three types:

- `speech.audio.delta` `{"audio": "<base64>"}`: the bytes of the audio the default answer carries for the
  same `response_format`, in order, so the decoded deltas joined are that audio.
- `speech.audio.marks` `{"words", "sounds"}`: an extension to OpenAI's stream, which has no such event. See below.
- `speech.audio.done` `{}`: the last event. It carries no `usage`, which deviates from OpenAI's.

A stream that ends without `done` failed, as WIRE-90 says. The status and headers leave before synthesis
as for the audio, so a failure aborts the connection and no `done` follows.

- WIRE-152: A voice sentence is a sentence its voice speaks as one piece of audio. A Piper voice sends one marks event for each of its
  sentences; a voice that marks nothing sends one after all its audio, with both kinds `null`. `words` is a list of the words of
  the sentence, `[]` where it holds none, or `null` where they could not be proven; the audio is sent anyway.
  A word missing from the list is unproven, which is not the same as silent. `sounds` is the same for the
  sounds of the sentence (WIRE-159): `[]` where it holds none. A kind whose `gives_marks` flag
  (WIRE-122) is false is always `null`; otherwise `null` means that sentence's marks failed.

- WIRE-153: A word is `{"text", "start_char", "end_char", "start", "end"}`. `start_char` and `end_char` count
  Unicode code points of `input` as received, unnormalized, `end_char` exclusive, so `text` is
  `input[start_char:end_char]`. `start` and `end` are seconds from the answer's first sample, continued across
  sentences, whatever the format: the header of a WAV does not count, and an Opus counts decoded time after its
  pre-skip. They are truncated to milliseconds, the starts of one answer increase, and no `end` lies after the audio.
  They are Piper's predicted durations for the word's sounds, not a measured onset; a cut at a played position is
  accurate to one Piper phoneme.

- WIRE-154: One word may cover several spoken words (`in the` is one entry): its `text` is the exact `input`
  span, without the whitespace and punctuation after it. A symbol spoken as several names (an emoji) is one entry.

- WIRE-155: The words of a sentence are given only when the service proved them: they follow what the voice
  spoke, in order, and no word is another's text. A sentence that fails a check has `words` and `sounds` `null`, and so has
  every sentence after it in the answer; the sentences before it keep theirs. Input with a raw phoneme block
  (`[[ ... ]]`) has no words in any sentence.

- WIRE-156: `gives_marks.words` and `gives_marks.sounds` are true for a Piper voice once its start load checked
  the means of the words (WIRE-122); where the check fails the voice speaks as before, its `sse` answers carry `null` words and sounds, and the
  service logs one error line for each such voice. The words never delay the audio: a sentence's marks follow
  its audio, and the next sentence waits for them, not the first. An answer that the caller leaves loses the marks not yet sent.

- WIRE-157: A marks event follows the audio of its sentence. With `opus`, up to one Ogg page of that
  sentence (at most 100 ms) may follow its marks; the times stay on the answer's own clock.

- WIRE-159: A sound is `{"ipa", "start", "end"}`: one spoken sound of a sentence, in order, `ipa` in NFC without a
  stress mark. `start` and `end` are on the clock of the words (WIRE-153). A sound is a base code point with the
  combining marks and the length marks `ː` and `ˑ` after it, and the tie bar `͡` joins the next base (`t͡ʃ` is one
  sound); spaces and punctuation are the gaps between sounds. A phoneme id the voice times as one unit is one sound
  whatever its base count, so a diphthong id `aɪ` is one sound with the id's own samples; the grouping applies within
  single-base ids. A sound lasts for the samples of its parts, and a stress mark's samples open the sound after it, so a sound may start before its word
  does. A sound ends inside the word it belongs to. A Piper sound carries no `estimated`; a voice whose times are not its own sets `estimated` to `true`, and a
  reader takes a sound without it as not estimated.

## Refusals

- WIRE-91: Every refusal is `Content-Type: application/json` in OpenAI's error form,
  `{"error": {"message", "type", "param", "code"}}`.
- WIRE-92: `message` is a fixed
  sentence per code and never echoes the request.
- WIRE-93: `type` is
  `invalid_request_error` for a 4xx and `server_error` for a 5xx.
- WIRE-94: `param` is one
  of `file`, `model`, `input`, `voice`, `language`, `response_format`, `stream_format`, `intent`, or `null`
  when no request field is at fault (an unknown field, a body that is not a JSON
  object). Apps read `code`. With several faults in one request, which field `param`
  names is not part of the contract, but for WIRE-139.

| ID | `code` | Status | When | Header |
|---|---|---|---|---|
| WIRE-95 | `unsupported_version` | 400 | any method or upgrade on `/v{n}` or `/v{n}/...` with `n` other than 1 | |
| WIRE-96 | `not_found` | 404 | any other path, method or upgrade under version 1 | |
| WIRE-97 | `unauthorized` | 401 | a token is missing, malformed, expired, for another issuer or audience, badly signed, or lacks a subject or a valid group claim | `WWW-Authenticate`: `Bearer` |
| WIRE-98 | `forbidden` | 403 | both tokens are valid but do not permit the request: not a workload token, an unregistered app, tokens of different clients or of one subject, a person outside the app group or the service group | |
| WIRE-100 | `invalid_request` | 422 | the body is not a JSON object, a field is missing, unknown or mistyped, a value is not accepted, or `input` is empty, without a letter or digit, or over the limit, or `instructions` are not accepted (WIRE-141, WIRE-142); for a recording: the form or `file` is missing, a field is unknown or not accepted, or the audio cannot be decoded (`param` `file`) | |
| WIRE-101 | `too_large` | 413 | the body is over `SPEECH_MAX_BODY_BYTES` (a recording: `SPEECH_MAX_RECORDING_BYTES`), or the recording is longer than `SPEECH_MAX_RECORDING_SECONDS` (`param` `file`) | |
| WIRE-102 | `busy` | 429 | the one synthesis (a recording: the one decode) is taken and its waiting places are taken (answered at once), or it is not free within its wait (a recording: or no place keeps it, or its upload outlasts the wait, WIRE-160, WIRE-162): `SPEECH_MAX_WAITING_REQUESTS` and `SPEECH_BUSY_WAIT_SECONDS`, for a recording `SPEECH_HEARING_MAX_WAITING_REQUESTS` and `SPEECH_HEARING_BUSY_WAIT_SECONDS` | `Retry-After`: the wait in whole seconds, at least 1 |
| WIRE-103 | `unavailable` | 503 | the tokens cannot be checked because the issuer's key set is unreachable and no cached key serves, or the named voice cannot speak: it is not offered, the start load still runs, its load failed or found its files missing, its engine process died while the request waited, or the drain began (`param` `voice`); for a recording: hearing is not loaded or configured, the drain began, or its decode failed | |

- WIRE-104: Apps read the `busy` wait with `busy_wait(status, headers)` in
  `speech_client.outcomes`: it returns the wait in whole seconds, `None` for any
  other status, and fails loud on a `busy` without whole seconds, at least 1, in `Retry-After`.

- WIRE-105: `n` is read as a decimal integer, so `/v01/...` is version 1.
- WIRE-106: A path that is no
  version path (`/v-1/...`, `/vx/...`, `/speak`) gets the app's default 404 with
  `{"detail": "Not Found"}`, outside the wire.

## Settings that shape the wire

- WIRE-107: `SPEECH_VOICES` (catalogue voice ids separated by commas; when absent, every
  catalogue voice of an engine the installation carries, which is Piper always and Qwen
  with `SPEECH_PROVIDER_ROOT` set; none when set empty): the voices offered. A Piper voice loads at start, a
  GPU voice on its first request (WIRE-09), or at start with WIRE-143. An unknown or
  repeated id, or a named Piper voice whose files are missing or fail to load, ends the
  start.
- `SPEECH_CATALOGUE_FILE` (absolute path; unset, the catalogue the service's package ships): the TOML file that holds
  the service's voices, one `[voices.<id>]` table each with its `engine`, `name`,
  `languages` and `pin`. The service does not start, and `speech-fetch-voice` does not
  fetch, when the file is unreadable or has an entry that names a wrong or unknown
  field; a voice `SPEECH_VOICES` names that the file lacks ends the start (WIRE-107).
  The ids the wire accepts stay the closed `VoiceId` set.
- WIRE-143: `SPEECH_LOAD_GPU_VOICES_AT_START` (`true` or `false`, default `false`): with
  `true` every offered GPU voice loads at start, after the Piper voices and before hearing,
  and a GPU voice whose files are missing or fail to load ends the start as a Piper voice's
  do. `speaking.ready` (WIRE-124) then waits for it.
- WIRE-108: `SPEECH_BUSY_WAIT_SECONDS` (default `5.0`, at least 0): how long a request
  waits for its voice's synthesis before `busy`.
- WIRE-109: `SPEECH_MAX_WAITING_REQUESTS` (default `3`, at least 0, where 0 allows no
  waiting): how many requests wait for each voice's synthesis.
- WIRE-110: `SPEECH_MAX_BODY_BYTES` (default `32768`, at least 1): the largest body. It
  holds a 4096-character `input` even when every character is escaped as a
  six-byte `\uXXXX`, plus the other fields.
- WIRE-111: `SPEECH_MAX_INPUT_CHARACTERS` (default `4096`, at least 1): the longest
  `input`.
- WIRE-112: `SPEECH_CONFIG_FILE` (absolute path, required): a TOML file that says whom the
  service trusts. The service does not start without a valid file.
- WIRE-113: The
  `[identity]` table holds `issuer`, `jwks_url` (https; it may carry a
  credential and is never printed), `audience`, `workload_role`, `group_claim`,
  `service_group` (a group path such as `/services/speech-service`).
- WIRE-114: Each `[apps.<name>]` table, at least one,
  holds `workload_client`, the client id of that app's workload token; a client
  belongs to one app, and the app group `/apps/<name>` follows from the name.
  Every registered app may name every offered voice and may hear.
- WIRE-116: Unknown keys are refused.
- WIRE-117: `SPEECH_HEARING_MODEL` (`large-v3` or `small`, none when absent): the hearing
  model the process loads at start. With none, hearing is `unavailable`.
- WIRE-118: `SPEECH_HEARING_BUSY_WAIT_SECONDS` (default `30.0`, at least 0) and
  `SPEECH_HEARING_MAX_WAITING_REQUESTS` (default `3`, at least 0): the
  decode's wait and its waiting places, as for speaking.
- WIRE-119: `SPEECH_MAX_HEARING_SESSIONS` (default `1`, at least 1): live sessions at
  once; one more is refused at the upgrade as `busy`.
- WIRE-120: `SPEECH_HEARING_FINAL_WAIT_SECONDS` (default `60`, above 0): how long a live
  final may wait for the decode, from its turn's end.
- WIRE-121: `SPEECH_MAX_RECORDING_BYTES` (default 25 MiB, at least 1) and
  `SPEECH_MAX_RECORDING_SECONDS` (default `300`, above 0): the largest request
  body and the longest recording.
- WIRE-169: `SPEECH_SOUND_EVENTS` (`true` or `false`, default `true`): with `false` the
  sound model does not load and an ask is echoed without its field (WIRE-165).
  `SPEECH_SOUND_MODEL_DIRECTORY` (absolute path, default `~/.cache/frame-mn10`, and
  `/opt/sound-model` in the images): the model that
  `scripts/build_sound_model.py` builds. Both take effect only with `SPEECH_HEARING_MODEL`; a malformed value
  still stops the start.
  With sounds on and the model missing the start ends, naming the setting, the build
  command and the switch; a model that does not load ends it too. The service never
  downloads it.
- WIRE-148: `SPEECH_LIVE_SILENCE_SECONDS` (default `0.5`, above 0, at most 10): the
  silence that ends a live turn.

## Health

- WIRE-122: `GET /health` answers without a token, so a supervisor can poll it:
  `{"speaking": {"ready", "voices": {"<voice id>": {"model", "name", "languages", "takes_instructions", "gives_marks", "ready", "sample_rate"}}}, "hearing": {"model", "ready"}, "card_memory_mb"}`.
- WIRE-123: It answers while the models are still loading, with `ready` false.
- WIRE-124: `speaking.ready` is the speaking service's own readiness: the start load has
  ended, the drain has not begun (WIRE-144), and every offered Piper voice (with WIRE-143, every offered voice)
  can speak. With no such voice offered, it is true once the start load ends.
- WIRE-125: `speaking.voices` lists
  the offered voices by id. Each reports, loaded or not, its `model` (the request's
  `model`), its `name`, its `languages` (the codes it speaks), whether it `takes_instructions` (WIRE-142), what it `gives_marks` as `{"words", "sounds"}` (WIRE-156; none until it is loaded; a client reads it as optional: a service from before the field sends none, read as words and sounds false) and its WAV `sample_rate` (a
  Piper voice's is 0 until it is loaded). A client reads each id as a string, so a voice id it
  does not know, under a model and in languages it knows, does not break its reading of
  health.
- WIRE-126: A voice's `ready` is true exactly when a request for it
  would be spoken now: never during the start load or once the drain began, and
  never for a GPU voice that is not loaded, failed, or whose engine process died.
- WIRE-127: The service builds the answer through `Health` in `speech_client.wire`,
  which ignores keys it does not know.
- WIRE-128: `hearing.model` is the configured hearing
  model, or `unavailable` when none is configured;
- WIRE-129: `hearing.ready` is false while
  it loads, once the drain began and for good when its load found no CUDA device or no room on the card,
  which leaves speaking up.

## Not offered

- WIRE-130: Version 1 has no voice administration: `/v1/voices` answers
  `not_found`, and the service mounts no `/hear` and no `/voices*` route at all.
- WIRE-131: The service serves no API documentation: `/docs`, `/redoc` and
  `/openapi.json` answer `404`.
