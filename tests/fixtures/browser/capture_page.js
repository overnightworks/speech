// The stand-in app's page: its switch starts and stops the package on a stream the page owns.
// The query names the scenario: `source` (device, wav, tone, two-tones, none), `hop` (the relay
// route), `worklet` (file, blob, held), `start` (click, load) and the message knobs: `messages`
// (set: pass `onMessage`), `stopAt` (call stop() at that message) and `throwFirst` (set: the first
// call throws). `window.capturePage` holds what the page saw, for the test to read.
import { startCapture } from "/dist/index.js";

const query = new URLSearchParams(location.search);
const WORKLET_PATHS = { file: "/dist/capture-worklet.js", held: "/held-worklet.js" };
const TWO_TONES = [440, 1250];

const seen = {
  started: null,
  ended: null,
  endedCount: 0,
  errors: [],
  sentBeforeEnd: null,
  socketsOpened: 0,
  contextsOpened: 0,
  sentBytes: [],
  messagesReceived: 0,
  messages: [],
  startedAtFirstMessage: null,
  messagesAfterEnded: 0,
  packageGetUserMediaCalls: 0,
  openedAt: null,
  startedAt: null,
  firstSendAt: null,
  contextStatesAtStart: [],
  stopToEndedMs: null,
  socketStatesAtEnd: [],
  contextStatesAtEnd: [],
  trackState: null,
  trackTones: [],
};
window.capturePage = seen;
const sockets = [];
const contexts = [];

addEventListener("error", (event) => seen.errors.push(String(event.message)));
addEventListener("unhandledrejection", (event) => seen.errors.push(String(event.reason)));

// What the package opens goes through these globals; the page itself keeps the originals.
const PageWebSocket = WebSocket;
const PageAudioContext = AudioContext;
window.WebSocket = class extends PageWebSocket {
  constructor(...parameters) {
    super(...parameters);
    sockets.push(this);
    seen.socketsOpened += 1;
    this.addEventListener("open", () => (seen.openedAt = performance.now()));
    this.addEventListener("message", () => (seen.messagesReceived += 1));
  }

  send(data) {
    seen.firstSendAt ??= performance.now();
    seen.sentBytes.push(data.byteLength);
    super.send(data);
  }
};
window.AudioContext = class extends PageAudioContext {
  constructor(...parameters) {
    super(...parameters);
    contexts.push(this);
    seen.contextsOpened += 1;
  }
};
const pageGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
navigator.mediaDevices.getUserMedia = (...parameters) => {
  seen.packageGetUserMediaCalls += 1;
  return pageGetUserMedia(...parameters);
};

function toneTrack(context, frequency) {
  const oscillator = new OscillatorNode(context, { frequency });
  const gain = new GainNode(context, { gain: 0.2 });
  const destination = new MediaStreamAudioDestinationNode(context);
  oscillator.connect(gain).connect(destination);
  oscillator.start();
  return destination.stream.getAudioTracks()[0];
}

// The audio of the stream, and what the page does once the capture streams.
async function prepareSource() {
  const kind = query.get("source");
  if (kind === "device") {
    return { stream: await pageGetUserMedia({ audio: true }) };
  }
  if (kind === "none") {
    return { stream: new MediaStream() };
  }
  const context = new PageAudioContext();
  if (kind === "tone") {
    return { stream: new MediaStream([toneTrack(context, Number(query.get("tone") ?? 300))]) };
  }
  if (kind === "two-tones") {
    // Ordered so that the first track's id sorts higher: a source node that picks the lowest id takes the other.
    const tracks = TWO_TONES.map((frequency) => ({ frequency, track: toneTrack(context, frequency) }));
    tracks.sort((one, other) => (one.track.id < other.track.id ? 1 : -1));
    seen.trackTones = tracks.map(({ frequency }) => frequency);
    return { stream: new MediaStream(tracks.map(({ track }) => track)) };
  }
  const recording = await context.decodeAudioData(await (await fetch("/fixture.wav")).arrayBuffer());
  const player = new AudioBufferSourceNode(context, { buffer: recording });
  const destination = new MediaStreamAudioDestinationNode(context);
  player.connect(destination);
  return { stream: destination.stream, onStreaming: () => player.start() };
}

async function workletUrl() {
  if (query.get("worklet") !== "blob") {
    return WORKLET_PATHS[query.get("worklet") ?? "file"];
  }
  const source = await (await fetch(WORKLET_PATHS.file)).text();
  return URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
}

function messageRecord(data) {
  if (typeof data === "string") return { kind: "text", text: data };
  if (data instanceof ArrayBuffer) return { kind: "binary", bytes: Array.from(new Uint8Array(data)) };
  return { kind: typeof data };
}

// The page's `onMessage`: it keeps what it was handed and acts on the knobs.
function onMessage(data) {
  seen.startedAtFirstMessage ??= seen.started === null ? "pending" : seen.started;
  if (seen.ended !== null) seen.messagesAfterEnded += 1;
  seen.messages.push(messageRecord(data));
  if (query.has("throwFirst") && seen.messages.length === 1) throw new Error("the page's callback threw");
  if (seen.messages.length === Number(query.get("stopAt"))) capture.stop();
}

let capture = null;
let track = null;
let stoppedAt = null;

async function switchOn() {
  const { stream, onStreaming } = await prepareSource();
  track = stream.getAudioTracks()[0] ?? null;
  capture = startCapture({
    stream,
    hopUrl: new URL(query.get("hop") ?? "/hear", location.href.replace(/^http/, "ws")),
    workletUrl: await workletUrl(),
    ...(query.has("messages") ? { onMessage } : {}),
  });
  capture.started.then((started) => {
    seen.started = started;
    seen.startedAt = performance.now();
    seen.contextStatesAtStart = contexts.map((context) => context.state);
    if (started === "streaming") onStreaming?.();
  });
  capture.ended.then((end) => {
    seen.ended = end;
    seen.endedCount += 1;
    if (stoppedAt !== null) seen.stopToEndedMs = performance.now() - stoppedAt;
    seen.socketStatesAtEnd = sockets.map((socket) => socket.readyState);
    seen.contextStatesAtEnd = contexts.map((context) => context.state);
    seen.trackState = track?.readyState ?? null;
  });
}

function switchOff() {
  stoppedAt = performance.now();
  capture.stop();
}

window.endTrack = () => {
  seen.sentBeforeEnd = seen.sentBytes.length;
  track.dispatchEvent(new Event("ended"));
};

document.getElementById("switch").addEventListener("click", () => (capture === null ? switchOn() : switchOff()));
if (query.get("start") === "load") {
  void switchOn();
}
