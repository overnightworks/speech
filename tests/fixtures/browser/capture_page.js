// The stand-in app's page: its switch starts and stops the package on a stream the page owns.
// The query names the scenario: `source` (device, wav, tone, two-tones, none), `hop` (the relay
// route), `worklet` (file, blob) and `start` (click, load). `window.capturePage` holds what the
// page saw, for the test to read.
import { startCapture } from "/dist/capture.js";

const query = new URLSearchParams(location.search);
const WORKLET_PATH = "/dist/capture-worklet.js";
const TWO_TONES = [440, 1250];

const seen = {
  started: null,
  ended: null,
  endedCount: 0,
  errors: [],
  violations: [],
  socketsOpened: 0,
  contextsOpened: 0,
  sentBytes: [],
  messagesReceived: 0,
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
document.addEventListener("securitypolicyviolation", (event) =>
  seen.violations.push({ directive: event.effectiveDirective, blocked: event.blockedURI }),
);

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
    return WORKLET_PATH;
  }
  const source = await (await fetch(WORKLET_PATH)).text();
  return URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
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

window.endTrack = () => track.dispatchEvent(new Event("ended"));

document.getElementById("switch").addEventListener("click", () => (capture === null ? switchOn() : switchOff()));
if (query.get("start") === "load") {
  void switchOn();
}
