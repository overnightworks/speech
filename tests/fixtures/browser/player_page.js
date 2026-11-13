// The stand-in app's page for the player. `window.playerPage` is what the test drives and reads.
//
// The measuring instrument wraps the Web Audio entry points before the page creates its player, and keeps
// the originals: what it wraps it records (edges to the output, sources, decodes, resumes), and its own
// analyser, built with the originals, reads the energy of what reaches the output.
import * as entry from "/dist/index.js";

const Page = {
  AnalyserNode,
  AudioBufferSourceNode,
  AudioContext,
  AudioDestinationNode,
  GainNode,
  connect: AudioNode.prototype.connect,
  createAnalyser: BaseAudioContext.prototype.createAnalyser,
  createBufferSource: BaseAudioContext.prototype.createBufferSource,
  decodeAudioData: BaseAudioContext.prototype.decodeAudioData,
  resume: AudioContext.prototype.resume,
  start: AudioBufferSourceNode.prototype.start,
};
const SAMPLE_MS = 20;
const SILENCE_POLL_MS = 10;
const SILENCE_GIVE_UP_MS = 5000;
const JUNK_BYTES = 512;
const KINDS = [
  ["GainNode", Page.GainNode],
  ["AudioBufferSourceNode", Page.AudioBufferSourceNode],
  ["AudioDestinationNode", Page.AudioDestinationNode],
  ["AnalyserNode", Page.AnalyserNode],
];

const seen = {
  exportNames: Object.keys(entry),
  sounds: [],
  log: [],
  errors: [],
  edges: [],
  bytes: {},
  analysersCreated: 0,
  sourcesCreated: 0,
  sourcesStarted: 0,
  sourcesEnded: 0,
  instrumentStops: 0,
  decodesCalled: 0,
  decodesSettled: 0,
  resumesCalled: 0,
  lastActionAt: null,
  endedAtAction: null,
  peakRms: 0,
};
const contexts = [];
const meters = new WeakMap();
const nodeIds = new WeakMap();
const trackedSources = new WeakSet();
const held = { decode: [], resume: [] };
const heldNow = [];
const byteStore = new Map();
let nextNodeId = 0;
let stopOnSourceEnd = false;
let rejectNextResume = false;
let outputGain = null;
let player = null;

addEventListener("error", (event) => seen.errors.push(String(event.message)));
addEventListener("unhandledrejection", (event) => seen.errors.push(String(event.reason)));

function idOf(node) {
  if (!nodeIds.has(node)) nodeIds.set(node, nextNodeId++);
  return nodeIds.get(node);
}

function kindOf(node) {
  return KINDS.find(([, constructor]) => node instanceof constructor)?.[0] ?? node?.constructor?.name;
}

function meterOf(context) {
  if (!meters.has(context)) meters.set(context, new Page.AnalyserNode(context));
  return meters.get(context);
}

function trackSource(source) {
  if (trackedSources.has(source)) return source;
  trackedSources.add(source);
  seen.sourcesCreated += 1;
  // Added at creation, so it runs before the `onended` the package sets.
  source.addEventListener("ended", () => {
    seen.sourcesEnded += 1;
    if (stopOnSourceEnd) {
      stopOnSourceEnd = false;
      seen.instrumentStops += 1;
      player.stop();
    }
  });
  return source;
}

function gateFor(kind, result) {
  const gate = held[kind].shift();
  if (gate === undefined) return result;
  heldNow.push(gate);
  return gate.promise.then(() => result);
}

window.AudioContext = class AudioContext extends Page.AudioContext {
  constructor(...parameters) {
    super(...parameters);
    contexts.push(this);
  }
};
window.AudioBufferSourceNode = class AudioBufferSourceNode extends Page.AudioBufferSourceNode {
  constructor(...parameters) {
    super(...parameters);
    trackSource(this);
  }
};
window.AnalyserNode = class AnalyserNode extends Page.AnalyserNode {
  constructor(...parameters) {
    super(...parameters);
    seen.analysersCreated += 1;
  }
};
BaseAudioContext.prototype.createBufferSource = function () {
  return trackSource(Page.createBufferSource.call(this));
};
BaseAudioContext.prototype.createAnalyser = function () {
  seen.analysersCreated += 1;
  return Page.createAnalyser.call(this);
};
Page.AudioBufferSourceNode.prototype.start = function (...parameters) {
  seen.sourcesStarted += 1;
  return Page.start.apply(this, parameters);
};
AudioNode.prototype.connect = function (target, ...parameters) {
  const toOutput = target instanceof Page.AudioDestinationNode;
  seen.edges.push({ from: idOf(this), fromKind: kindOf(this), to: target instanceof AudioNode ? idOf(target) : null, toKind: kindOf(target) });
  if (toOutput) {
    outputGain = this;
    Page.connect.call(this, meterOf(this.context));
  }
  return Page.connect.call(this, target, ...parameters);
};
BaseAudioContext.prototype.decodeAudioData = function (...parameters) {
  seen.decodesCalled += 1;
  const result = Page.decodeAudioData.apply(this, parameters);
  const settle = () => (seen.decodesSettled += 1);
  result.then(settle, settle);
  return gateFor("decode", result);
};
AudioContext.prototype.resume = function () {
  seen.resumesCalled += 1;
  if (rejectNextResume) {
    rejectNextResume = false;
    return Promise.reject(new DOMException("refused", "NotAllowedError"));
  }
  return gateFor("resume", Page.resume.call(this));
};

function rms() {
  const meter = meters.get(contexts[0]);
  if (meter === undefined) return 0;
  const samples = new Float32Array(meter.fftSize);
  meter.getFloatTimeDomainData(samples);
  return Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
}

setInterval(() => {
  const level = rms();
  seen.peakRms = Math.max(seen.peakRms, level);
  for (const sound of seen.sounds) {
    if (sound.startedValue === "playing" && sound.endedValue === null) sound.peakRms = Math.max(sound.peakRms, level);
  }
}, SAMPLE_MS);

function argumentOf(spec) {
  if (spec.url !== undefined) return spec.url;
  if (spec.urlObject !== undefined) return new URL(spec.urlObject, location.href);
  if (spec.bytes !== undefined) return byteStore.get(spec.bytes);
  return new Uint8Array(JUNK_BYTES).fill(7).buffer;
}

function handOver(spec) {
  const sound = { id: seen.sounds.length, startedValue: null, startedCount: 0, endedValue: null, endedCount: 0, endedAt: null, peakRms: 0 };
  seen.sounds.push(sound);
  const playback = player.play(argumentOf(spec));
  playback.started.then((value) => {
    sound.startedValue = value;
    sound.startedCount += 1;
    seen.log.push({ event: "started", id: sound.id, value, at: performance.now() });
  });
  playback.ended.then((end) => {
    sound.endedValue = end.kind;
    sound.endedCount += 1;
    sound.endedAt = performance.now();
    seen.log.push({ event: "ended", id: sound.id, value: end.kind, at: sound.endedAt });
    if (spec.bytes !== undefined) seen.bytes[spec.bytes].after = byteStore.get(spec.bytes).byteLength;
    window.noteEnded?.(sound.id);
  });
  return sound.id;
}

// A switch of the app: it records when it happened and what had ended by the end of its own task.
async function act(action) {
  seen.lastActionAt = performance.now();
  action();
  await Promise.resolve();
  seen.endedAtAction = seen.sounds.map((sound) => sound.endedValue);
}

window.playerPage = {
  seen,
  play: (...specs) => specs.map(handOver),
  prepareBytes: async (key, url) => {
    const bytes = await (await fetch(url)).arrayBuffer();
    byteStore.set(key, bytes);
    seen.bytes[key] = { before: bytes.byteLength, after: null };
  },
  stop: () => act(() => player.stop()),
  mute: (muted) => act(() => player.setMuted(muted)),
  close: () => act(() => player.close()),
  hold: (kind) => held[kind].push(Promise.withResolvers()),
  release: () => heldNow.splice(0).forEach((gate) => gate.resolve()),
  rejectNextResume: () => (rejectNextResume = true),
  stopOnNextSourceEnd: () => (stopOnSourceEnd = true),
  snapshot: () => ({
    ...seen,
    contextStates: contexts.map((context) => context.state),
    gainValue: outputGain?.gain.value ?? null,
    hasBeenActive: navigator.userActivation.hasBeenActive,
  }),
  async silentAfterAction(threshold) {
    const start = performance.now();
    while (rms() > threshold && performance.now() - start < SILENCE_GIVE_UP_MS) {
      await new Promise((resolve) => setTimeout(resolve, SILENCE_POLL_MS));
    }
    return performance.now() - seen.lastActionAt;
  },
};
player = entry.createPlayer();

// A sound handed over while the page loads, before any click: the autoplay policy refuses it.
// The test learns of its end through `window.noteEnded`, because any call of the test into the page counts as a click.
const atLoad = new URLSearchParams(location.search).get("handOver");
if (atLoad !== null) handOver({ url: atLoad });
